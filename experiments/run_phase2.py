"""Phase 2 (H1): does routing precision into attention reduce confident errors?

Design (SQuAD v1.1 validation — real passages, real questions):
For each item we run FIVE conditions:
  1. closed-book        : question only, greedy  -> parametric answer + pi_hat
  2. clean context, base: passage + question, no gating
  3. clean context,gated: same, attention biased toward context by gamma*(1-pi)
  4. conflict, base     : passage with the answer swapped for a wrong entity
  5. conflict, gated    : same, gated

What H1 predicts:
  - On CLEAN items the model gets wrong closed-book (low pi_hat), gating
    raises open-book accuracy: evidence is up-weighted exactly where the
    parametric prior is unreliable.
  - On CONFLICT items, behavior becomes precision-governed: low pi_hat ->
    follow the context; high pi_hat -> resist it. The base model's mix is
    arbitrary; the gated model's mix tracks pi_hat.
  - Fluency guard: answers stay well-formed (rate of empty/degenerate
    outputs does not rise).

Run from the repo root:
    caffeinate -i python experiments/run_phase2.py            # n=500, ~2-3h
    python experiments/run_phase2.py --n 50                   # smoke test

Resumable: appends one JSON line per item to phase2_squad_n{N}/items.jsonl.
"""
import os, sys, json, argparse

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from noema.semantic_entropy import NormalizedMatch
from noema.gated_attention import (FrozenRidgeHead, PrecisionGate,
                                   context_token_span, question_precision)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HEAD_NPZ = "phase1_triviaqa_n2000/ridge_head.npz"
PHASE1_ARRAYS = "phase1_triviaqa_n2000/arrays.npz"
GATE_LAYERS = range(16, 28)     # from the layer sweep: knowing-plateau onward
READ_LAYER = -6                 # matches the frozen head's training layer
PREFIX = "Answer the question using the context.\nContext: "


def device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_squad(n, distractors=0, seed=0):
    """Items with a conflict variant (answer span swapped for a donor answer)
    and, optionally, distractor passages mixed in — a realistic retrieval
    setting where the model must FIND the evidence, not just copy it."""
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split="validation")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))
    items, seen_ctx = [], set()
    max_len = 1200 if distractors else 2200
    for i in idx:
        x = ds[int(i)]
        golds = list(dict.fromkeys(x["answers"]["text"]))
        ans = golds[0]
        # one question per passage; answer must appear verbatim to swap
        if x["context"] in seen_ctx or ans not in x["context"]:
            continue
        if len(x["context"]) > max_len:   # keep prompts small for MPS
            continue
        seen_ctx.add(x["context"])
        items.append(dict(question=x["question"], context=x["context"],
                          golds=golds, answer=ans))
        if len(items) >= n:
            break
    # donor answers: shift by one so donor != own answer
    for j, it in enumerate(items):
        donor = items[(j + 1) % len(items)]["answer"]
        if donor.lower() == it["answer"].lower():
            donor = items[(j + 2) % len(items)]["answer"]
        gold_ctx = it["context"]
        confl_ctx = gold_ctx.replace(it["answer"], donor)
        if distractors:
            # distractor passages from other items; gold position random
            pool = [items[(j + 3 + d) % len(items)]["context"]
                    for d in range(distractors)]
            pos = int(rng.integers(0, distractors + 1))
            clean_parts = pool[:pos] + [gold_ctx] + pool[pos:]
            confl_parts = pool[:pos] + [confl_ctx] + pool[pos:]
            it["context"] = "\n\n".join(clean_parts)
            it["conflict_context"] = "\n\n".join(confl_parts)
        else:
            it["conflict_context"] = confl_ctx
        it["donor"] = donor
    return items


@torch.no_grad()
def gen_greedy(model, tok, prompt, device, max_new=24):
    enc = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc.input_ids.shape[1]:],
                      skip_special_tokens=True).strip().split("\n")[0]


def open_book_prompt(context, question):
    return f"{PREFIX}{context}\nQ: {question}\nA:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--distractors", type=int, default=2)
    ap.add_argument("--max-pi", type=float, default=1.0,
                    help="enrichment: run the full battery only on items with "
                         "pi_hat below this (screening pass is ~2s/item)")
    args = ap.parse_args()

    out_dir = (f"phase2_squad_n{args.n}_g{args.gamma:g}"
               f"_d{args.distractors}")
    if args.max_pi < 1.0:
        out_dir += f"_pi{args.max_pi:g}"
    os.makedirs(out_dir, exist_ok=True)
    items_path = os.path.join(out_dir, "items.jsonl")

    device, dtype = device_and_dtype()
    print(f"device={device} dtype={dtype} gamma={args.gamma} "
          f"distractors={args.distractors}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=dtype, attn_implementation="eager")
    model.to(device).eval()

    # frozen Phase-1 head; recover ybar from the phase-1 train half
    arr = np.load(PHASE1_ARRAYS)
    ybar = float(np.exp(-arr["SE"][:len(arr["SE"]) // 2]).mean())
    head = FrozenRidgeHead(HEAD_NPZ, ybar=ybar)
    match = NormalizedMatch()

    items = load_squad(args.n, distractors=args.distractors)
    print(f"{len(items)} SQuAD items with conflict variants")

    done = 0
    if os.path.exists(items_path):
        done = sum(1 for _ in open(items_path))
        print(f"Resuming at item {done}")

    gate = PrecisionGate(model, GATE_LAYERS, gamma=args.gamma)
    gate.clear()

    with open(items_path, "a") as fout:
        for i in range(done, len(items)):
            it = items[i]
            q, golds = it["question"], it["golds"]

            # 1. closed-book: pi_hat + parametric answer
            pi = question_precision(model, tok, q, head, device, READ_LAYER)
            if pi > args.max_pi:            # enrichment screen: skip battery
                fout.write(json.dumps(dict(i=i, pi=pi, skipped=True)) + "\n")
                fout.flush()
                continue
            closed = gen_greedy(
                model, tok,
                f"Answer the question as briefly as possible.\nQ: {q}\nA:",
                device)

            rec = dict(i=i, question=q, golds=golds, donor=it["donor"],
                       pi=pi, closed=closed,
                       closed_correct=int(any(match(closed, g) for g in golds)))

            # 2-5. open-book: clean/conflict x base/gated
            for cond, ctx in [("clean", it["context"]),
                              ("conflict", it["conflict_context"])]:
                prompt = open_book_prompt(ctx, q)
                span = context_token_span(tok, PREFIX, ctx)

                gate.clear()
                base = gen_greedy(model, tok, prompt, device)

                gate.set(span[0], span[1], pi)
                gated = gen_greedy(model, tok, prompt, device)
                gate.clear()

                rec[f"{cond}_base"] = base
                rec[f"{cond}_gated"] = gated
                if cond == "clean":
                    rec["clean_base_correct"] = int(any(match(base, g) for g in golds))
                    rec["clean_gated_correct"] = int(any(match(gated, g) for g in golds))
                else:
                    rec["conflict_base_follows"] = int(match(base, it["donor"]))
                    rec["conflict_gated_follows"] = int(match(gated, it["donor"]))
                    rec["conflict_base_parametric"] = int(any(match(base, g) for g in golds))
                    rec["conflict_gated_parametric"] = int(any(match(gated, g) for g in golds))

            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(items)}] pi={pi:.2f} "
                      f"closed={rec['closed_correct']} "
                      f"base={rec['clean_base_correct']} "
                      f"gated={rec['clean_gated_correct']}")

    gate.remove()
    analyze(items_path, out_dir, args.gamma)


def analyze(items_path, out_dir, gamma):
    recs = [json.loads(l) for l in open(items_path)]
    n_scanned = len(recs)
    recs = [r for r in recs if not r.get("skipped")]
    print(f"scanned {n_scanned}, kept {len(recs)} after pi screen")
    pi = np.array([r["pi"] for r in recs])
    lo, hi = np.quantile(pi, [1 / 3, 2 / 3])
    terc = np.digitize(pi, [lo, hi])          # 0 = low pi, 2 = high pi

    def rate(key, mask=None):
        v = np.array([r[key] for r in recs])
        return float(v.mean() if mask is None else v[mask].mean())

    res = dict(n=len(recs), gamma=gamma,
               pi_terciles=[float(lo), float(hi)],
               closed_book_acc=rate("closed_correct"),
               clean_acc_base=rate("clean_base_correct"),
               clean_acc_gated=rate("clean_gated_correct"),
               empty_base=float(np.mean([r["clean_base"] == "" for r in recs])),
               empty_gated=float(np.mean([r["clean_gated"] == "" for r in recs])))

    for t, name in [(0, "low_pi"), (1, "mid_pi"), (2, "high_pi")]:
        m = terc == t
        res[f"clean_base_{name}"] = rate("clean_base_correct", m)
        res[f"clean_gated_{name}"] = rate("clean_gated_correct", m)
        res[f"conflict_follows_base_{name}"] = rate("conflict_base_follows", m)
        res[f"conflict_follows_gated_{name}"] = rate("conflict_gated_follows", m)

    # H1 headline: accuracy on items the model does NOT know (closed wrong)
    unk = np.array([r["closed_correct"] == 0 for r in recs])
    res["unknown_items_frac"] = float(unk.mean())
    res["clean_base_on_unknown"] = rate("clean_base_correct", unk)
    res["clean_gated_on_unknown"] = rate("clean_gated_correct", unk)

    # paired bootstrap for the gated-minus-base difference on unknown items
    d = (np.array([r["clean_gated_correct"] for r in recs])[unk]
         - np.array([r["clean_base_correct"] for r in recs])[unk])
    rng = np.random.default_rng(0)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    res["gated_minus_base_unknown"] = float(d.mean())
    res["gated_minus_base_unknown_ci"] = [float(np.percentile(boots, 2.5)),
                                          float(np.percentile(boots, 97.5))]

    print(json.dumps(res, indent=2))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
