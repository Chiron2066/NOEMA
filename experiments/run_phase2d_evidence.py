"""Phase 2d: BOTH knobs — prior precision x evidence precision.

Phase 2's attention nulls used half the mechanism: a uniform bias over the
whole retrieved block, junk distractors included. Full precision weighting
scales trust in each evidence source by its own reliability. Here:

  evidence precision : per-passage relevance to the question (content-word
                       overlap — cheap, transparent, no gold leakage)
  gate               : attention bias ONLY on the top-scoring passage,
                       strength gamma * (1 - pi_hat) as before

Three arms on identical prompts (SQuAD + 3 distractors, clean contexts):
  base            : no gating
  block-gated     : Phase-2 uniform bias (the arm that went null)
  evidence-gated  : bias restricted to the most relevant passage

Target failure mode: answers extracted from the WRONG passage. Diagnostics
recorded per item: which passage the scorer picked (vs gold), and whether
the base answer text appears in a distractor rather than the gold passage.

Run from the repo root:
    caffeinate -i python experiments/run_phase2d_evidence.py --n 400
~5 ops/item: expect ~3-4h on MPS. Resumable.
"""
import os, sys, json, argparse, re, string

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

from noema.semantic_entropy import NormalizedMatch, normalize
from noema.gated_attention import (FrozenRidgeHead, PrecisionGate,
                                   question_precision)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HEAD_NPZ = "phase1_triviaqa_n2000/ridge_head.npz"
PHASE1_ARRAYS = "phase1_triviaqa_n2000/arrays.npz"
GATE_LAYERS = range(16, 28)
READ_LAYER = -6
PREFIX = "Answer the question using the context.\nContext: "
N_DISTRACTORS = 3

STOP = set("the a an of in on at to for with by from is are was were be been "
           "what which who whom whose when where why how did do does done "
           "and or not no it its this that these those as his her their".split())


def content_words(text):
    words = re.sub(f"[{re.escape(string.punctuation)}]", " ", text.lower()).split()
    return {w for w in words if w not in STOP and len(w) > 2}


def relevance(question, passage):
    q, p = content_words(question), content_words(passage)
    return len(q & p) / max(len(q), 1)


def device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_items(n, seed=0):
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split="validation")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))
    items, seen = [], set()
    for i in idx:
        x = ds[int(i)]
        golds = list(dict.fromkeys(x["answers"]["text"]))
        if x["context"] in seen or len(x["context"]) > 1200:
            continue
        seen.add(x["context"])
        items.append(dict(question=x["question"], gold_passage=x["context"],
                          golds=golds))
        if len(items) >= n:
            break
    for j, it in enumerate(items):
        distr = [items[(j + 1 + d) % len(items)]["gold_passage"]
                 for d in range(N_DISTRACTORS)]
        pos = int(rng.integers(0, N_DISTRACTORS + 1))
        passages = distr[:pos] + [it["gold_passage"]] + distr[pos:]
        it["passages"], it["gold_idx"] = passages, pos
    return items


def passage_token_spans(tok, passages):
    """Token span of each passage inside PREFIX + '\\n\\n'.join(passages)."""
    spans, sep = [], "\n\n"
    for k in range(len(passages)):
        pre = PREFIX + sep.join(passages[:k]) + (sep if k else "")
        start = len(tok(pre).input_ids)
        end = len(tok(pre + passages[k]).input_ids)
        spans.append((start, end))
    return spans


@torch.no_grad()
def gen_greedy(model, tok, prompt, device, max_new=24):
    enc = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc.input_ids.shape[1]:],
                      skip_special_tokens=True).strip().split("\n")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--gamma", type=float, default=2.0)
    args = ap.parse_args()

    out_dir = f"phase2d_evidence_n{args.n}_g{args.gamma:g}"
    os.makedirs(out_dir, exist_ok=True)
    items_path = os.path.join(out_dir, "items.jsonl")

    device, dtype = device_and_dtype()
    print(f"device={device} dtype={dtype} gamma={args.gamma}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=dtype, attn_implementation="eager")
    model.to(device).eval()

    arr = np.load(PHASE1_ARRAYS)
    ybar = float(np.exp(-arr["SE"][:len(arr["SE"]) // 2]).mean())
    head = FrozenRidgeHead(HEAD_NPZ, ybar=ybar)
    match = NormalizedMatch()

    items = load_items(args.n)
    print(f"{len(items)} items, {N_DISTRACTORS + 1} passages each")

    done = sum(1 for _ in open(items_path)) if os.path.exists(items_path) else 0
    if done:
        print(f"Resuming at {done}")

    gate = PrecisionGate(model, GATE_LAYERS, gamma=args.gamma)
    gate.clear()

    with open(items_path, "a") as fout:
        for i in range(done, len(items)):
            it = items[i]
            q, golds, passages = it["question"], it["golds"], it["passages"]
            ctx = "\n\n".join(passages)
            prompt = f"{PREFIX}{ctx}\nQ: {q}\nA:"
            spans = passage_token_spans(tok, passages)

            pi = question_precision(model, tok, q, head, device, READ_LAYER)
            closed = gen_greedy(
                model, tok,
                f"Answer the question as briefly as possible.\nQ: {q}\nA:",
                device)

            scores = [relevance(q, p) for p in passages]
            top = int(np.argmax(scores))

            gate.clear()
            base = gen_greedy(model, tok, prompt, device)
            gate.set(spans[0][0], spans[-1][1], pi)       # whole block
            block = gen_greedy(model, tok, prompt, device)
            gate.set(spans[top][0], spans[top][1], pi)    # top passage only
            evid = gen_greedy(model, tok, prompt, device)
            gate.clear()

            nb = normalize(base)
            rec = dict(
                i=i, pi=pi, gold_idx=it["gold_idx"], scorer_top=top,
                scorer_hit=int(top == it["gold_idx"]),
                closed_correct=int(any(match(closed, g) for g in golds)),
                base_correct=int(any(match(base, g) for g in golds)),
                block_correct=int(any(match(block, g) for g in golds)),
                evid_correct=int(any(match(evid, g) for g in golds)),
                base_from_distractor=int(bool(nb) and any(
                    nb in normalize(p) for k, p in enumerate(passages)
                    if k != it["gold_idx"]) and nb not in
                    normalize(it["gold_passage"])),
            )
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(items)}] pi={pi:.2f} scorer_hit="
                      f"{rec['scorer_hit']} base={rec['base_correct']} "
                      f"evid={rec['evid_correct']}")

    gate.remove()
    analyze(items_path, out_dir, args.gamma)


def analyze(items_path, out_dir, gamma):
    from math import comb
    recs = [json.loads(l) for l in open(items_path)]

    def mcnemar(a, b):
        a, b = np.array(a), np.array(b)
        wr = int(((a == 0) & (b == 1)).sum())
        rw = int(((a == 1) & (b == 0)).sum())
        n = wr + rw
        p = 1.0 if n == 0 else min(1.0, sum(
            comb(n, k) for k in range(min(wr, rw) + 1)) * 2 / 2 ** n)
        return wr, rw, p

    base = [r["base_correct"] for r in recs]
    block = [r["block_correct"] for r in recs]
    evid = [r["evid_correct"] for r in recs]
    res = dict(n=len(recs), gamma=gamma,
               scorer_top1=float(np.mean([r["scorer_hit"] for r in recs])),
               base_acc=float(np.mean(base)),
               block_gated_acc=float(np.mean(block)),
               evidence_gated_acc=float(np.mean(evid)),
               base_from_distractor_rate=float(
                   np.mean([r["base_from_distractor"] for r in recs])))
    for name, arm in [("evid_vs_base", evid), ("block_vs_base", block)]:
        wr, rw, p = mcnemar(base, arm)
        res[name] = dict(flips_gained=wr, flips_lost=rw, p=round(p, 4))
    wr, rw, p = mcnemar(block, evid)
    res["evid_vs_block"] = dict(flips_gained=wr, flips_lost=rw, p=round(p, 4))
    # conditional: items where the scorer found the gold passage
    hit = [r["scorer_hit"] == 1 for r in recs]
    if any(hit):
        res["on_scorer_hits"] = dict(
            n=int(np.sum(hit)),
            base=float(np.mean([b for b, h in zip(base, hit) if h])),
            evidence_gated=float(np.mean([e for e, h in zip(evid, hit) if h])))
    print(json.dumps(res, indent=2))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
