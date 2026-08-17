"""Phase 2b: lost-in-the-middle — does gating recover buried evidence?

The one retrieval regime where evidence-use demonstrably BREAKS (Liu et al.
2023): the gold passage sits in the middle of a long context and the model
ignores it. Headroom is guaranteed by construction, so this is the fair test
of whether precision-gated attention converts routing into accuracy.

Design: SQuAD questions, gold passage among 7 distractors (8 passages,
~2.5-3k tokens). Gold placed FIRST, MIDDLE, or LAST. For each placement:
base generation vs gated generation (bias toward the whole retrieved block,
strength gamma*(1-pi)). The paper figure: accuracy-by-position curves —
base should sag in the middle; gating should flatten the sag.

Run from the repo root:
    caffeinate -i python experiments/run_phase2b_lostmiddle.py --n 200

~6 ops/item with long prompts: expect ~3-5h for n=200 on MPS.
Resumable (checkpoints every item).
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
GATE_LAYERS = range(16, 28)
READ_LAYER = -6
PREFIX = "Answer the question using the context.\nContext: "
N_PASSAGES = 8                      # 1 gold + 7 distractors
POSITIONS = {"first": 0, "middle": N_PASSAGES // 2, "last": N_PASSAGES - 1}


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
        if x["context"] in seen or len(x["context"]) > 900:
            continue
        seen.add(x["context"])
        items.append(dict(question=x["question"], context=x["context"],
                          golds=golds))
        if len(items) >= n:
            break
    for j, it in enumerate(items):
        it["distractors"] = [items[(j + 1 + d) % len(items)]["context"]
                             for d in range(N_PASSAGES - 1)]
    return items


@torch.no_grad()
def gen_greedy(model, tok, prompt, device, max_new=24):
    enc = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc.input_ids.shape[1]:],
                      skip_special_tokens=True).strip().split("\n")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--gamma", type=float, default=2.0)
    args = ap.parse_args()

    out_dir = f"phase2b_lostmiddle_n{args.n}_g{args.gamma:g}"
    os.makedirs(out_dir, exist_ok=True)
    items_path = os.path.join(out_dir, "items.jsonl")

    device, dtype = device_and_dtype()
    print(f"device={device} dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=dtype, attn_implementation="eager")
    model.to(device).eval()

    arr = np.load(PHASE1_ARRAYS)
    ybar = float(np.exp(-arr["SE"][:len(arr["SE"]) // 2]).mean())
    head = FrozenRidgeHead(HEAD_NPZ, ybar=ybar)
    match = NormalizedMatch()

    items = load_items(args.n)
    print(f"{len(items)} items, {N_PASSAGES} passages each")

    done = sum(1 for _ in open(items_path)) if os.path.exists(items_path) else 0
    if done:
        print(f"Resuming at {done}")

    gate = PrecisionGate(model, GATE_LAYERS, gamma=args.gamma)
    gate.clear()

    with open(items_path, "a") as fout:
        for i in range(done, len(items)):
            it = items[i]
            q, golds = it["question"], it["golds"]
            pi = question_precision(model, tok, q, head, device, READ_LAYER)
            rec = dict(i=i, question=q, golds=golds, pi=pi)

            for name, pos in POSITIONS.items():
                passages = list(it["distractors"])
                passages.insert(pos, it["context"])
                ctx = "\n\n".join(passages)
                prompt = f"{PREFIX}{ctx}\nQ: {q}\nA:"
                span = context_token_span(tok, PREFIX, ctx)

                gate.clear()
                base = gen_greedy(model, tok, prompt, device)
                gate.set(span[0], span[1], pi)
                gated = gen_greedy(model, tok, prompt, device)
                gate.clear()

                rec[f"{name}_base"] = int(any(match(base, g) for g in golds))
                rec[f"{name}_gated"] = int(any(match(gated, g) for g in golds))

            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(items)}]")

    gate.remove()

    recs = [json.loads(l) for l in open(items_path)]
    res = dict(n=len(recs), gamma=args.gamma, n_passages=N_PASSAGES)
    for name in POSITIONS:
        b = np.mean([r[f"{name}_base"] for r in recs])
        g = np.mean([r[f"{name}_gated"] for r in recs])
        res[f"{name}_base"] = float(b)
        res[f"{name}_gated"] = float(g)
    print(json.dumps(res, indent=2))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
