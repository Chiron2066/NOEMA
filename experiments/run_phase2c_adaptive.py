"""Phase 2c: precision-gated ADAPTIVE RETRIEVAL — pi_hat governing action.

The cleanest accuracy-relevant use of the precision signal: when no context
is present, low pi_hat should TRIGGER retrieval; high pi_hat should skip it.
This is precision weighting at the policy level (the gate's 'retrieve'
action, H1/H4), and it needs self-contained questions where the model
legitimately knows a chunk (TriviaQA, ~40%), not context-dependent SQuAD.

Uses TriviaQA rc.wikipedia (validation): each question ships with real
Wikipedia evidence — the 'retrieval' arm uses the first evidence document
(truncated). Per item: pi_hat + closed-book answer + open-book answer.

Endpoints:
  - accuracy/retrieval-rate curve of the policy 'retrieve iff pi_hat < tau'
  - vs closed-book-always, retrieve-always, and the per-item oracle
  - headline: % of retrieval calls saved at <=1 point accuracy loss

Run from the repo root (first run downloads the rc.wikipedia config):
    caffeinate -i python experiments/run_phase2c_adaptive.py --n 800

~3 ops/item: expect ~2-3h for n=800 on MPS. Resumable.
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
from noema.gated_attention import FrozenRidgeHead, question_precision

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HEAD_NPZ = "phase1_triviaqa_n2000/ridge_head.npz"
PHASE1_ARRAYS = "phase1_triviaqa_n2000/arrays.npz"
READ_LAYER = -6
CTX_CHARS = 2500


def device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_items(n, seed=0):
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia",
                      split="validation")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))
    items = []
    for i in idx:
        x = ds[int(i)]
        wiki = x["entity_pages"]["wiki_context"]
        if not wiki:
            continue
        golds = [x["answer"]["value"]] + list(x["answer"].get("aliases", []))
        items.append(dict(question=x["question"], golds=golds[:20],
                          evidence=wiki[0][:CTX_CHARS]))
        if len(items) >= n:
            break
    return items


@torch.no_grad()
def gen_greedy(model, tok, prompt, device, max_new=32):
    enc = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc.input_ids.shape[1]:],
                      skip_special_tokens=True).strip().split("\n")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    args = ap.parse_args()

    out_dir = f"phase2c_adaptive_n{args.n}"
    os.makedirs(out_dir, exist_ok=True)
    items_path = os.path.join(out_dir, "items.jsonl")

    device, dtype = device_and_dtype()
    print(f"device={device} dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype)
    model.to(device).eval()

    arr = np.load(PHASE1_ARRAYS)
    ybar = float(np.exp(-arr["SE"][:len(arr["SE"]) // 2]).mean())
    head = FrozenRidgeHead(HEAD_NPZ, ybar=ybar)
    match = NormalizedMatch()

    items = load_items(args.n)
    print(f"{len(items)} TriviaQA items with Wikipedia evidence")

    done = sum(1 for _ in open(items_path)) if os.path.exists(items_path) else 0
    if done:
        print(f"Resuming at {done}")

    with open(items_path, "a") as fout:
        for i in range(done, len(items)):
            it = items[i]
            q, golds = it["question"], it["golds"]
            pi = question_precision(model, tok, q, head, device, READ_LAYER)
            closed = gen_greedy(
                model, tok,
                f"Answer the question as briefly as possible.\nQ: {q}\nA:",
                device)
            opened = gen_greedy(
                model, tok,
                f"Answer the question using the context.\n"
                f"Context: {it['evidence']}\nQ: {q}\nA:",
                device)
            rec = dict(
                i=i, pi=pi,
                closed_correct=int(any(match(closed, g) for g in golds)),
                open_correct=int(any(match(opened, g) for g in golds)))
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if (i + 1) % 25 == 0:
                print(f"[{i+1}/{len(items)}] pi={pi:.2f} "
                      f"closed={rec['closed_correct']} open={rec['open_correct']}")

    analyze(items_path, out_dir)


def analyze(items_path, out_dir):
    recs = [json.loads(l) for l in open(items_path)]
    pi = np.array([r["pi"] for r in recs])
    closed = np.array([r["closed_correct"] for r in recs])
    opened = np.array([r["open_correct"] for r in recs])

    res = dict(n=len(recs),
               closed_always=float(closed.mean()),
               retrieve_always=float(opened.mean()),
               oracle=float(np.maximum(closed, opened).mean()))
    curve = []
    for tau in np.arange(0.05, 1.001, 0.05):
        use = pi < tau
        curve.append(dict(tau=round(float(tau), 2),
                          acc=float(np.where(use, opened, closed).mean()),
                          retrieval_rate=float(use.mean())))
    res["curve"] = curve

    # headline: max retrieval saved with accuracy within 1pt of always-retrieve
    ok = [c for c in curve if c["acc"] >= res["retrieve_always"] - 0.01]
    if ok:
        best = min(ok, key=lambda c: c["retrieval_rate"])
        res["headline"] = dict(
            tau=best["tau"], acc=best["acc"],
            retrieval_rate=best["retrieval_rate"],
            retrieval_saved=round(1 - best["retrieval_rate"], 3))
    print(json.dumps(res, indent=2))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
