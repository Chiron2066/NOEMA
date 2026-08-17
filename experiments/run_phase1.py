"""Phase 1 on a real model: distill semantic entropy into a precision head.

Requires a GPU box:  pip install torch transformers datasets accelerate

Example:
    python experiments/run_phase1.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --dataset triviaqa --n 2000 --k 10 --layer -6

What it does (H2):
  1. For each QA item: greedy answer + K sampled answers (T=1.0).
  2. Hidden state of the last prompt token at --layer  (probe input).
  3. Bidirectional-entailment clustering (DeBERTa-MNLI) -> semantic entropy.
  4. Correctness of greedy answer vs gold (normalized match / NLI).
  5. Train TorchPrecisionHead by SE distillation (+ asymmetric grounding loss).
  6. Report: AUROC(sampled SE), AUROC(probe), ECE, AURC, gate outcomes.
"""
import argparse, json, os, sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from noema.semantic_entropy import (HFEntailment, NormalizedMatch,
                                    cluster_by_meaning, semantic_entropy)
from noema.precision_head import RidgePrecisionHead
from noema.metrics import auroc, ece, risk_coverage
from noema.gate import fit_generate_threshold, gate_outcomes


def load_qa(name: str, n: int):
    from datasets import load_dataset
    if name == "triviaqa":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        items = [(x["question"], x["answer"]["value"]) for x in ds.select(range(n))]
    elif name == "simpleqa":
        ds = load_dataset("basicv8vc/SimpleQA", split="test")
        items = [(x["problem"], x["answer"]) for x in ds.select(range(n))]
    elif name == "nq_open":
        ds = load_dataset("google-research-datasets/nq_open", split="validation")
        # gold may have several acceptable aliases; keep them all, joined
        items = [(x["question"], " ||| ".join(x["answer"])) for x in ds.select(range(n))]
    else:  # jsonl with {"question":..., "answer":...}
        items = []
        with open(name) as f:
            for line in f:
                d = json.loads(line)
                items.append((d["question"], d["answer"]))
                if len(items) >= n:
                    break
    return items


@torch.no_grad()
def answer_and_states(model, tok, question, k, layer, device, max_new=32):
    prompt = f"Answer the question as briefly as possible.\nQ: {question}\nA:"
    enc = tok(prompt, return_tensors="pt").to(device)

    # hidden state of last prompt token at chosen layer -> probe input
    hs = model(**enc, output_hidden_states=True).hidden_states[layer][0, -1]

    def gen(temperature, do_sample):
        kw = dict(temperature=temperature, top_p=0.95) if do_sample else {}
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=do_sample,
                             pad_token_id=tok.eos_token_id, **kw)
        return tok.decode(out[0, enc.input_ids.shape[1]:],
                          skip_special_tokens=True).strip().split("\n")[0]

    greedy = gen(1.0, False)
    samples = [gen(1.0, True) for _ in range(k)]
    return greedy, samples, hs.float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dataset", default="triviaqa")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--layer", type=int, default=-6)
    ap.add_argument("--nli", default="microsoft/deberta-large-mnli")
    ap.add_argument("--out", default="phase1_results")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None,
                    help="auto: cuda->bf16, mps->fp16, cpu->fp32")
    args = ap.parse_args()

    if args.device is None:
        args.device = ("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}.get(
        args.dtype,
        torch.bfloat16 if args.device == "cuda"
        else torch.float16 if args.device == "mps" else torch.float32)
    print(f"device={args.device} dtype={dtype}")

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype).to(args.device).eval()
    entail = HFEntailment(args.nli, device=args.device)
    match = NormalizedMatch()

    items = load_qa(args.dataset, args.n)
    ckpt = os.path.join(args.out, "arrays.npz")
    H, SE, correct = [], [], []
    if os.path.exists(ckpt):
        d = np.load(ckpt)
        H = [x for x in d["H"]]; SE = list(d["SE"]); correct = list(d["correct"])
        print(f"Resuming at item {len(SE)}")

    for i in range(len(SE), len(items)):
        q, gold = items[i]
        greedy, samples, h = answer_and_states(model, tok, q, args.k,
                                               args.layer, args.device)
        ids = cluster_by_meaning(samples, lambda a, b: entail(a, b, question=q))
        H.append(h.numpy()); SE.append(semantic_entropy(ids))
        correct.append(int(match(greedy, gold) or entail(greedy, gold, question=q)))
        if (i + 1) % 50 == 0 or i + 1 == len(items):
            np.savez(ckpt, H=np.array(H), SE=np.array(SE),
                     correct=np.array(correct))
            print(f"[{i+1}/{len(items)}] SE={SE[-1]:.2f} "
                  f"correct={correct[-1]} (checkpointed)")

    H = np.array(H); SE = np.array(SE)
    correct = np.array(correct); confab = 1 - correct

    # winning Phase-1 recipe: ridge on exp(-SE), lambda by 5-fold CV in train,
    # Platt on train correctness, single test evaluation
    n_tr = len(correct) // 2
    lams, folds = [1e2, 1e3, 1e4, 1e5], np.array_split(np.arange(n_tr), 5)
    best_lam, best_cv = 1e4, -1.0
    for lam in lams:
        scores = []
        for f in folds:
            m = np.ones(n_tr, bool); m[f] = False
            hd = RidgePrecisionHead(lam).fit(H[:n_tr][m], SE[:n_tr][m])
            a = auroc(-hd.predict(H[:n_tr][f]), confab[:n_tr][f])
            if not np.isnan(a):
                scores.append(a)
        if np.mean(scores) > best_cv:
            best_cv, best_lam = float(np.mean(scores)), lam
    head = RidgePrecisionHead(best_lam).fit(H[:n_tr], SE[:n_tr],
                                            correct=correct[:n_tr])
    pi = head.predict(H[n_tr:])

    se_te, confab_te, correct_te = SE[n_tr:], confab[n_tr:], correct[n_tr:]
    results = dict(
        model=args.model, n=len(correct), lam=best_lam,
        accuracy=float(correct.mean()),
        auroc_sampled_se=auroc(se_te, confab_te),
        auroc_precision_head=auroc(-pi, confab_te),
        ece=ece(pi, correct_te),
        aurc=risk_coverage(pi, correct_te)[2],
    )
    th = fit_generate_threshold(pi, correct_te, target_risk=0.05)
    results["gate@5%risk"] = gate_outcomes(pi, correct_te, th)
    np.savez(os.path.join(args.out, "ridge_head.npz"), w=head.w, mu=head.mu,
             sd=head.sd, platt_a=head.a, platt_b=head.b, ybar=head.ybar,
             lam=best_lam)
    print(json.dumps(results, indent=2))
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
