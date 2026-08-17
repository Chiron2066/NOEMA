"""Layer sweep: where does 'knowing' live in the network's depth?

Reuses SE + correctness from phase1_triviaqa_n2000/arrays.npz (same questions,
same order), so it only needs ONE forward pass per question to capture the
last-prompt-token hidden state at EVERY layer. Then trains the winning ridge
recipe per layer and reports test AUROC vs the sampled-SE ceiling (0.820).

Run from the repo root:
    caffeinate -i python experiments/run_layer_sweep.py

Resumable: checkpoints every 100 items to layer_sweep/states.npy.
Outputs:    layer_sweep/layer_sweep.json + layer_sweep/layer_sweep.png
"""
import os, sys, json

# --- environment quirks BEFORE heavy imports (corporate SSL + xet stalls) ---
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

from noema.metrics import auroc

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
N = 2000
PHASE1_DIR = "phase1_triviaqa_n2000"
OUT_DIR = "layer_sweep"
LAMBDAS = [1e2, 1e3, 1e4, 1e5]


def device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_questions(n):
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    return [x["question"] for x in ds.select(range(n))]


@torch.no_grad()
def collect_states(questions, device, dtype):
    """(n, n_layers, dim) float16 — last-prompt-token state at every layer."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype)
    model.to(device).eval()

    states_path = os.path.join(OUT_DIR, "states.npy")
    count_path = os.path.join(OUT_DIR, "done_count.txt")

    # probe once to size the array
    enc = tok("Q: probe\nA:", return_tensors="pt").to(device)
    hs = model(**enc, output_hidden_states=True).hidden_states
    n_layers, dim = len(hs), hs[0].shape[-1]

    if os.path.exists(states_path) and os.path.exists(count_path):
        states = np.load(states_path)
        start = int(open(count_path).read().strip())
        print(f"Resuming at item {start}")
    else:
        states = np.zeros((len(questions), n_layers, dim), dtype=np.float16)
        start = 0

    for i in range(start, len(questions)):
        prompt = (f"Answer the question as briefly as possible.\n"
                  f"Q: {questions[i]}\nA:")
        enc = tok(prompt, return_tensors="pt").to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        for L in range(n_layers):
            states[i, L] = hs[L][0, -1].float().cpu().numpy().astype(np.float16)
        if (i + 1) % 100 == 0 or i + 1 == len(questions):
            np.save(states_path, states)
            with open(count_path, "w") as f:
                f.write(str(i + 1))
            print(f"[{i+1}/{len(questions)}] checkpointed")
    return states


def fit_ridge(X, y, lam):
    """Returns w, ybar for centered ridge on standardized X."""
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ (y - y.mean())), y.mean()


def sweep_layer(H, se, confab, n_tr):
    """CV lambda inside train half, single test evaluation. Returns dict."""
    Htr, Hte = H[:n_tr].astype(np.float64), H[n_tr:].astype(np.float64)
    mu, sd = Htr.mean(0), Htr.std(0) + 1e-8
    Xtr, Xte = (Htr - mu) / sd, (Hte - mu) / sd
    ytr = np.exp(-se[:n_tr])

    # 5-fold CV for lambda (AUROC of -pred vs confab, matching Phase 1)
    folds = np.array_split(np.arange(n_tr), 5)
    best_lam, best_cv = None, -1.0
    for lam in LAMBDAS:
        scores = []
        for f in folds:
            m = np.ones(n_tr, bool); m[f] = False
            w, ybar = fit_ridge(Xtr[m], ytr[m], lam)
            pred = Xtr[f] @ w + ybar
            a = auroc(-pred, confab[:n_tr][f])
            if not np.isnan(a):
                scores.append(a)
        cv = float(np.mean(scores))
        if cv > best_cv:
            best_cv, best_lam = cv, lam

    w, ybar = fit_ridge(Xtr, ytr, best_lam)
    pred_te = Xte @ w + ybar
    au = auroc(-pred_te, confab[n_tr:])

    # bootstrap CI
    rng = np.random.default_rng(0)
    n_te = len(pred_te)
    boots = [auroc(-pred_te[idx], confab[n_tr:][idx])
             for idx in (rng.integers(0, n_te, n_te) for _ in range(500))]
    boots = [b for b in boots if not np.isnan(b)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(auroc=float(au), ci=[float(lo), float(hi)],
                lam=best_lam, cv_auroc=best_cv)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    arr = np.load(os.path.join(PHASE1_DIR, "arrays.npz"))
    se, correct = arr["SE"].astype(np.float64), arr["correct"]
    confab = 1 - correct
    assert len(se) == N, f"expected {N} items, found {len(se)}"

    device, dtype = device_and_dtype()
    print(f"device={device} dtype={dtype}")

    questions = load_questions(N)
    states = collect_states(questions, device, dtype)  # (N, L, dim)
    n_layers = states.shape[1]
    n_tr = N // 2

    results = {}
    for L in range(n_layers):
        r = sweep_layer(states[:, L, :], se, confab, n_tr)
        results[L] = r
        print(f"layer {L:2d}  AUROC {r['auroc']:.3f} "
              f"[{r['ci'][0]:.3f}-{r['ci'][1]:.3f}]  lam={r['lam']:.0e}")

    best = max(results, key=lambda L: results[L]["auroc"])
    summary = dict(model=MODEL, n=N, n_layers=n_layers,
                   se_ceiling=0.820, phase1_layer=n_layers - 6,
                   best_layer=int(best),
                   best_auroc=results[best]["auroc"],
                   per_layer=results)
    with open(os.path.join(OUT_DIR, "layer_sweep.json"), "w") as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = sorted(results)
        aus = [results[L]["auroc"] for L in xs]
        los = [results[L]["ci"][0] for L in xs]
        his = [results[L]["ci"][1] for L in xs]
        plt.figure(figsize=(9, 5))
        plt.fill_between(xs, los, his, alpha=0.2, label="95% CI")
        plt.plot(xs, aus, "o-", label="ridge head AUROC")
        plt.axhline(0.820, ls="--", c="gray", label="sampled SE ceiling (10 passes)")
        plt.axvline(n_layers - 6, ls=":", c="red", label="Phase 1 layer (-6)")
        plt.xlabel("layer (0 = embeddings)"); plt.ylabel("test AUROC")
        plt.title("Where knowing lives: confabulation detection by depth")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "layer_sweep.png"), dpi=150)
        print("Saved layer_sweep/layer_sweep.png")
    except ImportError:
        print("matplotlib not installed; skipped plot")

    print(f"\nBest layer: {best} (AUROC {results[best]['auroc']:.3f}); "
          f"Phase 1 used layer {n_layers - 6}.")


if __name__ == "__main__":
    main()
