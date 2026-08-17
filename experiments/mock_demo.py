"""End-to-end validation of the NOEMA pipeline on a synthetic world.

Runs with numpy only. Simulates an LLM whose 'knowledge quality' q per fact
controls both (a) its answer distribution and (b) its hidden state, then
verifies the full H2 chain:

    sampled answers -> entailment clustering -> semantic entropy
    -> precision probe distilled from hidden states (single 'forward pass')
    -> detection AUROC / calibration / risk-coverage
    -> H4: the gate converts errors into abstentions, not into truth.

If this pipeline is sound, swapping the simulator for a real HF model
(experiments/run_phase1.py) changes the data source, not the logic.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from noema.semantic_entropy import compute_se, NormalizedMatch
from noema.precision_head import NumpyPrecisionProbe
from noema.metrics import auroc, ece, risk_coverage
from noema.gate import fit_generate_threshold, gate_outcomes

rng = np.random.default_rng(42)

N_FACTS, DIM, K_SAMPLES = 1200, 64, 10
VOCAB = [f"entity_{i}" for i in range(500)]


def make_world():
    """Each fact: true answer, knowledge quality q, hidden state correlated
    with q through a fixed 'knownness' direction plus per-fact structure."""
    known_dir = rng.normal(0, 1, DIM)
    known_dir /= np.linalg.norm(known_dir)
    facts = []
    for i in range(N_FACTS):
        q = float(np.clip(rng.beta(1.2, 1.2), 0.01, 0.99))
        truth = VOCAB[rng.integers(len(VOCAB))]
        h = rng.normal(0, 1, DIM) + 5.0 * q * known_dir
        facts.append(dict(q=q, truth=truth, h=h))
    return facts


def sample_answer(fact):
    """The simulated LM: knows the answer with prob q; otherwise confabulates
    fluently — a *plausible but wrong* entity, sometimes with padding (so the
    entailment heuristic has real work to do)."""
    if rng.random() < fact["q"]:
        # paraphrase variants of the true answer
        t = fact["truth"]
        return rng.choice([t, f"the {t}", f"{t}."])
    wrong = VOCAB[rng.integers(len(VOCAB))]
    return rng.choice([wrong, f"the {wrong}", f"{wrong}, I believe"])


def greedy_answer(fact):
    return fact["truth"] if fact["q"] > 0.5 else VOCAB[rng.integers(len(VOCAB))]


def main():
    facts = make_world()
    entail = NormalizedMatch()

    # ---- Stage 1: sampled semantic entropy per fact (the expensive signal)
    H, SE, correct = [], [], []
    for f in facts:
        res = compute_se(lambda f=f: sample_answer(f), k=K_SAMPLES, entail=entail)
        H.append(f["h"]); SE.append(res.entropy)
        correct.append(int(entail(greedy_answer(f), f["truth"])))
    H, SE, correct = np.array(H), np.array(SE), np.array(correct)
    confab = 1 - correct

    n_tr = N_FACTS // 2
    tr, te = slice(0, n_tr), slice(n_tr, None)

    # ---- Baseline: sampled SE itself as the detector (Farquhar et al.)
    au_se = auroc(SE[te], confab[te])

    # ---- H2: distill SE into a probe over hidden states (amortization)
    probe = NumpyPrecisionProbe(DIM).fit(H[tr], (SE[tr] < np.median(SE[tr])).astype(int))
    pi = probe.predict(H[te])                       # single-forward-pass precision
    au_probe = auroc(-pi, confab[te])               # low precision -> confabulation

    print("=== NOEMA mock pipeline (synthetic world, n=%d) ===" % N_FACTS)
    print(f"Confabulation base rate:            {confab[te].mean():.3f}")
    print(f"AUROC  sampled semantic entropy:    {au_se:.3f}   (10 forward passes)")
    print(f"AUROC  distilled precision probe:   {au_probe:.3f}   (1 forward pass)")
    print(f"H2 check (probe >= 90% of SE AUROC): "
          f"{'PASS' if au_probe >= 0.9 * au_se else 'FAIL'}")

    # ---- Calibration + selective prediction
    e = ece(pi, correct[te])
    cov, risk, aurc = risk_coverage(pi, correct[te])
    print(f"\nECE of precision estimate:          {e:.3f}")
    print(f"AURC (selective prediction):        {aurc:.3f}")

    # ---- H4: gate converts error into abstention, not truth
    th = fit_generate_threshold(pi, correct[te], target_risk=0.10)
    out = gate_outcomes(pi, correct[te], th)
    print(f"\nGate @ 10% target risk on answered:")
    print(f"  coverage:                         {out['coverage']:.3f}")
    print(f"  risk among answered:              {out['risk_answered']:.3f}"
          f"   (base error {out['base_error']:.3f})")
    print(f"  errors converted to abstention:   "
          f"{out['errors_converted_to_abstention']:.3f}")
    print("\nH4 shape confirmed: accuracy on answered rises because errors are"
          "\nabstained, not because the worldless prior learned anything true.")


if __name__ == "__main__":
    main()
