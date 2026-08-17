"""Unit tests for the NOEMA pipeline (numpy only). Run: python tests/test_pipeline.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from noema.semantic_entropy import (NormalizedMatch, cluster_by_meaning,
                                    semantic_entropy)
from noema.metrics import auroc, ece, risk_coverage
from noema.gate import Action, decide, GateThresholds, fit_generate_threshold


def test_clustering_merges_paraphrases():
    ent = NormalizedMatch()
    ids = cluster_by_meaning(
        ["Paris", "the Paris", "paris.", "London", "Rome", "rome"], ent)
    assert ids[0] == ids[1] == ids[2], ids
    assert ids[3] not in (ids[0], ids[4]), ids
    assert ids[4] == ids[5], ids
    assert len(set(ids)) == 3


def test_semantic_entropy_bounds():
    assert semantic_entropy([0, 0, 0, 0]) == 0.0
    hi = semantic_entropy([0, 1, 2, 3])
    lo = semantic_entropy([0, 0, 0, 1])
    assert hi > lo > 0


def test_auroc_sanity():
    labels = np.array([1, 1, 1, 0, 0, 0])
    assert auroc(np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]), labels) == 1.0
    assert abs(auroc(np.array([0.5] * 6), labels) - 0.5) < 1e-9


def test_ece_perfect_calibration():
    conf = np.array([0.8] * 10)
    correct = np.array([1] * 8 + [0] * 2)
    assert ece(conf, correct) < 1e-9


def test_gate_logic():
    th = GateThresholds()
    assert decide(0.9, None, False, th) == Action.GENERATE
    assert decide(0.3, 0.8, True, th) == Action.RETRIEVE
    assert decide(0.5, None, False, th) == Action.HEDGE
    assert decide(0.1, None, False, th) == Action.ABSTAIN


def test_threshold_meets_target_risk():
    rng = np.random.default_rng(0)
    pi = rng.random(500)
    correct = (rng.random(500) < pi).astype(int)   # calibrated world
    th = fit_generate_threshold(pi, correct, target_risk=0.1)
    answered = pi >= th
    assert (1 - correct[answered]).mean() <= 0.1 + 1e-9


def test_risk_coverage_monotone_setup():
    pi = np.array([0.9, 0.8, 0.2, 0.1])
    correct = np.array([1, 1, 0, 0])
    cov, risk, aurc = risk_coverage(pi, correct)
    assert risk[0] == 0.0 and risk[-1] == 0.5 and aurc < 0.5


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
