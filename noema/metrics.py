"""Evaluation metrics for NOEMA (numpy-only, no sklearn dependency)."""
import numpy as np


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the rank-sum (Mann-Whitney) formulation.

    scores: higher = more likely positive. labels: 1 = positive (confabulation).
    """
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    # average ranks for ties (proper Mann-Whitney)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    confidences, correct = np.asarray(confidences, float), np.asarray(correct, int)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidences > lo) & (confidences <= hi)
        if m.sum() == 0:
            continue
        total += m.mean() * abs(correct[m].mean() - confidences[m].mean())
    return float(total)


def risk_coverage(confidences: np.ndarray, correct: np.ndarray):
    """Selective-prediction curve: answer only the most-confident fraction.

    Returns (coverage, risk) arrays and AURC (lower = better).
    """
    confidences, correct = np.asarray(confidences, float), np.asarray(correct, int)
    order = np.argsort(-confidences)
    err = 1 - correct[order]
    cum_risk = np.cumsum(err) / (np.arange(len(err)) + 1)
    coverage = (np.arange(len(err)) + 1) / len(err)
    aurc = float(np.trapezoid(cum_risk, coverage))
    return coverage, cum_risk, aurc


def confident_error_rate(confidences, correct, threshold: float) -> float:
    """The metric that matters: rate of high-confidence wrong answers (H1 endpoint)."""
    confidences, correct = np.asarray(confidences, float), np.asarray(correct, int)
    m = confidences >= threshold
    if m.sum() == 0:
        return 0.0
    return float((1 - correct[m]).mean() * m.mean())
