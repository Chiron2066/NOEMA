"""The four-way decoding gate: generate / retrieve / hedge / abstain.

Consumes two precision estimates:
  pi_prior    - reliability of parametric knowledge (from the precision head)
  pi_evidence - reliability of supplied context (retrieval score, consistency);
                None when no evidence pathway exists (H4 territory)
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import numpy as np


class Action(str, Enum):
    GENERATE = "generate"
    RETRIEVE = "retrieve"
    HEDGE = "hedge"
    ABSTAIN = "abstain"


@dataclass
class GateThresholds:
    generate: float = 0.75   # pi_prior above this: answer from the prior
    hedge: float = 0.45      # between hedge and generate: hedged answer
    evidence_ok: float = 0.5  # minimum usable evidence precision


def decide(pi_prior: float, pi_evidence: float | None,
           retrieval_available: bool, th: GateThresholds = GateThresholds()) -> Action:
    if pi_prior >= th.generate:
        return Action.GENERATE
    if retrieval_available and (pi_evidence is None or pi_evidence >= th.evidence_ok):
        return Action.RETRIEVE          # low precision *calls for* evidence
    if pi_prior >= th.hedge:
        return Action.HEDGE
    return Action.ABSTAIN


def fit_generate_threshold(pi: np.ndarray, correct: np.ndarray,
                           target_risk: float = 0.05) -> float:
    """Choose the generate-threshold giving <= target_risk error among answered
    items, maximizing coverage. (Selective-prediction operating point.)"""
    pi, correct = np.asarray(pi, float), np.asarray(correct, int)
    order = np.argsort(-pi)
    err = 1 - correct[order]
    cum_risk = np.cumsum(err) / (np.arange(len(err)) + 1)
    ok = np.where(cum_risk <= target_risk)[0]
    if len(ok) == 0:
        return float(pi.max()) + 1e-9   # abstain on everything
    return float(pi[order][ok[-1]])


def gate_outcomes(pi: np.ndarray, correct: np.ndarray, threshold: float):
    """Summary of what gating does to the error budget (H4 analysis).

    Returns dict with coverage, risk-on-answered, confident-error rate,
    and the abstention conversion: fraction of would-be errors converted
    to abstentions rather than answers."""
    pi, correct = np.asarray(pi, float), np.asarray(correct, int)
    answered = pi >= threshold
    cov = float(answered.mean())
    risk = float((1 - correct[answered]).mean()) if answered.any() else 0.0
    base_err = float((1 - correct).mean())
    errors_abstained = float(((1 - correct) & ~answered).sum()) / max(
        (1 - correct).sum(), 1)
    return dict(coverage=cov, risk_answered=risk, base_error=base_err,
                errors_converted_to_abstention=errors_abstained)
