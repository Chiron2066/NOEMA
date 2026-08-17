"""The annealing curriculum (H3): canalization defense during preference-tuning.

REBUS logic transposed to training: alternate *plasticity* phases (elevated
sampling temperature, representation noise, relaxed confidence penalty) with
*consolidation* phases (baseline entropy, calibration re-annealing). Success
criterion: preference reward within tolerance while ECE and output diversity
are preserved relative to standard RLHF.

This module is trainer-agnostic: it yields per-step knobs; wire them into a
DPO/PPO loop (e.g., trl) via callbacks.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class PhaseKnobs:
    phase: str              # "plasticity" | "consolidation"
    temperature: float      # sampling temperature for rollouts
    rep_noise_std: float    # gaussian noise on high-layer hidden states
    lambda_conf: float      # weight on the precision head's confidence term
    entropy_bonus: float    # policy entropy regularization weight


@dataclass
class AnnealingSchedule:
    """Cosine-gated alternation between plasticity and consolidation."""
    total_steps: int
    cycle_steps: int = 2000
    plasticity_frac: float = 0.3        # fraction of each cycle spent plastic
    temp_hi: float = 1.3
    temp_lo: float = 1.0
    noise_hi: float = 0.02
    lambda_conf_lo: float = 0.1         # relaxed during plasticity
    lambda_conf_hi: float = 1.0         # re-annealed during consolidation
    entropy_hi: float = 0.02
    entropy_lo: float = 0.002

    def knobs(self, step: int) -> PhaseKnobs:
        t = (step % self.cycle_steps) / self.cycle_steps
        if t < self.plasticity_frac:
            # ramp entropy up then down within the plastic window (dosed, not chronic)
            u = math.sin(math.pi * t / self.plasticity_frac)
            return PhaseKnobs("plasticity",
                              self.temp_lo + u * (self.temp_hi - self.temp_lo),
                              u * self.noise_hi,
                              self.lambda_conf_lo,
                              self.entropy_lo + u * (self.entropy_hi - self.entropy_lo))
        return PhaseKnobs("consolidation", self.temp_lo, 0.0,
                          self.lambda_conf_hi, self.entropy_lo)


# ---- Diversity + calibration tracking (the H3 measurements) ----------------

def distinct_n(samples: list[str], n: int = 2) -> float:
    grams, total = set(), 0
    for s in samples:
        toks = s.split()
        for i in range(len(toks) - n + 1):
            grams.add(tuple(toks[i:i + n])); total += 1
    return len(grams) / max(total, 1)


def semantic_diversity(cluster_ids: list[int]) -> float:
    """Effective number of meanings: exp(semantic entropy) / n."""
    from .semantic_entropy import semantic_entropy
    n = max(len(cluster_ids), 1)
    return math.exp(semantic_entropy(cluster_ids)) / n


@dataclass
class H3Tracker:
    """Log per-eval-point: reward, ECE, distinct-2, semantic diversity.
    H3 predicts: standard RLHF shows ECE decay co-occurring with diversity
    collapse; annealed runs decouple them."""
    history: list = None

    def __post_init__(self):
        self.history = []

    def log(self, step, reward, ece_val, d2, sem_div, phase):
        self.history.append(dict(step=step, reward=reward, ece=ece_val,
                                 distinct2=d2, sem_div=sem_div, phase=phase))

    def canalization_signature(self) -> dict:
        """Correlation between calibration decay and diversity collapse."""
        h = self.history
        if len(h) < 3:
            return {}
        ece_d = np.diff([x["ece"] for x in h])
        div_d = np.diff([x["sem_div"] for x in h])
        c = float(np.corrcoef(ece_d, div_d)[0, 1]) if ece_d.std() > 0 else 0.0
        return dict(ece_trend=float(np.polyfit(range(len(h)), [x["ece"] for x in h], 1)[0]),
                    diversity_trend=float(np.polyfit(range(len(h)), [x["sem_div"] for x in h], 1)[0]),
                    decay_coupling=c)
