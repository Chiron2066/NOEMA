"""Semantic entropy (Farquhar et al., Nature 2024) — the distillation target.

Pipeline: sample K answers -> cluster by bidirectional entailment -> entropy
over meaning-clusters. High SE predicts confabulation.

Entailment backends are pluggable:
  - NormalizedMatch: cheap heuristic (exact/substring match after normalization)
  - HFEntailment: DeBERTa-v3 NLI (requires transformers; use for real runs)
"""
from __future__ import annotations
import math
import re
import string
from dataclasses import dataclass
from typing import Callable, List, Sequence


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


class NormalizedMatch:
    """Heuristic entailment: bidirectional if normalized strings match or one
    contains the other. Good enough for short-form QA; replace with NLI for
    sentence-length answers."""

    def __call__(self, a: str, b: str) -> bool:
        na, nb = normalize(a), normalize(b)
        if not na or not nb:
            return na == nb
        return na == nb or na in nb or nb in na


class HFEntailment:
    """Bidirectional entailment with an NLI cross-encoder.

    Usage (GPU box):
        ent = HFEntailment("microsoft/deberta-large-mnli")
        clusters = cluster_by_meaning(answers, ent, question=q)
    """

    def __init__(self, model_name: str = "microsoft/deberta-large-mnli",
                 device: str = "cuda"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch  # noqa: F401
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval().to(device)
        self.device = device
        # auto-detect which output index means "entailment" from the model config
        # (label order differs between NLI models; hardcoding breaks silently)
        self.entail_idx = next(
            (int(k) for k, v in self.model.config.id2label.items()
             if "entail" in str(v).lower()), 2)

    def _entails(self, premise: str, hypothesis: str) -> bool:
        import torch
        with torch.no_grad():
            enc = self.tok(premise, hypothesis, return_tensors="pt",
                           truncation=True).to(self.device)
            pred = self.model(**enc).logits.argmax(-1).item()
        return pred == self.entail_idx

    def __call__(self, a: str, b: str, question: str = "") -> bool:
        ctx = f"{question} " if question else ""
        return (self._entails(ctx + a, ctx + b)
                and self._entails(ctx + b, ctx + a))


def cluster_by_meaning(answers: Sequence[str],
                       entail: Callable[[str, str], bool]) -> List[int]:
    """Greedy bidirectional-entailment clustering (Kuhn et al. 2023).

    Returns cluster id per answer. Each answer joins the first cluster whose
    representative it bidirectionally entails; else founds a new cluster.
    """
    reps: List[str] = []
    assign: List[int] = []
    for ans in answers:
        placed = False
        for cid, rep in enumerate(reps):
            if entail(ans, rep):
                assign.append(cid)
                placed = True
                break
        if not placed:
            reps.append(ans)
            assign.append(len(reps) - 1)
    return assign


def semantic_entropy(cluster_ids: Sequence[int]) -> float:
    """Discrete semantic entropy: entropy of the cluster-membership distribution."""
    n = len(cluster_ids)
    if n == 0:
        return 0.0
    counts = {}
    for c in cluster_ids:
        counts[c] = counts.get(c, 0) + 1
    return -sum((k / n) * math.log(k / n) for k in counts.values())


@dataclass
class SEResult:
    answers: List[str]
    cluster_ids: List[int]
    entropy: float
    n_clusters: int


def compute_se(sample_fn: Callable[[], str], k: int = 10,
               entail: Callable[[str, str], bool] | None = None) -> SEResult:
    """Sample k answers from `sample_fn` and compute semantic entropy."""
    entail = entail or NormalizedMatch()
    answers = [sample_fn() for _ in range(k)]
    ids = cluster_by_meaning(answers, entail)
    return SEResult(answers, ids, semantic_entropy(ids), len(set(ids)))
