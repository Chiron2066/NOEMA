"""NOEMA deployment wrapper: any HF causal LM + precision head + policy gate.

The 'improved model' as a single object. Every query returns an action taken
on the basis of the model's intrinsic precision estimate:

    generate  : answer directly (pi_hat >= tau_gen)
    retrieve  : call the provided retrieve_fn, answer from evidence
    hedge     : answer, explicitly marked low-confidence
    abstain   : decline (pi_hat below tau_abstain and no retrieval available)

Usage:
    lm = NoemaLM("Qwen/Qwen2.5-1.5B-Instruct",
                 head_npz="phase1_triviaqa_n2000/ridge_head.npz",
                 ybar_from="phase1_triviaqa_n2000/arrays.npz")
    out = lm.ask("Who wrote The Master and Margarita?")
    out = lm.ask("What changed in the 2026 tax code?", retrieve_fn=my_search)

    out -> dict(action, answer, pi, note)

Thresholds are operating points on the risk-coverage curve; recalibrate the
two Platt scalars on ~100 labeled in-domain examples per deployment domain
(see PAPER_NOTES: discrimination transfers, calibration needs a light refit).
"""
from __future__ import annotations
import numpy as np
import torch

from .gated_attention import FrozenRidgeHead, question_precision


class NoemaLM:
    def __init__(self, model_name: str, head_npz: str,
                 ybar_from: str | None = None, layer: int = -6,
                 tau_gen: float = 0.55, tau_abstain: float = 0.25,
                 device: str | None = None, max_new: int = 64):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if device is None:
            device = ("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype).to(device).eval()
        ybar = None
        if ybar_from:
            arr = np.load(ybar_from)
            ybar = float(np.exp(-arr["SE"][:len(arr["SE"]) // 2]).mean())
        self.head = FrozenRidgeHead(head_npz, ybar=ybar)
        self.layer, self.device, self.max_new = layer, device, max_new
        self.tau_gen, self.tau_abstain = tau_gen, tau_abstain

    @torch.no_grad()
    def _gen(self, prompt: str) -> str:
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**enc, max_new_tokens=self.max_new,
                                  do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, enc.input_ids.shape[1]:],
                               skip_special_tokens=True).strip().split("\n")[0]

    def precision(self, question: str) -> float:
        """pi_hat in (0,1): the model's intrinsic 'do I know this?' estimate."""
        return question_precision(self.model, self.tok, question, self.head,
                                  self.device, self.layer)

    def ask(self, question: str, retrieve_fn=None) -> dict:
        pi = self.precision(question)
        if pi >= self.tau_gen:
            ans = self._gen(f"Answer the question as briefly as possible."
                            f"\nQ: {question}\nA:")
            return dict(action="generate", answer=ans, pi=pi, note=None)
        if retrieve_fn is not None:
            evidence = retrieve_fn(question)
            if evidence:
                ans = self._gen(f"Answer the question using the context."
                                f"\nContext: {evidence[:4000]}"
                                f"\nQ: {question}\nA:")
                return dict(action="retrieve", answer=ans, pi=pi,
                            note="answered from retrieved evidence")
        if pi >= self.tau_abstain:
            ans = self._gen(f"Answer the question as briefly as possible."
                            f"\nQ: {question}\nA:")
            return dict(action="hedge", answer=ans, pi=pi,
                        note="low confidence — verify before relying on this")
        return dict(action="abstain", answer=None, pi=pi,
                    note="the model does not know this and no retrieval "
                         "source is available")
