"""Phase 2 (H1): precision-gated attention.

Mechanism
---------
1. Read the model's intrinsic precision pi_hat for the *question alone*
   (frozen Phase-1 ridge head on the layer −6 last-prompt-token state):
   "do I know this parametrically?"
2. When answering WITH a retrieved context, add a positive pre-softmax bias
   b = gamma * (1 - pi_hat) to attention logits at the context-token key
   positions, in the layers after the knowing-plateau begins (L16+).
   Low precision -> lean on evidence. High precision -> bias ~0, model
   behaves normally.

This is the machine analog of precision-weighting: sensory evidence is
up-weighted exactly when the prior is unreliable (REBUS in one line).

Stability notes (reviewer objection 2): pi_hat here is computed ONCE per
question, so the bias is constant across generated tokens — no per-token
fluctuation. The bias is bounded (gamma <= ~8 recommended) and applied only
to a subset of layers. Token-level dynamic gating is future work.

Requires the model to be loaded with attn_implementation="eager" so the
additive attention mask is always materialized.
"""
from __future__ import annotations
import numpy as np
import torch


class FrozenRidgeHead:
    """Loads phase1_triviaqa_n2000/ridge_head.npz (w, mu, sd, platt_a, platt_b)."""

    def __init__(self, path: str, ybar: float | None = None):
        d = np.load(path)
        self.w, self.mu, self.sd = d["w"], d["mu"], d["sd"]
        self.a, self.b = float(d["platt_a"]), float(d["platt_b"])
        # ybar (train mean of exp(-SE)) wasn't saved in the npz; recover it
        # from phase1 arrays and pass it in, else scores shift by a*ybar.
        if ybar is not None:
            self.ybar = float(ybar)
        else:
            self.ybar = float(d["ybar"]) if "ybar" in d.files else 0.0

    def predict(self, h: np.ndarray) -> float:
        """h: (dim,) hidden state -> pi_hat in (0,1)."""
        x = (np.asarray(h, float) - self.mu) / self.sd
        s = x @ self.w + self.ybar
        return float(1.0 / (1.0 + np.exp(-(self.a * s + self.b))))


class PrecisionGate:
    """Patches selected decoder layers to add a context-key attention bias.

    Usage:
        gate = PrecisionGate(model, layers=range(16, 28), gamma=4.0)
        gate.set(ctx_start, ctx_end, pi_hat)   # before generate()
        ...model.generate(...)...
        gate.clear()                            # bias off (model unchanged)
        gate.remove()                           # unpatch entirely
    """

    def __init__(self, model, layers, gamma: float = 4.0, max_bias: float = 8.0):
        self.model = model
        self.gamma = float(gamma)
        self.max_bias = float(max_bias)
        self._span = None          # (start, end) token indices in the prompt
        self._strength = 0.0       # gamma * (1 - pi_hat), clamped
        self._originals = []
        for idx in layers:
            layer = model.model.layers[idx]
            attn = layer.self_attn
            orig = attn.forward
            self._originals.append((attn, orig))
            attn.forward = self._make_wrapper(orig)

    def set(self, ctx_start: int, ctx_end: int, pi_hat: float):
        self._span = (int(ctx_start), int(ctx_end))
        self._strength = min(self.gamma * (1.0 - float(pi_hat)), self.max_bias)

    def clear(self):
        self._span, self._strength = None, 0.0

    def remove(self):
        for attn, orig in self._originals:
            attn.forward = orig
        self._originals = []

    def _make_wrapper(self, orig_forward):
        gate = self

        def forward(*args, **kwargs):
            if gate._span is not None and gate._strength > 0:
                mask = kwargs.get("attention_mask", None)
                if mask is not None:
                    kv_len = mask.shape[-1]
                    s, e = gate._span
                    e = min(e, kv_len)
                    if e > s:
                        bias = torch.zeros(
                            (1, 1, 1, kv_len), dtype=mask.dtype, device=mask.device)
                        bias[..., s:e] = gate._strength
                        kwargs["attention_mask"] = mask + bias
            return orig_forward(*args, **kwargs)

        return forward


def context_token_span(tok, prefix: str, context: str) -> tuple[int, int]:
    """Token index span of `context` inside prefix+context+... (BPE-approx)."""
    n_prefix = len(tok(prefix).input_ids)
    n_with = len(tok(prefix + context).input_ids)
    return n_prefix, n_with


@torch.no_grad()
def question_precision(model, tok, question: str, head: FrozenRidgeHead,
                       device: str, layer: int = -6) -> float:
    """pi_hat from the SAME question-only prompt template the head was
    trained on (Phase 1) — keeps the head fully in-distribution."""
    prompt = f"Answer the question as briefly as possible.\nQ: {question}\nA:"
    enc = tok(prompt, return_tensors="pt").to(device)
    hs = model(**enc, output_hidden_states=True).hidden_states[layer][0, -1]
    return head.predict(hs.float().cpu().numpy())
