"""The intrinsic precision head (H2).

Distills sampled semantic entropy into a single-forward-pass estimate from
hidden states. Two implementations:

  - NumpyPrecisionProbe: dependency-free logistic probe (mock/CI use)
  - TorchPrecisionHead:  the real thing, with the asymmetric calibration loss

Training signal, stage 1 (distillation):  regress/classify sampled SE.
Training signal, stage 2 (grounding):     asymmetric loss on verified answers
    L = alpha * pi * 1[wrong]  -  beta * pi * 1[right],  alpha >> beta
so confident-wrong costs far more than confident-right earns (Kalai et al. 2025).
"""
from __future__ import annotations
import numpy as np


class NumpyPrecisionProbe:
    """Logistic probe: p(low-SE | hidden_state) — i.e. estimated precision.

    Plain full-batch gradient descent; adequate for probe-sized problems.
    """

    def __init__(self, dim: int, lr: float = 0.5, epochs: int = 400,
                 l2: float = 1e-3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, dim)
        self.b = 0.0
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, H: np.ndarray, high_precision: np.ndarray):
        """H: (n, d) hidden states; high_precision: 1 if SE below median (reliable)."""
        H = np.asarray(H, float)
        y = np.asarray(high_precision, float)
        mu, sd = H.mean(0), H.std(0) + 1e-8
        self.mu, self.sd = mu, sd
        X = (H - mu) / sd
        n = len(y)
        for _ in range(self.epochs):
            p = self._sigmoid(X @ self.w + self.b)
            g = X.T @ (p - y) / n + self.l2 * self.w
            self.w -= self.lr * g
            self.b -= self.lr * float((p - y).mean())
        return self

    def predict(self, H: np.ndarray) -> np.ndarray:
        X = (np.asarray(H, float) - self.mu) / self.sd
        return self._sigmoid(X @ self.w + self.b)




class RidgePrecisionHead:
    """The recipe that won Phase 1 (TriviaQA n=2000, Qwen2.5-1.5B, layer -6):
    ridge regression on the soft target exp(-SE), lambda chosen by CV inside
    the training split, followed by Platt calibration on training correctness.

    Result: test AUROC 0.819 vs 0.820 for 10-sample semantic entropy
    (ratio 0.998), ECE 0.031. One forward pass matches ten.
    """

    def __init__(self, lam: float = 1e4):
        self.lam = lam

    def fit(self, H, se, correct=None):
        H = np.asarray(H, float)
        self.mu, self.sd = H.mean(0), H.std(0) + 1e-8
        X = (H - self.mu) / self.sd
        y = np.exp(-np.asarray(se, float))
        A = X.T @ X + self.lam * np.eye(X.shape[1])
        self.w = np.linalg.solve(A, X.T @ (y - y.mean()))
        self.ybar = y.mean()
        # Platt calibration -> pi in (0,1)
        self.a, self.b = 0.0, 0.0
        if correct is not None:
            s = X @ self.w + self.ybar
            c = np.asarray(correct, float)
            for _ in range(2000):
                p = 1 / (1 + np.exp(-(self.a * s + self.b)))
                self.a -= 0.5 * ((p - c) * s).mean()
                self.b -= 0.5 * (p - c).mean()
        return self

    def predict(self, H):
        X = (np.asarray(H, float) - self.mu) / self.sd
        s = X @ self.w + self.ybar
        return 1 / (1 + np.exp(-(self.a * s + self.b))) if self.a else s


try:
    import torch
    import torch.nn as nn

    class TorchPrecisionHead(nn.Module):
        """MLP head over a chosen layer's hidden state (last generated token).

        forward(h) -> pi_hat in (0,1): the model's intrinsic precision estimate.
        """

        def __init__(self, dim: int, hidden: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden), nn.GELU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, h):  # h: (batch, dim)
            return torch.sigmoid(self.net(h)).squeeze(-1)

    def distillation_loss(pi_hat, se, se_scale: float = 1.0):
        """Stage 1: pi_hat should track exp(-SE) (monotone map of sampled SE)."""
        target = torch.exp(-se / se_scale)
        return nn.functional.mse_loss(pi_hat, target)

    def asymmetric_calibration_loss(pi_hat, correct, alpha: float = 4.0,
                                    beta: float = 1.0):
        """Stage 2: confident-wrong penalized alpha/beta times harder than
        confident-right is rewarded. correct: float tensor in {0,1}."""
        return (alpha * pi_hat * (1 - correct) - beta * pi_hat * correct).mean()

    def train_head(head, H, se, correct=None, epochs=30, lr=1e-3,
                   lambda_cal=0.5, device="cuda"):
        """H: (n,d) float tensor; se: (n,) sampled semantic entropy;
        correct: optional (n,) verified correctness for stage-2 grounding."""
        head.to(device).train()
        opt = torch.optim.AdamW(head.parameters(), lr=lr)
        H, se = H.to(device), se.to(device)
        if correct is not None:
            correct = correct.to(device)
        for _ in range(epochs):
            opt.zero_grad()
            pi = head(H)
            loss = distillation_loss(pi, se)
            if correct is not None:
                loss = loss + lambda_cal * asymmetric_calibration_loss(pi, correct)
            loss.backward()
            opt.step()
        return head

except ImportError:  # torch unavailable: numpy probe only
    pass
