from __future__ import annotations

import numpy as np


class TrustMatrix:
    def __init__(self, num_agents: int, alpha: float, lambda_: float) -> None:
        self.num_agents = num_agents
        self.alpha = alpha
        self.lambda_ = lambda_
        self.raw = np.zeros((num_agents, num_agents), dtype=float)

    def reset(self) -> None:
        """Reset trust to uniform (zero raw scores). Call between episodes unless
        trust_persistence is enabled."""
        self.raw = np.zeros((self.num_agents, self.num_agents), dtype=float)

    def update(self, predictions: np.ndarray, realized_return: np.ndarray) -> None:
        """Update raw trust scores.

        predictions: shape (num_agents, num_assets) — each agent j's predicted return.
        realized_return: shape (num_assets,) — the actual market return.

        Note: in Phase 1 (random portfolios), predictions[j] does not depend on
        the observing agent i, so all trust rows raw[i, :] are identical. Agent-level
        differentiation in trust only emerges in Phases 2/3 where each agent acts
        differently and those actions drive the predictor indirectly.
        """
        if predictions.shape != (self.num_agents, realized_return.shape[0]):
            raise ValueError("predictions shape mismatch for trust update")

        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i == j:
                    continue
                l1_error = np.linalg.norm(predictions[j] - realized_return, ord=1)
                score = np.exp(-self.lambda_ * l1_error)
                self.raw[i, j] = (1.0 - self.alpha) * self.raw[i, j] + self.alpha * score

    def normalized(self) -> np.ndarray:
        """Row-wise softmax normalization, excluding self-trust (diagonal)."""
        normalized = np.zeros_like(self.raw)
        for i in range(self.num_agents):
            row = self.raw[i].copy()
            row[i] = -np.inf
            finite_mask = np.isfinite(row)
            if not finite_mask.any():
                # Fallback: uniform trust over peers
                normalized[i] = 1.0 / (self.num_agents - 1)
                normalized[i, i] = 0.0
                continue
            row_max = np.max(row[finite_mask])
            exps = np.exp(row - row_max)
            exps[i] = 0.0
            denom = np.sum(exps)
            if denom == 0.0:
                exps = np.ones(self.num_agents, dtype=float)
                exps[i] = 0.0
                denom = np.sum(exps)
            normalized[i] = exps / denom
        return normalized
