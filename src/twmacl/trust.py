from __future__ import annotations

import numpy as np


class TrustMatrix:
    def __init__(self, num_agents: int, alpha: float, lambda_: float) -> None:
        self.num_agents = num_agents
        self.alpha = alpha
        self.lambda_ = lambda_
        self.raw = np.zeros((num_agents, num_agents), dtype=float)

    def update(self, predictions: np.ndarray, realized_return: np.ndarray) -> None:
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
        normalized = np.zeros_like(self.raw)
        for i in range(self.num_agents):
            row = self.raw[i].copy()
            row[i] = -np.inf
            row_max = np.max(row[np.isfinite(row)])
            exps = np.exp(row - row_max)
            exps[i] = 0.0
            denom = np.sum(exps)
            if denom == 0.0:
                exps = np.ones(self.num_agents, dtype=float)
                exps[i] = 0.0
                denom = np.sum(exps)
            normalized[i] = exps / denom
        return normalized
