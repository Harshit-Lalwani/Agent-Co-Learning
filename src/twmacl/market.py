from __future__ import annotations

import numpy as np


class CorrelatedMarket:
    def __init__(self, mu: list[float], cov: list[list[float]], seed: int) -> None:
        self.mu = np.asarray(mu, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self.rng = np.random.default_rng(seed)

    def sample_return(self) -> np.ndarray:
        return self.rng.multivariate_normal(self.mu, self.cov)
