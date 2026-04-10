from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BasePredictor:
    def predict(self, step: int, return_history: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def trust_update_enabled(self, step: int) -> bool:
        return True


@dataclass
class MovingAveragePredictor(BasePredictor):
    num_agents: int
    num_assets: int
    window: int
    noise_std: float
    rng: np.random.Generator

    def trust_update_enabled(self, step: int) -> bool:
        return step >= self.window

    def predict(self, step: int, return_history: list[np.ndarray]) -> np.ndarray:
        if not return_history:
            base = np.zeros(self.num_assets, dtype=float)
        else:
            lookback = min(len(return_history), self.window)
            base = np.mean(np.asarray(return_history[-lookback:]), axis=0)
        noise = self.rng.normal(loc=0.0, scale=self.noise_std, size=(self.num_agents, self.num_assets))
        return base[None, :] + noise


@dataclass
class NoisyOraclePredictor(BasePredictor):
    num_agents: int
    mu: np.ndarray
    noise_std: float
    rng: np.random.Generator

    def predict(self, step: int, return_history: list[np.ndarray]) -> np.ndarray:
        noise = self.rng.normal(loc=0.0, scale=self.noise_std, size=(self.num_agents, self.mu.shape[0]))
        return self.mu[None, :] + noise


@dataclass
class RandomPredictor(BasePredictor):
    num_agents: int
    mu: np.ndarray
    cov: np.ndarray
    rng: np.random.Generator

    def predict(self, step: int, return_history: list[np.ndarray]) -> np.ndarray:
        return self.rng.multivariate_normal(self.mu, self.cov, size=self.num_agents)


def build_predictor(
    predictor_mode: str,
    num_agents: int,
    num_assets: int,
    predictor_window: int,
    noise_std: float,
    mu: np.ndarray,
    cov: np.ndarray,
    seed: int,
) -> BasePredictor:
    rng = np.random.default_rng(seed)
    if predictor_mode == "moving_average":
        return MovingAveragePredictor(
            num_agents=num_agents,
            num_assets=num_assets,
            window=predictor_window,
            noise_std=noise_std,
            rng=rng,
        )
    if predictor_mode == "noisy_oracle":
        return NoisyOraclePredictor(num_agents=num_agents, mu=mu, noise_std=noise_std, rng=rng)
    if predictor_mode == "random":
        return RandomPredictor(num_agents=num_agents, mu=mu, cov=cov, rng=rng)
    raise ValueError(f"Unsupported predictor_mode: {predictor_mode}")
