from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from twmacl.portfolio import project_l1_leverage


@dataclass
class ActionResult:
    mean: np.ndarray
    raw: np.ndarray
    executed: np.ndarray


@dataclass
class LinearGaussianPolicy:
    num_assets: int
    obs_dim: int
    leverage_cap: float
    learning_rate: float
    exploration_std: float
    reward_baseline_decay: float
    rng: np.random.Generator
    init_scale: float = 0.01

    def __post_init__(self) -> None:
        self.weights = self.rng.normal(loc=0.0, scale=self.init_scale, size=(self.num_assets, self.obs_dim))
        self.bias = np.zeros(self.num_assets, dtype=float)
        self.reward_baseline = 0.0

    def compute_mean(self, observation: np.ndarray) -> np.ndarray:
        return self.weights @ observation + self.bias

    def act(self, observation: np.ndarray, deterministic: bool = False, mean_override: np.ndarray | None = None) -> ActionResult:
        if mean_override is not None:
            mean = mean_override
        else:
            mean = self.compute_mean(observation)
            
        if deterministic or self.exploration_std == 0.0:
            raw = mean.copy()
        else:
            raw = mean + self.rng.normal(loc=0.0, scale=self.exploration_std, size=self.num_assets)
        executed = project_l1_leverage(raw, self.leverage_cap)
        return ActionResult(mean=mean, raw=raw, executed=executed)

    def update(self, observation: np.ndarray, action: ActionResult, reward: float, gradient_scale: float = 1.0) -> dict[str, float]:
        self.reward_baseline = (
            self.reward_baseline_decay * self.reward_baseline + (1.0 - self.reward_baseline_decay) * reward
        )
        advantage = reward - self.reward_baseline
        if self.exploration_std > 0.0:
            scale = max(self.exploration_std**2, 1e-12)
            score = (action.raw - action.mean) / scale
            self.weights += self.learning_rate * advantage * np.outer(score, observation) * gradient_scale
            self.bias += self.learning_rate * advantage * score * gradient_scale
        return {
            "reward_baseline": float(self.reward_baseline),
            "advantage": float(advantage),
        }