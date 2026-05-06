from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
        self.weights = self.rng.normal(
            loc=0.0, scale=self.init_scale, size=(self.num_assets, self.obs_dim)
        )
        self.bias = np.zeros(self.num_assets, dtype=float)
        self.reward_baseline = 0.0

        # Running statistics for reward normalization (Welford's online algorithm).
        # Normalizing raw rewards before computing advantages dramatically improves
        # REINFORCE stability when reward magnitudes are tiny (e.g., 1e-4 scale).
        self._reward_count: int = 0
        self._reward_mean: float = 0.0
        self._reward_M2: float = 0.0  # running sum of squared deviations

    def _normalize_reward(self, reward: float) -> float:
        """Online Welford normalization of reward signal."""
        self._reward_count += 1
        delta = reward - self._reward_mean
        self._reward_mean += delta / self._reward_count
        delta2 = reward - self._reward_mean
        self._reward_M2 += delta * delta2

        if self._reward_count < 2:
            return 0.0  # not enough data yet
        variance = self._reward_M2 / (self._reward_count - 1)
        std = max(variance ** 0.5, 1e-8)
        return (reward - self._reward_mean) / std

    def reset_normalizer(self) -> None:
        """Reset the running statistics for reward normalization. Call at episode start."""
        self._reward_count = 0
        self._reward_mean = 0.0
        self._reward_M2 = 0.0

    def compute_mean(self, observation: np.ndarray) -> np.ndarray:
        return self.weights @ observation + self.bias

    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
        mean_override: np.ndarray | None = None,
    ) -> ActionResult:
        if mean_override is not None:
            mean = mean_override
        else:
            mean = self.compute_mean(observation)

        if deterministic or self.exploration_std == 0.0:
            raw = mean.copy()
        else:
            raw = mean + self.rng.normal(
                loc=0.0, scale=self.exploration_std, size=self.num_assets
            )
        executed = project_l1_leverage(raw, self.leverage_cap)
        return ActionResult(mean=mean, raw=raw, executed=executed)

    def update(
        self,
        observation: np.ndarray,
        action: ActionResult,
        reward: float,
        gradient_scale: float = 1.0,
        lr_decay_factor: float = 1.0,
    ) -> dict[str, float]:
        """REINFORCE policy update with reward normalization and LR decay.

        Args:
            observation: The observation the agent received.
            action: The ActionResult from the corresponding act() call.
            reward: The realized portfolio return.
            gradient_scale: Multiplier on gradient (used by S2 to scale by (1-beta)).
            lr_decay_factor: Learning rate multiplier in [0,1] for LR annealing.
        """
        # Normalize reward to zero-mean, unit-variance before computing advantage.
        # This makes REINFORCE robust to tiny or varying reward magnitudes.
        normalized_reward = self._normalize_reward(reward)

        # Exponential moving average baseline on normalized rewards.
        self.reward_baseline = (
            self.reward_baseline_decay * self.reward_baseline
            + (1.0 - self.reward_baseline_decay) * normalized_reward
        )
        advantage = normalized_reward - self.reward_baseline

        if self.exploration_std > 0.0:
            effective_lr = self.learning_rate * lr_decay_factor * gradient_scale
            scale = max(self.exploration_std ** 2, 1e-12)
            score = (action.raw - action.mean) / scale
            self.weights += effective_lr * advantage * np.outer(score, observation)
            self.bias += effective_lr * advantage * score

        return {
            "reward_baseline": float(self.reward_baseline),
            "advantage": float(advantage),
            "normalized_reward": float(normalized_reward),
        }
    
class MLPGaussianPolicy:
    """PyTorch drop-in replacement for LinearGaussianPolicy."""
    def __init__(
        self,
        num_assets: int,
        obs_dim: int,
        leverage_cap: float,
        learning_rate: float,
        exploration_std: float,
        reward_baseline_decay: float,
        rng: np.random.Generator,
        init_scale: float = 0.01,
        hidden_dim: int = 64
    ) -> None:
        self.num_assets = num_assets
        self.obs_dim = obs_dim
        self.leverage_cap = leverage_cap
        self.learning_rate = learning_rate
        self.exploration_std = exploration_std
        self.reward_baseline_decay = reward_baseline_decay
        self.rng = rng
        
        # The Neural Network
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_assets)
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        self.reward_baseline = 0.0
        self._reward_count = 0
        self._reward_mean = 0.0
        self._reward_M2 = 0.0

    def _normalize_reward(self, reward: float) -> float:
        # Welford normalization ported from baseline
        self._reward_count += 1
        delta = reward - self._reward_mean
        self._reward_mean += delta / self._reward_count
        delta2 = reward - self._reward_mean
        self._reward_M2 += delta * delta2

        if self._reward_count < 2:
            return 0.0  
        variance = self._reward_M2 / (self._reward_count - 1)
        std = max(variance ** 0.5, 1e-8)
        return (reward - self._reward_mean) / std

    def reset_normalizer(self) -> None:
        self._reward_count = 0
        self._reward_mean = 0.0
        self._reward_M2 = 0.0

    def compute_mean(self, observation: np.ndarray) -> np.ndarray:
        obs_t = torch.FloatTensor(observation)
        with torch.no_grad():
            return self.network(obs_t).numpy()

    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
        mean_override: np.ndarray | None = None,
    ):
        from twmacl.policies import ActionResult
        from twmacl.portfolio import project_l1_leverage
        
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

    def update(
        self,
        observation: np.ndarray,
        action,
        reward: float,
        gradient_scale: float = 1.0,
        lr_decay_factor: float = 1.0,
    ) -> dict[str, float]:
        
        normalized_reward = self._normalize_reward(reward)
        self.reward_baseline = (
            self.reward_baseline_decay * self.reward_baseline
            + (1.0 - self.reward_baseline_decay) * normalized_reward
        )
        advantage = normalized_reward - self.reward_baseline

        if self.exploration_std > 0.0:
            obs_t = torch.FloatTensor(observation)
            action_raw_t = torch.FloatTensor(action.raw)
            
            mean_t = self.network(obs_t)
            dist = torch.distributions.Normal(mean_t, self.exploration_std)
            log_prob = dist.log_prob(action_raw_t).sum()
            
            # REINFORCE Loss scaled by S2 Trust factor
            loss = -log_prob * advantage * gradient_scale
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate * lr_decay_factor
                
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

        return {
            "reward_baseline": float(self.reward_baseline),
            "advantage": float(advantage),
            "normalized_reward": float(normalized_reward),
        }