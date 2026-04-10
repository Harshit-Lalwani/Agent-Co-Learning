from __future__ import annotations

import numpy as np


class RunningSharpe:
    def __init__(self, num_agents: int) -> None:
        self.num_agents = num_agents
        self.history: list[np.ndarray] = []

    def update(self, agent_returns: np.ndarray) -> np.ndarray:
        self.history.append(agent_returns.copy())
        arr = np.asarray(self.history)
        means = np.mean(arr, axis=0)
        stds = np.std(arr, axis=0)
        sharpe = np.zeros(self.num_agents, dtype=float)
        valid = stds > 1e-12
        sharpe[valid] = means[valid] / stds[valid]
        return sharpe

    def cumulative_return(self) -> np.ndarray:
        if not self.history:
            return np.zeros(self.num_agents, dtype=float)
        arr = np.asarray(self.history)
        return np.sum(arr, axis=0)
