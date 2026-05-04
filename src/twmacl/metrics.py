from __future__ import annotations

import numpy as np


class RunningSharpe:
    """Tracks per-agent Sharpe ratio over a stream of per-step returns.

    The running Sharpe is mean(r) / std(r), optionally annualized.
    For episode-level comparison use annualization_factor = steps_per_year.
    """

    def __init__(self, num_agents: int) -> None:
        self.num_agents = num_agents
        self.history: list[np.ndarray] = []

    def update(self, agent_returns: np.ndarray, annualization_factor: float = 1.0) -> np.ndarray:
        """Append one step's returns and return the running Sharpe vector.

        annualization_factor: multiply Sharpe by sqrt(annualization_factor).
            E.g. pass steps_per_year=252 to get an annualized Sharpe.
        """
        self.history.append(agent_returns.copy())
        arr = np.asarray(self.history)
        means = np.mean(arr, axis=0)
        stds = np.std(arr, axis=0)
        sharpe = np.zeros(self.num_agents, dtype=float)
        valid = stds > 1e-12
        sharpe[valid] = means[valid] / stds[valid] * np.sqrt(annualization_factor)
        return sharpe

    def cumulative_return(self) -> np.ndarray:
        """Sum of per-step returns (additive, not compounded)."""
        if not self.history:
            return np.zeros(self.num_agents, dtype=float)
        arr = np.asarray(self.history)
        return np.sum(arr, axis=0)

    def log_cumulative_return(self) -> np.ndarray:
        """Log cumulative return: sum of log(1 + r_t). More accurate for compounding."""
        if not self.history:
            return np.zeros(self.num_agents, dtype=float)
        arr = np.asarray(self.history)
        # Clip to avoid log of negative (shouldn't happen but guard for extreme leverage)
        return np.sum(np.log1p(np.clip(arr, -0.9999, None)), axis=0)

    def reset(self) -> None:
        """Clear history for re-use across episodes."""
        self.history = []
