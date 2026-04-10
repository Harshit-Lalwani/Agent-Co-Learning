from __future__ import annotations

import numpy as np


def entropy_per_agent(normalized_trust: np.ndarray) -> np.ndarray:
    num_agents = normalized_trust.shape[0]
    entropy = np.zeros(num_agents, dtype=float)
    for i in range(num_agents):
        probs = normalized_trust[i].copy()
        probs[i] = 0.0
        probs = probs[probs > 0]
        entropy[i] = -np.sum(probs * np.log(probs))
    return entropy


def asymmetry_index(normalized_trust: np.ndarray) -> float:
    m = normalized_trust.shape[0]
    total = 0.0
    count = 0
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            total += abs(normalized_trust[i, j] - normalized_trust[j, i])
            count += 1
    if count == 0:
        return 0.0
    return total / count


def rolling_entropy_slope(entropy_history: np.ndarray, window: int) -> np.ndarray:
    num_steps, num_agents = entropy_history.shape
    slopes = np.full(num_agents, np.nan, dtype=float)
    if num_steps < window:
        return slopes

    x = np.arange(window, dtype=float)
    x_centered = x - np.mean(x)
    denom = np.sum(x_centered**2)
    y_window = entropy_history[-window:]

    for agent_idx in range(num_agents):
        y = y_window[:, agent_idx]
        y_centered = y - np.mean(y)
        slopes[agent_idx] = np.sum(x_centered * y_centered) / denom
    return slopes


def convergence_flags(slopes: np.ndarray, threshold: float) -> np.ndarray:
    flags = np.zeros_like(slopes, dtype=bool)
    finite = np.isfinite(slopes)
    flags[finite] = np.abs(slopes[finite]) < threshold
    return flags
