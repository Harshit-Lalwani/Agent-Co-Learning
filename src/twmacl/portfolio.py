from __future__ import annotations

import numpy as np


def project_l1_leverage(weights: np.ndarray, leverage_cap: float) -> np.ndarray:
    l1_norm = float(np.sum(np.abs(weights)))
    if l1_norm <= leverage_cap or l1_norm == 0.0:
        return weights
    return (weights / l1_norm) * leverage_cap


def sample_no_learning_weights(
    num_agents: int,
    num_assets: int,
    leverage_cap: float,
    rng: np.random.Generator,
) -> np.ndarray:
    raw = rng.normal(loc=0.0, scale=1.0, size=(num_agents, num_assets))
    projected = np.zeros_like(raw)
    for agent_idx in range(num_agents):
        projected[agent_idx] = project_l1_leverage(raw[agent_idx], leverage_cap)
    return projected
