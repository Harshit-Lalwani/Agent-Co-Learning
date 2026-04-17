from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AgentState:
    last_reward: float = 0.0
    cumulative_reward: float = 0.0
    last_action: np.ndarray | None = None


class HistoryObservationBuilder:
    def __init__(self, num_assets: int, window: int) -> None:
        self.num_assets = num_assets
        self.window = window

    @property
    def observation_dim(self) -> int:
        return self.window * self.num_assets + 3

    def build(
        self,
        return_history: list[np.ndarray],
        step: int,
        episode_length: int,
        agent_state: AgentState,
    ) -> np.ndarray:
        recent = return_history[-self.window :]
        pad = self.window - len(recent)
        pieces: list[np.ndarray] = [np.zeros(self.num_assets, dtype=float) for _ in range(pad)]
        pieces.extend(np.asarray(ret, dtype=float) for ret in recent)
        history_vector = np.concatenate(pieces) if pieces else np.zeros(self.window * self.num_assets, dtype=float)
        step_fraction = 0.0 if episode_length <= 1 else step / float(episode_length - 1)
        private_state = np.array([agent_state.last_reward, agent_state.cumulative_reward, step_fraction], dtype=float)
        return np.concatenate([history_vector, private_state])