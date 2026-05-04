from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AgentState:
    last_reward: float = 0.0
    cumulative_reward: float = 0.0
    last_action: np.ndarray | None = None


class HistoryObservationBuilder:
    """Builds per-agent observations from market return history and private state.

    The observation vector is:
        [normalized_return_history (window * num_assets), last_reward, cumulative_reward, step_fraction]

    Return history is z-score normalized (zero mean, unit variance) across the window.
    This is important for REINFORCE stability — raw returns are ~1e-4 scale, which
    makes linear policy weights need to be correspondingly tiny without normalization.
    """

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
        history_vector = (
            np.concatenate(pieces) if pieces else np.zeros(self.window * self.num_assets, dtype=float)
        )

        # Z-score normalize the history vector so REINFORCE sees O(1) inputs.
        # Skip normalization when all zeros (padding only).
        if len(recent) >= 2:
            h_mean = history_vector.mean()
            h_std = history_vector.std()
            if h_std > 1e-12:
                history_vector = (history_vector - h_mean) / h_std

        step_fraction = 0.0 if episode_length <= 1 else step / float(episode_length - 1)
        private_state = np.array(
            [agent_state.last_reward, agent_state.cumulative_reward, step_fraction],
            dtype=float,
        )
        return np.concatenate([history_vector, private_state])