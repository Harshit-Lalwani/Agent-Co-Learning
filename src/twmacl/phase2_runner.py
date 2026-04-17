from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from twmacl.config import Phase2Config
from twmacl.diagnostics import asymmetry_index, entropy_per_agent
from twmacl.logging_io import ParquetSink, write_aggregate_index, write_summary
from twmacl.market import CorrelatedMarket
from twmacl.metrics import RunningSharpe
from twmacl.observation import AgentState, HistoryObservationBuilder
from twmacl.policies import ActionResult, LinearGaussianPolicy
from twmacl.predictors import build_predictor
from twmacl.trust import TrustMatrix


SCHEMA_VERSION = "2.0.0"


def _flatten_vector(vector: np.ndarray, prefix: str) -> dict[str, float]:
    return {f"{prefix}_{idx}": float(value) for idx, value in enumerate(vector)}


def _flatten_rect_matrix(matrix: np.ndarray, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            out[f"{prefix}_{i}_{j}"] = float(matrix[i, j])
    return out


def _flatten_square_matrix(matrix: np.ndarray, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j:
                continue
            out[f"{prefix}_{i}_{j}"] = float(matrix[i, j])
    return out


def _build_policies(config: Phase2Config, obs_dim: int, seed: int) -> list[LinearGaussianPolicy]:
    policies: list[LinearGaussianPolicy] = []
    for agent_idx in range(config.environment.num_agents):
        policies.append(
            LinearGaussianPolicy(
                num_assets=config.environment.num_assets,
                obs_dim=obs_dim,
                leverage_cap=config.environment.leverage_cap,
                learning_rate=config.learning.learning_rate,
                exploration_std=config.learning.exploration_std,
                reward_baseline_decay=config.learning.reward_baseline_decay,
                rng=np.random.default_rng(seed + 5_000 + agent_idx),
                init_scale=config.learning.policy_init_scale,
            )
        )
    return policies


def _episode_seed(base_seed: int, seed: int, episode: int, offset: int) -> int:
    return base_seed + seed * offset + episode


def _run_episode(
    *,
    config: Phase2Config,
    seed: int,
    episode: int,
    mode: str,
    policies: list[LinearGaussianPolicy],
    builder: HistoryObservationBuilder,
    trust: TrustMatrix,
    predictor_seed: int,
    episode_offset: int,
    sink: ParquetSink,
) -> dict[str, float | int | str]:
    num_agents = config.environment.num_agents
    num_assets = config.environment.num_assets
    episode_length = config.environment.episode_length

    market = CorrelatedMarket(
        config.market.mu,
        config.market.cov,
        seed=_episode_seed(seed, seed, episode, episode_offset),
    )
    predictor = build_predictor(
        predictor_mode=config.predictor.predictor_mode,
        num_agents=num_agents,
        num_assets=num_assets,
        predictor_window=config.predictor.predictor_window,
        noise_std=config.predictor.noise_std,
        mu=np.asarray(config.market.mu, dtype=float),
        cov=np.asarray(config.market.cov, dtype=float),
        seed=predictor_seed + episode,
    )

    history_returns: list[np.ndarray] = []
    agent_states = [AgentState() for _ in range(num_agents)]
    sharpe = RunningSharpe(num_agents=num_agents)
    cumulative_return = np.zeros(num_agents, dtype=float)
    final_entropy = np.zeros(num_agents, dtype=float)
    final_asymmetry = 0.0
    final_sharpe_running = np.zeros(num_agents, dtype=float)

    for step in range(episode_length):
        realized_return = market.sample_return()
        history_returns.append(realized_return)

        predictions = predictor.predict(step=step, return_history=history_returns[:-1])
        if predictor.trust_update_enabled(step):
            trust.update(predictions=predictions, realized_return=realized_return)

        normalized_trust = trust.normalized()
        entropy = entropy_per_agent(normalized_trust)
        final_entropy = entropy
        final_asymmetry = asymmetry_index(normalized_trust)

        observations = [builder.build(history_returns[:-1], step, episode_length, agent_state) for agent_state in agent_states]
        action_results: list[ActionResult] = [
            policy.act(observation, deterministic=(mode == "eval"))
            for policy, observation in zip(policies, observations, strict=True)
        ]
        executed_actions = np.vstack([result.executed for result in action_results])
        raw_actions = np.vstack([result.raw for result in action_results])
        agent_returns = np.sum(executed_actions * realized_return[None, :], axis=1)
        sharpe_running = sharpe.update(agent_returns)
        final_sharpe_running = sharpe_running
        cumulative_return += agent_returns

        step_updates: list[dict[str, float]] = []
        for agent_idx, policy in enumerate(policies):
            agent_state = agent_states[agent_idx]
            agent_state.last_reward = float(agent_returns[agent_idx])
            agent_state.cumulative_reward += float(agent_returns[agent_idx])
            agent_state.last_action = executed_actions[agent_idx].copy()

            update_info = {"reward_baseline": policy.reward_baseline, "advantage": 0.0}
            if mode == "train":
                update_info = policy.update(observations[agent_idx], action_results[agent_idx], float(agent_returns[agent_idx]))
            step_updates.append(update_info)

        step_row: dict[str, float | int | str | bool] = {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "episode": episode,
            "step": step,
            "mode": mode,
            "trust_update_applied": bool(predictor.trust_update_enabled(step)),
            "asymmetry_index": float(final_asymmetry),
        }
        step_row.update(_flatten_square_matrix(normalized_trust, "tau"))
        step_row.update(_flatten_vector(entropy, "entropy"))
        step_row.update(_flatten_vector(agent_returns, "agent_return"))
        step_row.update(_flatten_vector(cumulative_return, "cumulative_return"))
        step_row.update(_flatten_vector(sharpe_running, "sharpe_running"))
        step_row.update(_flatten_vector(np.array([info["reward_baseline"] for info in step_updates], dtype=float), "reward_baseline"))
        step_row.update(_flatten_vector(np.array([info["advantage"] for info in step_updates], dtype=float), "advantage"))
        step_row.update(_flatten_rect_matrix(executed_actions, "action"))
        step_row.update(_flatten_rect_matrix(raw_actions, "raw_action"))
        sink.append_step(step_row)

    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "episode": episode,
        "mode": mode,
        "final_asymmetry": float(final_asymmetry),
        "mean_final_entropy": float(np.mean(final_entropy)),
        "cumulative_return_mean": float(np.mean(cumulative_return)),
        "cumulative_return_std": float(np.std(cumulative_return)),
        "final_sharpe_mean": float(np.mean(final_sharpe_running)),
        "num_steps": episode_length,
    }


def run_phase2(config: Phase2Config) -> None:
    output_root = config.output_root_path()
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, float | int | str]] = []
    config_hash = config.config_hash()

    for seed in config.seed_values():
        run_start = time.perf_counter()
        num_agents = config.environment.num_agents
        num_assets = config.environment.num_assets
        builder = HistoryObservationBuilder(
            num_assets=num_assets,
            window=config.learning.observation_window,
        )
        policies = _build_policies(config, builder.observation_dim, seed)
        trust = TrustMatrix(num_agents=num_agents, alpha=config.trust.trust_alpha, lambda_=config.trust.trust_lambda)

        train_sink = ParquetSink()
        eval_sink = ParquetSink()

        train_summaries: list[dict[str, float | int | str]] = []
        for episode in range(config.learning.num_train_episodes):
            train_summaries.append(
                _run_episode(
                    config=config,
                    seed=seed,
                    episode=episode,
                    mode="train",
                    policies=policies,
                    builder=builder,
                    trust=trust,
                    predictor_seed=seed + 20_000,
                    episode_offset=10_000,
                    sink=train_sink,
                )
            )

        eval_policies = policies
        eval_summaries: list[dict[str, float | int | str]] = []
        for episode in range(config.learning.num_eval_episodes):
            eval_summaries.append(
                _run_episode(
                    config=config,
                    seed=seed,
                    episode=episode,
                    mode="eval",
                    policies=eval_policies,
                    builder=builder,
                    trust=trust,
                    predictor_seed=seed + 40_000,
                    episode_offset=20_000,
                    sink=eval_sink,
                )
            )

        train_step_file = output_root / "train" / "steps" / f"seed_{seed:05d}_steps.parquet"
        eval_step_file = output_root / "eval" / "steps" / f"seed_{seed:05d}_steps.parquet"
        train_sink.write_steps(train_step_file)
        eval_sink.write_steps(eval_step_file)

        policy_checkpoint = output_root / "checkpoints" / f"seed_{seed:05d}_policy.npz"
        policy_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_payload: dict[str, np.ndarray] = {}
        for agent_idx, policy in enumerate(policies):
            checkpoint_payload[f"weights_{agent_idx}"] = policy.weights
            checkpoint_payload[f"bias_{agent_idx}"] = policy.bias
            checkpoint_payload[f"reward_baseline_{agent_idx}"] = np.array([policy.reward_baseline], dtype=float)
        np.savez(policy_checkpoint, **checkpoint_payload)

        train_summary = train_summaries[-1]
        eval_summary = eval_summaries[-1]
        run_duration = time.perf_counter() - run_start
        summary: dict[str, float | int | str] = {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "config_hash": config_hash,
            "num_train_episodes": config.learning.num_train_episodes,
            "num_eval_episodes": config.learning.num_eval_episodes,
            "episode_length": config.environment.episode_length,
            "learning_rate": config.learning.learning_rate,
            "exploration_std": config.learning.exploration_std,
            "mode": "train_eval",
            "run_duration_seconds": float(run_duration),
            "final_train_asymmetry": float(train_summary["final_asymmetry"]),
            "final_eval_asymmetry": float(eval_summary["final_asymmetry"]),
        }
        summary.update(_flatten_vector(np.asarray([item["cumulative_return_mean"] for item in train_summaries], dtype=float), "train_cumulative_return_mean_episode"))
        summary.update(_flatten_vector(np.asarray([item["cumulative_return_mean"] for item in eval_summaries], dtype=float), "eval_cumulative_return_mean_episode"))
        summary.update(_flatten_vector(np.asarray([item["final_sharpe_mean"] for item in train_summaries], dtype=float), "train_final_sharpe_mean_episode"))
        summary.update(_flatten_vector(np.asarray([item["final_sharpe_mean"] for item in eval_summaries], dtype=float), "eval_final_sharpe_mean_episode"))

        run_file = output_root / "runs" / f"seed_{seed:05d}_summary.parquet"
        write_summary(summary, run_file)
        summaries.append(summary)

    aggregate_file = output_root / "aggregate" / "run_index.parquet"
    write_aggregate_index(summaries, aggregate_file)


__all__ = ["run_phase2"]