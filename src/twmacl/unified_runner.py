from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from twmacl.baselines import evaluate_markowitz, markowitz_sharpe_analytical
from twmacl.config import ExperimentRunConfig
from twmacl.diagnostics import asymmetry_index, entropy_per_agent
from twmacl.logging_io import ParquetSink, write_aggregate_index, write_summary
from twmacl.market import CorrelatedMarket
from twmacl.metrics import RunningSharpe
from twmacl.observation import AgentState, HistoryObservationBuilder
from twmacl.policies import ActionResult, LinearGaussianPolicy
from twmacl.predictors import build_predictor
from twmacl.trust import TrustMatrix


SCHEMA_VERSION = "3.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _build_policies(config: ExperimentRunConfig, obs_dim: int, seed: int) -> list[LinearGaussianPolicy]:
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


def _episode_seed(seed: int, episode: int, offset: int) -> int:
    return seed + episode * offset


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _run_episode(
    *,
    config: ExperimentRunConfig,
    seed: int,
    episode: int,
    total_episodes: int,
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
    steps_per_year = config.learning.steps_per_year
    beta = config.learning.imitation_beta

    # Reset trust at episode start unless persistence is enabled.
    if not config.trust.trust_persistence:
        trust.reset()

    market = CorrelatedMarket(
        config.market.mu,
        config.market.cov,
        seed=_episode_seed(seed, episode, episode_offset),
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

    # Learning rate decay: linearly anneal from 1.0 to 0.1 over training episodes.
    # Applied only during training; eval always uses the current (decayed) weights.
    lr_decay_factor = 1.0
    if mode == "train" and total_episodes > 1:
        lr_decay_factor = max(0.1, 1.0 - 0.9 * episode / (total_episodes - 1))

    # Reset reward normalizer at the start of each episode
    for policy in policies:
        policy.reset_normalizer()

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

        observations = [
            builder.build(history_returns[:-1], step, episode_length, agent_state)
            for agent_state in agent_states
        ]

        # ----------------------------------------------------------------
        # Action selection: S1 vs S2 branching
        # ----------------------------------------------------------------
        if beta > 0.0:
            # S2: trust-weighted policy imitation.
            # Hybrid mean: μ_hybrid_i = (1-β)μ_i + β Σ_j τ_ij μ_j
            base_means = np.array([
                policy.compute_mean(obs)
                for policy, obs in zip(policies, observations)
            ])
            peer_means = normalized_trust @ base_means
            hybrid_means = (1.0 - beta) * base_means + beta * peer_means
            action_results: list[ActionResult] = [
                policy.act(obs, deterministic=(mode == "eval"), mean_override=hybrid_means[i])
                for i, (policy, obs) in enumerate(zip(policies, observations, strict=True))
            ]
        else:
            # S1: independent learning — each agent uses its own policy mean.
            action_results = [
                policy.act(obs, deterministic=(mode == "eval"))
                for policy, obs in zip(policies, observations, strict=True)
            ]

        executed_actions = np.vstack([result.executed for result in action_results])
        raw_actions = np.vstack([result.raw for result in action_results])
        agent_returns = np.sum(executed_actions * realized_return[None, :], axis=1)
        sharpe_running = sharpe.update(agent_returns, annualization_factor=float(steps_per_year))
        final_sharpe_running = sharpe_running
        cumulative_return += agent_returns

        # ----------------------------------------------------------------
        # Policy updates (training only)
        # ----------------------------------------------------------------
        step_updates: list[dict[str, float]] = []
        for agent_idx, policy in enumerate(policies):
            agent_state = agent_states[agent_idx]
            agent_state.last_reward = float(agent_returns[agent_idx])
            agent_state.cumulative_reward += float(agent_returns[agent_idx])
            agent_state.last_action = executed_actions[agent_idx].copy()

            update_info = {"reward_baseline": policy.reward_baseline, "advantage": 0.0, "normalized_reward": 0.0}
            if mode == "train":
                # S2 scales gradient by (1-β) since the action is a blend.
                grad_scale = (1.0 - beta) if beta > 0.0 else 1.0
                update_info = policy.update(
                    observations[agent_idx],
                    action_results[agent_idx],
                    float(agent_returns[agent_idx]),
                    gradient_scale=grad_scale,
                    lr_decay_factor=lr_decay_factor,
                )
            step_updates.append(update_info)

        # ----------------------------------------------------------------
        # Step logging
        # ----------------------------------------------------------------
        step_row: dict[str, float | int | str | bool] = {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "episode": episode,
            "step": step,
            "mode": mode,
            "is_s2": bool(beta > 0.0),
            "imitation_beta": float(beta),
            "trust_update_applied": bool(predictor.trust_update_enabled(step)),
            "asymmetry_index": float(final_asymmetry),
            "lr_decay_factor": float(lr_decay_factor),
        }
        step_row.update(_flatten_square_matrix(normalized_trust, "tau"))
        step_row.update(_flatten_vector(entropy, "entropy"))
        step_row.update(_flatten_vector(agent_returns, "agent_return"))
        step_row.update(_flatten_vector(cumulative_return, "cumulative_return"))
        step_row.update(_flatten_vector(sharpe_running, "sharpe_running"))
        step_row.update(
            _flatten_vector(
                np.array([info["reward_baseline"] for info in step_updates], dtype=float),
                "reward_baseline",
            )
        )
        step_row.update(
            _flatten_vector(
                np.array([info["advantage"] for info in step_updates], dtype=float),
                "advantage",
            )
        )
        step_row.update(_flatten_rect_matrix(executed_actions, "action"))
        step_row.update(_flatten_rect_matrix(raw_actions, "raw_action"))
        sink.append_step(step_row)

    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "episode": episode,
        "mode": mode,
        "is_s2": bool(beta > 0.0),
        "imitation_beta": float(beta),
        "final_asymmetry": float(final_asymmetry),
        "mean_final_entropy": float(np.mean(final_entropy)),
        "cumulative_return_mean": float(np.mean(cumulative_return)),
        "cumulative_return_std": float(np.std(cumulative_return)),
        "final_sharpe_mean": float(np.mean(final_sharpe_running)),
        "num_steps": episode_length,
    }


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_experiment(config: ExperimentRunConfig) -> None:
    """Run S1 (beta=0) or S2 (beta>0) experiment across all seeds.

    Phases 2 and 3 from the plan both use this function — the strategy
    is entirely determined by config.learning.imitation_beta.
    """
    output_root = config.output_root_path()
    output_root.mkdir(parents=True, exist_ok=True)

    mu = np.asarray(config.market.mu, dtype=float)
    cov = np.asarray(config.market.cov, dtype=float)
    markowitz_analytical_sharpe = markowitz_sharpe_analytical(mu, cov)

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
        # Trust matrix is created once per seed; episodes reset it unless
        # trust_persistence=True in config.
        trust = TrustMatrix(
            num_agents=num_agents,
            alpha=config.trust.trust_alpha,
            lambda_=config.trust.trust_lambda,
        )

        # Markowitz benchmark for this seed (ground-truth upper bound).
        markowitz_bench = evaluate_markowitz(
            mu=mu,
            cov=cov,
            leverage_cap=config.environment.leverage_cap,
            num_steps=config.environment.episode_length,
            seed=seed,
            steps_per_year=float(config.learning.steps_per_year),
        )

        train_sink = ParquetSink()
        eval_sink = ParquetSink()
        num_train = config.learning.num_train_episodes
        num_eval = config.learning.num_eval_episodes

        train_summaries: list[dict[str, float | int | str]] = []
        for episode in range(num_train):
            train_summaries.append(
                _run_episode(
                    config=config,
                    seed=seed,
                    episode=episode,
                    total_episodes=num_train,
                    mode="train",
                    policies=policies,
                    builder=builder,
                    trust=trust,
                    predictor_seed=seed + 20_000,
                    episode_offset=10_000,
                    sink=train_sink,
                )
            )

        eval_summaries: list[dict[str, float | int | str]] = []
        for episode in range(num_eval):
            eval_summaries.append(
                _run_episode(
                    config=config,
                    seed=seed,
                    episode=episode,
                    total_episodes=num_eval,
                    mode="eval",
                    policies=policies,
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

        # Save policy checkpoints.
        policy_checkpoint = output_root / "checkpoints" / f"seed_{seed:05d}_policy.npz"
        policy_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_payload: dict[str, np.ndarray] = {}
        for agent_idx, policy in enumerate(policies):
            checkpoint_payload[f"weights_{agent_idx}"] = policy.weights
            checkpoint_payload[f"bias_{agent_idx}"] = policy.bias
            checkpoint_payload[f"reward_baseline_{agent_idx}"] = np.array(
                [policy.reward_baseline], dtype=float
            )
        np.savez(policy_checkpoint, **checkpoint_payload)

        # Per-seed summary.
        train_sharpe_by_ep = [float(item["final_sharpe_mean"]) for item in train_summaries]
        eval_sharpe_by_ep = [float(item["final_sharpe_mean"]) for item in eval_summaries]
        run_duration = time.perf_counter() - run_start

        summary: dict[str, float | int | str] = {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "config_hash": config_hash,
            "is_s2": bool(config.is_s2),
            "imitation_beta": float(config.learning.imitation_beta),
            "num_train_episodes": num_train,
            "num_eval_episodes": num_eval,
            "episode_length": config.environment.episode_length,
            "learning_rate": config.learning.learning_rate,
            "exploration_std": config.learning.exploration_std,
            "observation_window": config.learning.observation_window,
            "mode": "train_eval",
            "run_duration_seconds": float(run_duration),
            "final_train_asymmetry": float(train_summaries[-1]["final_asymmetry"]),
            "final_eval_asymmetry": float(eval_summaries[-1]["final_asymmetry"]),
            # Markowitz benchmark for comparison.
            "markowitz_sharpe_analytical": float(markowitz_analytical_sharpe),
            "markowitz_sharpe_simulated": float(markowitz_bench["sharpe"]),
            "markowitz_annualized_sharpe": float(markowitz_bench["annualized_sharpe"]),
            "markowitz_cumulative_return": float(markowitz_bench["cumulative_return"]),
        }
        summary.update(
            _flatten_vector(np.asarray(train_sharpe_by_ep, dtype=float), "train_final_sharpe_mean_episode")
        )
        summary.update(
            _flatten_vector(np.asarray(eval_sharpe_by_ep, dtype=float), "eval_final_sharpe_mean_episode")
        )
        summary.update(
            _flatten_vector(
                np.asarray([float(item["cumulative_return_mean"]) for item in train_summaries], dtype=float),
                "train_cumulative_return_mean_episode",
            )
        )
        summary.update(
            _flatten_vector(
                np.asarray([float(item["cumulative_return_mean"]) for item in eval_summaries], dtype=float),
                "eval_cumulative_return_mean_episode",
            )
        )

        run_file = output_root / "runs" / f"seed_{seed:05d}_summary.parquet"
        write_summary(summary, run_file)
        summaries.append(summary)

    aggregate_file = output_root / "aggregate" / "run_index.parquet"
    write_aggregate_index(summaries, aggregate_file)


__all__ = ["run_experiment"]
