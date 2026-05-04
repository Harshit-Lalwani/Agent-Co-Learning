from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from twmacl.baselines import evaluate_markowitz, markowitz_sharpe_analytical
from twmacl.config import Phase1Config
from twmacl.diagnostics import asymmetry_index, convergence_flags, entropy_per_agent, rolling_entropy_slope
from twmacl.logging_io import ParquetSink, write_aggregate_index, write_summary
from twmacl.market import CorrelatedMarket
from twmacl.metrics import RunningSharpe
from twmacl.portfolio import sample_no_learning_weights
from twmacl.predictors import build_predictor
from twmacl.trust import TrustMatrix


SCHEMA_VERSION = "1.0.0"


def _flatten_matrix(matrix: np.ndarray, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    m = matrix.shape[0]
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            out[f"{prefix}_{i}_{j}"] = float(matrix[i, j])
    return out


def _flatten_vector(vector: np.ndarray, prefix: str) -> dict[str, float]:
    return {f"{prefix}_{idx}": float(value) for idx, value in enumerate(vector)}


def run_phase1(config: Phase1Config) -> None:
    output_root = config.output_root_path()
    output_root.mkdir(parents=True, exist_ok=True)

    mu = np.asarray(config.market.mu, dtype=float)
    cov = np.asarray(config.market.cov, dtype=float)
    markowitz_analytical = markowitz_sharpe_analytical(mu, cov)

    summaries: list[dict[str, float | int | str]] = []
    config_hash = config.config_hash()

    for seed in config.seed_values():
        run_start = time.perf_counter()
        num_agents = config.environment.num_agents
        num_assets = config.environment.num_assets
        episode_length = config.environment.episode_length

        market = CorrelatedMarket(config.market.mu, config.market.cov, seed=seed)
        predictor = build_predictor(
            predictor_mode=config.predictor.predictor_mode,
            num_agents=num_agents,
            num_assets=num_assets,
            predictor_window=config.predictor.predictor_window,
            noise_std=config.predictor.noise_std,
            mu=np.asarray(config.market.mu, dtype=float),
            cov=np.asarray(config.market.cov, dtype=float),
            seed=seed + 10_000,
        )
        trust = TrustMatrix(
            num_agents=num_agents,
            alpha=config.trust.trust_alpha,
            lambda_=config.trust.trust_lambda,
        )
        sharpe = RunningSharpe(num_agents=num_agents)
        sink = ParquetSink()
        weights_rng = np.random.default_rng(seed + 20_000)

        return_history: list[np.ndarray] = []
        entropy_history: list[np.ndarray] = []
        convergence_counter = np.zeros(num_agents, dtype=int)
        convergence_time = np.full(num_agents, -1, dtype=int)
        max_leverage = np.zeros(num_agents, dtype=float)

        for step in range(episode_length):
            realized_return = market.sample_return()
            return_history.append(realized_return)

            weights = sample_no_learning_weights(
                num_agents=num_agents,
                num_assets=num_assets,
                leverage_cap=config.environment.leverage_cap,
                rng=weights_rng,
            )
            leverage_used = np.sum(np.abs(weights), axis=1)
            max_leverage = np.maximum(max_leverage, leverage_used)
            agent_returns = np.sum(weights * realized_return[None, :], axis=1)
            sharpe_running = sharpe.update(agent_returns)

            predictions = predictor.predict(step=step, return_history=return_history[:-1])
            trust_applied = predictor.trust_update_enabled(step)
            if trust_applied:
                trust.update(predictions=predictions, realized_return=realized_return)

            normalized_trust = trust.normalized()
            entropy = entropy_per_agent(normalized_trust)
            entropy_history.append(entropy)
            asymmetry = asymmetry_index(normalized_trust)

            slopes = rolling_entropy_slope(np.asarray(entropy_history), config.trust.entropy_window)
            converged = convergence_flags(slopes, config.trust.entropy_slope_threshold)

            for agent_idx in range(num_agents):
                if converged[agent_idx]:
                    convergence_counter[agent_idx] += 1
                    if (
                        convergence_counter[agent_idx] >= config.trust.convergence_persistence
                        and convergence_time[agent_idx] == -1
                    ):
                        convergence_time[agent_idx] = step
                else:
                    convergence_counter[agent_idx] = 0

            step_row: dict[str, float | int | bool] = {
                "schema_version": SCHEMA_VERSION,
                "seed": seed,
                "step": step,
                "trust_update_applied": bool(trust_applied),
                "asymmetry_index": float(asymmetry),
                "system_converged": bool(np.all(convergence_time >= 0)),
            }
            step_row.update(_flatten_matrix(normalized_trust, "tau"))
            step_row.update(_flatten_vector(entropy, "entropy"))
            step_row.update(_flatten_vector(slopes, "entropy_slope"))
            step_row.update(_flatten_vector(agent_returns, "agent_return"))
            step_row.update(_flatten_vector(sharpe_running, "sharpe_running"))
            step_row.update(_flatten_vector(converged.astype(float), "agent_converged"))
            sink.append_step(step_row)

        step_file = output_root / "steps" / f"seed_{seed:05d}_steps.parquet"
        sink.write_steps(step_file)

        run_duration = time.perf_counter() - run_start
        summary: dict[str, float | int | str] = {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "config_hash": config_hash,
            "num_steps": episode_length,
            "predictor_mode": config.predictor.predictor_mode,
            "run_duration_seconds": float(run_duration),
            "final_asymmetry": float(asymmetry_index(trust.normalized())),
        }
        summary.update(_flatten_vector(np.asarray(convergence_time, dtype=float), "convergence_time"))
        summary.update(_flatten_vector(entropy_history[-1], "final_entropy"))
        summary.update(_flatten_vector(sharpe.cumulative_return(), "cumulative_return"))
        summary.update(_flatten_vector(sharpe_running, "sharpe_ratio"))
        summary.update(_flatten_vector(max_leverage, "max_leverage_used"))
        summary["markowitz_sharpe_analytical"] = float(markowitz_analytical)

        run_file = output_root / "runs" / f"seed_{seed:05d}_summary.parquet"
        write_summary(summary, run_file)
        summaries.append(summary)

    aggregate_file = output_root / "aggregate" / "run_index.parquet"
    write_aggregate_index(summaries, aggregate_file)


__all__ = ["run_phase1"]
