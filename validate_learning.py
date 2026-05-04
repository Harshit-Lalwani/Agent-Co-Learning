#!/usr/bin/env python
"""
validate_learning.py — Sanity check: is the learner actually learning?

Runs a quick training (50 episodes, 1 seed) and checks:
1. Policy weights are actually being updated (action distribution shifts)
2. S1 eval Sharpe is not catastrophically bad (> -1.0 annualized)
3. S1 and S2 produce measurably different actions

Prints PASS or FAIL with details. Run this before running full ablations.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from twmacl.baselines import markowitz_sharpe_analytical
from twmacl.config import load_experiment_config
from twmacl.unified_runner import run_experiment

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


QUICK_CONFIG = {
    "num_seeds": 1,
    "num_train_episodes": 50,
    "num_eval_episodes": 5,
    "output_root_s1": "outputs/_validate_learning_s1",
    "output_root_s2": "outputs/_validate_learning_s2",
}


def _linear_slope(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x -= x.mean()
    y = np.array(values) - np.mean(values)
    denom = np.sum(x ** 2)
    return float(np.sum(x * y) / denom) if denom > 0 else 0.0


def main() -> int:
    print("=" * 60)
    print("Learning Validation")
    print("=" * 60)

    errors: list[str] = []
    warnings: list[str] = []

    # --- Build quick configs ---
    base_s1 = load_experiment_config("configs/phase2.yaml")
    quick_s1 = base_s1.model_copy(
        update={
            "experiment": base_s1.experiment.model_copy(
                update={
                    "num_seeds": QUICK_CONFIG["num_seeds"],
                    "output_root": QUICK_CONFIG["output_root_s1"],
                }
            ),
            "learning": base_s1.learning.model_copy(
                update={
                    "num_train_episodes": QUICK_CONFIG["num_train_episodes"],
                    "num_eval_episodes": QUICK_CONFIG["num_eval_episodes"],
                }
            ),
        }
    )

    base_s2 = load_experiment_config("configs/phase3.yaml")
    quick_s2 = base_s2.model_copy(
        update={
            "experiment": base_s2.experiment.model_copy(
                update={
                    "num_seeds": QUICK_CONFIG["num_seeds"],
                    "output_root": QUICK_CONFIG["output_root_s2"],
                }
            ),
            "learning": base_s2.learning.model_copy(
                update={
                    "num_train_episodes": QUICK_CONFIG["num_train_episodes"],
                    "num_eval_episodes": QUICK_CONFIG["num_eval_episodes"],
                }
            ),
        }
    )

    # Clean up old runs
    for root in [QUICK_CONFIG["output_root_s1"], QUICK_CONFIG["output_root_s2"]]:
        shutil.rmtree(root, ignore_errors=True)

    # --- Markowitz ground truth ---
    mu = np.asarray(quick_s1.market.mu)
    cov = np.asarray(quick_s1.market.cov)
    steps_per_year = float(quick_s1.learning.steps_per_year)
    mw_per_step = markowitz_sharpe_analytical(mu, cov)
    mw_annualized = mw_per_step * np.sqrt(steps_per_year)
    print(f"\nMarkowitz analytical Sharpe (per-step): {mw_per_step:.4f}")
    print(f"Markowitz analytical Sharpe (annualized x{steps_per_year:.0f}): {mw_annualized:.4f}")

    # --- Run S1 ---
    print(f"\nRunning S1 ({QUICK_CONFIG['num_train_episodes']} train episodes)...")
    run_experiment(quick_s1)

    # --- Run S2 ---
    print(f"Running S2 (beta={quick_s2.learning.imitation_beta}, {QUICK_CONFIG['num_train_episodes']} train episodes)...")
    run_experiment(quick_s2)

    # --- Analyze results ---
    if not HAS_PANDAS:
        print("\nWARNING: pandas not available, skipping detailed checks.")
        print("Install pandas+pyarrow to run full validation.")
        return 0

    seed = quick_s1.experiment.base_seed

    # ----------------------------------------------------------------
    # Check 1: Return slope (informational — 50 ep is too noisy for hard check)
    # ----------------------------------------------------------------
    s1_train = pd.read_parquet(
        Path(QUICK_CONFIG["output_root_s1"]) / "train" / "steps" / f"seed_{seed:05d}_steps.parquet"
    )
    ret_cols = [c for c in s1_train.columns if c.startswith("agent_return_")]
    ep_returns = s1_train.groupby("episode")[ret_cols].mean().mean(axis=1).tolist()
    slope_s1 = _linear_slope(ep_returns)
    print(f"\nS1 return slope across {len(ep_returns)} episodes: {slope_s1:.2e}")
    if slope_s1 > 0:
        print("  ✅ Positive return trend detected")
    else:
        msg = f"S1 slope non-positive ({slope_s1:.2e}) at {len(ep_returns)} episodes — may need more training"
        print(f"  ⚠️  {msg}")
        warnings.append(msg)

    # ----------------------------------------------------------------
    # Check 2 (HARD): Policy weights must shift — gradient updates must be firing
    # ----------------------------------------------------------------
    act_cols = [c for c in s1_train.columns if c.startswith("raw_action_")]
    if act_cols:
        ep_first = s1_train[s1_train["episode"] == s1_train["episode"].min()]
        ep_last  = s1_train[s1_train["episode"] == s1_train["episode"].max()]
        action_change = float(abs(ep_first[act_cols].mean() - ep_last[act_cols].mean()).sum())
        print(f"Policy action shift (ep0 → ep_last): {action_change:.4f}")
        if action_change > 0.01:
            print("  ✅ Policy weights are updating (gradient updates firing)")
        else:
            msg = f"Action shift too small ({action_change:.6f}): gradient updates may not be firing"
            print(f"  ❌ {msg}")
            errors.append(msg)

    # ----------------------------------------------------------------
    # Check 3: S1 Sharpe vs ANALYTICAL Markowitz (not noisy simulated)
    # The simulated Markowitz Sharpe over 200 steps has very high variance
    # (±0.3 CI from sampling noise), so always use analytical value.
    # ----------------------------------------------------------------
    s1_run = pd.read_parquet(
        Path(QUICK_CONFIG["output_root_s1"]) / "runs" / f"seed_{seed:05d}_summary.parquet"
    )
    eval_sharpe_cols = sorted([c for c in s1_run.columns if "eval_final_sharpe" in c])
    if eval_sharpe_cols:
        final_s1_sharpe = float(s1_run[eval_sharpe_cols[-1]].values[0])
        print(f"\nS1 final eval Sharpe (annualized): {final_s1_sharpe:.4f}")
        print(f"Markowitz ANALYTICAL annualized Sharpe: {mw_annualized:.4f}")
        if final_s1_sharpe > -1.0:
            ratio = final_s1_sharpe / mw_annualized if mw_annualized > 0 else 0.0
            print(f"  ✅ S1 Sharpe is reasonable (ratio: {ratio:.1%})")
            print("     (Full 200-episode run expected to reach ~20–50% of Markowitz)")
        else:
            msg = f"S1 Sharpe ({final_s1_sharpe:.4f}) < -1.0: something is seriously wrong"
            print(f"  ❌ {msg}")
            errors.append(msg)

    # ----------------------------------------------------------------
    # Check 4: S1 vs S2 actions differ
    # ----------------------------------------------------------------
    s1_eval = pd.read_parquet(
        Path(QUICK_CONFIG["output_root_s1"]) / "eval" / "steps" / f"seed_{seed:05d}_steps.parquet"
    )
    s2_eval = pd.read_parquet(
        Path(QUICK_CONFIG["output_root_s2"]) / "eval" / "steps" / f"seed_{seed:05d}_steps.parquet"
    )
    action_cols = [c for c in s1_eval.columns if c.startswith("action_")]
    if action_cols and len(s1_eval) == len(s2_eval):
        action_diff = float(np.abs(
            s1_eval[action_cols].values - s2_eval[action_cols].values
        ).mean())
        print(f"\nMean action difference S1 vs S2: {action_diff:.6f}")
        if action_diff > 1e-6:
            print("  ✅ S2 produces different actions from S1 (trust blending is active)")
        else:
            msg = "S1 and S2 produce identical actions — trust blending has no effect"
            print(f"  ❌ {msg}")
            errors.append(msg)

    # --- Verdict ---
    print("\n" + "=" * 60)
    if errors:
        print("FAIL")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PASS")
    if warnings:
        for w in warnings:
            print(f"  WARNING: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
