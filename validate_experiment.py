"""
validate_experiment.py — Replaces phase-2-validation.py and phase-3-validation.py.

Checks:
1. Outputs are non-empty and have the expected schema version
2. Runs with the same seed produce identical outputs (reproducibility)
3. S1 and S2 produce different actions (trust blending is active)
4. Trust resets between episodes (unless trust_persistence=True)
5. Schema version matches SCHEMA_VERSION
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from twmacl.config import load_experiment_config
from twmacl.unified_runner import SCHEMA_VERSION, run_experiment


def _build_quick(config_path: str, output_root: str):
    config = load_experiment_config(config_path)
    return config.model_copy(
        update={
            "experiment": config.experiment.model_copy(
                update={"num_seeds": 1, "base_seed": 2026, "output_root": output_root}
            ),
            "learning": config.learning.model_copy(
                update={"num_train_episodes": 3, "num_eval_episodes": 2}
            ),
        }
    )


def _load(root: Path, seed: int = 2026):
    return (
        pd.read_parquet(root / "train" / "steps" / f"seed_{seed:05d}_steps.parquet"),
        pd.read_parquet(root / "eval" / "steps" / f"seed_{seed:05d}_steps.parquet"),
        pd.read_parquet(root / "runs" / f"seed_{seed:05d}_summary.parquet"),
        pd.read_parquet(root / "aggregate" / "run_index.parquet"),
    )


def _stable(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["run_duration_seconds", "config_hash"], errors="ignore")


def main() -> int:
    errors: list[str] = []

    # ----------------------------------------------------------------
    # 1. S1 outputs are non-empty and schema version is correct
    # ----------------------------------------------------------------
    print("=== Check 1: S1 output schema ===")
    s1_root_a = Path("outputs/_val_s1_a")
    s1_root_b = Path("outputs/_val_s1_b")
    for p in [s1_root_a, s1_root_b]:
        shutil.rmtree(p, ignore_errors=True)

    cfg_s1_a = _build_quick("configs/phase2.yaml", str(s1_root_a))
    run_experiment(cfg_s1_a)
    train_a, eval_a, run_a, agg_a = _load(s1_root_a)

    for name, df in [("train", train_a), ("eval", eval_a), ("run", run_a), ("agg", agg_a)]:
        if df.empty:
            errors.append(f"S1 {name} output is empty")
        if "schema_version" in df.columns:
            versions = set(df["schema_version"].astype(str))
            if versions != {SCHEMA_VERSION}:
                errors.append(f"S1 {name} schema_version={versions}, expected {SCHEMA_VERSION!r}")
    if not errors:
        print("  ✅ S1 outputs non-empty, schema version correct")

    # ----------------------------------------------------------------
    # 2. S1 reproducibility
    # ----------------------------------------------------------------
    print("=== Check 2: S1 reproducibility ===")
    cfg_s1_b = _build_quick("configs/phase2.yaml", str(s1_root_b))
    run_experiment(cfg_s1_b)
    train_b, eval_b, run_b, agg_b = _load(s1_root_b)

    if not train_a.equals(train_b):
        errors.append("S1 train steps differ between identical runs (not reproducible)")
    if not eval_a.equals(eval_b):
        errors.append("S1 eval steps differ between identical runs (not reproducible)")
    if not _stable(run_a).equals(_stable(run_b)):
        errors.append("S1 run summary differs between identical runs")
    if not errors:
        print("  ✅ S1 is reproducible")

    # ----------------------------------------------------------------
    # 3. S2 outputs are non-empty
    # ----------------------------------------------------------------
    print("=== Check 3: S2 output schema ===")
    s2_root = Path("outputs/_val_s2")
    shutil.rmtree(s2_root, ignore_errors=True)
    cfg_s2 = _build_quick("configs/phase3.yaml", str(s2_root))
    run_experiment(cfg_s2)
    train_s2, eval_s2, run_s2, _ = _load(s2_root)

    for name, df in [("train", train_s2), ("eval", eval_s2), ("run", run_s2)]:
        if df.empty:
            errors.append(f"S2 {name} output is empty")
    if not errors:
        print("  ✅ S2 outputs non-empty")

    # ----------------------------------------------------------------
    # 4. S1 vs S2 actions differ (trust blending is active)
    # ----------------------------------------------------------------
    print("=== Check 4: S1 vs S2 action differentiation ===")
    action_cols = [c for c in eval_a.columns if c.startswith("action_")]
    s2_action_cols = [c for c in eval_s2.columns if c.startswith("action_")]
    if action_cols and s2_action_cols and len(eval_a) == len(eval_s2):
        diff = float(np.abs(eval_a[action_cols].values - eval_s2[s2_action_cols].values).mean())
        if diff > 1e-6:
            print(f"  ✅ S2 actions differ from S1 (mean L1={diff:.6f})")
        else:
            errors.append("S1 and S2 eval actions are identical — trust blending has no effect")
    else:
        print("  SKIP (row count mismatch or missing action columns)")

    # ----------------------------------------------------------------
    # 5. Trust resets between episodes (trust_persistence=False)
    # ----------------------------------------------------------------
    print("=== Check 5: Trust resets between episodes ===")
    # When trust_persistence=False, trust at start of episode 1 should be
    # identical to trust at start of episode 0 (both start from zeros).
    # We detect this by checking that tau values at step=0 of ep 0 and ep 1 match.
    tau_cols = [c for c in train_a.columns if c.startswith("tau_")]
    if tau_cols and not cfg_s1_a.trust.trust_persistence:
        ep0_step0 = train_a[(train_a["episode"] == 0) & (train_a["step"] == 0)][tau_cols].values
        ep1_step0 = train_a[(train_a["episode"] == 1) & (train_a["step"] == 0)][tau_cols].values
        if ep0_step0.shape == ep1_step0.shape and np.allclose(ep0_step0, ep1_step0):
            print("  ✅ Trust resets identically at start of each episode")
        else:
            errors.append("Trust at step=0 differs across episodes — reset may not be working")
    else:
        print("  SKIP (trust_persistence=True or no tau columns)")

    # ----------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------
    for p in [s1_root_a, s1_root_b, s2_root]:
        shutil.rmtree(p, ignore_errors=True)

    # ----------------------------------------------------------------
    # Verdict
    # ----------------------------------------------------------------
    print()
    if errors:
        print("NO-GO")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("GO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
