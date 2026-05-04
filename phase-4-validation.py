"""phase-4-validation.py — Check Phase 4 ablation outputs are ready for analysis."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def validate_phase4_data() -> int:
    root = Path("outputs/ablations")
    if not root.exists():
        print("FAIL: outputs/ablations does not exist — run scripts/run_ablations.py first")
        return 1

    runs = []
    for file in root.glob("**/runs/*_summary.parquet"):
        run_df = pd.read_parquet(file)
        run_df["experiment_name"] = file.parts[-3]
        runs.append(run_df)

    if not runs:
        print("FAIL: No run summaries found in outputs/ablations")
        return 1

    df = pd.concat(runs, ignore_index=True)
    errors: list[str] = []

    # Required experiment types
    baseline = df[df["experiment_name"] == "phase2_baseline"]
    ablations = df[df["experiment_name"] != "phase2_baseline"]

    if baseline.empty:
        errors.append("Missing phase2_baseline experiment")
    if ablations.empty:
        errors.append("Missing Phase 3 ablation experiments")

    # Required columns
    eval_sharpe_cols = [c for c in df.columns if "eval_final_sharpe_mean_episode" in c]
    if not eval_sharpe_cols:
        errors.append("Missing eval_final_sharpe_mean_episode columns")

    if "final_eval_asymmetry" not in df.columns:
        errors.append("Missing final_eval_asymmetry column")

    if "markowitz_annualized_sharpe" not in df.columns:
        errors.append("Missing markowitz_annualized_sharpe column (run scripts need rebuild)")

    # No NaN in numeric columns
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    nan_counts = df[numeric_cols].isna().sum()
    bad_cols = nan_counts[nan_counts > 0].index.tolist()
    if bad_cols:
        errors.append(f"NaN values found in columns: {bad_cols[:5]}")

    # Learning sanity: do train Sharpe values improve over episodes?
    train_ep_cols = sorted([c for c in df.columns if "train_final_sharpe_mean_episode" in c])
    if len(train_ep_cols) >= 2 and not baseline.empty:
        first_ep = baseline[train_ep_cols[0]].mean()
        last_ep = baseline[train_ep_cols[-1]].mean()
        slope = last_ep - first_ep
        if slope <= 0:
            errors.append(
                f"Baseline S1 Sharpe did not improve across training episodes "
                f"(first={first_ep:.4f}, last={last_ep:.4f}). "
                f"Learning may not be working."
            )

    # Print summary
    print("Phase 4 validation summary:")
    print(f"  Experiments:    {df['experiment_name'].nunique()}")
    print(f"  Total runs:     {len(df)}")
    print(f"  Baseline rows:  {len(baseline)}")
    print(f"  Ablation rows:  {len(ablations)}")
    if eval_sharpe_cols and not baseline.empty:
        bl_sharpe = baseline[eval_sharpe_cols[-1]].mean()
        print(f"  Baseline mean eval Sharpe: {bl_sharpe:.4f}")
    if "markowitz_annualized_sharpe" in df.columns:
        mw = df["markowitz_annualized_sharpe"].mean()
        print(f"  Markowitz annualized Sharpe: {mw:.4f}")
        if not baseline.empty and eval_sharpe_cols:
            ratio = bl_sharpe / mw if mw > 0 else 0.0
            print(f"  Baseline/Markowitz ratio: {ratio:.1%}")

    print()
    if errors:
        print("FAIL")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("GO")
    return 0


if __name__ == "__main__":
    raise SystemExit(validate_phase4_data())
