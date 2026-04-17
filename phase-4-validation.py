from __future__ import annotations

from pathlib import Path

import pandas as pd


def validate_phase4_data() -> int:
    root = Path("outputs/ablations")
    if not root.exists():
        print("FAIL: outputs/ablations does not exist")
        return 1

    runs = []
    for file in root.glob("**/runs/*_summary.parquet"):
        run_df = pd.read_parquet(file)
        run_df["experiment_name"] = file.parts[-3]
        runs.append(run_df)

    if not runs:
        print("FAIL: No runs found in outputs/ablations")
        return 1

    df = pd.concat(runs, ignore_index=True)
    baseline = df[df["experiment_name"] == "phase2_baseline"]
    ablations = df[df["experiment_name"] != "phase2_baseline"]

    if baseline.empty:
        print("FAIL: No phase2_baseline data")
        return 1
    
    if ablations.empty:
        print("FAIL: No Phase 3 ablations data")
        return 1

    eval_sharpe_cols = [c for c in df.columns if 'eval_final_sharpe_mean_episode' in c]
    if not eval_sharpe_cols:
        print("FAIL: Missing Sharpe metrics in outputs")
        return 1
    
    if 'final_eval_asymmetry' not in df.columns:
        print("FAIL: Missing final_eval_asymmetry in outputs")
        return 1

    print("Phase 4 validation: GO")
    print(f"Total runs loaded: {len(df)}")
    print(f"Baseline rows:     {len(baseline)}")
    print(f"Ablation rows:     {len(ablations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(validate_phase4_data())
