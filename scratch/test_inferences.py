from pathlib import Path
import pandas as pd
import numpy as np

def main():
    root = Path("outputs/ablations")
    if not root.exists():
        print("No output found.")
        return
        
    runs = []
    for file in root.glob("**/runs/*_summary.parquet"):
        run_df = pd.read_parquet(file)
        run_df["experiment_name"] = file.parts[-3]
        runs.append(run_df)
        
    if not runs:
        return
        
    df = pd.concat(runs, ignore_index=True)
    baseline = df[df["experiment_name"] == "phase2_baseline"]
    
    eval_cols = [c for c in df.columns if 'eval_final_sharpe_mean_episode' in c]
    target_col = eval_cols[-1]
    
    print("Baseline Sharpe:")
    print(baseline[target_col].describe())
    
    ablations = df[df["experiment_name"] != "phase2_baseline"]
    print("\nAblations Sharpe grouped by experiment:")
    print(ablations.groupby("experiment_name")[target_col].describe().sort_values("mean", ascending=False))
    
    print("\nCorrelation Asymmetry vs Sharpe:")
    corr = df["final_eval_asymmetry"].corr(df[target_col])
    print(corr)

if __name__ == "__main__":
    main()
