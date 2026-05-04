import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set style
sns.set_theme(style="whitegrid", context="talk")

def main():
    out_dir = Path("outputs/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ablation_dir = Path("outputs/ablations")
    
    print("Loading data...")
    # Load all step data for training
    train_steps_list = []
    for f in ablation_dir.rglob("**/train/steps/*.parquet"):
        try:
            df = pd.read_parquet(f, columns=["episode", "step", "asymmetry_index", 
                                             "entropy_0", "entropy_1", "entropy_2", "entropy_3", "entropy_4"])
            exp_name = f.parts[-4]
            df["experiment"] = exp_name
            seed = f.stem.split("_")[1]
            df["seed"] = seed
            df["entropy_mean"] = df[["entropy_0", "entropy_1", "entropy_2", "entropy_3", "entropy_4"]].mean(axis=1)
            train_steps_list.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            pass
            
    if not train_steps_list:
        print("No training data found!")
        return
        
    train_steps = pd.concat(train_steps_list, ignore_index=True)
    
    # Load all step data for eval
    eval_steps_list = []
    for f in ablation_dir.rglob("**/eval/steps/*.parquet"):
        try:
            df = pd.read_parquet(f, columns=["episode", "step", "asymmetry_index",
                                             "entropy_0", "entropy_1", "entropy_2", "entropy_3", "entropy_4"])
            exp_name = f.parts[-4]
            df["experiment"] = exp_name
            seed = f.stem.split("_")[1]
            df["seed"] = seed
            df["entropy_mean"] = df[["entropy_0", "entropy_1", "entropy_2", "entropy_3", "entropy_4"]].mean(axis=1)
            eval_steps_list.append(df)
        except Exception as e:
            pass
            
    if not eval_steps_list:
        print("No eval data found!")
        return
        
    eval_steps = pd.concat(eval_steps_list, ignore_index=True)
    
    # Load summary runs
    runs_list = []
    for f in ablation_dir.rglob("**/runs/*_summary.parquet"):
        try:
            df = pd.read_parquet(f)
            exp_name = f.parts[-3]
            df["experiment"] = exp_name
            runs_list.append(df)
        except Exception as e:
            pass
            
    runs = pd.concat(runs_list, ignore_index=True)
    
    # Helper to parse beta and predictor from experiment name
    def parse_exp(name):
        if "baseline" in name:
            return "Baseline", 0.0
        parts = name.replace("p", ".").split("_")
        pred = parts[1] + "_" + parts[2]
        beta = float(parts[-1])
        return pred, beta
        
    runs["predictor"] = runs["experiment"].apply(lambda x: parse_exp(x)[0])
    runs["beta"] = runs["experiment"].apply(lambda x: parse_exp(x)[1])
    
    print("Generating RQ1 Plot (Convergence / Entropy)...")
    # RQ1: Trust entropy over time (train phase)
    # We aggregate by episode for smooth plotting
    plt.figure(figsize=(10, 6))
    rq1_data = train_steps[train_steps["experiment"] != "phase2_baseline"]
    # Group by experiment, seed, episode to get mean per episode
    rq1_grouped = rq1_data.groupby(["experiment", "seed", "episode"])["entropy_mean"].mean().reset_index()
    
    sns.lineplot(data=rq1_grouped, x="episode", y="entropy_mean", hue="experiment", alpha=0.6)
    plt.title("RQ1: Trust Entropy Convergence Over Training Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Mean Trust Entropy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(out_dir / "rq1_entropy.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("Generating RQ2 Plot (Performance / Sharpe)...")
    # RQ2: Final Eval Sharpe Ratio vs S1 Baseline
    # Get the last eval sharpe column name
    eval_cols = sorted([c for c in runs.columns if 'eval_final_sharpe_mean' in c])
    if eval_cols:
        final_sharpe_col = eval_cols[-1]
        plt.figure(figsize=(10, 6))
        sns.barplot(data=runs, x="beta", y=final_sharpe_col, hue="predictor")
        
        # Add baseline as a horizontal line
        baseline_mean = runs[runs["experiment"] == "phase2_baseline"][final_sharpe_col].mean()
        plt.axhline(y=baseline_mean, color='r', linestyle='--', label=f'Baseline (S1) Sharpe: {baseline_mean:.4f}')
        
        # Add markowitz as a horizontal line
        markowitz = runs["markowitz_annualized_sharpe"].mean()
        plt.axhline(y=markowitz, color='g', linestyle='--', label=f'Markowitz Sharpe: {markowitz:.4f}')
        
        plt.title("RQ2: Performance (Sharpe Ratio) across Trust Weights (Beta)")
        plt.xlabel("Imitation Beta (Trust Weight)")
        plt.ylabel("Annualized Eval Sharpe Ratio")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(out_dir / "rq2_performance.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    print("Generating RQ3 Plot (Asymmetry)...")
    # RQ3: Asymmetry index over time
    plt.figure(figsize=(10, 6))
    rq3_grouped = rq1_data.groupby(["experiment", "seed", "episode"])["asymmetry_index"].mean().reset_index()
    sns.lineplot(data=rq3_grouped, x="episode", y="asymmetry_index", hue="experiment", alpha=0.6)
    plt.title("RQ3: Trust Asymmetry Index Over Training Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Asymmetry Index")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(out_dir / "rq3_asymmetry.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Done! Plots saved to outputs/plots/")

if __name__ == "__main__":
    main()
