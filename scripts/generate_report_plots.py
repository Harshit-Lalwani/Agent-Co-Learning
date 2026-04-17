import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate():
    root = Path("outputs/ablations")
    if not root.exists():
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
    ablations = df[df["experiment_name"] != "phase2_baseline"]

    eval_sharpe_cols = [c for c in df.columns if 'eval_final_sharpe_mean_episode' in c]
    if not eval_sharpe_cols:
        return
    target = eval_sharpe_cols[-1]

    plot_data = []
    for _, row in baseline.iterrows():
        plot_data.append({"Experiment": "Baseline", "Sharpe": row[target], "Beta": 0.0, "Predictor": "None"})
        
    for _, row in ablations.iterrows():
        name = row["experiment_name"]
        beta = name.split("_beta_")[1].replace("p", ".") if "_beta_" in name else "0.0"
        pred = name.split("_beta_")[0].replace("pred_", "") if "pred_" in name else "unknown"
        plot_data.append({"Experiment": f"{pred}_{beta}", "Sharpe": row[target], "Beta": float(beta), "Predictor": pred})

    plot_df = pd.DataFrame(plot_data)

    # Plot 1: Sharpe by Beta and Predictor
    plt.figure(figsize=(10, 6))
    s2_data = plot_df[plot_df["Experiment"] != "Baseline"]
    
    for pred in s2_data["Predictor"].unique():
        subset = s2_data[s2_data["Predictor"] == pred].groupby("Beta")["Sharpe"].mean().reset_index()
        plt.plot(subset["Beta"], subset["Sharpe"], marker="o", label=pred)
        
    baseline_mean = plot_df[plot_df["Experiment"] == "Baseline"]["Sharpe"].mean()
    plt.axhline(baseline_mean, color='r', linestyle='--', label=f"Baseline Mean ({baseline_mean:.4f})")
    plt.title("Portfolio Performance by Trust Imitation Weight (Beta)")
    plt.xlabel("Imitation Beta")
    plt.ylabel("Eval Sharpe Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(root / "plot_sharpe_beta.png")
    plt.close()

    # Plot 2: Scatter Asymmetry vs Sharpe
    plt.figure(figsize=(8, 6))
    corr = df["final_eval_asymmetry"].corr(df[target])
    
    for exp in df["experiment_name"].unique():
        subset = df[df["experiment_name"] == exp]
        plt.scatter(subset["final_eval_asymmetry"], subset[target], alpha=0.7, label=exp)

    plt.title(f"Asymmetry Index vs Sharpe Ratio (Pearson r: {corr:.3f})")
    plt.xlabel("Absolute Trust Asymmetry Index")
    plt.ylabel("Evaluation Sharpe Ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(root / "plot_asymmetry.png")
    plt.close()

if __name__ == "__main__":
    generate()
