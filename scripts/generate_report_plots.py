"""
generate_report_plots.py — Produce publication-ready plots from ablation outputs.

Requires: matplotlib, pandas, pyarrow, scipy
Run from repo root: python scripts/generate_report_plots.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not installed — correlation p-values will not be computed")


ROOT = Path("outputs/ablations")
OUT_DIR = ROOT  # save plots alongside data


def _load_ablation_runs() -> pd.DataFrame:
    runs = []
    for file in ROOT.glob("**/runs/*_summary.parquet"):
        df = pd.read_parquet(file)
        # experiment_name is the ablation folder (2 levels up from runs/)
        df["experiment_name"] = file.parts[-3]
        runs.append(df)
    if not runs:
        raise FileNotFoundError(f"No run summaries found under {ROOT}")
    return pd.concat(runs, ignore_index=True)


def _parse_ablation_name(name: str) -> tuple[str, float]:
    """Extract predictor and beta from an experiment folder name."""
    if name == "phase2_baseline":
        return "Baseline (S1)", 0.0
    # e.g. pred_moving_average_beta_0p2
    try:
        pred = name.split("_beta_")[0].replace("pred_", "")
        beta_str = name.split("_beta_")[1].replace("p", ".")
        return pred, float(beta_str)
    except (IndexError, ValueError):
        return name, 0.0


def plot_sharpe_vs_beta(df: pd.DataFrame, eval_col: str) -> None:
    """Line plot: mean eval Sharpe vs beta for each predictor vs S1 baseline."""
    baseline = df[df["experiment_name"] == "phase2_baseline"]
    ablations = df[df["experiment_name"] != "phase2_baseline"]

    rows = []
    for _, row in ablations.iterrows():
        pred, beta = _parse_ablation_name(row["experiment_name"])
        rows.append({"Predictor": pred, "Beta": beta, "Sharpe": row[eval_col]})
    abl_df = pd.DataFrame(rows)

    baseline_mean = float(baseline[eval_col].mean())
    baseline_std = float(baseline[eval_col].std()) if len(baseline) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(9, 5))
    for pred, grp in abl_df.groupby("Predictor"):
        summary = grp.groupby("Beta")["Sharpe"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            summary["Beta"],
            summary["mean"],
            yerr=summary["std"].fillna(0),
            marker="o",
            capsize=4,
            label=pred,
        )

    ax.axhline(baseline_mean, color="red", linestyle="--", linewidth=1.5,
               label=f"S1 Baseline ({baseline_mean:.4f})")
    if baseline_std > 0:
        ax.axhspan(baseline_mean - baseline_std, baseline_mean + baseline_std,
                   alpha=0.1, color="red")

    ax.set_xlabel("Imitation Beta (β)", fontsize=12)
    ax.set_ylabel("Annualized Eval Sharpe Ratio", fontsize=12)
    ax.set_title("RQ2: S2 Performance vs S1 Baseline", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "plot_sharpe_vs_beta.png"
    fig.savefig(out, dpi=150)
    fig.savefig(OUT_DIR / "plot_sharpe_vs_beta.pdf")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_asymmetry_vs_sharpe(df: pd.DataFrame, eval_col: str) -> None:
    """Scatter: trust asymmetry index vs Sharpe, coloured by strategy."""
    if "final_eval_asymmetry" not in df.columns:
        print("Skipping asymmetry plot: final_eval_asymmetry not in data")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for exp, grp in df.groupby("experiment_name"):
        label, _ = _parse_ablation_name(exp)
        ax.scatter(grp["final_eval_asymmetry"], grp[eval_col], alpha=0.7,
                   label=label, s=40)

    x = df["final_eval_asymmetry"].values
    y = df[eval_col].values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 3:
        r, p = (scipy_stats.pearsonr(x[mask], y[mask]) if HAS_SCIPY
                else (np.corrcoef(x[mask], y[mask])[0, 1], float("nan")))
        ax.set_title(f"RQ3: Asymmetry vs Sharpe  (r={r:.3f}, p={p:.3f})", fontsize=13)
    else:
        ax.set_title("RQ3: Asymmetry vs Sharpe", fontsize=13)

    ax.set_xlabel("Final Eval Asymmetry Index", fontsize=12)
    ax.set_ylabel("Annualized Eval Sharpe Ratio", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = OUT_DIR / "plot_asymmetry_vs_sharpe.png"
    fig.savefig(out, dpi=150)
    fig.savefig(OUT_DIR / "plot_asymmetry_vs_sharpe.pdf")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_markowitz_ratio(df: pd.DataFrame, eval_col: str) -> None:
    """Bar chart: learned Sharpe / Markowitz Sharpe per experiment."""
    if "markowitz_annualized_sharpe" not in df.columns:
        print("Skipping Markowitz ratio plot: markowitz_annualized_sharpe not in data")
        return

    df = df.copy()
    df["ratio"] = df[eval_col] / df["markowitz_annualized_sharpe"].replace(0, np.nan)
    summary = (
        df.groupby("experiment_name")["ratio"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    labels = [_parse_ablation_name(n)[0] + f"\n({n})" for n in summary["experiment_name"]]

    fig, ax = plt.subplots(figsize=(max(8, len(summary) * 0.8), 5))
    bars = ax.bar(range(len(summary)), summary["mean"], yerr=summary["std"].fillna(0),
                  capsize=4, color="steelblue", alpha=0.8)
    ax.axhline(1.0, color="gold", linestyle="--", label="Markowitz = 1.0")
    ax.axhline(0.0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Learned Sharpe / Markowitz Sharpe", fontsize=12)
    ax.set_title("Learned Policy Performance Relative to Markowitz Optimum", fontsize=13)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "plot_markowitz_ratio.png"
    fig.savefig(out, dpi=150)
    fig.savefig(OUT_DIR / "plot_markowitz_ratio.pdf")
    plt.close(fig)
    print(f"Saved: {out}")


def generate() -> None:
    if not ROOT.exists():
        print(f"outputs/ablations not found — run scripts/run_ablations.py first")
        return

    df = _load_ablation_runs()
    eval_cols = sorted([c for c in df.columns if "eval_final_sharpe_mean_episode" in c])
    if not eval_cols:
        print("No eval Sharpe columns found in data")
        return
    # Use the last eval episode column (most-trained Sharpe)
    target_col = eval_cols[-1]
    print(f"Using eval Sharpe column: {target_col}")
    print(f"Experiments loaded: {df['experiment_name'].nunique()}, total rows: {len(df)}")

    plot_sharpe_vs_beta(df, target_col)
    plot_asymmetry_vs_sharpe(df, target_col)
    plot_markowitz_ratio(df, target_col)
    print("Done.")


if __name__ == "__main__":
    generate()
