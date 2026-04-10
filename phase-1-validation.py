from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _run_pipeline(repo_root: Path, config_path: Path) -> None:
    cmd = [sys.executable, "run_phase1.py", "--config", str(config_path)]
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Pipeline failed for config "
            f"{config_path}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _seed_list(cfg: dict[str, Any]) -> list[int]:
    exp = cfg["experiment"]
    return [exp["base_seed"] + i for i in range(exp["num_seeds"])]


def _step_file(output_root: Path, seed: int) -> Path:
    return output_root / "steps" / f"seed_{seed:05d}_steps.parquet"


def _run_file(output_root: Path, seed: int) -> Path:
    return output_root / "runs" / f"seed_{seed:05d}_summary.parquet"


def _aggregate_file(output_root: Path) -> Path:
    return output_root / "aggregate" / "run_index.parquet"


def _check_files_exist(output_root: Path, seeds: list[int]) -> list[str]:
    errors: list[str] = []
    for seed in seeds:
        for file_path in (_step_file(output_root, seed), _run_file(output_root, seed)):
            if not file_path.exists():
                errors.append(f"missing output file: {file_path}")
    agg = _aggregate_file(output_root)
    if not agg.exists():
        errors.append(f"missing output file: {agg}")
    return errors


def _expected_step_columns(num_agents: int) -> set[str]:
    cols = {
        "seed",
        "step",
        "trust_update_applied",
        "asymmetry_index",
        "system_converged",
    }
    for i in range(num_agents):
        for j in range(num_agents):
            if i == j:
                continue
            cols.add(f"tau_{i}_{j}")
    for i in range(num_agents):
        cols.add(f"entropy_{i}")
        cols.add(f"entropy_slope_{i}")
        cols.add(f"agent_return_{i}")
        cols.add(f"sharpe_running_{i}")
        cols.add(f"agent_converged_{i}")
    return cols


def _expected_run_columns(num_agents: int) -> set[str]:
    cols = {
        "seed",
        "config_hash",
        "num_steps",
        "predictor_mode",
        "run_duration_seconds",
        "final_asymmetry",
    }
    for i in range(num_agents):
        cols.add(f"convergence_time_{i}")
        cols.add(f"final_entropy_{i}")
        cols.add(f"cumulative_return_{i}")
        cols.add(f"sharpe_ratio_{i}")
        cols.add(f"max_leverage_used_{i}")
    return cols


def _schema_check(output_root: Path, cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    seeds = _seed_list(cfg)
    num_agents = cfg["environment"]["num_agents"]

    steps = pd.read_parquet(_step_file(output_root, seeds[0]))
    run = pd.read_parquet(_run_file(output_root, seeds[0]))

    missing_step = sorted(_expected_step_columns(num_agents) - set(steps.columns))
    missing_run = sorted(_expected_run_columns(num_agents) - set(run.columns))

    if missing_step:
        errors.append(f"step schema missing columns: {missing_step}")
    if missing_run:
        errors.append(f"run schema missing columns: {missing_run}")

    if "schema_version" not in steps.columns:
        warnings.append("step schema_version missing (recommended before Phase 2)")
    if "schema_version" not in run.columns:
        warnings.append("run schema_version missing (recommended before Phase 2)")

    return errors, warnings


def _predictor_mode_validation(repo_root: Path, base_cfg: dict[str, Any], base_output_root: Path) -> list[str]:
    errors: list[str] = []
    modes = ["moving_average", "noisy_oracle", "random"]

    for mode in modes:
        cfg = copy.deepcopy(base_cfg)
        output_root = base_output_root / f"predictor_{mode}"
        cfg["predictor"]["predictor_mode"] = mode
        cfg["experiment"]["output_root"] = str(output_root)

        config_path = base_output_root / f"config_{mode}.yaml"
        _write_yaml(config_path, cfg)
        _run_pipeline(repo_root, config_path)

        seeds = _seed_list(cfg)
        errors.extend(_check_files_exist(output_root, seeds))

        run_df = pd.read_parquet(_run_file(output_root, seeds[0]))
        if not pd.api.types.is_numeric_dtype(run_df["final_asymmetry"]):
            errors.append(f"final_asymmetry not numeric for mode {mode}")
        if run_df["final_asymmetry"].isna().any():
            errors.append(f"final_asymmetry contains NaN for mode {mode}")

    return errors


def _reproducibility_validation(repo_root: Path, base_cfg: dict[str, Any], base_output_root: Path) -> list[str]:
    errors: list[str] = []

    cfg_a = copy.deepcopy(base_cfg)
    cfg_b = copy.deepcopy(base_cfg)
    out_a = base_output_root / "repro_a"
    out_b = base_output_root / "repro_b"
    cfg_a["experiment"]["output_root"] = str(out_a)
    cfg_b["experiment"]["output_root"] = str(out_b)

    cfg_a_path = base_output_root / "config_repro_a.yaml"
    cfg_b_path = base_output_root / "config_repro_b.yaml"
    _write_yaml(cfg_a_path, cfg_a)
    _write_yaml(cfg_b_path, cfg_b)

    _run_pipeline(repo_root, cfg_a_path)
    _run_pipeline(repo_root, cfg_b_path)

    for seed in _seed_list(base_cfg):
        df_a = pd.read_parquet(_step_file(out_a, seed))
        df_b = pd.read_parquet(_step_file(out_b, seed))
        if not df_a.equals(df_b):
            errors.append(f"repro mismatch for seed {seed}")

    return errors


def _convergence_sanity_validation(base_cfg: dict[str, Any], output_root: Path) -> list[str]:
    errors: list[str] = []

    seeds = _seed_list(base_cfg)
    window = base_cfg["trust"]["entropy_window"]
    predictor_window = base_cfg["predictor"]["predictor_window"]

    steps = pd.read_parquet(_step_file(output_root, seeds[0]))

    early_updates = int(steps.loc[steps["step"] < predictor_window, "trust_update_applied"].sum())
    if early_updates != 0:
        errors.append(f"trust updates applied during burn-in: {early_updates}")

    early_conv = steps.loc[steps["step"] < (window - 1), "system_converged"].any()
    if bool(early_conv):
        errors.append("system_converged became true before entropy window was available")

    system_series = steps["system_converged"].astype(int)
    if (system_series.diff().fillna(0) < 0).any():
        errors.append("system_converged is not monotonic")

    return errors


def _edge_case_validation(repo_root: Path, base_cfg: dict[str, Any], base_output_root: Path) -> list[str]:
    errors: list[str] = []

    edge_cfgs: list[tuple[str, dict[str, Any]]] = []

    cfg_min = copy.deepcopy(base_cfg)
    cfg_min["environment"]["num_agents"] = 2
    cfg_min["environment"]["num_assets"] = 1
    cfg_min["environment"]["episode_length"] = 80
    cfg_min["environment"]["leverage_cap"] = 1.0
    cfg_min["predictor"]["predictor_window"] = 5
    cfg_min["predictor"]["noise_std"] = 0.0
    cfg_min["trust"]["entropy_window"] = 5
    cfg_min["trust"]["convergence_persistence"] = 3
    cfg_min["market"]["mu"] = [0.0005]
    cfg_min["market"]["cov"] = [[0.0003]]
    cfg_min["experiment"]["output_root"] = str(base_output_root / "edge_min")
    edge_cfgs.append(("edge_min", cfg_min))

    cfg_lambda = copy.deepcopy(base_cfg)
    cfg_lambda["trust"]["trust_lambda"] = 10.0
    cfg_lambda["predictor"]["noise_std"] = 0.0
    cfg_lambda["experiment"]["output_root"] = str(base_output_root / "edge_high_lambda")
    edge_cfgs.append(("edge_high_lambda", cfg_lambda))

    for name, cfg in edge_cfgs:
        config_path = base_output_root / f"config_{name}.yaml"
        _write_yaml(config_path, cfg)
        _run_pipeline(repo_root, config_path)

        seed = _seed_list(cfg)[0]
        run_df = pd.read_parquet(_run_file(Path(cfg["experiment"]["output_root"]), seed))
        num_df = run_df.select_dtypes(include=["number"])
        if num_df.isna().any().any():
            errors.append(f"NaN detected in run summary for {name}")

    return errors


def _print_summary_table(base_cfg: dict[str, Any], output_root: Path) -> None:
    seeds = _seed_list(base_cfg)
    agg = pd.read_parquet(_aggregate_file(output_root))

    print("\n=== Phase 1 Validation Summary ===")
    print(f"seeds: {len(seeds)}")
    print(f"mean_final_asymmetry: {agg['final_asymmetry'].mean():.6f}")
    print(f"mean_run_duration_seconds: {agg['run_duration_seconds'].mean():.6f}")
    print("top cumulative_return_0:")
    show = agg[["seed", "cumulative_return_0"]].sort_values("cumulative_return_0", ascending=False)
    print(show.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 closure validator")
    parser.add_argument("--config", default="configs/phase1.yaml", help="Base config path")
    parser.add_argument(
        "--workspace", default="outputs/phase1_validation", help="Validation workspace output root"
    )
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep validation artifacts")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    base_config_path = (repo_root / args.config).resolve()
    workspace = (repo_root / args.workspace).resolve()

    if workspace.exists() and not args.keep_artifacts:
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_yaml(base_config_path)

    all_errors: list[str] = []
    all_warnings: list[str] = []

    # 1) Predictor mode validation and baseline output generation
    predictor_errors = _predictor_mode_validation(repo_root, base_cfg, workspace)
    all_errors.extend(predictor_errors)

    baseline_cfg = copy.deepcopy(base_cfg)
    baseline_output = workspace / "baseline"
    baseline_cfg["experiment"]["output_root"] = str(baseline_output)
    baseline_cfg_path = workspace / "config_baseline.yaml"
    _write_yaml(baseline_cfg_path, baseline_cfg)
    _run_pipeline(repo_root, baseline_cfg_path)

    seeds = _seed_list(base_cfg)
    all_errors.extend(_check_files_exist(baseline_output, seeds))

    # 2) Full-seed reproducibility
    all_errors.extend(_reproducibility_validation(repo_root, base_cfg, workspace))

    # 3) Convergence sanity
    all_errors.extend(_convergence_sanity_validation(base_cfg, baseline_output))

    # 4) Schema contract
    schema_errors, schema_warnings = _schema_check(baseline_output, base_cfg)
    all_errors.extend(schema_errors)
    all_warnings.extend(schema_warnings)

    # 5) Edge-case stress checks
    all_errors.extend(_edge_case_validation(repo_root, base_cfg, workspace))

    # 6) Summary report
    _print_summary_table(base_cfg, baseline_output)

    print("\n=== Validation Result ===")
    if all_errors:
        print("NO-GO")
        print("errors:")
        for err in all_errors:
            print(f"- {err}")
    else:
        if all_warnings:
            print("GO (with warnings)")
        else:
            print("GO")

    if all_warnings:
        print("warnings:")
        for warn in all_warnings:
            print(f"- {warn}")

    metadata = {
        "errors": all_errors,
        "warnings": all_warnings,
        "num_errors": len(all_errors),
        "num_warnings": len(all_warnings),
    }
    report_path = workspace / "validation_report.json"
    report_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"report: {report_path}")

    return 1 if all_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
