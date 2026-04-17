from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from twmacl.config import load_phase2_config
from twmacl.phase2_runner import SCHEMA_VERSION, run_phase2


def _build_smoke_config(output_root: str):
    config = load_phase2_config("configs/phase2.yaml")
    return config.model_copy(
        update={
            "experiment": config.experiment.model_copy(
                update={"num_seeds": 1, "base_seed": 2026, "output_root": output_root}
            ),
            "learning": config.learning.model_copy(update={"num_train_episodes": 2, "num_eval_episodes": 1}),
        }
    )


def _load_outputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(root / "train" / "steps" / "seed_02026_steps.parquet")
    eval_ = pd.read_parquet(root / "eval" / "steps" / "seed_02026_steps.parquet")
    run = pd.read_parquet(root / "runs" / "seed_02026_summary.parquet")
    agg = pd.read_parquet(root / "aggregate" / "run_index.parquet")
    return train, eval_, run, agg


def _stable_view(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["run_duration_seconds", "config_hash"], errors="ignore").copy()


def main() -> int:
    output_root_a = Path("outputs/phase2_validation_a")
    output_root_b = Path("outputs/phase2_validation_b")

    for path in (output_root_a, output_root_b):
        if path.exists():
            shutil.rmtree(path)

    config_a = _build_smoke_config(str(output_root_a))
    run_phase2(config_a)
    train_a, eval_a, run_a, agg_a = _load_outputs(output_root_a)

    assert not train_a.empty, "train outputs are empty"
    assert not eval_a.empty, "eval outputs are empty"
    assert not run_a.empty, "run summary is empty"
    assert not agg_a.empty, "aggregate outputs are empty"
    assert set(run_a["schema_version"].astype(str)) == {SCHEMA_VERSION}
    assert set(train_a["schema_version"].astype(str)) == {SCHEMA_VERSION}
    assert set(eval_a["schema_version"].astype(str)) == {SCHEMA_VERSION}

    config_b = _build_smoke_config(str(output_root_b))
    run_phase2(config_b)
    train_b, eval_b, run_b, agg_b = _load_outputs(output_root_b)

    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(eval_a, eval_b)
    pd.testing.assert_frame_equal(_stable_view(run_a), _stable_view(run_b))
    pd.testing.assert_frame_equal(_stable_view(agg_a), _stable_view(agg_b))

    print("Phase 2 validation: GO")
    print(f"train rows: {len(train_a)}")
    print(f"eval rows: {len(eval_a)}")
    print(f"schema_version: {SCHEMA_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())