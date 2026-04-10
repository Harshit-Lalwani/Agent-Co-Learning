from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class ParquetSink:
    def __init__(self) -> None:
        self.step_rows: list[dict[str, Any]] = []

    def append_step(self, row: dict[str, Any]) -> None:
        self.step_rows.append(row)

    def write_steps(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.step_rows)
        df.to_parquet(output_path, index=False)


def write_summary(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_parquet(output_path, index=False)


def write_aggregate_index(summaries: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_parquet(output_path, index=False)
