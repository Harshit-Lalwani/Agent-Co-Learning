from __future__ import annotations

import argparse

from twmacl.config import load_phase2_config
from twmacl.phase2_runner import run_phase2


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 2 baseline learning")
    parser.add_argument("--config", default="configs/phase2.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config = load_phase2_config(args.config)
    run_phase2(config)
    return 0