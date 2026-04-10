from __future__ import annotations

import argparse

from twmacl.config import load_config
from twmacl.runner import run_phase1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 1 exploratory trust diagnostics")
    parser.add_argument("--config", default="configs/phase1.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    run_phase1(config)
    return 0
