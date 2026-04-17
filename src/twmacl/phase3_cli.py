from __future__ import annotations

import argparse

from twmacl.config import load_phase3_config
from twmacl.phase3_runner import run_phase3


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 3 trust-weighted policy imitation learning")
    parser.add_argument("--config", default="configs/phase3.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config = load_phase3_config(args.config)
    run_phase3(config)
    return 0
