from __future__ import annotations

import argparse

from twmacl.config import load_experiment_config
from twmacl.unified_runner import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run an S1 or S2 experiment. Strategy is determined by imitation_beta in the config:\n"
            "  imitation_beta=0.0 → S1 (independent learning)\n"
            "  imitation_beta>0.0 → S2 (trust-weighted imitation)"
        )
    )
    parser.add_argument(
        "--config",
        default="configs/phase2.yaml",
        help="Path to YAML experiment config",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    strategy = "S2 (trust-weighted)" if config.is_s2 else "S1 (independent)"
    print(f"Running {strategy} experiment: beta={config.learning.imitation_beta}")
    run_experiment(config)
    return 0
