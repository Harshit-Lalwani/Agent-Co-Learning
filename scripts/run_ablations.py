from __future__ import annotations

import itertools
import shutil
import time
from pathlib import Path

from twmacl.config import load_phase2_config, load_phase3_config
from twmacl.phase2_runner import run_phase2
from twmacl.phase3_runner import run_phase3


def main() -> int:
    base_config = load_phase3_config("configs/phase3.yaml")

    ablation_root = Path("outputs/ablations")
    if ablation_root.exists():
        shutil.rmtree(ablation_root)
    ablation_root.mkdir(parents=True, exist_ok=True)

    # Re-run Phase 2 Baseline 
    p2_config = load_phase2_config("configs/phase2.yaml")
    p2_mod = p2_config.model_copy(
        update={
            "experiment": p2_config.experiment.model_copy(
                update={"output_root": str(ablation_root / "phase2_baseline")}
            )
        }
    )
    print("Running Phase 2 Baseline...")
    run_phase2(p2_mod)

    # Execute Phase 3 Parameter sweeps
    betas = [0.1, 0.2, 0.5, 0.8]
    predictors = ["moving_average", "noisy_oracle"]

    total_runs = len(betas) * len(predictors)
    run_index = 1

    start_time = time.time()

    for beta, predictor in itertools.product(betas, predictors):
        print(f"[{run_index}/{total_runs}] Running Phase 3 ablation: predictor={predictor}, beta={beta}")
        
        run_name = f"pred_{predictor}_beta_{beta}".replace(".", "p")
        config_mod = base_config.model_copy(
            update={
                "learning": base_config.learning.model_copy(update={"imitation_beta": beta}),
                "predictor": base_config.predictor.model_copy(update={"predictor_mode": predictor}),
                "experiment": base_config.experiment.model_copy(
                    update={"output_root": str(ablation_root / run_name)}
                )
            }
        )
        
        run_phase3(config_mod)
        run_index += 1

    print(f"Completed all ablations in {time.time() - start_time:.2f} seconds.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
