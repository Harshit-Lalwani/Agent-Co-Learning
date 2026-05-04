from __future__ import annotations

import itertools
import shutil
import time
from pathlib import Path

from twmacl.config import load_experiment_config
from twmacl.unified_runner import run_experiment


def main() -> int:
    base_config = load_experiment_config("configs/phase3.yaml")

    ablation_root = Path("outputs/ablations")
    if ablation_root.exists():
        shutil.rmtree(ablation_root)
    ablation_root.mkdir(parents=True, exist_ok=True)

    # --- Phase 2 Baseline (S1, beta=0) ---
    p2_config = load_experiment_config("configs/phase2.yaml")
    p2_mod = p2_config.model_copy(
        update={
            "experiment": p2_config.experiment.model_copy(
                update={"output_root": str(ablation_root / "phase2_baseline")}
            )
        }
    )
    print("Running Phase 2 Baseline (S1, beta=0.0)...")
    run_experiment(p2_mod)

    # --- Phase 3 Parameter Sweeps (S2, varying beta and predictor) ---
    betas = [0.1, 0.2, 0.5, 0.8]
    predictors = ["moving_average", "noisy_oracle"]

    configs_to_run = []
    
    for beta, predictor in itertools.product(betas, predictors):
        run_name = f"pred_{predictor}_beta_{beta}".replace(".", "p")
        config_mod = base_config.model_copy(
            update={
                "learning": base_config.learning.model_copy(update={"imitation_beta": beta}),
                "predictor": base_config.predictor.model_copy(
                    update={"predictor_mode": predictor}
                ),
                "experiment": base_config.experiment.model_copy(
                    update={"output_root": str(ablation_root / run_name)}
                ),
            }
        )
        configs_to_run.append((run_name, config_mod))

    total_runs = len(configs_to_run)
    print(f"Starting {total_runs} ablations using CPU multiprocessing...")
    start_time = time.time()

    import concurrent.futures
    import os

    max_workers = os.cpu_count() or 4
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_experiment, config): name for name, config in configs_to_run}
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                future.result()
                completed += 1
                print(f"[{completed}/{total_runs}] Completed: {name}")
            except Exception as exc:
                print(f"Run {name} generated an exception: {exc}")

    elapsed = time.time() - start_time
    print(f"All ablations complete in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
