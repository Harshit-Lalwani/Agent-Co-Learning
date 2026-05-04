import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import time
import sys

sys.path.insert(0, 'src')
from twmacl.config import load_experiment_config
from twmacl.unified_runner import run_experiment

def main():
    # Write a phase 5 config
    config_dict = {
        "environment": {
            "num_agents": 5,
            "num_assets": 4,
            "episode_length": 200,
            "leverage_cap": 2.0
        },
        "trust": {
            "trust_alpha": 0.2,
            "trust_lambda": 2.0,
            "entropy_window": 30,
            "entropy_slope_threshold": 0.00005,
            "convergence_persistence": 30
        },
        "predictor": {
            "predictor_mode": "noisy_oracle",
            "predictor_window": 20,
            "noise_std": 1.0,
            "expert_agent_idx": 0,
            "expert_noise_std": 0.001
        },
        "market": {
            "mu": [0.05, 0.02, 0.08, -0.01],
            "cov": [
                [ 0.04,  0.01,  0.00, -0.01],
                [ 0.01,  0.03,  0.01,  0.00],
                [ 0.00,  0.01,  0.05,  0.02],
                [-0.01,  0.00,  0.02,  0.06]
            ]
        },
        "learning": {
            "num_train_episodes": 200,
            "num_eval_episodes": 10,
            "observation_window": 5,
            "learning_rate": 0.05,
            "exploration_std": 0.1,
            "reward_baseline_decay": 0.95,
            "policy_init_scale": 0.01,
            "imitation_beta": 0.5,
            "policy_mode": "linear_gaussian",
            "observation_mode": "history_with_private_state",
            "steps_per_year": 252
        },
        "experiment": {
            "num_seeds": 5,
            "base_seed": 1000,
            "output_root": "outputs/phase5_expert"
        }
    }
    
    config_path = Path("configs/phase5_expert.yaml")
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)
        
    print("Running Phase 5 Expert Experiment...")
    t0 = time.time()
    config = load_experiment_config(config_path)
    run_experiment(config)
    print(f"Done in {time.time()-t0:.1f}s")
    
    # Analyze
    ablation_dir = Path("outputs/phase5_expert")
    train_steps_list = []
    for f in ablation_dir.rglob("train/steps/*.parquet"):
        df = pd.read_parquet(f, columns=["episode", "step", "asymmetry_index", 
                                         "entropy_0", "entropy_1", "entropy_2", "entropy_3", "entropy_4"])
        df["entropy_mean"] = df[["entropy_0", "entropy_1", "entropy_2", "entropy_3", "entropy_4"]].mean(axis=1)
        train_steps_list.append(df)
        
    train_steps = pd.concat(train_steps_list, ignore_index=True)
    
    ep0 = train_steps[train_steps["episode"] == 0]["entropy_mean"].mean()
    ep199 = train_steps[train_steps["episode"] == 199]["entropy_mean"].mean()
    
    asym0 = train_steps[train_steps["episode"] == 0]["asymmetry_index"].mean()
    asym199 = train_steps[train_steps["episode"] == 199]["asymmetry_index"].mean()
    
    print(f"--- Phase 5 Results ---")
    print(f"Entropy Ep 0:   {ep0:.4f}")
    print(f"Entropy Ep 199: {ep199:.4f}")
    print(f"Asymmetry Ep 0:   {asym0:.4f}")
    print(f"Asymmetry Ep 199: {asym199:.4f}")

if __name__ == "__main__":
    main()
