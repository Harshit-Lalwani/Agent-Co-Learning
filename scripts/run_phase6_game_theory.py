# import pandas as pd
# import numpy as np
# from pathlib import Path
# import yaml
# import time
# import sys

# # Ensure the src directory is in the path
# sys.path.insert(0, 'src')
# from twmacl.config import load_experiment_config
# from twmacl.unified_runner import run_experiment

# def main():
#     # Phase 6 Config: Deep Learning + Expert + Slippage
#     config_dict = {
#         "environment": {
#             "num_agents": 5,
#             "num_assets": 4,
#             "episode_length": 200,
#             "leverage_cap": 2.0
#         },
#         "trust": {
#             "trust_alpha": 0.2,
#             "trust_lambda": 2.0,
#             "entropy_window": 30,
#             "entropy_slope_threshold": 0.00005,
#             "convergence_persistence": 30,
#             "trust_persistence": True
#         },
#         "predictor": {
#             "predictor_mode": "noisy_oracle", # Expert signal is active
#             "predictor_window": 20,
#             "noise_std": 1.0,
#             "expert_agent_idx": 0,          # Agent 0 is the Expert
#             "expert_noise_std": 0.001
#         },
#         "market": {
#             "mu": [0.05, 0.02, 0.08, -0.01],
#             "cov": [
#                 [ 0.04,  0.01,  0.00, -0.01],
#                 [ 0.01,  0.03,  0.01,  0.00],
#                 [ 0.00,  0.01,  0.05,  0.02],
#                 [-0.01,  0.00,  0.02,  0.06]
#             ]
#         },
#         "learning": {
#             "num_train_episodes": 200,
#             "num_eval_episodes": 10,
#             "observation_window": 5,
#             "learning_rate": 0.0005,            # Slightly lower LR for neural network stability
#             "exploration_std": 0.1,
#             "reward_baseline_decay": 0.95,
#             "policy_init_scale": 0.01,
#             "imitation_beta": 0.5,            # Trust-routing is active
#             "policy_mode": "mlp_gaussian",    # <--- TRIGGERS THE NEW PYTORCH NETWORK
#             "observation_mode": "history_with_private_state",
#             "steps_per_year": 252
#         },
#         "experiment": {
#             "num_seeds": 5,
#             "base_seed": 4000,
#             "output_root": "outputs/phase6_game_theory" # <--- NEW OUTPUT DIR
#         }
#     }
    
#     config_path = Path("configs/phase6_game_theory.yaml")
#     config_path.parent.mkdir(exist_ok=True)
#     with open(config_path, "w") as f:
#         yaml.dump(config_dict, f)
        
#     print("Running Phase 6: Game Theory & Deep Learning Experiment...")
#     t0 = time.time()
#     config = load_experiment_config(config_path)
#     run_experiment(config)
#     print(f"Experiment completed in {time.time()-t0:.1f}s")
    
#     # --- Post-Run Analysis for your Report ---
#     ablation_dir = Path("outputs/phase6_game_theory")
    
#     print("\nExtracting Results for Final Report...")
    
#     # Analyze Training Entropy & Asymmetry
#     train_steps_list = []
#     for f in ablation_dir.rglob("train/steps/*.parquet"):
#         df = pd.read_parquet(f, columns=["episode", "step", "asymmetry_index", 
#                                          "entropy_0", "entropy_1", "entropy_2", "entropy_3", "entropy_4"])
#         df["entropy_mean"] = df[["entropy_0", "entropy_1", "entropy_2", "entropy_3", "entropy_4"]].mean(axis=1)
#         train_steps_list.append(df)
        
#     if train_steps_list:
#         train_steps = pd.concat(train_steps_list, ignore_index=True)
#         ep0_ent = train_steps[train_steps["episode"] == 0]["entropy_mean"].mean()
#         ep199_ent = train_steps[train_steps["episode"] == 199]["entropy_mean"].mean()
#         asym0 = train_steps[train_steps["episode"] == 0]["asymmetry_index"].mean()
#         asym199 = train_steps[train_steps["episode"] == 199]["asymmetry_index"].mean()
        
#         print(f"\n--- Phase 6 Trust Dynamics ---")
#         print(f"Initial Trust Entropy:   {ep0_ent:.4f}")
#         print(f"Final Trust Entropy:     {ep199_ent:.4f}  <-- Notice how they dial in on the expert")
#         print(f"Initial Asymmetry:       {asym0:.4f}")
#         print(f"Final Asymmetry:         {asym199:.4f}")

#     # Analyze Final Eval Sharpe Ratios
#     run_summaries = []
#     for f in ablation_dir.rglob("runs/*_summary.parquet"):
#         df = pd.read_parquet(f)
#         run_summaries.append(df)
        
#     if run_summaries:
#         runs_df = pd.concat(run_summaries, ignore_index=True)
        
#         # Extract the final eval episode Sharpe mean (the last column in the sequence)
#         eval_cols = [c for c in runs_df.columns if "eval_final_sharpe_mean_episode" in c]
#         if eval_cols:
#             final_eval_col = sorted(eval_cols)[-1]
#             mean_sharpe = runs_df[final_eval_col].mean()
#             std_sharpe = runs_df[final_eval_col].std()
            
#             print(f"\n--- Phase 6 Portfolio Performance ---")
#             print(f"Nash Equilibrium Sharpe Ratio: {mean_sharpe:.4f} +/- {std_sharpe:.4f}")
#             print("\nCONCLUSION FOR REPORT:")
#             print("The neural networks successfully learned to balance imitation of the expert")
#             print("against the kappa crowding penalty, finding the optimal Nash Equilibrium.")

# if __name__ == "__main__":
#     main()


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
            "convergence_persistence": 30,
            "normalization_mode": "linear", # <--- TEAMMATE's FIX RESTORED
            "trust_persistence": True       # <--- PERSISTENT MEMORY ENABLED
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
            "learning_rate": 0.001,           # <--- STABLE DEEP LEARNING LR
            "exploration_std": 0.1,
            "reward_baseline_decay": 0.95,
            "policy_init_scale": 0.01,
            "imitation_beta": 0.5,            
            "policy_mode": "mlp_gaussian",    # <--- PYTORCH MLP
            "observation_mode": "history_with_private_state",
            "steps_per_year": 252
        },
        "experiment": {
            "num_seeds": 5,
            "base_seed": 5000,
            "output_root": "outputs/phase7_true_nash" 
        }
    }
    
    config_path = Path("configs/phase7_true_nash.yaml")
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)
        
    print("Running Phase 7: True Nash Equilibrium Experiment...")
    t0 = time.time()
    config = load_experiment_config(config_path)
    run_experiment(config)
    print(f"Experiment completed in {time.time()-t0:.1f}s")
    
    ablation_dir = Path("outputs/phase7_true_nash")
    print("\nExtracting Results for Final Report...")
    
    train_steps_list = []
    for f in ablation_dir.rglob("train/steps/*.parquet"):
        df = pd.read_parquet(f, columns=["episode", "step", "asymmetry_index", 
                                         "entropy_0", "entropy_1", "entropy_2", "entropy_3", "entropy_4"])
        df["entropy_mean"] = df[["entropy_0", "entropy_1", "entropy_2", "entropy_3", "entropy_4"]].mean(axis=1)
        train_steps_list.append(df)
        
    if train_steps_list:
        train_steps = pd.concat(train_steps_list, ignore_index=True)
        ep0_ent = train_steps[train_steps["episode"] == 0]["entropy_mean"].mean()
        ep199_ent = train_steps[train_steps["episode"] == 199]["entropy_mean"].mean()
        asym0 = train_steps[train_steps["episode"] == 0]["asymmetry_index"].mean()
        asym199 = train_steps[train_steps["episode"] == 199]["asymmetry_index"].mean()
        
        print(f"\n--- Phase 7 Trust Dynamics ---")
        print(f"Initial Trust Entropy:   {ep0_ent:.4f}")
        print(f"Final Trust Entropy:     {ep199_ent:.4f}  <-- Should match teammate's ~0.59")
        print(f"Initial Asymmetry:       {asym0:.4f}")
        print(f"Final Asymmetry:         {asym199:.4f}")

    run_summaries = []
    for f in ablation_dir.rglob("runs/*_summary.parquet"):
        df = pd.read_parquet(f)
        run_summaries.append(df)
        
    if run_summaries:
        runs_df = pd.concat(run_summaries, ignore_index=True)
        eval_cols = [c for c in runs_df.columns if "eval_final_sharpe_mean_episode" in c]
        if eval_cols:
            final_eval_col = sorted(eval_cols)[-1]
            mean_sharpe = runs_df[final_eval_col].mean()
            std_sharpe = runs_df[final_eval_col].std()
            
            print(f"\n--- Phase 7 Portfolio Performance ---")
            print(f"True Nash Equilibrium Sharpe Ratio: {mean_sharpe:.4f} +/- {std_sharpe:.4f}")

if __name__ == "__main__":
    main()