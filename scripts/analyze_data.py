import pandas as pd
import numpy as np

import sys
sys.path.insert(0, 'src')
from twmacl.market import CorrelatedMarket
from twmacl.config import load_experiment_config

def main():
    config = load_experiment_config("configs/phase2.yaml")
    mu = np.array(config.market.mu)
    cov = np.array(config.market.cov)
    
    random_sharpes = []
    np.random.seed(42)
    
    for i in range(50):
        market = CorrelatedMarket(mu=mu, cov=cov, seed=42+i)
        returns = []
        for step in range(200):
            actions = np.random.randn(5, 4)
            norms = np.abs(actions).sum(axis=1, keepdims=True)
            actions = actions / np.maximum(norms, 1e-8) * 2.0
            
            r = market.sample_return()
            agent_returns = np.sum(actions * r[None, :], axis=1)
            returns.append(agent_returns)
            
        returns_array = np.array(returns)
        mean_ret = returns_array.mean(axis=0)
        std_ret = returns_array.std(axis=0) + 1e-8
        sr = (mean_ret / std_ret) * np.sqrt(252)
        random_sharpes.append(sr.mean())
        
    print(f"Random Agent Annualized Sharpe: {np.mean(random_sharpes):.4f} +/- {np.std(random_sharpes):.4f}")

if __name__ == "__main__":
    main()
