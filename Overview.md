# Mathematics for Finance — Project Proposal  
## Trust-Weighted Multi-Agent Co-Learning for Portfolio Optimization

---

## Motivation

Portfolio-managing agents operating in shared markets can observe peer decisions and potentially improve their own strategies by selectively incorporating peer signals. We fix a trust mechanism and check which learning strategies are beneficial, stable, and robust.

---

## Problem Formulation

M agents each manage a portfolio over N risky assets in a discrete-time market.

Each agent *i* holds a trust vector τᵢᵗ over peers, updated via a standard exponential moving average on realized signal quality:

τᵢⱼ^{t+1} = (1−α)τᵢⱼ^t + α · exp(−λ|ŝ_j^t − r^t|₁)

where ŝ_j^t is agent j’s predicted return and r^t is the realized return.

- Trust is initialized uniformly  
- The aggregated peer signal used in decision-making is the trust-normalized weighted mean of peer predictions  

Each agent maximizes its own risk-adjusted return:

E[Σ_t (r_{i,t} − γ/2 · Var(r_{i,t}))]

---

## Learning Strategies

- **S1: Independent Q-Learning (Baseline)**  
- **S2: Trust-Weighted Policy Imitation**  
  Agent blends own action with trust-weighted peer actions:

  â_i = (1−β)a_i + βΣ_jτ_{ij}a_j

---

## Research Questions

| RQ | Question | Metric |
|----|----------|--------|
| 1 | Does trust converge, and how quickly? | Trust entropy H(τ_i^t) over time |
| 2 | Does cooperation via trust improve portfolio performance? | Sharpe ratio and cumulative return vs. S1 baseline |
| 3 | Does asymmetric trust (τ_{ij} ≠ τ_{ji}) emerge and persist? | Asymmetry index Σ_j\|τ_{ij}−τ_{ji}\|/M; correlation with performance gap |

---

## Implementation

- The market environment will be implemented as a custom Gymnasium simulator  
- Agents will be implemented in PyTorch  
- The three strategies will be evaluated across two conditions:
  - Homogeneous agents (same information set)
  - Heterogeneous agents (asset specialization)

---

## Bonus: Regime-Switching Extensions

### Problem Extension
- Market extended to a **discrete-time regime-switching setting**

### Additional Research Question

| RQ | Question | Metric |
|----|----------|--------|
| B1 | Is trust robust to market regime shifts? | Trust drift and recovery time post-regime switch |

### Additional Experiments

- Explore regime-switching returns in various regime shift scenarios