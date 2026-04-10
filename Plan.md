# Trust-Weighted Multi-Agent Co-Learning: Project Plan

## Phase Roadmap

### Phase 1: Exploratory Phase (5 Days)
Goal: validate trust dynamics, diagnostics, and metric pipeline before any RL training.

### Phase 2: Baseline Learning (S1)
Goal: implement and stabilize independent baseline learning, then benchmark risk-return behavior.

### Phase 3: Trust-Weighted Strategy (S2)
Goal: add trust-weighted policy imitation and evaluate improvement versus S1.

### Phase 4: Evaluation and Report
Goal: perform multi-seed statistical analysis, ablations, and final RQ1-RQ3 conclusions.

## Detailed Plan: Phase 1 (Exploratory)

### Objective
Build a configurable, reproducible experimental scaffold to understand the mechanism first:
- market generation
- trust updates
- diagnostic simulation
- convergence and performance metrics

No RL training in this phase.

### Locked Design Choices
- Market model: synthetic correlated returns (multivariate normal)
- Action space: long-short with leverage cap
- Costs: no transaction costs
- Trust normalization: row-wise softmax over peers
- Convergence rule: entropy slope threshold
- Logging format: Parquet
- Reproducibility: 5 seeds by default (configurable)
- Peer prediction source default: moving-average predictor (configurable)

### Configuration-First Requirement (No Hard Coding)
All values must be externalized to a config file:
- `num_agents` (default: 5)
- `num_assets` (default: 4)
- `leverage_cap` (default: 2.0)
- `num_seeds` (default: 5)
- `episode_length`
- `trust_alpha` ($\alpha$)
- `trust_lambda` ($\lambda$)
- `entropy_window` ($W$)
- `entropy_slope_threshold` ($\delta$)
- `predictor_mode` (default: `moving_average`)
- `predictor_window` (for moving-average predictor)
- market return parameters (`mu`, `cov`)
- random seeds and output paths

### Core Mathematical Definitions
Trust update:

$$
\tau_{ij}^{t+1} = (1-\alpha)\tau_{ij}^{t} + \alpha\exp\left(-\lambda\left\|\hat{s}_j^t-r^t\right\|_1\right)
$$

Row-wise softmax trust normalization:

$$
\tilde{\tau}_{ij}^t = \frac{\exp(\tau_{ij}^t)}{\sum_{k \ne i}\exp(\tau_{ik}^t)}
$$

Trust entropy (per agent):

$$
H_i^t = -\sum_{j \ne i}\tilde{\tau}_{ij}^t\log\tilde{\tau}_{ij}^t
$$

Trust asymmetry index:

$$
A^t = \frac{1}{M(M-1)}\sum_{i \ne j}\left|\tilde{\tau}_{ij}^t-\tilde{\tau}_{ji}^t\right|
$$

Convergence detector via entropy slope:
- Compute rolling slope of $H_i^t$ over window $W$
- Agent-level convergence when $|\text{slope}(H_i)| < \delta$
- System-level convergence when all agents satisfy criterion for a persistence window

### Peer Prediction Generator (Clarified)
Since Phase 1 has no RL training, each agent still needs a prediction $\hat{s}_j^t$ for trust updates. Use configurable predictor modes:
1. `moving_average` (default):

$$
\hat{s}_j^t = \frac{1}{K}\sum_{u=t-K}^{t-1} r^u + \epsilon_j^t
$$

2. `noisy_oracle` (ablation):

$$
\hat{s}_j^t = \mu_t + \epsilon_j^t
$$

3. `random` (ablation baseline):
- Draw predictions from a fixed reference distribution independent of recent history.

### Implementation Work Plan (5 Days)
1. Day 1
- Create config schema and parser
- Implement correlated return generator
- Define portfolio constraints and leverage projection

2. Day 2
- Implement trust update and row-wise softmax normalization
- Add shape/range validation checks
- Add deterministic seed handling

3. Day 3
- Implement diagnostics runner (no-learning policies)
- Implement predictor module with `moving_average`, `noisy_oracle`, `random`
- Run first end-to-end dry run

4. Day 4
- Implement metric computation pipeline
- Store step-level and run-level outputs to Parquet
- Add plotting scripts for entropy and asymmetry trajectories

5. Day 5
- Multi-seed reproducibility pass
- Validate convergence detector behavior
- Freeze Phase 1 report template and outputs

### Required Outputs from Phase 1
- Per-step Parquet logs:
  - normalized trust matrix entries
  - entropy per agent
  - asymmetry index
  - portfolio return series
- Per-run Parquet summary:
  - convergence flags and times
  - Sharpe and cumulative return plumbing
  - seed-level metadata and config snapshot

### Phase 1 Exit Criteria
1. One-command exploratory run executes end-to-end for all configured seeds.
2. All required metrics are written in Parquet format.
3. Re-running with identical seed reproduces identical outputs.
4. Entropy-slope convergence detection runs automatically and produces consistent flags.

### Notes for Phase 2 Handoff
- Keep all interfaces stable (environment, predictor, metrics, logger).
- Add RL on top of the same config + logging pipeline.
- Maintain strict comparability between S1 and S2 experiments.
