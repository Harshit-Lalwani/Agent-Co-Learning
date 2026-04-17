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

## Detailed Plan: Phase 2 (Baseline Learning / S1)

### Objective
Introduce actual learning while keeping the environment fixed and the experiment modular:
- independent agent policies
- per-agent observations and rewards
- training and evaluation loops
- benchmark return, risk, and stability against the Phase 1 scaffold

The focus is a clean baseline, not trust weighting yet.

### Locked Design Choices
- Keep the same synthetic market as Phase 1
- Keep the same action space and leverage cap
- Keep the same logging contract where possible
- Keep trust diagnostics available for analysis, but do not use trust to drive actions yet
- Use independent agents as the baseline S1 condition

### Configuration-First Requirement (No Hard Coding)
All new learning behavior must be externalized to config:
- learning algorithm selection
- observation builder mode
- reward shaping parameters, if any
- exploration schedule
- optimizer settings
- replay or rollout settings, if used
- episode count and evaluation frequency
- checkpoint and artifact paths

### Core System Pieces
1. Environment wrapper
- Preserve the Phase 1 market simulator interface
- Expose reset/step semantics suitable for training
- Keep deterministic seeding behavior

2. Observation builder
- Define what each agent can observe at each step
- Start with a homogeneous baseline observation set
- Keep the observation schema modular so heterogeneous views can be added later

3. Agent policy
- Implement one independent policy per agent
- Keep the policy API separate from the environment API
- Support action sampling during training and deterministic evaluation

4. Learning update
- Add the baseline learning rule behind a small interface
- Keep the optimizer and update logic isolated from rollout collection
- Make the update path swappable for future S2 logic

5. Evaluation and logging
- Reuse the Phase 1 logger structure where possible
- Record learning curves, portfolio metrics, and stability diagnostics
- Keep seed-level summaries comparable to Phase 1 outputs

### Suggested Baseline Behavior
- Each agent learns from its own observation stream
- Each agent chooses portfolio weights subject to the same leverage constraint
- Rewards come from realized portfolio returns
- Exploration is explicit and configurable
- Trust values remain diagnostic only in this phase

### Implementation Work Plan (5 Days)
1. Day 1
- Define the learning-facing environment and observation schema
- Add agent policy and learner interfaces
- Reuse the Phase 1 config structure for new learning settings

2. Day 2
- Implement a stable independent baseline learner
- Connect action selection to the existing portfolio constraint logic
- Add training/evaluation mode separation

3. Day 3
- Wire in per-agent rollout collection and update steps
- Add learning-curve metrics and checkpointing
- Verify deterministic runs for a fixed seed

4. Day 4
- Extend logging for training summaries and evaluation summaries
- Keep trust diagnostics available as passive analysis outputs
- Run a first multi-seed baseline sweep

5. Day 5
- Compare baseline learning against Phase 1 no-learning results
- Check stability, variance, and seed sensitivity
- Freeze the S1 interface for Phase 3 trust-weighted work

### Required Outputs from Phase 2
- Per-step training traces for each agent
- Per-episode or per-run summaries with return and risk metrics
- Seed-level evaluation summaries
- Stable checkpoints or saved policies
- Diagnostics that remain comparable with Phase 1

### Phase 2 Exit Criteria
1. Independent baseline learning runs end-to-end for multiple seeds.
2. Training and evaluation outputs are reproducible for fixed seeds.
3. Baseline performance metrics are written in the same style as Phase 1 outputs.
4. The agent/policy interfaces are stable enough to support Phase 3 trust-weighted learning.

### Notes for Phase 3 Handoff
- Keep the policy API modular so trust weighting can wrap it later.
- Keep observation and logging schemas stable across S1 and S2.
- Preserve seed alignment so direct comparisons remain valid.

## Detailed Plan: Phase 3 (Trust-Weighted Strategy / S2)

### Objective
Integrate the trust metric calculated passively in Phase 1 and 2 directly into the action selection loop of the agents. This will enable trust-weighted policy imitation where agents blend their intended portfolio allocations with those of highly trusted peers.

### Implementation Setup
- Create a `Phase3Config` building upon the S1 framework, injecting an `imitation_beta` constant to govern the strength of the trust weighting mechanism.
- Abstract the action computation of the `LinearGaussianPolicy` to separate base mean construction from stochastic sampling.
- Implement `phase3_runner.py` wherein independent agent actions ($\mu_i$) are aggregated into hybrid means via $\mu_{i, hybrid} = (1-\beta)\mu_i + \beta \sum_j \tau_{ij}\mu_j$.
- Adjust the advantage gradients inside the policy update mathematically by factoring in the scalar $(1-\beta)$.
- Maintain rigid schema outputs identical to Phase 2 to allow for fully harmonized comparisons.

### Phase 3 Exit Criteria
1. The S2 strategy executes successfully end-to-end identically to S1.
2. The exact same seed permutations are perfectly tracked and evaluated without errors.
3. Trust outputs dictate differing actions compared to the baseline run.

## Detailed Plan: Phase 4 (Evaluation and Report)

### Objective
Synthesize the multi-seed outputs of the independent S1 models and the collaborative S2 models into an analytical, statically sound final project report that addresses the core research questions.

### Execution Path
- **Parameter Sweeps**: Run large scale ablations using `scripts/run_ablations.py` across multiple `imitation_beta` scalars and predictors (`moving_average` vs `noisy_oracle`).
- **Jupyter Report**: Synthesize outputs using standard pandas pipelines into `phase4_evaluation_report.ipynb`.
- **RQ1**: Plot trust entropy and the asymmetry index to verify trust matrix convergence.
- **RQ2**: Verify the performance difference directly between the S1 and S2 strategies using statistical Sharpe box plots.
- **RQ3**: Plot terminal asymmetry thresholds against individual evaluation Sharpe scores to study trust-driven inequality.
