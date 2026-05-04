from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class EnvironmentConfig(BaseModel):
    num_agents: int = 5
    num_assets: int = 4
    episode_length: int = 200
    leverage_cap: float = 2.0

    @field_validator("num_agents")
    @classmethod
    def _validate_num_agents(cls, value: int) -> int:
        if value < 2:
            raise ValueError("num_agents must be >= 2")
        return value

    @field_validator("num_assets")
    @classmethod
    def _validate_num_assets(cls, value: int) -> int:
        if value < 1:
            raise ValueError("num_assets must be >= 1")
        return value

    @field_validator("episode_length")
    @classmethod
    def _validate_episode_length(cls, value: int) -> int:
        if value < 2:
            raise ValueError("episode_length must be >= 2")
        return value

    @field_validator("leverage_cap")
    @classmethod
    def _validate_leverage_cap(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("leverage_cap must be > 0")
        return value


class TrustConfig(BaseModel):
    trust_alpha: float = 0.2
    trust_lambda: float = 2.0
    entropy_window: int = 30
    entropy_slope_threshold: float = 0.00005
    convergence_persistence: int = 30
    # If True, trust is not reset between episodes (accumulates across training).
    # If False (default), trust resets at the start of each episode.
    trust_persistence: bool = False
    normalization_mode: Literal["softmax", "linear"] = "softmax"

    @field_validator("trust_alpha")
    @classmethod
    def _validate_alpha(cls, value: float) -> float:
        if not 0 < value <= 1:
            raise ValueError("trust_alpha must be in (0, 1]")
        return value

    @field_validator("trust_lambda")
    @classmethod
    def _validate_lambda(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("trust_lambda must be > 0")
        return value

    @field_validator("entropy_window", "convergence_persistence")
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("value must be >= 1")
        return value

    @field_validator("entropy_slope_threshold")
    @classmethod
    def _validate_threshold(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("entropy_slope_threshold must be > 0")
        return value


class PredictorConfig(BaseModel):
    predictor_mode: Literal["moving_average", "noisy_oracle", "random"] = "moving_average"
    predictor_window: int = 20
    noise_std: float = 0.01
    expert_agent_idx: int | None = None
    expert_noise_std: float = 0.001

    @field_validator("predictor_window")
    @classmethod
    def _validate_window(cls, value: int) -> int:
        if value < 1:
            raise ValueError("predictor_window must be >= 1")
        return value

    @field_validator("noise_std", "expert_noise_std")
    @classmethod
    def _validate_noise_std(cls, value: float) -> float:
        if value < 0:
            raise ValueError("noise_std and expert_noise_std must be >= 0")
        return value


class MarketConfig(BaseModel):
    mu: list[float]
    cov: list[list[float]]


class ExperimentConfig(BaseModel):
    num_seeds: int = 5
    base_seed: int = 2026
    output_root: str = "outputs/phase1"

    @field_validator("num_seeds")
    @classmethod
    def _validate_num_seeds(cls, value: int) -> int:
        if value < 1:
            raise ValueError("num_seeds must be >= 1")
        return value


class LearningConfig(BaseModel):
    num_train_episodes: int = 200
    num_eval_episodes: int = 10
    observation_window: int = 5
    learning_rate: float = 0.05
    exploration_std: float = 0.1
    reward_baseline_decay: float = 0.95
    policy_init_scale: float = 0.01
    # imitation_beta=0.0 → S1 (independent learning)
    # imitation_beta>0.0 → S2 (trust-weighted imitation)
    imitation_beta: float = 0.0
    policy_mode: Literal["linear_gaussian"] = "linear_gaussian"
    observation_mode: Literal["history_with_private_state"] = "history_with_private_state"
    # Steps per year for annualized Sharpe (252 = daily, 1 = raw)
    steps_per_year: int = 252

    @field_validator("num_train_episodes", "num_eval_episodes", "observation_window")
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("value must be >= 1")
        return value

    @field_validator("learning_rate")
    @classmethod
    def _validate_learning_rate(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("learning_rate must be > 0")
        return value

    @field_validator("exploration_std", "policy_init_scale")
    @classmethod
    def _validate_nonnegative_float(cls, value: float) -> float:
        if value < 0:
            raise ValueError("value must be >= 0")
        return value

    @field_validator("reward_baseline_decay")
    @classmethod
    def _validate_decay(cls, value: float) -> float:
        if not 0 <= value < 1:
            raise ValueError("reward_baseline_decay must be in [0, 1)")
        return value

    @field_validator("imitation_beta")
    @classmethod
    def _validate_imitation_beta(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("imitation_beta must be in [0, 1]")
        return value

    @field_validator("steps_per_year")
    @classmethod
    def _validate_steps_per_year(cls, value: int) -> int:
        if value < 1:
            raise ValueError("steps_per_year must be >= 1")
        return value


def _validate_market_shape(market: MarketConfig, num_assets: int, cls_name: str) -> None:
    if len(market.mu) != num_assets:
        raise ValueError(f"{cls_name}: market.mu length must match environment.num_assets")
    if len(market.cov) != num_assets:
        raise ValueError(f"{cls_name}: market.cov rows must match environment.num_assets")
    for row in market.cov:
        if len(row) != num_assets:
            raise ValueError(f"{cls_name}: market.cov must be square with size num_assets")


class Phase1Config(BaseModel):
    environment: EnvironmentConfig
    trust: TrustConfig
    predictor: PredictorConfig
    market: MarketConfig
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    @model_validator(mode="after")
    def _validate_market_shape(self) -> "Phase1Config":
        _validate_market_shape(self.market, self.environment.num_assets, "Phase1Config")
        return self

    def seed_values(self) -> list[int]:
        return [self.experiment.base_seed + i for i in range(self.experiment.num_seeds)]

    def output_root_path(self) -> Path:
        return Path(self.experiment.output_root)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(), sort_keys=True)

    def config_hash(self) -> str:
        return hashlib.md5(self.canonical_json().encode("utf-8"), usedforsecurity=False).hexdigest()


class ExperimentRunConfig(BaseModel):
    """Unified config for S1 (imitation_beta=0) and S2 (imitation_beta>0) experiments."""

    environment: EnvironmentConfig
    trust: TrustConfig
    predictor: PredictorConfig
    learning: LearningConfig
    market: MarketConfig
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    @model_validator(mode="after")
    def _validate_market_shape(self) -> "ExperimentRunConfig":
        _validate_market_shape(self.market, self.environment.num_assets, "ExperimentRunConfig")
        return self

    def seed_values(self) -> list[int]:
        return [self.experiment.base_seed + i for i in range(self.experiment.num_seeds)]

    def output_root_path(self) -> Path:
        return Path(self.experiment.output_root)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(), sort_keys=True)

    def config_hash(self) -> str:
        return hashlib.md5(self.canonical_json().encode("utf-8"), usedforsecurity=False).hexdigest()

    @property
    def is_s2(self) -> bool:
        """True if this is S2 (trust-weighted imitation), False if S1 (independent)."""
        return self.learning.imitation_beta > 0.0


def load_config(config_path: str | Path) -> Phase1Config:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Phase1Config.model_validate(raw)


def load_experiment_config(config_path: str | Path) -> ExperimentRunConfig:
    """Load a unified S1/S2 experiment config. imitation_beta=0 → S1, >0 → S2."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ExperimentRunConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# Backwards-compatibility shims — kept so existing scripts don't break.
# Will be removed after all callers are updated.
# ---------------------------------------------------------------------------

Phase2Config = ExperimentRunConfig
Phase3Config = ExperimentRunConfig


def load_phase2_config(config_path: str | Path) -> ExperimentRunConfig:
    return load_experiment_config(config_path)


def load_phase3_config(config_path: str | Path) -> ExperimentRunConfig:
    return load_experiment_config(config_path)
