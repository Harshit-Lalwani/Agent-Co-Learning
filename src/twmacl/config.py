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
    episode_length: int = 300
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
    entropy_slope_threshold: float = 0.0005
    convergence_persistence: int = 10

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

    @field_validator("predictor_window")
    @classmethod
    def _validate_window(cls, value: int) -> int:
        if value < 1:
            raise ValueError("predictor_window must be >= 1")
        return value

    @field_validator("noise_std")
    @classmethod
    def _validate_noise_std(cls, value: float) -> float:
        if value < 0:
            raise ValueError("noise_std must be >= 0")
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


class Phase1Config(BaseModel):
    environment: EnvironmentConfig
    trust: TrustConfig
    predictor: PredictorConfig
    market: MarketConfig
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    @model_validator(mode="after")
    def _validate_market_shape(self) -> "Phase1Config":
        num_assets = self.environment.num_assets
        if len(self.market.mu) != num_assets:
            raise ValueError("market.mu length must match environment.num_assets")
        if len(self.market.cov) != num_assets:
            raise ValueError("market.cov rows must match environment.num_assets")
        for row in self.market.cov:
            if len(row) != num_assets:
                raise ValueError("market.cov must be square with size num_assets")
        return self

    def seed_values(self) -> list[int]:
        return [self.experiment.base_seed + i for i in range(self.experiment.num_seeds)]

    def output_root_path(self) -> Path:
        return Path(self.experiment.output_root)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(), sort_keys=True)

    def config_hash(self) -> str:
        return hashlib.md5(self.canonical_json().encode("utf-8"), usedforsecurity=False).hexdigest()


def load_config(config_path: str | Path) -> Phase1Config:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Phase1Config.model_validate(raw)
