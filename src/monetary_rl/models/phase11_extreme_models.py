from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ZLBTrapBenchmarkConfig:
    name: str
    state_names: list[str]
    action_name: str
    inflation_target: float
    neutral_rate: float
    discount_factor: float
    loss_weights: dict[str, float]
    A: np.ndarray
    B: np.ndarray
    Sigma: np.ndarray
    lower_bound: float
    accommodation_buffer: float
    recession_threshold: float
    inflation_threshold: float
    recession_drag: float
    deflation_drag: float
    calibration_notes: dict[str, str]

    @classmethod
    def from_json(cls, path: str | Path) -> "ZLBTrapBenchmarkConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=raw["name"],
            state_names=raw["state_names"],
            action_name=raw["action_name"],
            inflation_target=float(raw["inflation_target"]),
            neutral_rate=float(raw["neutral_rate"]),
            discount_factor=float(raw["discount_factor"]),
            loss_weights={k: float(v) for k, v in raw["loss_weights"].items()},
            A=np.asarray(raw["A"], dtype=float),
            B=np.asarray(raw["B"], dtype=float),
            Sigma=np.asarray(raw["Sigma"], dtype=float),
            lower_bound=float(raw["lower_bound"]),
            accommodation_buffer=float(raw["accommodation_buffer"]),
            recession_threshold=float(raw["recession_threshold"]),
            inflation_threshold=float(raw["inflation_threshold"]),
            recession_drag=float(raw["recession_drag"]),
            deflation_drag=float(raw["deflation_drag"]),
            calibration_notes=raw["calibration_notes"],
        )


class ZLBTrapBenchmarkModel:
    def __init__(self, config: ZLBTrapBenchmarkConfig) -> None:
        self.config = config
        self.A = config.A
        self.B = config.B
        self.Sigma = config.Sigma

    @property
    def state_dim(self) -> int:
        return self.A.shape[0]

    @property
    def action_dim(self) -> int:
        return self.B.shape[1]

    def state_transition(self, state: np.ndarray, action: float, shock: np.ndarray | None = None) -> np.ndarray:
        state_vec = np.asarray(state, dtype=float).reshape(self.state_dim)
        action_vec = np.asarray([action], dtype=float)
        shock_vec = np.zeros(self.state_dim, dtype=float) if shock is None else np.asarray(shock, dtype=float).reshape(self.state_dim)
        next_state = self.A @ state_vec + (self.B @ action_vec).reshape(self.state_dim) + self.Sigma @ shock_vec

        inflation_gap = float(state_vec[0])
        output_gap = float(state_vec[1])
        policy_gap = max(float(action) - self.config.lower_bound - self.config.accommodation_buffer, 0.0)
        recession_tail = max(-output_gap - self.config.recession_threshold, 0.0)
        deflation_tail = max(-inflation_gap - self.config.inflation_threshold, 0.0)
        if recession_tail > 0.0 and policy_gap > 0.0:
            next_state[1] -= self.config.recession_drag * recession_tail * (1.0 + recession_tail) * policy_gap
            next_state[0] -= self.config.deflation_drag * recession_tail * (1.0 + deflation_tail) * policy_gap
        return next_state

    def stage_loss(self, state: np.ndarray, action: float) -> float:
        state_vec = np.asarray(state, dtype=float).reshape(self.state_dim)
        inflation_gap, output_gap, lagged_rate_gap = state_vec
        rate_change = float(action) - lagged_rate_gap
        weights = self.config.loss_weights
        return (
            weights["inflation"] * inflation_gap ** 2
            + weights["output_gap"] * output_gap ** 2
            + weights["rate_smoothing"] * rate_change ** 2
        )


@dataclass
class ThresholdAsymmetricBenchmarkConfig:
    name: str
    state_names: list[str]
    action_name: str
    inflation_target: float
    neutral_rate: float
    discount_factor: float
    loss_weights: dict[str, float]
    A: np.ndarray
    B: np.ndarray
    Sigma: np.ndarray
    inflation_threshold: float
    output_gap_threshold: float
    inflation_tail_weight: float
    output_tail_weight: float
    tail_power: float
    calibration_notes: dict[str, str]

    @classmethod
    def from_json(cls, path: str | Path) -> "ThresholdAsymmetricBenchmarkConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=raw["name"],
            state_names=raw["state_names"],
            action_name=raw["action_name"],
            inflation_target=float(raw["inflation_target"]),
            neutral_rate=float(raw["neutral_rate"]),
            discount_factor=float(raw["discount_factor"]),
            loss_weights={k: float(v) for k, v in raw["loss_weights"].items()},
            A=np.asarray(raw["A"], dtype=float),
            B=np.asarray(raw["B"], dtype=float),
            Sigma=np.asarray(raw["Sigma"], dtype=float),
            inflation_threshold=float(raw["inflation_threshold"]),
            output_gap_threshold=float(raw["output_gap_threshold"]),
            inflation_tail_weight=float(raw["inflation_tail_weight"]),
            output_tail_weight=float(raw["output_tail_weight"]),
            tail_power=float(raw["tail_power"]),
            calibration_notes=raw["calibration_notes"],
        )


class ThresholdAsymmetricBenchmarkModel:
    def __init__(self, config: ThresholdAsymmetricBenchmarkConfig) -> None:
        self.config = config
        self.A = config.A
        self.B = config.B
        self.Sigma = config.Sigma

    @property
    def state_dim(self) -> int:
        return self.A.shape[0]

    @property
    def action_dim(self) -> int:
        return self.B.shape[1]

    def state_transition(self, state: np.ndarray, action: float, shock: np.ndarray | None = None) -> np.ndarray:
        state_vec = np.asarray(state, dtype=float).reshape(self.state_dim)
        action_vec = np.asarray([action], dtype=float)
        shock_vec = np.zeros(self.state_dim, dtype=float) if shock is None else np.asarray(shock, dtype=float).reshape(self.state_dim)
        return self.A @ state_vec + (self.B @ action_vec).reshape(self.state_dim) + self.Sigma @ shock_vec

    def stage_loss(self, state: np.ndarray, action: float) -> float:
        state_vec = np.asarray(state, dtype=float).reshape(self.state_dim)
        inflation_gap, output_gap, lagged_rate_gap = state_vec
        rate_change = float(action) - lagged_rate_gap
        weights = self.config.loss_weights
        base_loss = (
            weights["inflation"] * inflation_gap ** 2
            + weights["output_gap"] * output_gap ** 2
            + weights["rate_smoothing"] * rate_change ** 2
        )
        inflation_tail = max(inflation_gap - self.config.inflation_threshold, 0.0)
        output_tail = max(-output_gap - self.config.output_gap_threshold, 0.0)
        tail_loss = (
            self.config.inflation_tail_weight * inflation_tail ** self.config.tail_power
            + self.config.output_tail_weight * output_tail ** self.config.tail_power
        )
        return base_loss + tail_loss
