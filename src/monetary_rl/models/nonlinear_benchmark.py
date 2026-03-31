from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class NonlinearBenchmarkConfig:
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
    inflation_gap_sq: float
    output_gap_sq: float
    inflation_output_cross: float
    calibration_notes: dict[str, str]

    @classmethod
    def from_json(cls, path: str | Path) -> "NonlinearBenchmarkConfig":
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
            inflation_gap_sq=float(raw["inflation_gap_sq"]),
            output_gap_sq=float(raw["output_gap_sq"]),
            inflation_output_cross=float(raw["inflation_output_cross"]),
            calibration_notes=raw["calibration_notes"],
        )


class NonlinearBenchmarkModel:
    """Benchmark extension with a nonlinear Phillips curve."""

    def __init__(self, config: NonlinearBenchmarkConfig) -> None:
        self.config = config
        self.A = config.A
        self.B = config.B
        self.Sigma = config.Sigma

        if self.A.shape != (3, 3):
            raise ValueError("A must be 3x3 for the nonlinear benchmark.")
        if self.B.shape != (3, 1):
            raise ValueError("B must be 3x1 for the nonlinear benchmark.")
        if self.Sigma.shape != (3, 3):
            raise ValueError("Sigma must be 3x3 for the nonlinear benchmark.")

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
        linear_next = self.A @ state_vec + (self.B @ action_vec).reshape(self.state_dim)

        inflation_gap, output_gap, _ = state_vec
        nonlinear_term = (
            self.config.inflation_gap_sq * inflation_gap * abs(inflation_gap)
            + self.config.output_gap_sq * output_gap * abs(output_gap)
            + self.config.inflation_output_cross * inflation_gap * output_gap
        )
        linear_next[0] += nonlinear_term
        return linear_next + self.Sigma @ shock_vec

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
