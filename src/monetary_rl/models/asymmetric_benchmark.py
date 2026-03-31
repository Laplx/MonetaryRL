from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AsymmetricBenchmarkConfig:
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
    inflation_upside_extra: float
    output_gap_downside_extra: float
    calibration_notes: dict[str, str]

    @classmethod
    def from_json(cls, path: str | Path) -> "AsymmetricBenchmarkConfig":
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
            inflation_upside_extra=float(raw["inflation_upside_extra"]),
            output_gap_downside_extra=float(raw["output_gap_downside_extra"]),
            calibration_notes=raw["calibration_notes"],
        )


class AsymmetricBenchmarkModel:
    """Linear transition benchmark with asymmetric policymaker loss."""

    def __init__(self, config: AsymmetricBenchmarkConfig) -> None:
        self.config = config
        self.A = config.A
        self.B = config.B
        self.Sigma = config.Sigma

        if self.A.shape != (3, 3):
            raise ValueError("A must be 3x3 for the asymmetric benchmark.")
        if self.B.shape != (3, 1):
            raise ValueError("B must be 3x1 for the asymmetric benchmark.")
        if self.Sigma.shape != (3, 3):
            raise ValueError("Sigma must be 3x3 for the asymmetric benchmark.")

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
        return next_state

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
        asym_loss = (
            self.config.inflation_upside_extra * max(inflation_gap, 0.0) ** 2
            + self.config.output_gap_downside_extra * max(-output_gap, 0.0) ** 2
        )
        return base_loss + asym_loss
