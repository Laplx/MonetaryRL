from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class LQBenchmarkConfig:
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
    calibration_notes: dict[str, str]

    @classmethod
    def from_json(cls, path: str | Path) -> "LQBenchmarkConfig":
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
            calibration_notes=raw["calibration_notes"],
        )


class LQBenchmarkModel:
    """Stylized 3-state LQ benchmark for monetary policy."""

    def __init__(self, config: LQBenchmarkConfig) -> None:
        self.config = config
        self.A = config.A
        self.B = config.B
        self.Sigma = config.Sigma

        if self.A.shape != (3, 3):
            raise ValueError("A must be 3x3 for the Phase 3 benchmark.")
        if self.B.shape != (3, 1):
            raise ValueError("B must be 3x1 for the Phase 3 benchmark.")
        if self.Sigma.shape != (3, 3):
            raise ValueError("Sigma must be 3x3 for the Phase 3 benchmark.")

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
        return (
            weights["inflation"] * inflation_gap ** 2
            + weights["output_gap"] * output_gap ** 2
            + weights["rate_smoothing"] * rate_change ** 2
        )

    def qnr_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weights = self.config.loss_weights
        q = np.diag([weights["inflation"], weights["output_gap"], weights["rate_smoothing"]])
        n = np.array([[0.0], [0.0], [-weights["rate_smoothing"]]])
        r = np.array([[weights["rate_smoothing"]]])
        return q, n, r

    def closed_loop_matrix(self, feedback_matrix: np.ndarray) -> np.ndarray:
        feedback = np.asarray(feedback_matrix, dtype=float).reshape(self.action_dim, self.state_dim)
        return self.A - self.B @ feedback

    def simulate(
        self,
        initial_state: np.ndarray,
        policy: Callable[[np.ndarray, int], float],
        horizon: int,
        seed: int = 0,
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        states = np.zeros((horizon + 1, self.state_dim), dtype=float)
        actions = np.zeros(horizon, dtype=float)
        losses = np.zeros(horizon, dtype=float)
        shocks = np.zeros((horizon, self.state_dim), dtype=float)

        states[0] = np.asarray(initial_state, dtype=float).reshape(self.state_dim)
        for t in range(horizon):
            actions[t] = float(policy(states[t].copy(), t))
            losses[t] = self.stage_loss(states[t], actions[t])
            shocks[t] = rng.standard_normal(self.state_dim)
            states[t + 1] = self.state_transition(states[t], actions[t], shocks[t])

        discounted_losses = losses * (self.config.discount_factor ** np.arange(horizon))
        return {
            "states": states,
            "actions": actions,
            "losses": losses,
            "discounted_losses": discounted_losses,
            "total_discounted_loss": np.array([discounted_losses.sum()], dtype=float),
            "shocks": shocks,
        }
