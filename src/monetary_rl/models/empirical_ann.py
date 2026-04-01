from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EmpiricalANNConfig:
    name: str
    inflation_target: float
    neutral_rate: float
    discount_factor: float
    loss_weights: dict[str, float]
    output_regressors: list[str]
    inflation_regressors: list[str]
    state_max_lag: int
    sample_start: str
    sample_end: str
    action_low: float
    action_high: float
    output_spec: dict[str, Any]
    inflation_spec: dict[str, Any]


class EmpiricalANNModel:
    """Empirical nonlinear ANN environment used for Phase 9 supplementary analysis."""

    def __init__(self, config: EmpiricalANNConfig, output_pipeline: Any, inflation_pipeline: Any) -> None:
        self.config = config
        self.output_pipeline = output_pipeline
        self.inflation_pipeline = inflation_pipeline

    @property
    def observation_dim(self) -> int:
        return 3

    @property
    def full_state_dim(self) -> int:
        return 3 * self.config.state_max_lag - 1

    @property
    def shock_dim(self) -> int:
        return 2

    def observe(self, full_state: np.ndarray) -> np.ndarray:
        state = np.asarray(full_state, dtype=float).reshape(self.full_state_dim)
        return np.array(
            [
                state[0] - self.config.inflation_target,
                state[2],
                state[4] - self.config.neutral_rate,
            ],
            dtype=np.float32,
        )

    def action_to_level(self, action_gap: float) -> float:
        return self.config.neutral_rate + float(action_gap)

    def stage_loss(self, full_state: np.ndarray, action_gap: float) -> float:
        inflation_gap, output_gap, lagged_rate_gap = self.observe(full_state)
        rate_change = float(action_gap) - float(lagged_rate_gap)
        weights = self.config.loss_weights
        return (
            weights["inflation"] * float(inflation_gap) ** 2
            + weights["output_gap"] * float(output_gap) ** 2
            + weights["rate_smoothing"] * rate_change ** 2
        )

    def _inflation_value(self, state: np.ndarray, lag: int) -> float:
        if lag == 1:
            return float(state[0])
        if lag == 2:
            return float(state[1])
        return float(state[5 + 3 * (lag - 3)])

    def _output_gap_value(self, state: np.ndarray, lag: int) -> float:
        if lag == 1:
            return float(state[2])
        if lag == 2:
            return float(state[3])
        return float(state[6 + 3 * (lag - 3)])

    def _policy_rate_value(self, state: np.ndarray, lag: int, current_rate_level: float) -> float:
        if lag == 1:
            return float(current_rate_level)
        if lag == 2:
            return float(state[4])
        return float(state[7 + 3 * (lag - 3)])

    def _feature_row(
        self,
        regressors: list[str],
        state: np.ndarray,
        current_rate_level: float,
        output_gap_next: float | None = None,
    ) -> np.ndarray:
        values: list[float] = []
        for reg in regressors:
            if reg == "output_gap":
                if output_gap_next is None:
                    raise ValueError("output_gap current regressor requires output_gap_next.")
                values.append(float(output_gap_next))
            elif reg.startswith("output_gap_lag"):
                lag = int(reg.removeprefix("output_gap_lag"))
                values.append(self._output_gap_value(state, lag))
            elif reg.startswith("inflation_lag"):
                lag = int(reg.removeprefix("inflation_lag"))
                values.append(self._inflation_value(state, lag))
            elif reg.startswith("policy_rate_lag"):
                lag = int(reg.removeprefix("policy_rate_lag"))
                values.append(self._policy_rate_value(state, lag, current_rate_level))
            else:
                raise ValueError(f"Unsupported ANN regressor: {reg}")
        return np.asarray(values, dtype=float).reshape(1, -1)

    def state_transition(self, full_state: np.ndarray, action_gap: float, shock: np.ndarray | None = None) -> np.ndarray:
        state = np.asarray(full_state, dtype=float).reshape(self.full_state_dim)
        shock_vec = np.zeros(self.shock_dim, dtype=float) if shock is None else np.asarray(shock, dtype=float).reshape(self.shock_dim)
        current_rate_level = self.action_to_level(action_gap)

        output_features = self._feature_row(self.config.output_regressors, state, current_rate_level)
        output_gap_next = float(self.output_pipeline.predict(output_features)[0]) + float(shock_vec[0])

        inflation_features = self._feature_row(
            self.config.inflation_regressors,
            state,
            current_rate_level,
            output_gap_next=output_gap_next,
        )
        inflation_next = float(self.inflation_pipeline.predict(inflation_features)[0]) + float(shock_vec[1])

        next_state = [inflation_next, float(state[0]), output_gap_next, float(state[2]), current_rate_level]
        for lag in range(2, self.config.state_max_lag):
            next_state.extend(
                [
                    self._inflation_value(state, lag),
                    self._output_gap_value(state, lag),
                    self._policy_rate_value(state, lag, current_rate_level),
                ]
            )
        return np.asarray(next_state, dtype=float)
