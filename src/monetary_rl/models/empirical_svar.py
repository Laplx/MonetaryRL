from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EmpiricalSVARConfig:
    name: str
    inflation_target: float
    neutral_rate: float
    discount_factor: float
    loss_weights: dict[str, float]
    output_gap_coefficients: dict[str, float]
    inflation_coefficients: dict[str, float]
    sample_start: str
    sample_end: str
    action_low: float
    action_high: float


class EmpiricalSVARModel:
    """Empirical recursive SVAR environment used for Phase 8 counterfactuals."""

    def __init__(self, config: EmpiricalSVARConfig) -> None:
        self.config = config

    @property
    def observation_dim(self) -> int:
        return 3

    @property
    def full_state_dim(self) -> int:
        return 5

    @property
    def shock_dim(self) -> int:
        return 2

    def observe(self, full_state: np.ndarray) -> np.ndarray:
        state = np.asarray(full_state, dtype=float).reshape(self.full_state_dim)
        inflation_t, _, output_gap_t, _, lagged_rate_level = state
        return np.array(
            [
                inflation_t - self.config.inflation_target,
                output_gap_t,
                lagged_rate_level - self.config.neutral_rate,
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

    def state_transition(self, full_state: np.ndarray, action_gap: float, shock: np.ndarray | None = None) -> np.ndarray:
        state = np.asarray(full_state, dtype=float).reshape(self.full_state_dim)
        shock_vec = np.zeros(self.shock_dim, dtype=float) if shock is None else np.asarray(shock, dtype=float).reshape(self.shock_dim)

        inflation_t, inflation_tm1, output_gap_t, output_gap_tm1, lagged_rate_level = state
        current_rate_level = self.action_to_level(action_gap)

        output_coef = self.config.output_gap_coefficients
        inflation_coef = self.config.inflation_coefficients

        output_gap_next = (
            output_coef["const"]
            + output_coef["output_gap_lag1"] * output_gap_t
            + output_coef["inflation_lag1"] * inflation_t
            + output_coef["policy_rate_lag1"] * current_rate_level
            + output_coef["policy_rate_lag2"] * lagged_rate_level
            + shock_vec[0]
        )

        inflation_next = (
            inflation_coef["const"]
            + inflation_coef["output_gap"] * output_gap_next
            + inflation_coef["output_gap_lag1"] * output_gap_t
            + inflation_coef["output_gap_lag2"] * output_gap_tm1
            + inflation_coef["inflation_lag1"] * inflation_t
            + inflation_coef["inflation_lag2"] * inflation_tm1
            + inflation_coef["policy_rate_lag1"] * current_rate_level
            + shock_vec[1]
        )

        return np.array(
            [
                inflation_next,
                inflation_t,
                output_gap_next,
                output_gap_t,
                current_rate_level,
            ],
            dtype=float,
        )
