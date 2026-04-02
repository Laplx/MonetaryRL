from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FiniteHorizonDPConfig:
    horizon: int
    discount_factor: float
    action_low: float
    action_high: float
    action_points: int
    state_low: tuple[float, float, float]
    state_high: tuple[float, float, float]
    state_points: tuple[int, int, int]
    state_abs_limit: float
    terminal_penalty: float

    def build_state_grids(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(
            np.linspace(low, high, points, dtype=float)
            for low, high, points in zip(self.state_low, self.state_high, self.state_points)
        )

    def build_action_grid(self) -> np.ndarray:
        return np.linspace(self.action_low, self.action_high, self.action_points, dtype=float)


def three_point_normal_quadrature(shock_dim: int) -> tuple[np.ndarray, np.ndarray]:
    nodes_1d = np.array([-np.sqrt(3.0), 0.0, np.sqrt(3.0)], dtype=float)
    weights_1d = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=float)
    mesh = np.meshgrid(*([nodes_1d] * shock_dim), indexing="ij")
    weight_mesh = np.meshgrid(*([weights_1d] * shock_dim), indexing="ij")
    nodes = np.stack([axis.reshape(-1) for axis in mesh], axis=1)
    weights = np.prod(np.stack([axis.reshape(-1) for axis in weight_mesh], axis=1), axis=1)
    return nodes, weights


def _prepare_grid_points(grids: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    mesh = np.meshgrid(*grids, indexing="ij")
    return np.column_stack([axis.reshape(-1) for axis in mesh])


def _interp3_batch(
    values: np.ndarray,
    grids: tuple[np.ndarray, np.ndarray, np.ndarray],
    points: np.ndarray,
) -> np.ndarray:
    gx, gy, gz = grids
    pts = np.asarray(points, dtype=float).copy()
    pts[:, 0] = np.clip(pts[:, 0], gx[0], gx[-1])
    pts[:, 1] = np.clip(pts[:, 1], gy[0], gy[-1])
    pts[:, 2] = np.clip(pts[:, 2], gz[0], gz[-1])

    ix1 = np.searchsorted(gx, pts[:, 0], side="right")
    iy1 = np.searchsorted(gy, pts[:, 1], side="right")
    iz1 = np.searchsorted(gz, pts[:, 2], side="right")

    ix1 = np.clip(ix1, 1, len(gx) - 1)
    iy1 = np.clip(iy1, 1, len(gy) - 1)
    iz1 = np.clip(iz1, 1, len(gz) - 1)

    ix0 = ix1 - 1
    iy0 = iy1 - 1
    iz0 = iz1 - 1

    x0 = gx[ix0]
    x1 = gx[ix1]
    y0 = gy[iy0]
    y1 = gy[iy1]
    z0 = gz[iz0]
    z1 = gz[iz1]

    tx = np.where(x1 > x0, (pts[:, 0] - x0) / (x1 - x0), 0.0)
    ty = np.where(y1 > y0, (pts[:, 1] - y0) / (y1 - y0), 0.0)
    tz = np.where(z1 > z0, (pts[:, 2] - z0) / (z1 - z0), 0.0)

    v000 = values[ix0, iy0, iz0]
    v001 = values[ix0, iy0, iz1]
    v010 = values[ix0, iy1, iz0]
    v011 = values[ix0, iy1, iz1]
    v100 = values[ix1, iy0, iz0]
    v101 = values[ix1, iy0, iz1]
    v110 = values[ix1, iy1, iz0]
    v111 = values[ix1, iy1, iz1]

    c00 = v000 * (1.0 - tx) + v100 * tx
    c01 = v001 * (1.0 - tx) + v101 * tx
    c10 = v010 * (1.0 - tx) + v110 * tx
    c11 = v011 * (1.0 - tx) + v111 * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    return c0 * (1.0 - tz) + c1 * tz


def _batch_stage_loss(model, states: np.ndarray, action: float) -> np.ndarray:
    inflation_gap = states[:, 0]
    output_gap = states[:, 1]
    lagged_rate_gap = states[:, 2]
    rate_change = action - lagged_rate_gap
    weights = model.config.loss_weights
    losses = (
        weights["inflation"] * inflation_gap ** 2
        + weights["output_gap"] * output_gap ** 2
        + weights["rate_smoothing"] * rate_change ** 2
    )
    if hasattr(model.config, "inflation_upside_extra"):
        losses = losses + (
            model.config.inflation_upside_extra * np.maximum(inflation_gap, 0.0) ** 2
            + model.config.output_gap_downside_extra * np.maximum(-output_gap, 0.0) ** 2
        )
    return losses


def _batch_state_transition(model, states: np.ndarray, action: float, shock: np.ndarray) -> np.ndarray:
    linear_next = states @ model.A.T + action * model.B.reshape(1, -1)
    if hasattr(model.config, "inflation_gap_sq"):
        inflation_gap = states[:, 0]
        output_gap = states[:, 1]
        nonlinear_term = (
            model.config.inflation_gap_sq * inflation_gap * np.abs(inflation_gap)
            + model.config.output_gap_sq * output_gap * np.abs(output_gap)
            + model.config.inflation_output_cross * inflation_gap * output_gap
        )
        linear_next[:, 0] += nonlinear_term
    transition_shock = model.Sigma @ shock.reshape(-1, 1)
    return linear_next + transition_shock.reshape(1, -1)


class FiniteHorizonGridPolicy:
    def __init__(self, grids: tuple[np.ndarray, np.ndarray, np.ndarray], policy_tables: np.ndarray) -> None:
        self.grids = grids
        self.policy_tables = policy_tables

    def __call__(self, state: np.ndarray, t: int) -> float:
        index = min(max(int(t), 0), self.policy_tables.shape[0] - 1)
        point = np.asarray(state, dtype=float).reshape(1, 3)
        return float(_interp3_batch(self.policy_tables[index], self.grids, point)[0])


@dataclass
class FiniteHorizonDPSolution:
    grids: tuple[np.ndarray, np.ndarray, np.ndarray]
    action_grid: np.ndarray
    policy_tables: np.ndarray
    value_table: np.ndarray

    def policy(self) -> FiniteHorizonGridPolicy:
        return FiniteHorizonGridPolicy(self.grids, self.policy_tables)


def solve_finite_horizon_dp(
    model,
    config: FiniteHorizonDPConfig,
    shock_nodes: np.ndarray,
    shock_weights: np.ndarray,
) -> FiniteHorizonDPSolution:
    grids = config.build_state_grids()
    action_grid = config.build_action_grid()
    states = _prepare_grid_points(grids)
    nx, ny, nz = (len(axis) for axis in grids)

    value_next = np.zeros((nx, ny, nz), dtype=float)
    policy_tables = np.zeros((config.horizon, nx, ny, nz), dtype=float)

    for t in range(config.horizon - 1, -1, -1):
        best_values = np.full(states.shape[0], np.inf, dtype=float)
        best_actions = np.full(states.shape[0], action_grid[0], dtype=float)

        for action in action_grid:
            stage_loss = _batch_stage_loss(model, states, float(action))
            expected_cost = stage_loss.copy()

            for shock, weight in zip(shock_nodes, shock_weights):
                next_states = _batch_state_transition(model, states, float(action), shock)
                invalid = ~np.all(np.isfinite(next_states), axis=1)
                exploded = invalid | (np.max(np.abs(next_states), axis=1) > config.state_abs_limit)
                safe_next_states = np.nan_to_num(
                    next_states,
                    nan=0.0,
                    posinf=config.state_abs_limit,
                    neginf=-config.state_abs_limit,
                )
                continuation = _interp3_batch(value_next, grids, safe_next_states)
                expected_cost += weight * (
                    exploded.astype(float) * config.terminal_penalty
                    + (~exploded).astype(float) * config.discount_factor * continuation
                )

            improved = expected_cost < best_values
            best_values[improved] = expected_cost[improved]
            best_actions[improved] = action

        value_next = best_values.reshape(nx, ny, nz)
        policy_tables[t] = best_actions.reshape(nx, ny, nz)

    return FiniteHorizonDPSolution(
        grids=grids,
        action_grid=action_grid,
        policy_tables=policy_tables,
        value_table=value_next,
    )
