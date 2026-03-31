from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def evaluate_policy(env, policy_fn: Callable[[np.ndarray, int], float], episodes: int, gamma: float, seed: int) -> dict:
    rewards: list[float] = []
    discounted_losses: list[float] = []
    trajectories = []
    clip_count = 0
    explosion_count = 0
    step_count = 0
    abs_action_sum = 0.0

    for ep in range(episodes):
        state = env.reset(seed=seed + ep)
        done = False
        discount = 1.0
        total_reward = 0.0
        total_discounted_loss = 0.0
        traj = []
        t = 0

        while not done:
            raw_action = float(policy_fn(state.copy(), t))
            next_state, reward, done, info = env.step(raw_action)
            total_reward += reward
            total_discounted_loss += (-reward) * discount
            discount *= gamma
            clip_count += int(abs(float(info["raw_action"]) - float(info["action"])) > 1e-8)
            explosion_count += int(bool(info.get("exploded", False)))
            abs_action_sum += abs(float(info["action"]))
            step_count += 1
            traj.append(
                {
                    "period": t,
                    "inflation_gap": float(state[0]),
                    "output_gap": float(state[1]),
                    "lagged_policy_rate_gap": float(state[2]),
                    "action": float(info["action"]),
                    "loss": float(info["loss"]),
                }
            )
            state = next_state
            t += 1

        rewards.append(total_reward)
        discounted_losses.append(total_discounted_loss)
        if ep == 0:
            trajectories = traj

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
        "mean_discounted_loss": float(np.mean(discounted_losses)),
        "std_discounted_loss": float(np.std(discounted_losses, ddof=1)) if len(discounted_losses) > 1 else 0.0,
        "mean_abs_action": abs_action_sum / step_count if step_count else 0.0,
        "clip_rate": clip_count / step_count if step_count else 0.0,
        "explosion_rate": explosion_count / episodes if episodes else 0.0,
        "first_trajectory": trajectories,
    }


def fit_linear_policy_response(policy_name: str, policy_fn, low: tuple[float, float, float], high: tuple[float, float, float], grid_points: int = 5) -> dict[str, float]:
    grids = [np.linspace(lo, hi, grid_points) for lo, hi in zip(low, high)]
    rows = []
    for inflation_gap in grids[0]:
        for output_gap in grids[1]:
            for lagged_rate_gap in grids[2]:
                state = np.array([inflation_gap, output_gap, lagged_rate_gap], dtype=float)
                action = float(policy_fn(state, 0))
                rows.append([1.0, inflation_gap, output_gap, lagged_rate_gap, action])

    design = np.asarray(rows, dtype=float)
    X = design[:, :4]
    y = design[:, 4]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    return {
        "policy": policy_name,
        "intercept": float(beta[0]),
        "inflation_gap": float(beta[1]),
        "output_gap": float(beta[2]),
        "lagged_policy_rate_gap": float(beta[3]),
        "fit_rmse": rmse,
    }


def simulate_with_common_shocks(model, policy_map: dict[str, callable], initial_state: np.ndarray, shocks: np.ndarray, action_low: float, action_high: float) -> pd.DataFrame:
    rows = []
    for policy_name, policy_fn in policy_map.items():
        state = initial_state.copy()
        for t in range(shocks.shape[0]):
            raw_action = float(policy_fn(state.copy(), t))
            action = float(np.clip(raw_action, action_low, action_high))
            loss = float(model.stage_loss(state, action))
            rows.append(
                {
                    "policy": policy_name,
                    "period": t,
                    "inflation_gap": float(state[0]),
                    "output_gap": float(state[1]),
                    "lagged_policy_rate_gap": float(state[2]),
                    "action": action,
                    "loss": loss,
                }
            )
            state = model.state_transition(state, action, shocks[t])
    return pd.DataFrame(rows)
