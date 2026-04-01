from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monetary_rl.models.empirical_svar import EmpiricalSVARModel


@dataclass
class EmpiricalEnvConfig:
    horizon: int = 80
    action_low: float = -2.0
    action_high: float = 8.0
    state_abs_limit: float = 25.0
    terminal_penalty: float = 50.0
    seed: int = 0


class EmpiricalSVAREnv:
    """Bootstrap-shock empirical SVAR environment with benchmark-style observations."""

    def __init__(
        self,
        model: EmpiricalSVARModel,
        initial_states: np.ndarray,
        shock_pool: np.ndarray,
        config: EmpiricalEnvConfig,
    ) -> None:
        self.model = model
        self.initial_states = np.asarray(initial_states, dtype=float)
        self.shock_pool = np.asarray(shock_pool, dtype=float)
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.full_state = self.initial_states[0].copy()
        self.t = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        idx = int(self.rng.integers(low=0, high=len(self.initial_states)))
        self.full_state = self.initial_states[idx].copy()
        self.t = 0
        return self.model.observe(self.full_state)

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict]:
        clipped_action = float(np.clip(action, self.config.action_low, self.config.action_high))
        shock = self.shock_pool[int(self.rng.integers(low=0, high=len(self.shock_pool)))]
        loss = float(self.model.stage_loss(self.full_state, clipped_action))
        reward = -loss
        next_full_state = self.model.state_transition(self.full_state, clipped_action, shock)
        exploded = (not np.all(np.isfinite(next_full_state))) or bool(np.any(np.abs(next_full_state) > self.config.state_abs_limit))
        if exploded:
            reward -= self.config.terminal_penalty
            next_full_state = np.nan_to_num(
                next_full_state,
                nan=0.0,
                posinf=self.config.state_abs_limit,
                neginf=-self.config.state_abs_limit,
            )
            next_full_state = np.clip(next_full_state, -self.config.state_abs_limit, self.config.state_abs_limit)
        self.full_state = next_full_state.astype(float)
        self.t += 1
        done = self.t >= self.config.horizon or exploded
        info = {
            "loss": loss,
            "raw_action": float(action),
            "action": clipped_action,
            "exploded": exploded,
            "policy_rate": self.model.action_to_level(clipped_action),
            "inflation": float(self.full_state[0]),
            "output_gap": float(self.full_state[2]),
            "lagged_policy_rate": float(self.full_state[4]),
        }
        return self.model.observe(self.full_state), reward, done, info
