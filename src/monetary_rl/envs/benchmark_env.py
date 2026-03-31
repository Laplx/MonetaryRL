from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monetary_rl.models import LQBenchmarkModel


@dataclass
class BenchmarkEnvConfig:
    horizon: int = 60
    action_low: float = -6.0
    action_high: float = 6.0
    initial_state_low: tuple[float, float, float] = (-2.0, -2.0, -2.0)
    initial_state_high: tuple[float, float, float] = (2.0, 2.0, 2.0)
    seed: int = 0


class LQBenchmarkEnv:
    """Simple numpy environment for the stylized LQ benchmark."""

    def __init__(self, model: LQBenchmarkModel, config: BenchmarkEnvConfig) -> None:
        self.model = model
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.state = np.zeros(self.model.state_dim, dtype=np.float32)
        self.t = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        low = np.asarray(self.config.initial_state_low, dtype=np.float32)
        high = np.asarray(self.config.initial_state_high, dtype=np.float32)
        self.state = self.rng.uniform(low=low, high=high).astype(np.float32)
        self.t = 0
        return self.state.copy()

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict]:
        clipped_action = float(np.clip(action, self.config.action_low, self.config.action_high))
        shock = self.rng.standard_normal(self.model.state_dim)
        loss = float(self.model.stage_loss(self.state, clipped_action))
        reward = -loss
        next_state = self.model.state_transition(self.state, clipped_action, shock).astype(np.float32)
        self.state = next_state
        self.t += 1
        done = self.t >= self.config.horizon
        info = {"loss": loss, "raw_action": float(action), "action": clipped_action}
        return next_state.copy(), reward, done, info

