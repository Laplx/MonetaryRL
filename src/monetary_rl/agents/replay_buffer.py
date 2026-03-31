from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, state_dim: int, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, state: np.ndarray, action: float, reward: float, next_state: np.ndarray, done: bool) -> None:
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx, 0] = action
        self.rewards[idx, 0] = reward
        self.next_states[idx] = next_state
        self.dones[idx, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )
