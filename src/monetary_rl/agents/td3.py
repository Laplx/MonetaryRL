from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .replay_buffer import ReplayBuffer


@dataclass
class TD3Config:
    total_steps: int = 20000
    warmup_steps: int = 1000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    hidden_size: int = 64
    replay_capacity: int = 100000
    exploration_noise: float = 0.2
    target_noise: float = 0.15
    target_noise_clip: float = 0.3
    policy_delay: int = 2
    eval_episodes: int = 12
    eval_interval: int = 2000
    state_scale: tuple[float, float, float] = (2.5, 2.5, 3.0)
    seed: int = 0

    @classmethod
    def from_json(cls, path: str | Path) -> "TD3Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**raw)


class DeterministicActor(nn.Module):
    def __init__(self, state_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([states, actions], dim=-1))


class TD3Trainer:
    def __init__(self, env, config: TD3Config, device: str = "cpu") -> None:
        self.env = env
        self.config = config
        self.device = torch.device(device)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        action_low = float(env.config.action_low)
        action_high = float(env.config.action_high)
        self.action_scale = torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32, device=self.device)
        self.action_bias = torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32, device=self.device)
        self.action_low = action_low
        self.action_high = action_high
        self.state_scale = torch.tensor(config.state_scale, dtype=torch.float32, device=self.device)

        state_dim = env.model.state_dim
        self.actor = DeterministicActor(state_dim, config.hidden_size).to(self.device)
        self.actor_target = DeterministicActor(state_dim, config.hidden_size).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.q1 = Critic(state_dim, config.hidden_size).to(self.device)
        self.q2 = Critic(state_dim, config.hidden_size).to(self.device)
        self.target_q1 = Critic(state_dim, config.hidden_size).to(self.device)
        self.target_q2 = Critic(state_dim, config.hidden_size).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=config.critic_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=config.critic_lr)
        self.buffer = ReplayBuffer(state_dim=state_dim, capacity=config.replay_capacity, seed=config.seed)
        self.update_counter = 0

    def _normalize_states(self, states: np.ndarray | torch.Tensor) -> torch.Tensor:
        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        return states_tensor / self.state_scale

    def _actor_action(self, states: torch.Tensor, actor: nn.Module) -> torch.Tensor:
        raw = actor(states)
        return torch.tanh(raw) * self.action_scale + self.action_bias

    def _sample_action(self, state: np.ndarray) -> float:
        state_tensor = self._normalize_states(state).unsqueeze(0)
        with torch.no_grad():
            action = self._actor_action(state_tensor, self.actor)
        noise = np.random.normal(0.0, self.config.exploration_noise)
        return float(np.clip(action.item() + noise, self.action_low, self.action_high))

    def _deterministic_action(self, state: np.ndarray) -> float:
        state_tensor = self._normalize_states(state).unsqueeze(0)
        with torch.no_grad():
            action = self._actor_action(state_tensor, self.actor)
        return float(action.item())

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for src, tgt in zip(source.parameters(), target.parameters()):
                tgt.data.mul_(1.0 - self.config.tau).add_(self.config.tau * src.data)

    def _update(self) -> None:
        self.update_counter += 1
        states, actions, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)
        states_t = self._normalize_states(states)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = self._normalize_states(next_states)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_actions = self._actor_action(next_states_t, self.actor_target)
            noise = torch.randn_like(next_actions) * self.config.target_noise
            noise = noise.clamp(-self.config.target_noise_clip, self.config.target_noise_clip)
            next_actions = (next_actions + noise).clamp(self.action_low, self.action_high)
            target_q1 = self.target_q1(next_states_t, next_actions)
            target_q2 = self.target_q2(next_states_t, next_actions)
            target = rewards_t + self.config.gamma * (1.0 - dones_t) * torch.min(target_q1, target_q2)

        q1_loss = ((self.q1(states_t, actions_t) - target) ** 2).mean()
        q2_loss = ((self.q2(states_t, actions_t) - target) ** 2).mean()
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        if self.update_counter % self.config.policy_delay == 0:
            actor_actions = self._actor_action(states_t, self.actor)
            actor_loss = -self.q1(states_t, actor_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.q1, self.target_q1)
            self._soft_update(self.q2, self.target_q2)

    def train(self) -> dict:
        state = self.env.reset(seed=self.config.seed)
        training_log: list[dict] = []

        for step in range(1, self.config.total_steps + 1):
            if step <= self.config.warmup_steps:
                action = float(np.random.uniform(self.action_low, self.action_high))
            else:
                action = self._sample_action(state)

            next_state, reward, done, _ = self.env.step(action)
            self.buffer.add(state, action, reward, next_state, done)
            state = next_state

            if done:
                state = self.env.reset()

            if self.buffer.size >= self.config.batch_size and step > self.config.warmup_steps:
                self._update()

            if step % self.config.eval_interval == 0 or step == self.config.total_steps:
                eval_stats = self.evaluate(self.config.eval_episodes, seed=60_000 + step)
                training_log.append(
                    {
                        "step": step,
                        "eval_mean_reward": eval_stats["mean_reward"],
                        "eval_mean_discounted_loss": eval_stats["mean_discounted_loss"],
                        "eval_clip_rate": eval_stats["clip_rate"],
                    }
                )

        return {
            "config": asdict(self.config),
            "training_log": training_log,
            "actor_state_dict": self.actor.state_dict(),
        }

    def evaluate(self, episodes: int, seed: int = 0) -> dict:
        from monetary_rl.evaluation import evaluate_policy

        return evaluate_policy(self.env, lambda state, t: self._deterministic_action(state), episodes, self.env.model.config.discount_factor, seed)
