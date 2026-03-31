from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .replay_buffer import ReplayBuffer


@dataclass
class SACConfig:
    total_steps: int = 20000
    warmup_steps: int = 1000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha: float = 0.1
    hidden_size: int = 64
    replay_capacity: int = 100000
    eval_episodes: int = 12
    eval_interval: int = 2000
    state_scale: tuple[float, float, float] = (2.5, 2.5, 3.0)
    seed: int = 0

    @classmethod
    def from_json(cls, path: str | Path) -> "SACConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**raw)


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_size, 1)
        self.log_std_head = nn.Linear(hidden_size, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(states)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(-5.0, 1.0)
        return mean, log_std


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


class SACTrainer:
    def __init__(self, env, config: SACConfig, device: str = "cpu") -> None:
        self.env = env
        self.config = config
        self.device = torch.device(device)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        action_low = float(env.config.action_low)
        action_high = float(env.config.action_high)
        self.action_scale = torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32, device=self.device)
        self.action_bias = torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32, device=self.device)
        self.state_scale = torch.tensor(config.state_scale, dtype=torch.float32, device=self.device)

        state_dim = env.model.state_dim
        self.actor = GaussianActor(state_dim, config.hidden_size).to(self.device)
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

    def _normalize_states(self, states: np.ndarray | torch.Tensor) -> torch.Tensor:
        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        return states_tensor / self.state_scale

    def _sample_action(self, state: np.ndarray) -> float:
        state_tensor = self._normalize_states(state).unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.actor(state_tensor)
            std = log_std.exp()
            dist = Normal(mean, std)
            pre_tanh = dist.rsample()
            action = torch.tanh(pre_tanh) * self.action_scale + self.action_bias
        return float(action.item())

    def _deterministic_action(self, state: np.ndarray) -> float:
        state_tensor = self._normalize_states(state).unsqueeze(0)
        with torch.no_grad():
            mean, _ = self.actor(state_tensor)
            action = torch.tanh(mean) * self.action_scale + self.action_bias
        return float(action.item())

    def _sample_action_and_log_prob(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.actor(states)
        std = log_std.exp()
        dist = Normal(mean, std)
        pre_tanh = dist.rsample()
        tanh_pre = torch.tanh(pre_tanh)
        actions = tanh_pre * self.action_scale + self.action_bias
        correction = torch.log(self.action_scale * (1.0 - tanh_pre.pow(2)) + 1e-6)
        log_prob = dist.log_prob(pre_tanh) - correction
        return actions, log_prob.sum(dim=-1, keepdim=True)

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for src, tgt in zip(source.parameters(), target.parameters()):
                tgt.data.mul_(1.0 - self.config.tau).add_(self.config.tau * src.data)

    def _update(self) -> None:
        states, actions, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)
        states_t = self._normalize_states(states)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = self._normalize_states(next_states)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_actions, next_log_prob = self._sample_action_and_log_prob(next_states_t)
            target_q1 = self.target_q1(next_states_t, next_actions)
            target_q2 = self.target_q2(next_states_t, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.config.alpha * next_log_prob
            target = rewards_t + self.config.gamma * (1.0 - dones_t) * target_q

        q1_loss = ((self.q1(states_t, actions_t) - target) ** 2).mean()
        q2_loss = ((self.q2(states_t, actions_t) - target) ** 2).mean()
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        new_actions, log_prob = self._sample_action_and_log_prob(states_t)
        q_min = torch.min(self.q1(states_t, new_actions), self.q2(states_t, new_actions))
        actor_loss = (self.config.alpha * log_prob - q_min).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)

    def train(self) -> dict:
        state = self.env.reset(seed=self.config.seed)
        training_log: list[dict] = []
        episode_reward = 0.0

        for step in range(1, self.config.total_steps + 1):
            if step <= self.config.warmup_steps:
                action = float(np.random.uniform(self.env.config.action_low, self.env.config.action_high))
            else:
                action = self._sample_action(state)

            next_state, reward, done, _ = self.env.step(action)
            self.buffer.add(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state

            if done:
                state = self.env.reset()
                episode_reward = 0.0

            if self.buffer.size >= self.config.batch_size and step > self.config.warmup_steps:
                self._update()

            if step % self.config.eval_interval == 0 or step == self.config.total_steps:
                eval_stats = self.evaluate(self.config.eval_episodes, seed=50_000 + step)
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
        }

    def evaluate(self, episodes: int, seed: int = 0) -> dict:
        from monetary_rl.evaluation import evaluate_policy

        return evaluate_policy(self.env, lambda state, t: self._deterministic_action(state), episodes, self.env.model.config.discount_factor, seed)
