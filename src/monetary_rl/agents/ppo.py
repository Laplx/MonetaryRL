from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


@dataclass
class PPOConfig:
    total_updates: int = 120
    rollout_steps: int = 1024
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    train_epochs: int = 10
    minibatch_size: int = 256
    entropy_coef: float = 1e-3
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_size: int = 64
    linear_policy: bool = True
    state_scale: tuple[float, float, float] = (2.5, 2.5, 3.0)
    eval_episodes: int = 32
    eval_interval: int = 1
    seed: int = 0

    @classmethod
    def from_json(cls, path: str | Path) -> "PPOConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**raw)


class PolicyValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_size: int, linear_policy: bool) -> None:
        super().__init__()
        self.linear_policy = linear_policy
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        if linear_policy:
            self.policy_head = nn.Linear(state_dim, 1)
            nn.init.zeros_(self.policy_head.weight)
            nn.init.zeros_(self.policy_head.bias)
        else:
            self.policy_head = nn.Linear(hidden_size, 1)
        self.value_head = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.tensor([-0.7], dtype=torch.float32))

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.body(states)
        if self.linear_policy:
            mean = self.policy_head(states)
        else:
            mean = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean.squeeze(-1), std.squeeze(-1), value


class PPOTrainer:
    def __init__(self, env, config: PPOConfig, device: str = "cpu") -> None:
        self.env = env
        self.config = config
        self.device = torch.device(device)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.state_scale = torch.tensor(config.state_scale, dtype=torch.float32, device=self.device)
        action_low = float(env.config.action_low)
        action_high = float(env.config.action_high)
        self.action_scale = torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32, device=self.device)
        self.action_bias = torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32, device=self.device)
        self.model = PolicyValueNet(env.model.state_dim, config.hidden_size, config.linear_policy).to(self.device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": list(self.model.body.parameters()) + list(self.model.policy_head.parameters()) + [self.model.log_std], "lr": config.policy_lr},
                {"params": self.model.value_head.parameters(), "lr": config.value_lr},
            ]
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return state_tensor / self.state_scale

    def _sample_squashed_action(self, mean: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = Normal(mean, std)
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh) * self.action_scale + self.action_bias
        log_prob = self._squashed_log_prob(dist, pre_tanh)
        return action, log_prob

    def _squashed_log_prob(self, dist: Normal, pre_tanh: torch.Tensor) -> torch.Tensor:
        tanh_pre = torch.tanh(pre_tanh)
        correction = torch.log(self.action_scale * (1.0 - tanh_pre.pow(2)) + 1e-6)
        return dist.log_prob(pre_tanh) - correction

    def _inverse_squash_action(self, action: torch.Tensor) -> torch.Tensor:
        normalized = ((action - self.action_bias) / self.action_scale).clamp(-0.999999, 0.999999)
        return 0.5 * (torch.log1p(normalized) - torch.log1p(-normalized))

    def _sample_action(self, state: np.ndarray) -> tuple[float, float, float]:
        state_tensor = self._normalize_state(state).unsqueeze(0)
        with torch.no_grad():
            mean, std, value = self.model(state_tensor)
            action, log_prob = self._sample_squashed_action(mean, std)
        return float(action.item()), float(log_prob.item()), float(value.item())

    def _deterministic_action(self, state: np.ndarray) -> float:
        state_tensor = self._normalize_state(state).unsqueeze(0)
        with torch.no_grad():
            mean, _, _ = self.model(state_tensor)
            action = torch.tanh(mean) * self.action_scale + self.action_bias
        return float(action.item())

    def _compute_gae(self, rewards, values, dones, last_value):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * next_value * nonterminal - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * nonterminal * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def train(self) -> dict:
        training_log: list[dict] = []
        state = self.env.reset(seed=self.config.seed)

        for update in range(self.config.total_updates):
            states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
            episode_rewards = []
            running_episode_reward = 0.0
            rollout_clip_count = 0
            rollout_action_sum = 0.0

            for _ in range(self.config.rollout_steps):
                action, log_prob, value = self._sample_action(state)
                next_state, reward, done, info = self.env.step(action)
                states.append(state.copy())
                actions.append(float(info["action"]))
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(float(done))
                values.append(value)
                running_episode_reward += reward
                rollout_action_sum += abs(float(info["action"]))
                rollout_clip_count += int(abs(float(info["raw_action"]) - float(info["action"])) > 1e-8)
                state = next_state

                if done:
                    episode_rewards.append(running_episode_reward)
                    running_episode_reward = 0.0
                    state = self.env.reset()

            last_value = 0.0
            if not dones[-1]:
                with torch.no_grad():
                    _, _, last_value_tensor = self.model(self._normalize_state(state).unsqueeze(0))
                    last_value = float(last_value_tensor.item())

            rewards_np = np.asarray(rewards, dtype=np.float32)
            values_np = np.asarray(values, dtype=np.float32)
            dones_np = np.asarray(dones, dtype=np.float32)
            advantages, returns = self._compute_gae(rewards_np, values_np, dones_np, last_value)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            states_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device) / self.state_scale
            actions_tensor = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device)
            old_log_probs_tensor = torch.as_tensor(np.asarray(log_probs), dtype=torch.float32, device=self.device)
            returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
            advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

            num_samples = len(states)
            for _ in range(self.config.train_epochs):
                indices = np.random.permutation(num_samples)
                for start in range(0, num_samples, self.config.minibatch_size):
                    batch_idx = indices[start : start + self.config.minibatch_size]
                    batch_states = states_tensor[batch_idx]
                    batch_actions = actions_tensor[batch_idx]
                    batch_old_log_probs = old_log_probs_tensor[batch_idx]
                    batch_returns = returns_tensor[batch_idx]
                    batch_advantages = advantages_tensor[batch_idx]

                    mean, std, values_pred = self.model(batch_states)
                    dist = Normal(mean, std)
                    pre_tanh_actions = self._inverse_squash_action(batch_actions)
                    new_log_probs = self._squashed_log_prob(dist, pre_tanh_actions)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                    policy_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
                    value_loss = ((values_pred - batch_returns) ** 2).mean()
                    loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()

            if update % self.config.eval_interval == 0 or update == self.config.total_updates - 1:
                eval_stats = self.evaluate(self.config.eval_episodes, seed=10_000 + update)
                training_log.append(
                    {
                        "update": update,
                        "mean_episode_reward_in_rollout": float(np.mean(episode_rewards)) if episode_rewards else float("nan"),
                        "rollout_clip_rate": rollout_clip_count / self.config.rollout_steps,
                        "rollout_mean_abs_action": rollout_action_sum / self.config.rollout_steps,
                        "eval_mean_reward": eval_stats["mean_reward"],
                        "eval_mean_discounted_loss": eval_stats["mean_discounted_loss"],
                        "eval_clip_rate": eval_stats["clip_rate"],
                    }
                )

        return {
            "training_log": training_log,
            "config": asdict(self.config),
            "eval_stats": self.evaluate(self.config.eval_episodes * 2, seed=99_999),
            "state_scale": list(self.config.state_scale),
            "policy_state_dict": self.model.state_dict(),
        }

    def evaluate(self, episodes: int, seed: int = 0) -> dict:
        rewards = []
        discounted_losses = []
        trajectories = []
        clip_count = 0
        step_count = 0
        abs_action_sum = 0.0
        for ep in range(episodes):
            state = self.env.reset(seed=seed + ep)
            done = False
            episode_reward = 0.0
            discounted_loss = 0.0
            discount = 1.0
            traj = []
            while not done:
                action = self._deterministic_action(state)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                discounted_loss += (-reward) * discount
                discount *= self.config.gamma
                clip_count += int(abs(float(info["raw_action"]) - float(info["action"])) > 1e-8)
                abs_action_sum += abs(float(info["action"]))
                step_count += 1
                traj.append(
                    {
                        "inflation_gap": float(state[0]),
                        "output_gap": float(state[1]),
                        "lagged_policy_rate_gap": float(state[2]),
                        "action": float(info["action"]),
                        "loss": float(info["loss"]),
                    }
                )
                state = next_state
            rewards.append(episode_reward)
            discounted_losses.append(discounted_loss)
            if ep == 0:
                trajectories = traj
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
            "mean_discounted_loss": float(np.mean(discounted_losses)),
            "std_discounted_loss": float(np.std(discounted_losses, ddof=1)) if len(discounted_losses) > 1 else 0.0,
            "mean_abs_action": abs_action_sum / step_count if step_count else 0.0,
            "clip_rate": clip_count / step_count if step_count else 0.0,
            "first_trajectory": trajectories,
        }
