from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class LinearPolicySearchConfig:
    iterations: int = 40
    population_size: int = 96
    elite_frac: float = 0.125
    episodes_per_candidate: int = 8
    eval_episodes: int = 64
    init_std: float = 1.0
    min_std: float = 0.05
    seed: int = 0


class LinearPolicySearch:
    """Cross-entropy search for a deterministic linear policy a_t = k @ s_t."""

    def __init__(self, env, config: LinearPolicySearchConfig) -> None:
        self.env = env
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.state_dim = env.model.state_dim
        self.mean = np.zeros(self.state_dim, dtype=float)
        self.std = np.full(self.state_dim, config.init_std, dtype=float)

    def _evaluate_theta(self, theta: np.ndarray, episodes: int, seed: int) -> dict[str, float]:
        discounted_losses: list[float] = []
        rewards: list[float] = []
        clip_count = 0
        step_count = 0
        abs_action_sum = 0.0

        for ep in range(episodes):
            state = self.env.reset(seed=seed + ep)
            done = False
            discount = 1.0
            discounted_loss = 0.0
            total_reward = 0.0
            while not done:
                raw_action = float(theta @ state)
                next_state, reward, done, info = self.env.step(raw_action)
                total_reward += reward
                discounted_loss += (-reward) * discount
                discount *= self.env.model.config.discount_factor
                clip_count += int(abs(float(info["raw_action"]) - float(info["action"])) > 1e-8)
                abs_action_sum += abs(float(info["action"]))
                step_count += 1
                state = next_state

            discounted_losses.append(discounted_loss)
            rewards.append(total_reward)

        return {
            "mean_discounted_loss": float(np.mean(discounted_losses)),
            "std_discounted_loss": float(np.std(discounted_losses, ddof=1)) if len(discounted_losses) > 1 else 0.0,
            "mean_reward": float(np.mean(rewards)),
            "clip_rate": clip_count / step_count if step_count else 0.0,
            "mean_abs_action": abs_action_sum / step_count if step_count else 0.0,
        }

    def train(self) -> dict:
        elite_count = max(1, int(round(self.config.population_size * self.config.elite_frac)))
        history: list[dict[str, float]] = []
        best_theta = self.mean.copy()
        best_stats = self._evaluate_theta(best_theta, self.config.episodes_per_candidate, seed=100_000)

        for iteration in range(self.config.iterations):
            population = self.rng.normal(loc=self.mean, scale=self.std, size=(self.config.population_size, self.state_dim))
            scores = np.zeros(self.config.population_size, dtype=float)
            for idx, theta in enumerate(population):
                stats = self._evaluate_theta(theta, self.config.episodes_per_candidate, seed=iteration * 10_000 + idx * 100)
                scores[idx] = stats["mean_discounted_loss"]
                if stats["mean_discounted_loss"] < best_stats["mean_discounted_loss"]:
                    best_theta = theta.copy()
                    best_stats = stats

            elite_idx = np.argsort(scores)[:elite_count]
            elite = population[elite_idx]
            self.mean = elite.mean(axis=0)
            self.std = np.maximum(elite.std(axis=0), self.config.min_std)
            history.append(
                {
                    "iteration": iteration,
                    "best_population_loss": float(scores[elite_idx[0]]),
                    "mean_population_loss": float(scores.mean()),
                    "elite_mean_loss": float(scores[elite_idx].mean()),
                    "mean_norm": float(np.linalg.norm(self.mean)),
                    "std_norm": float(np.linalg.norm(self.std)),
                }
            )

        final_stats = self._evaluate_theta(best_theta, self.config.eval_episodes, seed=900_000)
        return {
            "config": asdict(self.config),
            "history": history,
            "best_theta": best_theta.tolist(),
            "best_stats": final_stats,
        }
