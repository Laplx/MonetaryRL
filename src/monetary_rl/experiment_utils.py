from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from monetary_rl.agents import (
    LinearPolicySearch,
    LinearPolicySearchConfig,
    PPOConfig,
    PPOTrainer,
    SACConfig,
    SACTrainer,
    TD3Config,
    TD3Trainer,
)
from monetary_rl.evaluation import evaluate_policy


def zero_gap_policy(state: np.ndarray, t: int) -> float:
    del state, t
    return 0.0


def load_taylor_rule(path: str | Path) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    coeffs = raw["coefficients"]
    return {
        "alpha": float(coeffs["const"]),
        "phi_pi": float(coeffs["inflation"]),
        "phi_x": float(coeffs["output_gap"]),
        "phi_i": float(coeffs["policy_rate_lag1"]),
    }


def build_taylor_gap_policy(rule: dict, model_config):
    pi_star = model_config.inflation_target
    i_star = model_config.neutral_rate
    intercept = rule["alpha"] + rule["phi_pi"] * pi_star + rule["phi_i"] * i_star - i_star

    def policy(state: np.ndarray, t: int) -> float:
        del t
        inflation_gap, output_gap, lagged_rate_gap = np.asarray(state, dtype=float)
        return (
            intercept
            + rule["phi_pi"] * inflation_gap
            + rule["phi_x"] * output_gap
            + rule["phi_i"] * lagged_rate_gap
        )

    return policy, intercept


def run_linear_search(env, eval_episodes: int, seed: int = 123) -> tuple[np.ndarray, dict, callable]:
    search = LinearPolicySearch(
        env,
        LinearPolicySearchConfig(
            seed=seed,
            iterations=24,
            population_size=48,
            episodes_per_candidate=4,
            eval_episodes=eval_episodes,
        ),
    )
    result = search.train()
    theta = np.asarray(result["best_theta"], dtype=float)

    def policy(state: np.ndarray, t: int) -> float:
        del t
        return float(theta @ np.asarray(state, dtype=float))

    return theta, result, policy


def run_ppo(env, config_path: str | Path, eval_episodes: int, seed: int, **overrides) -> tuple[dict, callable]:
    config = PPOConfig.from_json(config_path)
    config = replace(config, eval_episodes=eval_episodes, seed=seed, **overrides)
    trainer = PPOTrainer(env, config)
    result = trainer.train()

    def policy(state: np.ndarray, t: int) -> float:
        del t
        return trainer._deterministic_action(state)

    return result, policy


def run_sac(env, config_path: str | Path, eval_episodes: int, seed: int, **overrides) -> tuple[dict, callable]:
    config = SACConfig.from_json(config_path)
    config = replace(config, eval_episodes=eval_episodes, seed=seed, **overrides)
    trainer = SACTrainer(env, config)
    result = trainer.train()

    def policy(state: np.ndarray, t: int) -> float:
        del t
        return trainer._deterministic_action(state)

    return result, policy


def run_td3(env, config_path: str | Path, eval_episodes: int, seed: int, **overrides) -> tuple[dict, callable]:
    config = TD3Config.from_json(config_path)
    config = replace(config, eval_episodes=eval_episodes, seed=seed, **overrides)
    trainer = TD3Trainer(env, config)
    result = trainer.train()

    def policy(state: np.ndarray, t: int) -> float:
        del t
        return trainer._deterministic_action(state)

    return result, policy


def policy_row(policy_name: str, env, policy_fn, episodes: int, seed: int) -> dict:
    stats = evaluate_policy(env, policy_fn, episodes, env.model.config.discount_factor, seed)
    return {"policy": policy_name, **stats}


def training_log_frame(result: dict, algo: str, seed: int) -> pd.DataFrame:
    log = pd.DataFrame(result["training_log"])
    log.insert(0, "seed", seed)
    log.insert(0, "algo", algo)
    return log
