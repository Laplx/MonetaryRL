from __future__ import annotations

from pathlib import Path

from monetary_rl.envs import BenchmarkEnvConfig, LQBenchmarkEnv
from monetary_rl.models import (
    AsymmetricBenchmarkConfig,
    AsymmetricBenchmarkModel,
    LQBenchmarkConfig,
    LQBenchmarkModel,
    NonlinearBenchmarkConfig,
    NonlinearBenchmarkModel,
)


ROOT = Path(__file__).resolve().parents[2]
LINEAR_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_lq.json"
PPO_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_ppo_tuned.json"
SAC_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_sac.json"
TD3_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_td3.json"
TAYLOR_RULE_PATH = ROOT / "outputs" / "phase2" / "taylor_rule.json"

SEEDS = [7, 29, 43]
EVAL_EPISODES = 32
COMMON_SHOCK_HORIZON = 20
PPO_OVERRIDES = {
    "total_updates": 160,
    "rollout_steps": 512,
    "train_epochs": 4,
    "eval_interval": 20,
}
OFFPOLICY_OVERRIDES = {
    "total_steps": 12000,
    "eval_interval": 3000,
}

NEW_ENV_SPECS = [
    {
        "env_id": "nonlinear_very_strong",
        "group": "nonlinear",
        "tier": "very_strong",
        "model_kind": "nonlinear",
        "config_path": str(ROOT / "src" / "monetary_rl" / "config" / "nonlinear_phillips_very_strong.json"),
        "env_kwargs": {
            "horizon": 60,
            "action_low": -3.5,
            "action_high": 3.5,
            "initial_state_low": (-2.25, -2.25, -2.0),
            "initial_state_high": (2.25, 2.25, 2.0),
            "state_abs_limit": 35.0,
            "terminal_penalty": 150.0,
            "seed": 0,
        },
        "common_initial_state": [1.2, -1.4, 0.0],
        "dp_kwargs": {
            "state_low": (-5.0, -5.0, -3.5),
            "state_high": (5.0, 5.0, 3.5),
            "state_points": (17, 17, 17),
            "action_points": 41,
        },
    },
    {
        "env_id": "nonlinear_extreme",
        "group": "nonlinear",
        "tier": "extreme",
        "model_kind": "nonlinear",
        "config_path": str(ROOT / "src" / "monetary_rl" / "config" / "nonlinear_phillips_extreme.json"),
        "env_kwargs": {
            "horizon": 60,
            "action_low": -2.75,
            "action_high": 2.75,
            "initial_state_low": (-2.75, -2.75, -2.5),
            "initial_state_high": (2.75, 2.75, 2.5),
            "state_abs_limit": 40.0,
            "terminal_penalty": 200.0,
            "seed": 0,
        },
        "common_initial_state": [1.4, -1.6, 0.0],
        "dp_kwargs": {
            "state_low": (-6.0, -6.0, -2.75),
            "state_high": (6.0, 6.0, 2.75),
            "state_points": (17, 17, 17),
            "action_points": 37,
        },
    },
    {
        "env_id": "zlb_very_strong",
        "group": "zlb",
        "tier": "very_strong",
        "model_kind": "linear",
        "config_path": str(LINEAR_CONFIG_PATH),
        "env_kwargs": {
            "horizon": 60,
            "action_low": 0.0,
            "action_high": 5.0,
            "initial_state_low": (-3.5, -3.5, -0.75),
            "initial_state_high": (0.10, 0.10, 0.10),
            "state_abs_limit": 25.0,
            "terminal_penalty": 70.0,
            "seed": 0,
        },
        "common_initial_state": [-1.5, -2.0, -0.4],
        "dp_kwargs": {
            "state_low": (-5.0, -5.0, 0.0),
            "state_high": (1.5, 1.5, 5.0),
            "state_points": (17, 17, 17),
            "action_points": 41,
        },
    },
    {
        "env_id": "zlb_extreme",
        "group": "zlb",
        "tier": "extreme",
        "model_kind": "linear",
        "config_path": str(LINEAR_CONFIG_PATH),
        "env_kwargs": {
            "horizon": 60,
            "action_low": 0.25,
            "action_high": 4.0,
            "initial_state_low": (-4.5, -4.5, -0.25),
            "initial_state_high": (0.02, 0.02, 0.02),
            "state_abs_limit": 25.0,
            "terminal_penalty": 80.0,
            "seed": 0,
        },
        "common_initial_state": [-1.8, -2.5, -0.1],
        "dp_kwargs": {
            "state_low": (-5.5, -5.5, 0.25),
            "state_high": (1.0, 1.0, 4.0),
            "state_points": (17, 17, 17),
            "action_points": 37,
        },
    },
    {
        "env_id": "asymmetric_very_strong",
        "group": "asymmetric",
        "tier": "very_strong",
        "model_kind": "asymmetric",
        "config_path": str(ROOT / "src" / "monetary_rl" / "config" / "asymmetric_loss_very_strong.json"),
        "env_kwargs": {
            "horizon": 60,
            "action_low": -6.0,
            "action_high": 6.0,
            "initial_state_low": (-2.5, -2.5, -2.0),
            "initial_state_high": (2.5, 2.5, 2.0),
            "state_abs_limit": 25.0,
            "terminal_penalty": 60.0,
            "seed": 0,
        },
        "common_initial_state": [1.0, -1.0, 0.0],
        "dp_kwargs": {
            "state_low": (-4.5, -4.5, -6.0),
            "state_high": (4.5, 4.5, 6.0),
            "state_points": (17, 17, 17),
            "action_points": 49,
        },
    },
    {
        "env_id": "asymmetric_extreme",
        "group": "asymmetric",
        "tier": "extreme",
        "model_kind": "asymmetric",
        "config_path": str(ROOT / "src" / "monetary_rl" / "config" / "asymmetric_loss_extreme.json"),
        "env_kwargs": {
            "horizon": 60,
            "action_low": -6.0,
            "action_high": 6.0,
            "initial_state_low": (-2.75, -2.75, -2.25),
            "initial_state_high": (2.75, 2.75, 2.25),
            "state_abs_limit": 25.0,
            "terminal_penalty": 60.0,
            "seed": 0,
        },
        "common_initial_state": [1.2, -1.2, 0.0],
        "dp_kwargs": {
            "state_low": (-5.0, -5.0, -6.0),
            "state_high": (5.0, 5.0, 6.0),
            "state_points": (17, 17, 17),
            "action_points": 49,
        },
    },
]

TIER_ORDER = ["mild", "medium", "strong", "very_strong", "extreme"]


def make_model(spec: dict):
    if spec["model_kind"] == "linear":
        config = LQBenchmarkConfig.from_json(spec["config_path"])
        model = LQBenchmarkModel(config)
    elif spec["model_kind"] == "nonlinear":
        config = NonlinearBenchmarkConfig.from_json(spec["config_path"])
        model = NonlinearBenchmarkModel(config)
    elif spec["model_kind"] == "asymmetric":
        config = AsymmetricBenchmarkConfig.from_json(spec["config_path"])
        model = AsymmetricBenchmarkModel(config)
    else:
        raise ValueError(f"Unknown model kind: {spec['model_kind']}")
    return config, model


def make_env(spec: dict):
    _, model = make_model(spec)
    env = LQBenchmarkEnv(model, BenchmarkEnvConfig(**spec["env_kwargs"]))
    return env
