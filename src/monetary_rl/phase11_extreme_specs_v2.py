from __future__ import annotations

from pathlib import Path

from monetary_rl.envs import BenchmarkEnvConfig, LQBenchmarkEnv
from monetary_rl.models import (
    LQBenchmarkConfig,
    LQBenchmarkModel,
    NonlinearBenchmarkConfig,
    NonlinearBenchmarkModel,
    ThresholdAsymmetricBenchmarkConfig,
    ThresholdAsymmetricBenchmarkModel,
    ZLBTrapBenchmarkConfig,
    ZLBTrapBenchmarkModel,
)


ROOT = Path(__file__).resolve().parents[2]
LINEAR_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_lq.json"
PPO_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_ppo_tuned.json"
SAC_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_sac.json"
TD3_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_td3.json"
TAYLOR_RULE_PATH = ROOT / "outputs" / "phase2" / "taylor_rule.json"

SEEDS = [43]
EVAL_EPISODES = 48
COMMON_SHOCK_HORIZON = 20
PPO_OVERRIDES = {
    "total_updates": 320,
    "rollout_steps": 1024,
    "train_epochs": 10,
    "eval_interval": 20,
}
OFFPOLICY_OVERRIDES = {
    "total_steps": 28000,
    "warmup_steps": 1500,
    "eval_interval": 4000,
}

NEW_ENV_SPECS_V2 = [
    {
        "env_id": "nonlinear_extreme_v2",
        "group": "nonlinear",
        "tier": "extreme_v2",
        "model_kind": "nonlinear",
        "config_path": str(ROOT / "src" / "monetary_rl" / "config" / "nonlinear_phillips_extreme.json"),
        "env_kwargs": {
            "horizon": 60,
            "action_low": -3.0,
            "action_high": 3.0,
            "initial_state_low": (-3.2, -3.2, -2.8),
            "initial_state_high": (3.2, 3.2, 2.8),
            "state_abs_limit": 45.0,
            "terminal_penalty": 220.0,
            "seed": 0,
        },
        "common_initial_state": [1.5, -1.8, 0.0],
        "dp_kwargs": {
            "state_low": (-6.5, -6.5, -3.0),
            "state_high": (6.5, 6.5, 3.0),
            "state_points": (17, 17, 17),
            "action_points": 41,
        },
    },
    {
        "env_id": "nonlinear_hyper",
        "group": "nonlinear",
        "tier": "hyper",
        "model_kind": "nonlinear",
        "config_path": str(ROOT / "src" / "monetary_rl" / "config" / "nonlinear_phillips_hyper.json"),
        "env_kwargs": {
            "horizon": 60,
            "action_low": -2.75,
            "action_high": 2.75,
            "initial_state_low": (-3.0, -3.0, -2.5),
            "initial_state_high": (3.0, 3.0, 2.5),
            "state_abs_limit": 45.0,
            "terminal_penalty": 220.0,
            "seed": 0,
        },
        "common_initial_state": [1.6, -2.0, 0.0],
        "dp_kwargs": {
            "state_low": (-6.5, -6.5, -2.75),
            "state_high": (6.5, 6.5, 2.75),
            "state_points": (17, 17, 17),
            "action_points": 39,
        },
    },
    {
        "env_id": "zlb_trap_very_strong",
        "group": "zlb",
        "tier": "very_strong_v2",
        "model_kind": "zlb_trap",
        "config_path": str(ROOT / "src" / "monetary_rl" / "config" / "zlb_trap_very_strong.json"),
        "env_kwargs": {
            "horizon": 60,
            "action_low": 0.0,
            "action_high": 5.0,
            "initial_state_low": (-4.0, -4.5, -0.75),
            "initial_state_high": (0.15, 0.15, 0.15),
            "state_abs_limit": 35.0,
            "terminal_penalty": 120.0,
            "seed": 0,
        },
        "common_initial_state": [-1.8, -2.4, -0.2],
        "dp_kwargs": {
            "state_low": (-6.0, -7.0, 0.0),
            "state_high": (1.5, 1.5, 5.0),
            "state_points": (17, 17, 17),
            "action_points": 41,
        },
    },
    {
        "env_id": "zlb_trap_extreme",
        "group": "zlb",
        "tier": "extreme_v2",
        "model_kind": "zlb_trap",
        "config_path": str(ROOT / "src" / "monetary_rl" / "config" / "zlb_trap_extreme.json"),
        "env_kwargs": {
            "horizon": 60,
            "action_low": 0.25,
            "action_high": 4.25,
            "initial_state_low": (-5.0, -5.5, -0.25),
            "initial_state_high": (0.05, 0.05, 0.05),
            "state_abs_limit": 40.0,
            "terminal_penalty": 140.0,
            "seed": 0,
        },
        "common_initial_state": [-2.1, -2.9, 0.0],
        "dp_kwargs": {
            "state_low": (-7.0, -8.0, 0.25),
            "state_high": (1.0, 1.0, 4.25),
            "state_points": (17, 17, 17),
            "action_points": 39,
        },
    },
    {
        "env_id": "asymmetric_threshold_very_strong",
        "group": "asymmetric",
        "tier": "very_strong_v2",
        "model_kind": "threshold_asymmetric",
        "config_path": str(ROOT / "src" / "monetary_rl" / "config" / "asymmetric_threshold_very_strong.json"),
        "env_kwargs": {
            "horizon": 60,
            "action_low": -6.0,
            "action_high": 6.0,
            "initial_state_low": (-3.0, -3.0, -2.25),
            "initial_state_high": (3.0, 3.0, 2.25),
            "state_abs_limit": 30.0,
            "terminal_penalty": 80.0,
            "seed": 0,
        },
        "common_initial_state": [1.1, -1.5, 0.0],
        "dp_kwargs": {
            "state_low": (-5.5, -5.5, -6.0),
            "state_high": (5.5, 5.5, 6.0),
            "state_points": (17, 17, 17),
            "action_points": 49,
        },
    },
    {
        "env_id": "asymmetric_threshold_extreme",
        "group": "asymmetric",
        "tier": "extreme_v2",
        "model_kind": "threshold_asymmetric",
        "config_path": str(ROOT / "src" / "monetary_rl" / "config" / "asymmetric_threshold_extreme.json"),
        "env_kwargs": {
            "horizon": 60,
            "action_low": -6.0,
            "action_high": 6.0,
            "initial_state_low": (-3.25, -3.25, -2.5),
            "initial_state_high": (3.25, 3.25, 2.5),
            "state_abs_limit": 30.0,
            "terminal_penalty": 90.0,
            "seed": 0,
        },
        "common_initial_state": [1.3, -1.8, 0.0],
        "dp_kwargs": {
            "state_low": (-6.0, -6.0, -6.0),
            "state_high": (6.0, 6.0, 6.0),
            "state_points": (17, 17, 17),
            "action_points": 49,
        },
    },
]

TIER_ORDER_V2 = ["mild", "medium", "strong", "very_strong_v2", "extreme_v2", "hyper"]


def make_model(spec: dict):
    if spec["model_kind"] == "linear":
        config = LQBenchmarkConfig.from_json(spec["config_path"])
        model = LQBenchmarkModel(config)
    elif spec["model_kind"] == "nonlinear":
        config = NonlinearBenchmarkConfig.from_json(spec["config_path"])
        model = NonlinearBenchmarkModel(config)
    elif spec["model_kind"] == "zlb_trap":
        config = ZLBTrapBenchmarkConfig.from_json(spec["config_path"])
        model = ZLBTrapBenchmarkModel(config)
    elif spec["model_kind"] == "threshold_asymmetric":
        config = ThresholdAsymmetricBenchmarkConfig.from_json(spec["config_path"])
        model = ThresholdAsymmetricBenchmarkModel(config)
    else:
        raise ValueError(f"Unknown model kind: {spec['model_kind']}")
    return config, model


def make_env(spec: dict):
    _, model = make_model(spec)
    env = LQBenchmarkEnv(model, BenchmarkEnvConfig(**spec["env_kwargs"]))
    return env
