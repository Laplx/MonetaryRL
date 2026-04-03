from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monetary_rl.agents.ppo import PPOConfig, PPOTrainer
from monetary_rl.evaluation import evaluate_policy
from monetary_rl.phase10_utils import build_linear_policy

from phase10_pyfrbus_native_utils import (
    PyfrbusEnvConfig,
    PyfrbusNativeEnv,
    baseline_observation_bounds,
    evaluate_policy_fixed_point,
    fit_surrogate,
    revealed_loss_weights,
    save_json,
    summarize_results_md,
)


OUTPUT_DIR = ROOT / "outputs" / "phase12" / "pyfrbus_warmstart_ppo"
LINEAR_DIR = OUTPUT_DIR / "linear_stage"
NONLINEAR_DIR = OUTPUT_DIR / "nonlinear_stage"
SEEDS_LINEAR = [7, 43, 99]
SEEDS_NONLINEAR = [43]
TRAIN_ENV_CONFIG = PyfrbusEnvConfig()
PYFRBUS_BASELINE = pd.read_csv(ROOT / "outputs" / "phase10" / "external_model_robustness" / "pyfrbus_summary.csv")
BASELINE_ARTIFICIAL = float(PYFRBUS_BASELINE.loc[PYFRBUS_BASELINE["policy_name"] == "pyfrbus_baseline", "total_discounted_loss"].iloc[0])
BASELINE_REVEALED = float(
    PYFRBUS_BASELINE.loc[PYFRBUS_BASELINE["policy_name"] == "pyfrbus_baseline", "total_discounted_revealed_loss"].iloc[0]
)
FINE05_PATH = ROOT / "outputs" / "phase10" / "pyfrbus_native" / "a_tuning" / "best_linear_rule.csv"


def compute_state_scale() -> tuple[float, float, float]:
    low, high = baseline_observation_bounds()
    scale = [max(abs(lo), abs(hi), 0.05) for lo, hi in zip(low, high)]
    return tuple(float(x) for x in scale)


STATE_SCALE = compute_state_scale()
LINEAR_SETTINGS = PPOConfig(
    total_updates=12,
    rollout_steps=64,
    gamma=float(json.loads((ROOT / "src" / "monetary_rl" / "config" / "benchmark_lq.json").read_text(encoding="utf-8"))["discount_factor"]),
    gae_lambda=0.95,
    clip_ratio=0.10,
    policy_lr=5e-5,
    value_lr=2e-4,
    train_epochs=4,
    minibatch_size=64,
    entropy_coef=0.0,
    value_coef=0.5,
    max_grad_norm=0.5,
    hidden_size=64,
    linear_policy=True,
    state_scale=STATE_SCALE,
    eval_episodes=4,
    eval_interval=999999,
)
NONLINEAR_SETTINGS = PPOConfig(
    total_updates=10,
    rollout_steps=64,
    gamma=LINEAR_SETTINGS.gamma,
    gae_lambda=0.95,
    clip_ratio=0.10,
    policy_lr=3e-5,
    value_lr=2e-4,
    train_epochs=4,
    minibatch_size=64,
    entropy_coef=0.0,
    value_coef=0.5,
    max_grad_norm=0.5,
    hidden_size=64,
    linear_policy=False,
    state_scale=STATE_SCALE,
    eval_episodes=4,
    eval_interval=999999,
)


def load_fine05_row() -> pd.Series:
    return pd.read_csv(FINE05_PATH).iloc[0]


def build_fine05_policy():
    row = load_fine05_row()
    return build_linear_policy(
        "fine_05",
        float(row["intercept"]),
        float(row["inflation_coeff"]),
        float(row["output_coeff"]),
        float(row["lagged_rate_coeff"]),
    )


def build_env(config: PyfrbusEnvConfig | None = None) -> PyfrbusNativeEnv:
    return PyfrbusNativeEnv(config or TRAIN_ENV_CONFIG, revealed_loss_weights())


class SafePyfrbusNativeEnv(PyfrbusNativeEnv):
    def step(self, action: float):
        if self.t >= len(self.periods):
            clipped_action = float(np.clip(action, self.config.action_low, self.config.action_high))
            return (
                np.zeros(3, dtype=float),
                0.0,
                True,
                {
                    "loss": 0.0,
                    "raw_action": float(action),
                    "action": clipped_action,
                    "exploded": False,
                    "policy_rate": self.model.action_to_level(clipped_action),
                    "inflation_gap": np.nan,
                    "output_gap": np.nan,
                    "rate_change": np.nan,
                },
            )
        return super().step(action)


def build_train_env(config: PyfrbusEnvConfig | None = None) -> SafePyfrbusNativeEnv:
    return SafePyfrbusNativeEnv(config or TRAIN_ENV_CONFIG, revealed_loss_weights())


def build_ppo_policy_from_checkpoint(checkpoint_path: Path, config_payload: dict[str, Any], env: PyfrbusNativeEnv):
    trainer = PPOTrainer(env, PPOConfig(**config_payload))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    trainer.model.load_state_dict(checkpoint)

    def policy(state: np.ndarray, t: int) -> float:
        del t
        return trainer._deterministic_action(np.asarray(state, dtype=float))

    return policy


def clipped_action_targets(policy_fn, states: np.ndarray, env: PyfrbusNativeEnv) -> np.ndarray:
    actions = np.asarray([float(policy_fn(state, 0)) for state in states], dtype=np.float32)
    return np.clip(actions, env.config.action_low + 1e-4, env.config.action_high - 1e-4)


def deterministic_actions_from_trainer(trainer: PPOTrainer, states: torch.Tensor) -> torch.Tensor:
    states = states.to(trainer.device)
    normalized = states / trainer.state_scale
    mean, _, _ = trainer.model(normalized)
    return torch.tanh(mean) * trainer.action_scale + trainer.action_bias


def sample_states(num_samples: int, seed: int) -> np.ndarray:
    low, high = baseline_observation_bounds()
    rng = np.random.default_rng(seed)
    return rng.uniform(low=np.asarray(low, dtype=np.float32), high=np.asarray(high, dtype=np.float32), size=(num_samples, 3)).astype(np.float32)


def clone_policy_actions(
    trainer: PPOTrainer,
    target_policy_fn,
    *,
    num_samples: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    train_body: bool,
) -> dict[str, float]:
    states_np = sample_states(num_samples, seed)
    targets_np = clipped_action_targets(target_policy_fn, states_np, trainer.env)
    states = torch.as_tensor(states_np, dtype=torch.float32, device=trainer.device)
    targets = torch.as_tensor(targets_np, dtype=torch.float32, device=trainer.device)

    if train_body:
        params = list(trainer.model.body.parameters()) + list(trainer.model.policy_head.parameters())
    else:
        params = list(trainer.model.policy_head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng(seed + 10_000)
    losses: list[float] = []

    for _ in range(steps):
        idx = rng.integers(0, len(states_np), size=batch_size)
        batch_states = states[idx]
        batch_targets = targets[idx]
        pred_actions = deterministic_actions_from_trainer(trainer, batch_states)
        loss = F.mse_loss(pred_actions, batch_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    with torch.no_grad():
        pred_actions = deterministic_actions_from_trainer(trainer, states)
        rmse = torch.sqrt(torch.mean((pred_actions - targets) ** 2)).item()
    trainer.model.log_std.data.fill_(-4.5)
    return {
        "clone_rmse": float(rmse),
        "clone_loss_last": float(losses[-1]) if losses else 0.0,
        "clone_loss_mean": float(np.mean(losses)) if losses else 0.0,
    }


def fixed_point_with_improvements(policy_name: str, policy_fn) -> dict[str, Any]:
    summary = evaluate_policy_fixed_point(policy_name, policy_fn)
    summary["artificial_improvement_vs_baseline_pct"] = (BASELINE_ARTIFICIAL - float(summary["total_discounted_loss"])) / BASELINE_ARTIFICIAL * 100.0
    summary["revealed_improvement_vs_baseline_pct"] = (BASELINE_REVEALED - float(summary["total_discounted_revealed_loss"])) / BASELINE_REVEALED * 100.0
    return summary


def env_eval(policy_fn, seed: int) -> dict[str, Any]:
    env = build_env()
    return evaluate_policy(env, policy_fn, episodes=4, gamma=env.model.config.discount_factor, seed=seed)


def save_policy_artifacts(run_dir: Path, prefix: str, trainer: PPOTrainer, config: PPOConfig, fixed_point: dict[str, Any], surrogate: dict[str, Any], env_stats: dict[str, Any]) -> None:
    torch.save(trainer.model.state_dict(), run_dir / f"{prefix}.pt")
    save_json(run_dir / f"{prefix}_config.json", asdict(config))
    pd.DataFrame([fixed_point]).to_csv(run_dir / f"{prefix}_fixed_point.csv", index=False)
    pd.DataFrame([surrogate]).to_csv(run_dir / f"{prefix}_surrogate.csv", index=False)
    pd.DataFrame(env_stats["first_trajectory"]).to_csv(run_dir / f"{prefix}_first_trajectory.csv", index=False)


def train_linear_seed(seed: int, fine_policy_fn) -> dict[str, Any]:
    run_dir = LINEAR_DIR / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "result_row.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    config = PPOConfig(**{**asdict(LINEAR_SETTINGS), "seed": seed})
    trainer = PPOTrainer(build_train_env(), config)
    clone_stats = clone_policy_actions(
        trainer,
        fine_policy_fn,
        num_samples=4096,
        steps=1200,
        batch_size=256,
        lr=5e-3,
        seed=seed,
        train_body=False,
    )

    init_policy_fn = lambda state, t: trainer._deterministic_action(np.asarray(state, dtype=float))
    init_fixed = fixed_point_with_improvements(f"ppo_pyfrbus_linear_warmstart_init_seed_{seed}", init_policy_fn)
    init_env_stats = env_eval(init_policy_fn, 20_000 + seed)
    init_surrogate = fit_surrogate(f"ppo_pyfrbus_linear_warmstart_init_seed_{seed}", init_policy_fn)
    save_policy_artifacts(run_dir, "ppo_pyfrbus_linear_warmstart_init", trainer, config, init_fixed, init_surrogate, init_env_stats)

    result = trainer.train()
    pd.DataFrame(result["training_log"]).to_csv(run_dir / "training_log.csv", index=False)
    torch.save(result["policy_state_dict"], run_dir / "ppo_pyfrbus_linear_warmstart.pt")
    save_json(run_dir / "ppo_pyfrbus_linear_warmstart_config.json", result["config"])

    trained_policy_fn = lambda state, t: trainer._deterministic_action(np.asarray(state, dtype=float))
    trained_fixed = fixed_point_with_improvements(f"ppo_pyfrbus_linear_warmstart_seed_{seed}", trained_policy_fn)
    trained_env_stats = env_eval(trained_policy_fn, 21_000 + seed)
    trained_surrogate = fit_surrogate(f"ppo_pyfrbus_linear_warmstart_seed_{seed}", trained_policy_fn)
    pd.DataFrame([trained_fixed]).to_csv(run_dir / "ppo_pyfrbus_linear_warmstart_fixed_point.csv", index=False)
    pd.DataFrame([trained_surrogate]).to_csv(run_dir / "ppo_pyfrbus_linear_warmstart_surrogate.csv", index=False)
    pd.DataFrame(trained_env_stats["first_trajectory"]).to_csv(run_dir / "ppo_pyfrbus_linear_warmstart_first_trajectory.csv", index=False)

    row = {
        "seed": seed,
        "stage": "linear",
        **clone_stats,
        "init_checkpoint_path": str(run_dir / "ppo_pyfrbus_linear_warmstart_init.pt"),
        "init_config_path": str(run_dir / "ppo_pyfrbus_linear_warmstart_init_config.json"),
        "init_fixed_point_artificial_loss": init_fixed["total_discounted_loss"],
        "init_fixed_point_revealed_loss": init_fixed["total_discounted_revealed_loss"],
        "init_revealed_improvement_vs_baseline_pct": init_fixed["revealed_improvement_vs_baseline_pct"],
        "trained_checkpoint_path": str(run_dir / "ppo_pyfrbus_linear_warmstart.pt"),
        "trained_config_path": str(run_dir / "ppo_pyfrbus_linear_warmstart_config.json"),
        "trained_env_discounted_loss": trained_env_stats["mean_discounted_loss"],
        "trained_fixed_point_artificial_loss": trained_fixed["total_discounted_loss"],
        "trained_fixed_point_revealed_loss": trained_fixed["total_discounted_revealed_loss"],
        "trained_revealed_improvement_vs_baseline_pct": trained_fixed["revealed_improvement_vs_baseline_pct"],
        "trained_surrogate_rmse": trained_surrogate["fit_rmse"],
    }
    save_json(cache_path, row)
    return row


def choose_best_linear_variant(linear_df: pd.DataFrame) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for row in linear_df.to_dict("records"):
        candidates.append(
            {
                "seed": int(row["seed"]),
                "variant": "init",
                "checkpoint_path": row["init_checkpoint_path"],
                "config_path": row["init_config_path"],
                "fixed_point_revealed_loss": float(row["init_fixed_point_revealed_loss"]),
            }
        )
        candidates.append(
            {
                "seed": int(row["seed"]),
                "variant": "trained",
                "checkpoint_path": row["trained_checkpoint_path"],
                "config_path": row["trained_config_path"],
                "fixed_point_revealed_loss": float(row["trained_fixed_point_revealed_loss"]),
            }
        )
    return sorted(candidates, key=lambda item: item["fixed_point_revealed_loss"])[0]


def train_nonlinear_seed(seed: int, target_checkpoint: Path, target_config: dict[str, Any]) -> dict[str, Any]:
    run_dir = NONLINEAR_DIR / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "result_row.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    target_policy_fn = build_ppo_policy_from_checkpoint(target_checkpoint, target_config, build_env())
    config = PPOConfig(**{**asdict(NONLINEAR_SETTINGS), "seed": seed})
    trainer = PPOTrainer(build_train_env(), config)
    clone_stats = clone_policy_actions(
        trainer,
        target_policy_fn,
        num_samples=6144,
        steps=1600,
        batch_size=256,
        lr=1e-3,
        seed=seed,
        train_body=True,
    )

    init_policy_fn = lambda state, t: trainer._deterministic_action(np.asarray(state, dtype=float))
    init_fixed = fixed_point_with_improvements(f"ppo_pyfrbus_nonlinear_warmstart_init_seed_{seed}", init_policy_fn)
    init_env_stats = env_eval(init_policy_fn, 30_000 + seed)
    init_surrogate = fit_surrogate(f"ppo_pyfrbus_nonlinear_warmstart_init_seed_{seed}", init_policy_fn)
    save_policy_artifacts(run_dir, "ppo_pyfrbus_nonlinear_warmstart_init", trainer, config, init_fixed, init_surrogate, init_env_stats)

    result = trainer.train()
    pd.DataFrame(result["training_log"]).to_csv(run_dir / "training_log.csv", index=False)
    torch.save(result["policy_state_dict"], run_dir / "ppo_pyfrbus_nonlinear_warmstart.pt")
    save_json(run_dir / "ppo_pyfrbus_nonlinear_warmstart_config.json", result["config"])

    trained_policy_fn = lambda state, t: trainer._deterministic_action(np.asarray(state, dtype=float))
    trained_fixed = fixed_point_with_improvements(f"ppo_pyfrbus_nonlinear_warmstart_seed_{seed}", trained_policy_fn)
    trained_env_stats = env_eval(trained_policy_fn, 31_000 + seed)
    trained_surrogate = fit_surrogate(f"ppo_pyfrbus_nonlinear_warmstart_seed_{seed}", trained_policy_fn)
    pd.DataFrame([trained_fixed]).to_csv(run_dir / "ppo_pyfrbus_nonlinear_warmstart_fixed_point.csv", index=False)
    pd.DataFrame([trained_surrogate]).to_csv(run_dir / "ppo_pyfrbus_nonlinear_warmstart_surrogate.csv", index=False)
    pd.DataFrame(trained_env_stats["first_trajectory"]).to_csv(run_dir / "ppo_pyfrbus_nonlinear_warmstart_first_trajectory.csv", index=False)

    row = {
        "seed": seed,
        "stage": "nonlinear",
        **clone_stats,
        "target_checkpoint_path": str(target_checkpoint),
        "target_variant": target_config["linear_policy"] if "linear_policy" in target_config else True,
        "init_checkpoint_path": str(run_dir / "ppo_pyfrbus_nonlinear_warmstart_init.pt"),
        "init_config_path": str(run_dir / "ppo_pyfrbus_nonlinear_warmstart_init_config.json"),
        "init_fixed_point_artificial_loss": init_fixed["total_discounted_loss"],
        "init_fixed_point_revealed_loss": init_fixed["total_discounted_revealed_loss"],
        "init_revealed_improvement_vs_baseline_pct": init_fixed["revealed_improvement_vs_baseline_pct"],
        "trained_checkpoint_path": str(run_dir / "ppo_pyfrbus_nonlinear_warmstart.pt"),
        "trained_config_path": str(run_dir / "ppo_pyfrbus_nonlinear_warmstart_config.json"),
        "trained_env_discounted_loss": trained_env_stats["mean_discounted_loss"],
        "trained_fixed_point_artificial_loss": trained_fixed["total_discounted_loss"],
        "trained_fixed_point_revealed_loss": trained_fixed["total_discounted_revealed_loss"],
        "trained_revealed_improvement_vs_baseline_pct": trained_fixed["revealed_improvement_vs_baseline_pct"],
        "trained_surrogate_rmse": trained_surrogate["fit_rmse"],
    }
    save_json(cache_path, row)
    return row


def reference_rows(fine_row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant": "pyfrbus_baseline",
                "group": "reference",
                "artificial_loss": BASELINE_ARTIFICIAL,
                "revealed_loss": BASELINE_REVEALED,
                "revealed_improvement_vs_baseline_pct": 0.0,
            },
            {
                "variant": "fine_05",
                "group": "reference",
                "artificial_loss": float(fine_row["total_discounted_loss"]),
                "revealed_loss": float(fine_row["total_discounted_revealed_loss"]),
                "revealed_improvement_vs_baseline_pct": (BASELINE_REVEALED - float(fine_row["total_discounted_revealed_loss"])) / BASELINE_REVEALED * 100.0,
            },
        ]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LINEAR_DIR.mkdir(parents=True, exist_ok=True)
    NONLINEAR_DIR.mkdir(parents=True, exist_ok=True)

    fine_row = load_fine05_row()
    fine_policy_fn = build_fine05_policy()

    linear_rows = [train_linear_seed(seed, fine_policy_fn) for seed in SEEDS_LINEAR]
    linear_df = pd.DataFrame(linear_rows).sort_values("trained_fixed_point_revealed_loss").reset_index(drop=True)
    linear_df.to_csv(OUTPUT_DIR / "linear_seed_summary.csv", index=False)

    best_linear = choose_best_linear_variant(linear_df)
    best_linear_config = json.loads(Path(best_linear["config_path"]).read_text(encoding="utf-8"))
    nonlinear_rows = [
        train_nonlinear_seed(seed, Path(best_linear["checkpoint_path"]), best_linear_config) for seed in SEEDS_NONLINEAR
    ]
    nonlinear_df = pd.DataFrame(nonlinear_rows).sort_values("trained_fixed_point_revealed_loss").reset_index(drop=True)
    nonlinear_df.to_csv(OUTPUT_DIR / "nonlinear_seed_summary.csv", index=False)

    comparison_rows = []
    comparison_rows.extend(reference_rows(fine_row).to_dict("records"))

    best_linear_row = linear_df.iloc[0]
    comparison_rows.append(
        {
            "variant": "ppo_linear_warmstart_best",
            "group": "phase12_linear",
            "artificial_loss": float(best_linear_row["trained_fixed_point_artificial_loss"]),
            "revealed_loss": float(best_linear_row["trained_fixed_point_revealed_loss"]),
            "revealed_improvement_vs_baseline_pct": float(best_linear_row["trained_revealed_improvement_vs_baseline_pct"]),
        }
    )
    comparison_rows.append(
        {
            "variant": f"ppo_linear_best_{best_linear['variant']}",
            "group": "phase12_linear_best_variant",
            "artificial_loss": float(best_linear_row["init_fixed_point_artificial_loss"] if best_linear["variant"] == "init" else best_linear_row["trained_fixed_point_artificial_loss"]),
            "revealed_loss": float(best_linear["fixed_point_revealed_loss"]),
            "revealed_improvement_vs_baseline_pct": (BASELINE_REVEALED - float(best_linear["fixed_point_revealed_loss"])) / BASELINE_REVEALED * 100.0,
        }
    )
    if not nonlinear_df.empty:
        best_nonlinear_row = nonlinear_df.iloc[0]
        comparison_rows.append(
            {
                "variant": "ppo_nonlinear_warmstart_best",
                "group": "phase12_nonlinear",
                "artificial_loss": float(best_nonlinear_row["trained_fixed_point_artificial_loss"]),
                "revealed_loss": float(best_nonlinear_row["trained_fixed_point_revealed_loss"]),
                "revealed_improvement_vs_baseline_pct": float(best_nonlinear_row["trained_revealed_improvement_vs_baseline_pct"]),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values("revealed_loss").reset_index(drop=True)
    comparison_df.to_csv(OUTPUT_DIR / "comparison.csv", index=False)

    meta_df = pd.DataFrame(
        [
            {"key": "state_scale", "value": list(STATE_SCALE)},
            {"key": "linear_settings", "value": asdict(LINEAR_SETTINGS)},
            {"key": "nonlinear_settings", "value": asdict(NONLINEAR_SETTINGS)},
            {"key": "best_linear_variant", "value": best_linear},
        ]
    )
    meta_df.to_json(OUTPUT_DIR / "meta.json", orient="records", indent=2, force_ascii=False)

    summarize_results_md(
        OUTPUT_DIR / "summary.md",
        "Phase 12 PyFRBUS Warm-start PPO",
        [
            ("Comparison", comparison_df.round(6)),
            ("Linear Seeds", linear_df.round(6)),
            ("Nonlinear Seeds", nonlinear_df.round(6)),
        ],
    )


if __name__ == "__main__":
    main()
