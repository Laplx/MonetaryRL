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
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monetary_rl.agents.ppo import PPOConfig, PPOTrainer
from monetary_rl.evaluation import evaluate_policy
from monetary_rl.phase10_utils import build_linear_policy

from phase10_external_model_robustness import run_pyfrbus_fixed_point
from phase10_pyfrbus_native_utils import (
    PyfrbusEnvConfig,
    PyfrbusNativeEnv,
    evaluate_policy_fixed_point,
    fit_surrogate,
    revealed_loss_weights,
    save_json,
    summarize_results_md,
)


OUTPUT_DIR = ROOT / "outputs" / "phase12" / "pyfrbus_nonlinear_search"
SEED = 43
TRAIN_ENV_CONFIG = PyfrbusEnvConfig()
FINE05_PATH = ROOT / "outputs" / "phase10" / "pyfrbus_native" / "a_tuning" / "best_linear_rule.csv"
PYFRBUS_BASELINE = pd.read_csv(ROOT / "outputs" / "phase10" / "external_model_robustness" / "pyfrbus_summary.csv")
BASELINE_ARTIFICIAL = float(PYFRBUS_BASELINE.loc[PYFRBUS_BASELINE["policy_name"] == "pyfrbus_baseline", "total_discounted_loss"].iloc[0])
BASELINE_REVEALED = float(
    PYFRBUS_BASELINE.loc[PYFRBUS_BASELINE["policy_name"] == "pyfrbus_baseline", "total_discounted_revealed_loss"].iloc[0]
)
FINE05_REVEALED = float(pd.read_csv(FINE05_PATH).iloc[0]["total_discounted_revealed_loss"])
BASE_CONFIG = PPOConfig(
    total_updates=4,
    rollout_steps=64,
    gamma=float(json.loads((ROOT / "src" / "monetary_rl" / "config" / "benchmark_lq.json").read_text(encoding="utf-8"))["discount_factor"]),
    gae_lambda=0.95,
    clip_ratio=0.05,
    policy_lr=5e-6,
    value_lr=2e-4,
    train_epochs=4,
    minibatch_size=64,
    entropy_coef=0.0,
    value_coef=0.5,
    max_grad_norm=0.5,
    hidden_size=64,
    linear_policy=False,
    state_scale=(0.25, 0.5, 1.25),
    eval_episodes=4,
    eval_interval=999999,
    seed=SEED,
)
VARIANTS = [
    {"name": "residual_init_only", "clone_steps": 2400, "clone_lr": 7e-4, "jitter_scale": 0.20, "total_updates": 0, "policy_lr": 5e-6, "clip_ratio": 0.05},
    {"name": "residual_u2_lr5e6", "clone_steps": 2400, "clone_lr": 7e-4, "jitter_scale": 0.20, "total_updates": 2, "policy_lr": 5e-6, "clip_ratio": 0.05},
    {"name": "residual_u4_lr5e6", "clone_steps": 2400, "clone_lr": 7e-4, "jitter_scale": 0.20, "total_updates": 4, "policy_lr": 5e-6, "clip_ratio": 0.05},
    {"name": "residual_u4_lr1e5", "clone_steps": 2600, "clone_lr": 7e-4, "jitter_scale": 0.18, "total_updates": 4, "policy_lr": 1e-5, "clip_ratio": 0.05},
    {"name": "residual_u6_lr5e6", "clone_steps": 2600, "clone_lr": 6e-4, "jitter_scale": 0.15, "total_updates": 6, "policy_lr": 5e-6, "clip_ratio": 0.04},
    {"name": "residual_u8_lr5e6", "clone_steps": 3000, "clone_lr": 6e-4, "jitter_scale": 0.12, "total_updates": 8, "policy_lr": 5e-6, "clip_ratio": 0.04},
]


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


class ResidualPolicyValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.linear_head = nn.Linear(state_dim, 1)
        self.residual_head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.value_head = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.tensor([-4.5], dtype=torch.float32))

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.body(states)
        mean = self.linear_head(states) + self.residual_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean.squeeze(-1), std.squeeze(-1), value


def build_eval_env() -> PyfrbusNativeEnv:
    return PyfrbusNativeEnv(TRAIN_ENV_CONFIG, revealed_loss_weights())


def build_train_env() -> SafePyfrbusNativeEnv:
    return SafePyfrbusNativeEnv(TRAIN_ENV_CONFIG, revealed_loss_weights())


def load_fine05_policy():
    row = pd.read_csv(FINE05_PATH).iloc[0]
    return build_linear_policy(
        "fine_05",
        float(row["intercept"]),
        float(row["inflation_coeff"]),
        float(row["output_coeff"]),
        float(row["lagged_rate_coeff"]),
    )


def focused_states(policy_fn, jitter_scale: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    window, _ = run_pyfrbus_fixed_point("fine05_focus", policy_fn)
    path = window.copy().reset_index(drop=True)
    rates = path["rff"].to_numpy(dtype=float)
    lagged = np.concatenate([[2.0], rates[:-1]]) - 2.0
    base_states = np.column_stack(
        [
            path["picxfe"].to_numpy(dtype=float) - path["pitarg"].to_numpy(dtype=float),
            path["xgap"].to_numpy(dtype=float),
            lagged,
        ]
    ).astype(np.float32)
    rng = np.random.default_rng(seed)
    state_std = np.maximum(base_states.std(axis=0), np.array([1e-3, 1e-3, 1e-3], dtype=np.float32))
    jitter = rng.normal(0.0, state_std * jitter_scale, size=(len(base_states) * 128, 3)).astype(np.float32)
    tiled = np.repeat(base_states, 128, axis=0)
    states = np.vstack([base_states, tiled + jitter]).astype(np.float32)
    targets = np.asarray([float(policy_fn(state, 0)) for state in states], dtype=np.float32)
    return states, targets


def initialize_residual_model(trainer: PPOTrainer, states: np.ndarray, targets: np.ndarray) -> None:
    trainer.model = ResidualPolicyValueNet(3, trainer.config.hidden_size).to(trainer.device)
    trainer.optimizer = torch.optim.Adam(
        [
            {"params": list(trainer.model.body.parameters()) + list(trainer.model.linear_head.parameters()) + list(trainer.model.residual_head.parameters()) + [trainer.model.log_std], "lr": trainer.config.policy_lr},
            {"params": trainer.model.value_head.parameters(), "lr": trainer.config.value_lr},
        ]
    )
    action_scale = float(trainer.action_scale.item())
    action_bias = float(trainer.action_bias.item())
    clipped = np.clip(targets, float(trainer.env.config.action_low) + 1e-5, float(trainer.env.config.action_high) - 1e-5)
    normalized_target = np.clip((clipped - action_bias) / action_scale, -0.999, 0.999)
    pre_tanh = np.arctanh(normalized_target)
    states_norm = states / np.asarray(trainer.config.state_scale, dtype=np.float32)
    X = np.column_stack([np.ones(len(states_norm), dtype=np.float32), states_norm]).astype(np.float32)
    beta, *_ = np.linalg.lstsq(X, pre_tanh.astype(np.float32), rcond=None)
    with torch.no_grad():
        trainer.model.linear_head.bias.fill_(float(beta[0]))
        trainer.model.linear_head.weight.copy_(torch.as_tensor(beta[1:].reshape(1, -1), dtype=torch.float32, device=trainer.device))
        trainer.model.log_std.fill_(-4.5)


def deterministic_actions(trainer: PPOTrainer, states: torch.Tensor) -> torch.Tensor:
    normalized = states.to(trainer.device) / trainer.state_scale
    mean, _, _ = trainer.model(normalized)
    return torch.tanh(mean) * trainer.action_scale + trainer.action_bias


def clone_to_targets(trainer: PPOTrainer, states: np.ndarray, targets: np.ndarray, steps: int, lr: float, seed: int) -> dict[str, float]:
    states_t = torch.as_tensor(states, dtype=torch.float32, device=trainer.device)
    targets_t = torch.as_tensor(targets, dtype=torch.float32, device=trainer.device)
    optimizer = torch.optim.Adam(
        list(trainer.model.body.parameters()) + list(trainer.model.linear_head.parameters()) + list(trainer.model.residual_head.parameters()),
        lr=lr,
    )
    rng = np.random.default_rng(seed)
    losses = []
    batch_size = min(512, len(states))
    for _ in range(steps):
        idx = rng.integers(0, len(states), size=batch_size)
        pred = deterministic_actions(trainer, states_t[idx])
        loss = F.mse_loss(pred, targets_t[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    with torch.no_grad():
        rmse = torch.sqrt(torch.mean((deterministic_actions(trainer, states_t) - targets_t) ** 2)).item()
    return {
        "clone_rmse": float(rmse),
        "clone_loss_mean": float(np.mean(losses)) if losses else 0.0,
        "clone_loss_last": float(losses[-1]) if losses else 0.0,
    }


def build_policy_fn(trainer: PPOTrainer):
    def policy(state: np.ndarray, t: int) -> float:
        del t
        return trainer._deterministic_action(np.asarray(state, dtype=float))

    return policy


def fixed_point_metrics(name: str, policy_fn) -> dict[str, Any]:
    result = evaluate_policy_fixed_point(name, policy_fn)
    result["artificial_improvement_vs_baseline_pct"] = (BASELINE_ARTIFICIAL - float(result["total_discounted_loss"])) / BASELINE_ARTIFICIAL * 100.0
    result["revealed_improvement_vs_baseline_pct"] = (BASELINE_REVEALED - float(result["total_discounted_revealed_loss"])) / BASELINE_REVEALED * 100.0
    result["beats_fine05"] = float(result["total_discounted_revealed_loss"]) < FINE05_REVEALED
    result["gap_vs_fine05_bp"] = (FINE05_REVEALED - float(result["total_discounted_revealed_loss"])) * 10000.0
    return result


def train_variant(spec: dict[str, Any], fine_policy_fn) -> dict[str, Any]:
    run_dir = OUTPUT_DIR / spec["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "result_row.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    config = PPOConfig(**{**asdict(BASE_CONFIG), "total_updates": spec["total_updates"], "policy_lr": spec["policy_lr"], "clip_ratio": spec["clip_ratio"], "seed": SEED})
    trainer = PPOTrainer(build_train_env(), config)
    states, targets = focused_states(fine_policy_fn, jitter_scale=spec["jitter_scale"], seed=SEED)
    initialize_residual_model(trainer, states, targets)
    clone_stats = clone_to_targets(trainer, states, targets, steps=spec["clone_steps"], lr=spec["clone_lr"], seed=SEED)

    init_policy = build_policy_fn(trainer)
    init_fixed = fixed_point_metrics(f"{spec['name']}_init", init_policy)
    init_surrogate = fit_surrogate(f"{spec['name']}_init", init_policy)
    init_env = evaluate_policy(build_eval_env(), init_policy, episodes=4, gamma=trainer.env.model.config.discount_factor, seed=40_000 + SEED)
    torch.save(trainer.model.state_dict(), run_dir / "init.pt")
    save_json(run_dir / "init_config.json", asdict(config))
    pd.DataFrame([init_fixed]).to_csv(run_dir / "init_fixed_point.csv", index=False)
    pd.DataFrame([init_surrogate]).to_csv(run_dir / "init_surrogate.csv", index=False)
    pd.DataFrame(init_env["first_trajectory"]).to_csv(run_dir / "init_first_trajectory.csv", index=False)

    if spec["total_updates"] > 0:
        result = trainer.train()
        pd.DataFrame(result["training_log"]).to_csv(run_dir / "training_log.csv", index=False)
        torch.save(result["policy_state_dict"], run_dir / "trained.pt")
        save_json(run_dir / "trained_config.json", result["config"])
    else:
        result = None
        torch.save(trainer.model.state_dict(), run_dir / "trained.pt")
        save_json(run_dir / "trained_config.json", asdict(config))

    trained_policy = build_policy_fn(trainer)
    trained_fixed = fixed_point_metrics(spec["name"], trained_policy)
    trained_surrogate = fit_surrogate(spec["name"], trained_policy)
    trained_env = evaluate_policy(build_eval_env(), trained_policy, episodes=4, gamma=trainer.env.model.config.discount_factor, seed=41_000 + SEED)
    pd.DataFrame([trained_fixed]).to_csv(run_dir / "trained_fixed_point.csv", index=False)
    pd.DataFrame([trained_surrogate]).to_csv(run_dir / "trained_surrogate.csv", index=False)
    pd.DataFrame(trained_env["first_trajectory"]).to_csv(run_dir / "trained_first_trajectory.csv", index=False)

    row = {
        "variant": spec["name"],
        **spec,
        **clone_stats,
        "init_artificial_loss": init_fixed["total_discounted_loss"],
        "init_revealed_loss": init_fixed["total_discounted_revealed_loss"],
        "init_beats_fine05": init_fixed["beats_fine05"],
        "trained_artificial_loss": trained_fixed["total_discounted_loss"],
        "trained_revealed_loss": trained_fixed["total_discounted_revealed_loss"],
        "trained_beats_fine05": trained_fixed["beats_fine05"],
        "trained_gap_vs_fine05_bp": trained_fixed["gap_vs_fine05_bp"],
        "trained_revealed_improvement_vs_baseline_pct": trained_fixed["revealed_improvement_vs_baseline_pct"],
        "trained_surrogate_rmse": trained_surrogate["fit_rmse"],
    }
    save_json(cache_path, row)
    return row


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fine_policy_fn = load_fine05_policy()
    rows = [train_variant(spec, fine_policy_fn) for spec in VARIANTS]
    summary = pd.DataFrame(rows).sort_values("trained_revealed_loss").reset_index(drop=True)
    summary.to_csv(OUTPUT_DIR / "search_summary.csv", index=False)

    reference = pd.DataFrame(
        [
            {"variant": "fine_05", "trained_artificial_loss": pd.read_csv(FINE05_PATH).iloc[0]["total_discounted_loss"], "trained_revealed_loss": FINE05_REVEALED, "trained_revealed_improvement_vs_baseline_pct": (BASELINE_REVEALED - FINE05_REVEALED) / BASELINE_REVEALED * 100.0, "trained_beats_fine05": False, "trained_gap_vs_fine05_bp": 0.0},
            {"variant": "phase12_prev_best", "trained_artificial_loss": 0.012773957729508432, "trained_revealed_loss": 0.01605471350463282, "trained_revealed_improvement_vs_baseline_pct": 13.486763563889937, "trained_beats_fine05": False, "trained_gap_vs_fine05_bp": (FINE05_REVEALED - 0.01605471350463282) * 10000.0},
        ]
    )
    comparison = pd.concat([reference, summary[["variant", "trained_artificial_loss", "trained_revealed_loss", "trained_revealed_improvement_vs_baseline_pct", "trained_beats_fine05", "trained_gap_vs_fine05_bp"]]], ignore_index=True)
    comparison = comparison.sort_values("trained_revealed_loss").reset_index(drop=True)
    comparison.to_csv(OUTPUT_DIR / "comparison.csv", index=False)

    summarize_results_md(
        OUTPUT_DIR / "summary.md",
        "Phase 12 PyFRBUS Nonlinear PPO Search",
        [
            ("Comparison", comparison.round(6)),
            ("Variant Summary", summary.round(6)),
        ],
    )


if __name__ == "__main__":
    main()
