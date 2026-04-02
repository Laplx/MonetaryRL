from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monetary_rl.agents.sac import SACConfig, SACTrainer
from monetary_rl.evaluation import evaluate_policy

from phase10_pyfrbus_native_utils import (
    PyfrbusEnvConfig,
    PyfrbusNativeEnv,
    build_sac_policy_from_checkpoint,
    evaluate_policy_fixed_point,
    fit_surrogate,
    revealed_loss_weights,
    save_json,
    summarize_results_md,
)


OUTPUT_DIR = ROOT / "outputs" / "phase10" / "pyfrbus_native" / "b_native_revealed_training"
SEEDS = [7, 43, 99]
ENV_CONFIG = PyfrbusEnvConfig()
PYFRBUS_BASELINE = pd.read_csv(ROOT / "outputs" / "phase10" / "external_model_robustness" / "pyfrbus_summary.csv")
BASELINE_ARTIFICIAL = float(PYFRBUS_BASELINE.loc[PYFRBUS_BASELINE["policy_name"] == "pyfrbus_baseline", "total_discounted_loss"].iloc[0])
BASELINE_REVEALED = float(
    PYFRBUS_BASELINE.loc[PYFRBUS_BASELINE["policy_name"] == "pyfrbus_baseline", "total_discounted_revealed_loss"].iloc[0]
)
SAC_SETTINGS = SACConfig(
    total_steps=1800,
    warmup_steps=200,
    batch_size=128,
    gamma=float(json.loads((ROOT / "src" / "monetary_rl" / "config" / "benchmark_lq.json").read_text(encoding="utf-8"))["discount_factor"]),
    tau=0.01,
    actor_lr=2e-4,
    critic_lr=4e-4,
    alpha=0.05,
    hidden_size=64,
    replay_capacity=20000,
    eval_episodes=6,
    eval_interval=999999,
    state_scale=(0.25, 0.50, 1.25),
)


def train_one(seed: int) -> dict[str, object]:
    run_dir = OUTPUT_DIR / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "result_row.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    fixed_path = run_dir / "fixed_point_summary.csv"
    surrogate_path = run_dir / "linear_surrogate.csv"
    checkpoint_path = run_dir / "sac_pyfrbus_revealed_native.pt"
    if fixed_path.exists() and surrogate_path.exists() and checkpoint_path.exists():
        fixed_point = pd.read_csv(fixed_path).iloc[0].to_dict()
        surrogate = pd.read_csv(surrogate_path).iloc[0].to_dict()
        row = {
            "seed": seed,
            "checkpoint_path": str(checkpoint_path),
            "mean_discounted_loss_env": np.nan,
            "mean_reward_env": np.nan,
            "clip_rate_env": np.nan,
            "explosion_rate_env": np.nan,
            "fixed_point_artificial_loss": fixed_point["total_discounted_loss"],
            "fixed_point_revealed_loss": fixed_point["total_discounted_revealed_loss"],
            "fixed_point_revealed_improvement_vs_baseline_pct": (BASELINE_REVEALED - float(fixed_point["total_discounted_revealed_loss"])) / BASELINE_REVEALED * 100.0,
            "fixed_point_artificial_improvement_vs_baseline_pct": (BASELINE_ARTIFICIAL - float(fixed_point["total_discounted_loss"])) / BASELINE_ARTIFICIAL * 100.0,
            "surrogate_intercept": surrogate["intercept"],
            "surrogate_inflation_coeff": surrogate["inflation_gap"],
            "surrogate_output_coeff": surrogate["output_gap"],
            "surrogate_lagged_rate_coeff": surrogate["lagged_policy_rate_gap"],
            "surrogate_fit_rmse": surrogate["fit_rmse"],
        }
        save_json(cache_path, row)
        return row
    env = PyfrbusNativeEnv(ENV_CONFIG, revealed_loss_weights())
    config = SACConfig(**{**asdict(SAC_SETTINGS), "seed": seed})
    trainer = SACTrainer(env, config)
    result = trainer.train()
    checkpoint_path = run_dir / "sac_pyfrbus_revealed_native.pt"
    torch.save(result["actor_state_dict"], checkpoint_path)
    save_json(run_dir / "sac_pyfrbus_revealed_native_config.json", result["config"])

    policy_fn = lambda state, t: trainer._deterministic_action(np.asarray(state, dtype=float))
    env_eval = PyfrbusNativeEnv(ENV_CONFIG, revealed_loss_weights())
    env_stats = evaluate_policy(env_eval, policy_fn, episodes=8, gamma=env_eval.model.config.discount_factor, seed=10_000 + seed)
    pd.DataFrame(result["training_log"]).to_csv(run_dir / "training_log.csv", index=False)
    pd.DataFrame(env_stats["first_trajectory"]).to_csv(run_dir / "first_trajectory.csv", index=False)

    fixed_point = evaluate_policy_fixed_point(f"sac_pyfrbus_revealed_native_seed_{seed}", policy_fn)
    surrogate = fit_surrogate(f"sac_pyfrbus_revealed_native_seed_{seed}", policy_fn)
    pd.DataFrame([fixed_point]).to_csv(run_dir / "fixed_point_summary.csv", index=False)
    pd.DataFrame([surrogate]).to_csv(run_dir / "linear_surrogate.csv", index=False)
    row = {
        "seed": seed,
        "checkpoint_path": str(checkpoint_path),
        "mean_discounted_loss_env": env_stats["mean_discounted_loss"],
        "mean_reward_env": env_stats["mean_reward"],
        "clip_rate_env": env_stats["clip_rate"],
        "explosion_rate_env": env_stats["explosion_rate"],
        "fixed_point_artificial_loss": fixed_point["total_discounted_loss"],
        "fixed_point_revealed_loss": fixed_point["total_discounted_revealed_loss"],
        "fixed_point_revealed_improvement_vs_baseline_pct": (BASELINE_REVEALED - float(fixed_point["total_discounted_revealed_loss"])) / BASELINE_REVEALED * 100.0,
        "fixed_point_artificial_improvement_vs_baseline_pct": (BASELINE_ARTIFICIAL - float(fixed_point["total_discounted_loss"])) / BASELINE_ARTIFICIAL * 100.0,
        "surrogate_intercept": surrogate["intercept"],
        "surrogate_inflation_coeff": surrogate["inflation_gap"],
        "surrogate_output_coeff": surrogate["output_gap"],
        "surrogate_lagged_rate_coeff": surrogate["lagged_policy_rate_gap"],
        "surrogate_fit_rmse": surrogate["fit_rmse"],
    }
    save_json(cache_path, row)
    return row


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [train_one(seed) for seed in SEEDS]
    summary = pd.DataFrame(rows).sort_values("fixed_point_revealed_loss").reset_index(drop=True)
    summary.to_csv(OUTPUT_DIR / "native_training_summary.csv", index=False)

    best = summary.iloc[0]
    best_seed = int(best["seed"])
    best_dir = OUTPUT_DIR / f"seed_{best_seed}"
    best_config = json.loads((best_dir / "sac_pyfrbus_revealed_native_config.json").read_text(encoding="utf-8"))
    env = PyfrbusNativeEnv(ENV_CONFIG, revealed_loss_weights())
    best_policy_fn = build_sac_policy_from_checkpoint(best_dir / "sac_pyfrbus_revealed_native.pt", best_config, env)
    best_fixed = evaluate_policy_fixed_point("sac_pyfrbus_revealed_native_best", best_policy_fn)
    best_fixed["improvement_vs_pyfrbus_baseline_pct"] = (BASELINE_ARTIFICIAL - float(best_fixed["total_discounted_loss"])) / BASELINE_ARTIFICIAL * 100.0
    best_fixed["improvement_vs_pyfrbus_baseline_revealed_pct"] = (
        (BASELINE_REVEALED - float(best_fixed["total_discounted_revealed_loss"])) / BASELINE_REVEALED * 100.0
    )
    best_surrogate = fit_surrogate("sac_pyfrbus_revealed_native_best", best_policy_fn)
    pd.DataFrame([best_fixed]).to_csv(OUTPUT_DIR / "best_policy_fixed_point_summary.csv", index=False)
    pd.DataFrame([best_surrogate]).to_csv(OUTPUT_DIR / "best_policy_linear_surrogate.csv", index=False)

    summarize_results_md(
        OUTPUT_DIR / "native_training_summary.md",
        "PyFRBUS Native Revealed-Loss SAC Training",
        [
            ("Seed Ranking", summary.round(6)),
            ("Best Fixed-Point Result", pd.DataFrame([best_fixed]).round(6)),
            ("Best Linear Surrogate", pd.DataFrame([best_surrogate]).round(6)),
        ],
    )


if __name__ == "__main__":
    main()
