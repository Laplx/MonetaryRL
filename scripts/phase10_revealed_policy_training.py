from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monetary_rl.phase10_utils import (
    build_ann_context,
    build_svar_context,
    clone_context_with_loss_weights,
    fit_linear_policy_response,
    make_empirical_env,
    policy_row,
    run_ppo,
    training_log_frame,
    train_policy_bundle,
)


PPO_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_ppo_tuned.json"
SAC_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_sac.json"
TD3_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_td3.json"
REVEALED_DIR = ROOT / "outputs" / "phase10" / "revealed_welfare"
OUTPUT_ROOT = ROOT / "outputs" / "phase10" / "revealed_policy_training"


def revealed_loss_weights() -> dict[str, float]:
    payload = json.loads((REVEALED_DIR / "revealed_weights.json").read_text(encoding="utf-8"))
    return {
        "inflation": float(payload["inflation_weight"]),
        "output_gap": float(payload["output_gap_weight"]),
        "rate_smoothing": float(payload["rate_smoothing_weight"]),
    }


def train_nonlinear_ppo_variant(context, output_dir: Path, env_name: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy_name = f"ppo_{env_name}_revealed_direct_nonlinear"
    env = make_empirical_env(context, seed=seed)
    result, policy_fn = run_ppo(env, PPO_CONFIG_PATH, eval_episodes=16, seed=seed, linear_policy=False)
    log_df = training_log_frame(result, "ppo", seed)
    log_df.insert(0, "training_env", env_name)
    log_df.insert(0, "policy_name", policy_name)

    checkpoint_path = output_dir / "checkpoints" / f"{policy_name}.pt"
    config_path = output_dir / "configs" / f"{policy_name}.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result["policy_state_dict"], checkpoint_path)
    config_path.write_text(json.dumps(result["config"], indent=2, ensure_ascii=False), encoding="utf-8")

    eval_env = make_empirical_env(context, seed=seed + 500)
    eval_stats = policy_row(policy_name, eval_env, policy_fn, episodes=32, seed=seed + 10_000)
    coeff_row = fit_linear_policy_response(
        policy_name,
        policy_fn,
        context.observation_low,
        context.observation_high,
        grid_points=7,
    )
    registry_df = pd.DataFrame(
        [
            {
                "policy_name": policy_name,
                "rule_family": f"{env_name}_revealed_direct",
                "source_env": env_name,
                "training_env": env_name,
                "callable_type": "checkpoint",
                "algo": "ppo",
                "seed": seed,
                "policy_parameterization": "nonlinear_policy",
                "checkpoint_path": str(checkpoint_path),
                "config_path": str(config_path),
                "intercept": coeff_row["intercept"],
                "inflation_gap": coeff_row["inflation_gap"],
                "output_gap": coeff_row["output_gap"],
                "lagged_policy_rate_gap": coeff_row["lagged_policy_rate_gap"],
                "fit_rmse": coeff_row["fit_rmse"],
                "mean_discounted_loss": eval_stats["mean_discounted_loss"],
                "std_discounted_loss": eval_stats["std_discounted_loss"],
                "mean_reward": eval_stats["mean_reward"],
                "clip_rate": eval_stats["clip_rate"],
                "explosion_rate": eval_stats["explosion_rate"],
                "note": f"Direct-trained nonlinear-policy PPO in the {env_name.upper()}_REVEALED empirical environment. Training objective uses revealed welfare weights.",
            }
        ]
    )
    coeff_df = pd.DataFrame(
        [
            {
                "policy": policy_name,
                "training_env": env_name,
                "algo": "ppo",
                "seed": seed,
                "policy_parameterization": "nonlinear_policy",
                **coeff_row,
            }
        ]
    )
    return registry_df, coeff_df, log_df


def ensure_env_bundle(env_name: str, context, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = OUTPUT_ROOT / env_name
    registry_path = output_dir / "policy_registry.csv"
    coeff_path = output_dir / "policy_coefficients.csv"
    log_path = output_dir / "training_logs.csv"

    if registry_path.exists() and coeff_path.exists() and log_path.exists():
        registry_df = pd.read_csv(registry_path)
        coeff_df = pd.read_csv(coeff_path)
        log_df = pd.read_csv(log_path)
    else:
        registry_df, coeff_df, log_df = train_policy_bundle(
            context=context,
            output_dir=output_dir,
            ppo_config_path=PPO_CONFIG_PATH,
            sac_config_path=SAC_CONFIG_PATH,
            td3_config_path=TD3_CONFIG_PATH,
            seed=seed,
            ppo_linear_policy=True,
            ppo_policy_name=f"ppo_{env_name}_revealed_direct",
        )
        registry_df["policy_name"] = registry_df["policy_name"].replace(
            {
                f"td3_{context.name}_direct": f"td3_{env_name}_revealed_direct",
                f"sac_{context.name}_direct": f"sac_{env_name}_revealed_direct",
            }
        )
        coeff_df["policy"] = coeff_df["policy"].replace(
            {
                f"td3_{context.name}_direct": f"td3_{env_name}_revealed_direct",
                f"sac_{context.name}_direct": f"sac_{env_name}_revealed_direct",
            }
        )
        log_df["policy_name"] = log_df["policy_name"].replace(
            {
                f"td3_{context.name}_direct": f"td3_{env_name}_revealed_direct",
                f"sac_{context.name}_direct": f"sac_{env_name}_revealed_direct",
            }
        )
        registry_df["rule_family"] = f"{env_name}_revealed_direct"
        registry_df["source_env"] = env_name
        registry_df["training_env"] = env_name
        registry_df["note"] = registry_df["note"].astype(str) + " Training objective uses revealed welfare weights."
        coeff_df["training_env"] = env_name
        log_df["training_env"] = env_name

    nonlinear_name = f"ppo_{env_name}_revealed_direct_nonlinear"
    if nonlinear_name not in set(registry_df["policy_name"].astype(str)):
        nonlinear_registry, nonlinear_coeff, nonlinear_log = train_nonlinear_ppo_variant(context, output_dir, env_name, seed)
        registry_df = pd.concat([registry_df, nonlinear_registry], ignore_index=True, sort=False)
        coeff_df = pd.concat([coeff_df, nonlinear_coeff], ignore_index=True, sort=False)
        log_df = pd.concat([log_df, nonlinear_log], ignore_index=True, sort=False)

    registry_df = registry_df.sort_values("mean_discounted_loss").reset_index(drop=True)
    coeff_sort_cols = [col for col in ["algo", "policy_parameterization", "seed", "policy"] if col in coeff_df.columns]
    if coeff_sort_cols:
        coeff_df = coeff_df.sort_values(coeff_sort_cols).reset_index(drop=True)
    log_sort_cols = [col for col in ["policy_name", "episode"] if col in log_df.columns]
    if log_sort_cols:
        log_df = log_df.sort_values(log_sort_cols).reset_index(drop=True)

    registry_df.to_csv(registry_path, index=False)
    coeff_df.to_csv(coeff_path, index=False)
    log_df.to_csv(log_path, index=False)
    return registry_df, coeff_df, log_df


def rewrite_registry_family(registry_df: pd.DataFrame, coeff_df: pd.DataFrame, env_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    family = f"{env_name}_revealed_direct"
    mapping = {
        f"ppo_{env_name}_revealed_direct": f"ppo_{env_name}_revealed_direct",
        f"td3_{env_name}_revealed_direct": f"td3_{env_name}_revealed_direct",
        f"sac_{env_name}_revealed_direct": f"sac_{env_name}_revealed_direct",
    }
    registry_df = registry_df.copy()
    coeff_df = coeff_df.copy()
    registry_df["policy_name"] = registry_df["policy_name"].map(mapping)
    registry_df["rule_family"] = family
    registry_df["source_env"] = env_name
    registry_df["training_env"] = env_name
    registry_df["note"] = registry_df["note"].astype(str) + " Training objective uses revealed welfare weights."
    coeff_df["policy"] = coeff_df["policy"].map(mapping)
    coeff_df["training_env"] = env_name
    return registry_df, coeff_df


def write_summary(all_registry: pd.DataFrame, all_coeff: pd.DataFrame, weights: dict[str, float]) -> None:
    lines = [
        "# Phase 10 Revealed-Welfare RL Training",
        "",
        "## Revealed Loss Weights",
        "",
        pd.DataFrame([weights]).round(6).to_markdown(index=False),
        "",
        "## Training Evaluation",
        "",
        all_registry[
            [
                "policy_name",
                "training_env",
                "algo",
                "policy_parameterization",
                "mean_discounted_loss",
                "std_discounted_loss",
                "mean_reward",
                "clip_rate",
                "explosion_rate",
            ]
        ]
        .round(6)
        .to_markdown(index=False),
        "",
        "## Approximate Policy Coefficients",
        "",
        all_coeff[
            [
                "policy",
                "training_env",
                "algo",
                "policy_parameterization",
                "intercept",
                "inflation_gap",
                "output_gap",
                "lagged_policy_rate_gap",
                "fit_rmse",
            ]
        ]
        .round(6)
        .to_markdown(index=False),
    ]
    (OUTPUT_ROOT / "revealed_policy_training_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    weights = revealed_loss_weights()

    contexts = {
        "svar": clone_context_with_loss_weights(build_svar_context(ROOT), weights, "revealed"),
        "ann": clone_context_with_loss_weights(build_ann_context(ROOT), weights, "revealed"),
    }

    registry_frames: list[pd.DataFrame] = []
    coeff_frames: list[pd.DataFrame] = []
    log_frames: list[pd.DataFrame] = []

    for env_name, context in contexts.items():
        registry_df, coeff_df, log_df = ensure_env_bundle(env_name, context, seed=43)
        registry_frames.append(registry_df)
        coeff_frames.append(coeff_df)
        log_frames.append(log_df)

    all_registry = pd.concat(registry_frames, ignore_index=True).sort_values("mean_discounted_loss").reset_index(drop=True)
    all_coeff = pd.concat(coeff_frames, ignore_index=True).reset_index(drop=True)
    all_logs = pd.concat(log_frames, ignore_index=True).reset_index(drop=True)
    all_registry.to_csv(OUTPUT_ROOT / "policy_registry.csv", index=False)
    all_coeff.to_csv(OUTPUT_ROOT / "policy_coefficients.csv", index=False)
    all_logs.to_csv(OUTPUT_ROOT / "training_logs.csv", index=False)
    write_summary(all_registry, all_coeff, weights)


if __name__ == "__main__":
    main()
