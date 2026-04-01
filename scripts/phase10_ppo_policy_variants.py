from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monetary_rl.evaluation import fit_linear_policy_response
from monetary_rl.experiment_utils import policy_row, run_ppo, training_log_frame
from monetary_rl.phase10_utils import build_ann_context, build_svar_context, make_empirical_env


PPO_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_ppo_tuned.json"
OUTPUT_DIR = ROOT / "outputs" / "phase10" / "ppo_policy_variants"
DIRECT_ROOT = ROOT / "outputs" / "phase10"


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _copy_linear_variant(env_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    direct_dir = DIRECT_ROOT / f"{env_name}_direct"
    registry_df = pd.read_csv(direct_dir / "policy_registry.csv")
    coeff_df = pd.read_csv(direct_dir / "policy_coefficients.csv")
    linear_row = registry_df.loc[registry_df["policy_name"] == f"ppo_{env_name}_direct"].copy()
    linear_coeff = coeff_df.loc[coeff_df["policy"] == f"ppo_{env_name}_direct"].copy()
    linear_row.loc[:, "policy_name"] = f"ppo_{env_name}_direct_linear"
    linear_row.loc[:, "policy_parameterization"] = "linear_policy"
    linear_row.loc[:, "note"] = f"Reused linear-policy PPO direct result from outputs/phase10/{env_name}_direct/."
    linear_coeff.loc[:, "policy"] = f"ppo_{env_name}_direct_linear"
    linear_coeff.loc[:, "policy_parameterization"] = "linear_policy"
    return linear_row, linear_coeff


def run_variant(env_name: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context = build_svar_context(ROOT) if env_name == "svar" else build_ann_context(ROOT)
    variant_dir = OUTPUT_DIR / env_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    linear_registry, linear_coeff = _copy_linear_variant(env_name)

    nonlinear_name = f"ppo_{env_name}_direct_nonlinear"
    env = make_empirical_env(context, seed=seed)
    result, policy_fn = run_ppo(env, PPO_CONFIG_PATH, eval_episodes=16, seed=seed, linear_policy=False)
    nonlinear_log = training_log_frame(result, "ppo", seed)
    nonlinear_log.insert(0, "policy_name", nonlinear_name)
    nonlinear_log.insert(0, "training_env", env_name)

    checkpoint_path = variant_dir / "checkpoints" / f"{nonlinear_name}.pt"
    config_path = variant_dir / "configs" / f"{nonlinear_name}.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result["policy_state_dict"], checkpoint_path)
    _save_json(config_path, result["config"])

    eval_env = make_empirical_env(context, seed=seed + 500)
    eval_stats = policy_row(nonlinear_name, eval_env, policy_fn, episodes=32, seed=seed + 10_000)
    coeff_row = fit_linear_policy_response(
        nonlinear_name,
        policy_fn,
        context.observation_low,
        context.observation_high,
        grid_points=7,
    )
    nonlinear_registry = pd.DataFrame(
        [
            {
                "policy_name": nonlinear_name,
                "rule_family": f"{env_name}_direct",
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
                "note": f"Direct-trained nonlinear-policy PPO in the {env_name.upper()} empirical environment.",
            }
        ]
    )
    nonlinear_coeff = pd.DataFrame(
        [
            {
                "policy": nonlinear_name,
                "training_env": env_name,
                "algo": "ppo",
                "seed": seed,
                "policy_parameterization": "nonlinear_policy",
                **coeff_row,
            }
        ]
    )

    registry_df = pd.concat([linear_registry, nonlinear_registry], ignore_index=True)
    coeff_df = pd.concat([linear_coeff, nonlinear_coeff], ignore_index=True)
    log_df = nonlinear_log.copy()

    registry_df.to_csv(variant_dir / "policy_registry.csv", index=False)
    coeff_df.to_csv(variant_dir / "policy_coefficients.csv", index=False)
    log_df.to_csv(variant_dir / "training_logs.csv", index=False)
    return registry_df, coeff_df, log_df


def write_summary(summary_df: pd.DataFrame, coeff_df: pd.DataFrame) -> None:
    lines = [
        "# Phase 10 PPO Policy Variants",
        "",
        "## Variant Evaluation",
        "",
        summary_df[
            [
                "policy_name",
                "training_env",
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
        "## Approximate Coefficients",
        "",
        coeff_df[
            [
                "policy",
                "training_env",
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
        "",
        "## Notes",
        "",
        "- Linear-policy PPO reuses the main direct-training artifact to avoid duplicating the same run.",
        "- Nonlinear-policy PPO is newly trained here and kept separate from the main Phase 10 unified rule set.",
    ]
    (OUTPUT_DIR / "ppo_policy_variants_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO linear/nonlinear policy variants in empirical environments.")
    parser.add_argument("--env", choices=["both", "svar", "ann"], default="both")
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    envs = ["svar", "ann"] if args.env == "both" else [args.env]
    registry_frames: list[pd.DataFrame] = []
    coeff_frames: list[pd.DataFrame] = []
    for env_name in envs:
        registry_df, coeff_df, _ = run_variant(env_name, args.seed)
        registry_frames.append(registry_df)
        coeff_frames.append(coeff_df)
    summary_df = pd.concat(registry_frames, ignore_index=True).sort_values("mean_discounted_loss").reset_index(drop=True)
    coeff_df = pd.concat(coeff_frames, ignore_index=True).reset_index(drop=True)
    summary_df.to_csv(OUTPUT_DIR / "policy_registry.csv", index=False)
    coeff_df.to_csv(OUTPUT_DIR / "policy_coefficients.csv", index=False)
    write_summary(summary_df, coeff_df)


if __name__ == "__main__":
    main()
