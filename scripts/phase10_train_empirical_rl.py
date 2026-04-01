from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monetary_rl.phase10_utils import build_ann_context, build_svar_context, train_policy_bundle


PPO_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_ppo_tuned.json"
SAC_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_sac.json"
TD3_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_td3.json"
OUTPUT_ROOT = ROOT / "outputs" / "phase10"


def write_summary(output_dir: Path, registry_df: pd.DataFrame, coeff_df: pd.DataFrame) -> None:
    lines = [
        "# Phase 10 Direct Empirical RL Summary",
        "",
        "## Training Evaluation",
        "",
        registry_df[
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
        coeff_df[
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
        "",
        "## Notes",
        "",
        "- `Phase 8/9` transfer baseline is preserved; this directory adds direct-trained empirical RL rules only.",
        "- `PPO` here follows the current main setting (`linear_policy=True`); nonlinear PPO is handled separately in `phase10_ppo_policy_variants.py`.",
        "- Benchmark and empirical environments remain strictly separated.",
    ]
    (output_dir / "direct_training_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_env(env_name: str, seed: int) -> None:
    context = build_svar_context(ROOT) if env_name == "svar" else build_ann_context(ROOT)
    output_dir = OUTPUT_ROOT / f"{env_name}_direct"
    registry_df, coeff_df, _ = train_policy_bundle(
        context=context,
        output_dir=output_dir,
        ppo_config_path=PPO_CONFIG_PATH,
        sac_config_path=SAC_CONFIG_PATH,
        td3_config_path=TD3_CONFIG_PATH,
        seed=seed,
        ppo_linear_policy=True,
    )
    write_summary(output_dir, registry_df, coeff_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO/TD3/SAC directly in SVAR and ANN empirical environments.")
    parser.add_argument("--env", choices=["both", "svar", "ann"], default="both")
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    envs = ["svar", "ann"] if args.env == "both" else [args.env]
    for env_name in envs:
        run_env(env_name, args.seed)


if __name__ == "__main__":
    main()
