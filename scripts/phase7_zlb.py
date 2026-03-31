from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monetary_rl.evaluation import fit_linear_policy_response, simulate_with_common_shocks
from monetary_rl.experiment_utils import (
    build_taylor_gap_policy,
    load_taylor_rule,
    policy_row,
    run_linear_search,
    run_ppo,
    run_sac,
    run_td3,
    training_log_frame,
    zero_gap_policy,
)
from monetary_rl.envs import BenchmarkEnvConfig, LQBenchmarkEnv
from monetary_rl.models import LQBenchmarkConfig, LQBenchmarkModel
from monetary_rl.solvers import build_optimal_linear_policy, solve_discounted_lq_riccati


OUTPUT_DIR = ROOT / "outputs" / "phase7" / "zlb"
BENCHMARK_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_lq.json"
PPO_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_ppo_tuned.json"
SAC_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_sac.json"
TD3_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_td3.json"
TAYLOR_RULE_PATH = ROOT / "outputs" / "phase2" / "taylor_rule.json"

EVAL_EPISODES = 48


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_config = LQBenchmarkConfig.from_json(BENCHMARK_CONFIG_PATH)
    env_config = BenchmarkEnvConfig(horizon=60, action_low=-benchmark_config.neutral_rate, action_high=6.0, seed=0)
    model = LQBenchmarkModel(benchmark_config)
    env = LQBenchmarkEnv(model, env_config)

    taylor_rule = load_taylor_rule(TAYLOR_RULE_PATH)
    taylor_policy, taylor_intercept = build_taylor_gap_policy(taylor_rule, benchmark_config)
    riccati_solution = solve_discounted_lq_riccati(model)
    riccati_policy = build_optimal_linear_policy(riccati_solution)

    linear_theta, linear_result, linear_policy = run_linear_search(LQBenchmarkEnv(model, env_config), EVAL_EPISODES, seed=123)
    ppo_result, ppo_policy = run_ppo(
        LQBenchmarkEnv(model, env_config),
        PPO_CONFIG_PATH,
        EVAL_EPISODES,
        seed=43,
        total_updates=180,
        rollout_steps=768,
        train_epochs=6,
        eval_interval=10,
    )
    sac_result, sac_policy = run_sac(
        LQBenchmarkEnv(model, env_config),
        SAC_CONFIG_PATH,
        EVAL_EPISODES,
        seed=43,
        total_steps=14000,
    )
    td3_result, td3_policy = run_td3(
        LQBenchmarkEnv(model, env_config),
        TD3_CONFIG_PATH,
        EVAL_EPISODES,
        seed=43,
        total_steps=14000,
    )

    policy_df = pd.DataFrame(
        [
            policy_row("zero_policy", env, zero_gap_policy, EVAL_EPISODES, 30_000),
            policy_row("riccati_rule_with_zlb", env, riccati_policy, EVAL_EPISODES, 30_000),
            policy_row("empirical_taylor", env, taylor_policy, EVAL_EPISODES, 30_000),
            policy_row("linear_policy_search", env, linear_policy, EVAL_EPISODES, 30_000),
            policy_row("ppo_zlb", env, ppo_policy, EVAL_EPISODES, 30_000),
            policy_row("sac_zlb", env, sac_policy, EVAL_EPISODES, 30_000),
            policy_row("td3_zlb", env, td3_policy, EVAL_EPISODES, 30_000),
        ]
    )
    best_loss = float(policy_df["mean_discounted_loss"].min())
    policy_df["loss_gap_vs_best_pct"] = (policy_df["mean_discounted_loss"] - best_loss) / best_loss * 100.0
    policy_df.to_csv(OUTPUT_DIR / "policy_evaluation.csv", index=False)

    training_log_df = pd.concat(
        [
            training_log_frame(ppo_result, "ppo_zlb", 43),
            training_log_frame(sac_result, "sac_zlb", 43),
            training_log_frame(td3_result, "td3_zlb", 43),
        ],
        ignore_index=True,
        sort=False,
    )
    training_log_df.to_csv(OUTPUT_DIR / "training_logs.csv", index=False)

    coeff_df = pd.DataFrame(
        [
            {
                "policy": "riccati_rule_with_zlb",
                "intercept": 0.0,
                "inflation_gap": float(riccati_solution.K[0, 0]),
                "output_gap": float(riccati_solution.K[0, 1]),
                "lagged_policy_rate_gap": float(riccati_solution.K[0, 2]),
                "fit_rmse": 0.0,
            },
            {
                "policy": "empirical_taylor",
                "intercept": taylor_intercept,
                "inflation_gap": taylor_rule["phi_pi"],
                "output_gap": taylor_rule["phi_x"],
                "lagged_policy_rate_gap": taylor_rule["phi_i"],
                "fit_rmse": 0.0,
            },
            {
                "policy": "linear_policy_search",
                "intercept": 0.0,
                "inflation_gap": float(linear_theta[0]),
                "output_gap": float(linear_theta[1]),
                "lagged_policy_rate_gap": float(linear_theta[2]),
                "fit_rmse": 0.0,
            },
            fit_linear_policy_response("ppo_zlb", ppo_policy, env_config.initial_state_low, env_config.initial_state_high),
            fit_linear_policy_response("sac_zlb", sac_policy, env_config.initial_state_low, env_config.initial_state_high),
            fit_linear_policy_response("td3_zlb", td3_policy, env_config.initial_state_low, env_config.initial_state_high),
        ]
    )
    coeff_df.to_csv(OUTPUT_DIR / "policy_coefficients.csv", index=False)

    rng = np.random.default_rng(20260403)
    shocks = rng.standard_normal((20, model.state_dim))
    trajectory_df = simulate_with_common_shocks(
        model,
        {
            "zero_policy": zero_gap_policy,
            "riccati_rule_with_zlb": riccati_policy,
            "empirical_taylor": taylor_policy,
            "linear_policy_search": linear_policy,
            "ppo_zlb": ppo_policy,
            "sac_zlb": sac_policy,
            "td3_zlb": td3_policy,
        },
        initial_state=np.array([-0.5, -1.25, -1.5], dtype=float),
        shocks=shocks,
        action_low=env_config.action_low,
        action_high=env_config.action_high,
    )
    trajectory_df.to_csv(OUTPUT_DIR / "common_shock_trajectories.csv", index=False)

    summary_json = {
        "zlb_action_low": env_config.action_low,
        "zlb_action_high": env_config.action_high,
        "riccati_K": np.round(riccati_solution.K, 6).tolist(),
        "linear_policy_theta": np.round(linear_theta, 6).tolist(),
        "taylor_gap_intercept": taylor_intercept,
        "linear_policy_search": linear_result["config"],
    }
    (OUTPUT_DIR / "zlb_summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Phase 7 ZLB Summary")
    lines.append("")
    lines.append("## Policy Performance")
    lines.append("")
    lines.append(
        policy_df[
            [
                "policy",
                "mean_discounted_loss",
                "std_discounted_loss",
                "mean_reward",
                "mean_abs_action",
                "clip_rate",
                "loss_gap_vs_best_pct",
            ]
        ].round(6).to_markdown(index=False)
    )
    lines.append("")
    lines.append("## Approximate Policy Coefficients")
    lines.append("")
    lines.append(coeff_df.round(6).to_markdown(index=False))
    lines.append("")
    lines.append("## Training Snapshot")
    lines.append("")
    lines.append(training_log_df.groupby("algo").tail(3).round(6).to_markdown(index=False))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This environment imposes a ZLB by constraining the policy-rate gap to stay above -neutral_rate, i.e. actual nominal rate remains non-negative.")
    lines.append("- The Riccati rule shown here is the unconstrained linear benchmark rule executed through the ZLB-constrained environment, not a constrained optimum.")
    lines.append("- Clip rates now matter economically in this environment because hitting the lower bound is part of the policy behavior.")
    lines.append("")
    (OUTPUT_DIR / "zlb_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
