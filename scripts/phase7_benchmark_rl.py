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


OUTPUT_DIR = ROOT / "outputs" / "phase7" / "benchmark_rl"
BENCHMARK_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_lq.json"
PPO_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_ppo_tuned.json"
SAC_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_sac.json"
TD3_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_td3.json"
TAYLOR_RULE_PATH = ROOT / "outputs" / "phase2" / "taylor_rule.json"

EVAL_EPISODES = 48
ENV_CONFIG = BenchmarkEnvConfig(horizon=60, seed=0)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_config = LQBenchmarkConfig.from_json(BENCHMARK_CONFIG_PATH)
    model = LQBenchmarkModel(benchmark_config)
    env = LQBenchmarkEnv(model, ENV_CONFIG)

    taylor_rule = load_taylor_rule(TAYLOR_RULE_PATH)
    taylor_policy, taylor_intercept = build_taylor_gap_policy(taylor_rule, benchmark_config)
    riccati_solution = solve_discounted_lq_riccati(model)
    riccati_policy = build_optimal_linear_policy(riccati_solution)

    linear_theta, linear_result, linear_policy = run_linear_search(LQBenchmarkEnv(model, ENV_CONFIG), EVAL_EPISODES, seed=123)
    ppo_result, ppo_policy = run_ppo(
        LQBenchmarkEnv(model, ENV_CONFIG),
        PPO_CONFIG_PATH,
        EVAL_EPISODES,
        seed=43,
        total_updates=180,
        rollout_steps=768,
        train_epochs=6,
        eval_interval=10,
    )
    sac_result, sac_policy = run_sac(
        LQBenchmarkEnv(model, ENV_CONFIG),
        SAC_CONFIG_PATH,
        EVAL_EPISODES,
        seed=43,
        total_steps=14000,
    )
    td3_result, td3_policy = run_td3(
        LQBenchmarkEnv(model, ENV_CONFIG),
        TD3_CONFIG_PATH,
        EVAL_EPISODES,
        seed=43,
        total_steps=14000,
    )

    policy_df = pd.DataFrame(
        [
            policy_row("zero_policy", env, zero_gap_policy, EVAL_EPISODES, 10_000),
            policy_row("riccati_optimal", env, riccati_policy, EVAL_EPISODES, 10_000),
            policy_row("empirical_taylor", env, taylor_policy, EVAL_EPISODES, 10_000),
            policy_row("linear_policy_search", env, linear_policy, EVAL_EPISODES, 10_000),
            policy_row("ppo_tuned", env, ppo_policy, EVAL_EPISODES, 10_000),
            policy_row("sac", env, sac_policy, EVAL_EPISODES, 10_000),
            policy_row("td3", env, td3_policy, EVAL_EPISODES, 10_000),
        ]
    )
    riccati_loss = float(policy_df.loc[policy_df["policy"] == "riccati_optimal", "mean_discounted_loss"].iloc[0])
    policy_df["loss_gap_vs_riccati_pct"] = (policy_df["mean_discounted_loss"] - riccati_loss) / riccati_loss * 100.0
    policy_df.to_csv(OUTPUT_DIR / "policy_evaluation.csv", index=False)

    training_log_df = pd.concat(
        [
            training_log_frame(ppo_result, "ppo_tuned", 43),
            training_log_frame(sac_result, "sac", 43),
            training_log_frame(td3_result, "td3", 43),
        ],
        ignore_index=True,
        sort=False,
    )
    training_log_df.to_csv(OUTPUT_DIR / "training_logs.csv", index=False)

    coeff_df = pd.DataFrame(
        [
            {
                "policy": "riccati_optimal",
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
            fit_linear_policy_response("ppo_tuned", ppo_policy, ENV_CONFIG.initial_state_low, ENV_CONFIG.initial_state_high),
            fit_linear_policy_response("sac", sac_policy, ENV_CONFIG.initial_state_low, ENV_CONFIG.initial_state_high),
            fit_linear_policy_response("td3", td3_policy, ENV_CONFIG.initial_state_low, ENV_CONFIG.initial_state_high),
        ]
    )
    coeff_df.to_csv(OUTPUT_DIR / "policy_coefficients.csv", index=False)

    rng = np.random.default_rng(20260401)
    shocks = rng.standard_normal((20, model.state_dim))
    trajectory_df = simulate_with_common_shocks(
        model,
        {
            "zero_policy": zero_gap_policy,
            "riccati_optimal": riccati_policy,
            "empirical_taylor": taylor_policy,
            "linear_policy_search": linear_policy,
            "ppo_tuned": ppo_policy,
            "sac": sac_policy,
            "td3": td3_policy,
        },
        initial_state=np.array([1.0, -1.0, 0.0], dtype=float),
        shocks=shocks,
        action_low=ENV_CONFIG.action_low,
        action_high=ENV_CONFIG.action_high,
    )
    trajectory_df.to_csv(OUTPUT_DIR / "common_shock_trajectories.csv", index=False)

    summary_json = {
        "riccati_K": np.round(riccati_solution.K, 6).tolist(),
        "linear_policy_theta": np.round(linear_theta, 6).tolist(),
        "taylor_gap_intercept": taylor_intercept,
        "ppo_config": ppo_result["config"],
        "sac_config": sac_result["config"],
        "td3_config": td3_result["config"],
        "linear_policy_search": linear_result["config"],
    }
    (OUTPUT_DIR / "benchmark_rl_summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Phase 7 Benchmark RL Strengthening Summary")
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
                "loss_gap_vs_riccati_pct",
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
    lines.append("- This file strengthens the benchmark RL line after Phase 6 by adding tuned PPO, SAC and TD3 under a unified evaluation protocol.")
    lines.append("- Linear policy search remains a strong benchmark-specific baseline because the underlying benchmark is still an LQ problem.")
    lines.append("- Empirical Taylor rule is kept as an external benchmark policy, not an endogenous benchmark component.")
    lines.append("")
    (OUTPUT_DIR / "benchmark_rl_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
