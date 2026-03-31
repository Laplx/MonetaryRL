from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monetary_rl.agents import LinearPolicySearch, LinearPolicySearchConfig, PPOConfig, PPOTrainer
from monetary_rl.envs import BenchmarkEnvConfig, LQBenchmarkEnv
from monetary_rl.models import LQBenchmarkConfig, LQBenchmarkModel
from monetary_rl.solvers import build_optimal_linear_policy, solve_discounted_lq_riccati


OUTPUT_DIR = ROOT / "outputs" / "phase6"
BENCHMARK_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_lq.json"
PPO_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_ppo.json"
TAYLOR_RULE_PATH = ROOT / "outputs" / "phase2" / "taylor_rule.json"

EVAL_HORIZON = 60
EVAL_EPISODES = 48
PPO_SEEDS = [7, 29, 43]
GRID_POINTS = 5


def zero_gap_policy(state: np.ndarray, t: int) -> float:
    del state, t
    return 0.0


def load_taylor_rule() -> dict:
    raw = json.loads(TAYLOR_RULE_PATH.read_text(encoding="utf-8"))
    coeffs = raw["coefficients"]
    alpha = float(coeffs["const"])
    phi_pi = float(coeffs["inflation"])
    phi_x = float(coeffs["output_gap"])
    phi_i = float(coeffs["policy_rate_lag1"])
    return {
        "alpha": alpha,
        "phi_pi": phi_pi,
        "phi_x": phi_x,
        "phi_i": phi_i,
    }


def build_taylor_gap_policy(rule: dict, benchmark_config: LQBenchmarkConfig):
    pi_star = benchmark_config.inflation_target
    i_star = benchmark_config.neutral_rate
    intercept = rule["alpha"] + rule["phi_pi"] * pi_star + rule["phi_i"] * i_star - i_star

    def policy(state: np.ndarray, t: int) -> float:
        del t
        inflation_gap, output_gap, lagged_rate_gap = np.asarray(state, dtype=float)
        return (
            intercept
            + rule["phi_pi"] * inflation_gap
            + rule["phi_x"] * output_gap
            + rule["phi_i"] * lagged_rate_gap
        )

    return policy, intercept


def evaluate_policy(env: LQBenchmarkEnv, policy_fn, episodes: int, gamma: float, seed: int) -> dict:
    rewards: list[float] = []
    discounted_losses: list[float] = []
    trajectories = []
    clip_count = 0
    step_count = 0
    abs_action_sum = 0.0

    for ep in range(episodes):
        state = env.reset(seed=seed + ep)
        done = False
        discount = 1.0
        total_reward = 0.0
        total_discounted_loss = 0.0
        traj = []
        t = 0

        while not done:
            raw_action = float(policy_fn(state.copy(), t))
            next_state, reward, done, info = env.step(raw_action)
            total_reward += reward
            total_discounted_loss += (-reward) * discount
            discount *= gamma
            clip_count += int(abs(float(info["raw_action"]) - float(info["action"])) > 1e-8)
            abs_action_sum += abs(float(info["action"]))
            step_count += 1
            traj.append(
                {
                    "period": t,
                    "inflation_gap": float(state[0]),
                    "output_gap": float(state[1]),
                    "lagged_policy_rate_gap": float(state[2]),
                    "action": float(info["action"]),
                    "loss": float(info["loss"]),
                }
            )
            state = next_state
            t += 1

        rewards.append(total_reward)
        discounted_losses.append(total_discounted_loss)
        if ep == 0:
            trajectories = traj

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
        "mean_discounted_loss": float(np.mean(discounted_losses)),
        "std_discounted_loss": float(np.std(discounted_losses, ddof=1)) if len(discounted_losses) > 1 else 0.0,
        "mean_abs_action": abs_action_sum / step_count if step_count else 0.0,
        "clip_rate": clip_count / step_count if step_count else 0.0,
        "first_trajectory": trajectories,
    }


def fit_linear_policy_response(policy_name: str, policy_fn, env_config: BenchmarkEnvConfig) -> dict[str, float]:
    grid = np.linspace(env_config.initial_state_low[0], env_config.initial_state_high[0], GRID_POINTS)
    rows = []
    for inflation_gap in grid:
        for output_gap in grid:
            for lagged_rate_gap in grid:
                state = np.array([inflation_gap, output_gap, lagged_rate_gap], dtype=float)
                action = float(policy_fn(state, 0))
                rows.append([1.0, inflation_gap, output_gap, lagged_rate_gap, action])

    design = np.asarray(rows, dtype=float)
    X = design[:, :4]
    y = design[:, 4]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    return {
        "policy": policy_name,
        "intercept": float(beta[0]),
        "inflation_gap": float(beta[1]),
        "output_gap": float(beta[2]),
        "lagged_policy_rate_gap": float(beta[3]),
        "fit_rmse": rmse,
    }


def simulate_with_common_shocks(
    model: LQBenchmarkModel,
    policy_map: dict[str, callable],
    initial_state: np.ndarray,
    shocks: np.ndarray,
    action_low: float,
    action_high: float,
) -> pd.DataFrame:
    rows = []
    for policy_name, policy_fn in policy_map.items():
        state = initial_state.copy()
        for t in range(shocks.shape[0]):
            raw_action = float(policy_fn(state.copy(), t))
            action = float(np.clip(raw_action, action_low, action_high))
            loss = float(model.stage_loss(state, action))
            rows.append(
                {
                    "policy": policy_name,
                    "period": t,
                    "inflation_gap": float(state[0]),
                    "output_gap": float(state[1]),
                    "lagged_policy_rate_gap": float(state[2]),
                    "action": action,
                    "loss": loss,
                }
            )
            state = model.state_transition(state, action, shocks[t])
    return pd.DataFrame(rows)


def summarize_phase6(
    benchmark_config: LQBenchmarkConfig,
    env_config: BenchmarkEnvConfig,
    riccati_solution,
    policy_eval_df: pd.DataFrame,
    ppo_seed_df: pd.DataFrame,
    coeff_df: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    linear_search_result: dict,
    taylor_rule: dict,
    taylor_intercept: float,
) -> None:
    riccati_loss = float(policy_eval_df.loc[policy_eval_df["policy"] == "riccati_optimal", "mean_discounted_loss"].iloc[0])
    policy_eval_df = policy_eval_df.copy()
    policy_eval_df["loss_gap_vs_riccati_pct"] = (policy_eval_df["mean_discounted_loss"] - riccati_loss) / riccati_loss * 100.0

    lines: list[str] = []
    lines.append("# Phase 6 Benchmark Comparison Summary")
    lines.append("")
    lines.append("## Benchmark Protocol")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Horizon | {env_config.horizon} |")
    lines.append(f"| Evaluation episodes per policy | {EVAL_EPISODES} |")
    lines.append(f"| PPO seeds | {PPO_SEEDS} |")
    lines.append(f"| Action bounds | [{env_config.action_low:.1f}, {env_config.action_high:.1f}] |")
    lines.append(f"| Riccati K | {np.round(riccati_solution.K, 6).tolist()} |")
    lines.append("")

    lines.append("## Policy Performance")
    lines.append("")
    lines.append(
        policy_eval_df[
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

    lines.append("## PPO Seed Stability")
    lines.append("")
    lines.append(ppo_seed_df.round(6).to_markdown(index=False))
    lines.append("")

    lines.append("## Approximate Policy Coefficients")
    lines.append("")
    lines.append(coeff_df.round(6).to_markdown(index=False))
    lines.append("")

    lines.append("## Common-shock Trajectory Excerpt")
    lines.append("")
    lines.append(trajectory_df.head(40).round(6).to_markdown(index=False))
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Phase 6 compares policies under one unified benchmark evaluation protocol instead of mixing Phase 4 fixed-state examples with Phase 5 random-start evaluation.")
    lines.append("- Empirical Taylor rule is treated as an external rule estimated in Phase 2 and converted into benchmark gap form before simulation.")
    lines.append(f"- Taylor gap-form intercept used in benchmark simulations: {taylor_intercept:.6f}.")
    lines.append("- PPO now uses a squashed Gaussian action parameterization so the optimized action distribution matches the bounded action actually executed by the environment.")
    lines.append(
        f"- Linear policy search is included as an additional continuous-control baseline; its best coefficients are {np.round(np.asarray(linear_search_result['best_theta']), 6).tolist()}."
    )
    lines.append("- The benchmark remains a stylized LQ model calibrated to realistic magnitudes, not the empirical SVAR environment itself.")
    lines.append("")

    (OUTPUT_DIR / "phase6_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_config = LQBenchmarkConfig.from_json(BENCHMARK_CONFIG_PATH)
    base_ppo_config = PPOConfig.from_json(PPO_CONFIG_PATH)
    model = LQBenchmarkModel(benchmark_config)
    env_config = BenchmarkEnvConfig(horizon=EVAL_HORIZON, seed=0)
    riccati_solution = solve_discounted_lq_riccati(model)
    riccati_policy = build_optimal_linear_policy(riccati_solution)

    taylor_rule = load_taylor_rule()
    taylor_policy, taylor_intercept = build_taylor_gap_policy(taylor_rule, benchmark_config)

    eval_env = LQBenchmarkEnv(model, env_config)

    zero_stats = evaluate_policy(eval_env, zero_gap_policy, EVAL_EPISODES, gamma=benchmark_config.discount_factor, seed=10_000)
    riccati_stats = evaluate_policy(eval_env, riccati_policy, EVAL_EPISODES, gamma=benchmark_config.discount_factor, seed=10_000)
    taylor_stats = evaluate_policy(eval_env, taylor_policy, EVAL_EPISODES, gamma=benchmark_config.discount_factor, seed=10_000)

    linear_search_env = LQBenchmarkEnv(model, env_config)
    linear_search = LinearPolicySearch(
        linear_search_env,
        LinearPolicySearchConfig(
            seed=123,
            iterations=24,
            population_size=48,
            episodes_per_candidate=4,
            eval_episodes=EVAL_EPISODES,
        ),
    )
    linear_search_result = linear_search.train()
    linear_theta = np.asarray(linear_search_result["best_theta"], dtype=float)

    def linear_search_policy(state: np.ndarray, t: int) -> float:
        del t
        return float(linear_theta @ np.asarray(state, dtype=float))

    linear_stats = evaluate_policy(eval_env, linear_search_policy, EVAL_EPISODES, gamma=benchmark_config.discount_factor, seed=10_000)

    ppo_seed_rows = []
    ppo_training_logs = []
    best_ppo_loss = None
    best_ppo_policy = None
    best_ppo_seed = None

    for seed in PPO_SEEDS:
        ppo_config = replace(
            base_ppo_config,
            seed=seed,
            train_epochs=min(base_ppo_config.train_epochs, 4),
            eval_episodes=min(base_ppo_config.eval_episodes, 12),
        )
        ppo_env = LQBenchmarkEnv(model, env_config)
        trainer = PPOTrainer(ppo_env, ppo_config)
        training_result = trainer.train()

        def ppo_policy(state: np.ndarray, t: int, _trainer=trainer) -> float:
            del t
            return _trainer._deterministic_action(state)

        stats = evaluate_policy(eval_env, ppo_policy, EVAL_EPISODES, gamma=benchmark_config.discount_factor, seed=10_000)
        ppo_seed_rows.append(
            {
                "seed": seed,
                "total_updates": ppo_config.total_updates,
                "mean_discounted_loss": stats["mean_discounted_loss"],
                "std_discounted_loss": stats["std_discounted_loss"],
                "mean_reward": stats["mean_reward"],
                "clip_rate": stats["clip_rate"],
                "mean_abs_action": stats["mean_abs_action"],
                "final_eval_mean_discounted_loss": training_result["training_log"][-1]["eval_mean_discounted_loss"],
            }
        )
        seed_log = pd.DataFrame(training_result["training_log"])
        seed_log.insert(0, "seed", seed)
        ppo_training_logs.append(seed_log)

        if best_ppo_loss is None or stats["mean_discounted_loss"] < best_ppo_loss:
            best_ppo_loss = stats["mean_discounted_loss"]
            best_ppo_policy = ppo_policy
            best_ppo_seed = seed

    ppo_seed_df = pd.DataFrame(ppo_seed_rows).sort_values("mean_discounted_loss").reset_index(drop=True)
    ppo_training_log_df = pd.concat(ppo_training_logs, ignore_index=True)
    ppo_training_log_df.to_csv(OUTPUT_DIR / "ppo_training_log_multi_seed.csv", index=False)
    ppo_seed_df.to_csv(OUTPUT_DIR / "ppo_seed_summary.csv", index=False)

    best_ppo_stats = evaluate_policy(eval_env, best_ppo_policy, EVAL_EPISODES, gamma=benchmark_config.discount_factor, seed=10_000)
    policy_eval_df = pd.DataFrame(
        [
            {"policy": "zero_policy", **zero_stats},
            {"policy": "riccati_optimal", **riccati_stats},
            {"policy": "empirical_taylor", **taylor_stats},
            {"policy": "linear_policy_search", **linear_stats},
            {"policy": f"ppo_best_seed_{best_ppo_seed}", **best_ppo_stats},
        ]
    )
    policy_eval_df.to_csv(OUTPUT_DIR / "policy_evaluation.csv", index=False)

    coeff_rows = [
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
    ]
    coeff_rows.append(fit_linear_policy_response(f"ppo_best_seed_{best_ppo_seed}", best_ppo_policy, env_config))
    coeff_df = pd.DataFrame(coeff_rows)
    coeff_df.to_csv(OUTPUT_DIR / "policy_coefficients.csv", index=False)

    rng = np.random.default_rng(20260331)
    initial_state = np.array([1.0, -1.0, 0.0], dtype=float)
    shocks = rng.standard_normal((20, model.state_dim))
    trajectory_df = simulate_with_common_shocks(
        model,
        {
            "zero_policy": zero_gap_policy,
            "riccati_optimal": riccati_policy,
            "empirical_taylor": taylor_policy,
            "linear_policy_search": linear_search_policy,
            f"ppo_best_seed_{best_ppo_seed}": best_ppo_policy,
        },
        initial_state=initial_state,
        shocks=shocks,
        action_low=env_config.action_low,
        action_high=env_config.action_high,
    )
    trajectory_df.to_csv(OUTPUT_DIR / "common_shock_trajectories.csv", index=False)

    summary_json = {
        "benchmark_config": str(BENCHMARK_CONFIG_PATH.relative_to(ROOT)),
        "ppo_config": str(PPO_CONFIG_PATH.relative_to(ROOT)),
        "taylor_rule_source": str(TAYLOR_RULE_PATH.relative_to(ROOT)),
        "ppo_seeds": PPO_SEEDS,
        "best_ppo_seed": best_ppo_seed,
        "best_ppo_loss": best_ppo_loss,
        "riccati_K": np.round(riccati_solution.K, 6).tolist(),
        "linear_policy_theta": np.round(linear_theta, 6).tolist(),
        "taylor_gap_intercept": taylor_intercept,
    }
    (OUTPUT_DIR / "phase6_summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")

    summarize_phase6(
        benchmark_config=benchmark_config,
        env_config=env_config,
        riccati_solution=riccati_solution,
        policy_eval_df=policy_eval_df,
        ppo_seed_df=ppo_seed_df,
        coeff_df=coeff_df,
        trajectory_df=trajectory_df,
        linear_search_result=linear_search_result,
        taylor_rule=taylor_rule,
        taylor_intercept=taylor_intercept,
    )


if __name__ == "__main__":
    main()
