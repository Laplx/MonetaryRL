from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monetary_rl.agents import PPOConfig, PPOTrainer
from monetary_rl.envs import BenchmarkEnvConfig, LQBenchmarkEnv
from monetary_rl.models import LQBenchmarkConfig, LQBenchmarkModel
from monetary_rl.solvers import build_optimal_linear_policy, solve_discounted_lq_riccati


OUTPUT_DIR = ROOT / "outputs" / "phase5"
BENCHMARK_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_lq.json"
PPO_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_ppo.json"


def zero_gap_policy(state: np.ndarray, t: int) -> float:
    del state, t
    return 0.0


def evaluate_policy(env: LQBenchmarkEnv, policy_fn, episodes: int, gamma: float, seed: int) -> dict:
    rewards = []
    discounted_losses = []
    first_trajectory = []
    for ep in range(episodes):
        state = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        total_discounted_loss = 0.0
        discount = 1.0
        trajectory = []
        t = 0
        while not done:
            action = float(policy_fn(state.copy(), t))
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            total_discounted_loss += (-reward) * discount
            discount *= gamma
            trajectory.append(
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
            first_trajectory = trajectory
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
        "mean_discounted_loss": float(np.mean(discounted_losses)),
        "std_discounted_loss": float(np.std(discounted_losses, ddof=1)) if len(discounted_losses) > 1 else 0.0,
        "first_trajectory": first_trajectory,
    }


def write_summary(training_result: dict, zero_stats: dict, optimal_stats: dict, rl_stats: dict, riccati_solution) -> None:
    training_log = pd.DataFrame(training_result["training_log"])
    training_log.to_csv(OUTPUT_DIR / "training_log.csv", index=False)

    zero_loss = zero_stats["mean_discounted_loss"]
    opt_loss = optimal_stats["mean_discounted_loss"]
    rl_loss = rl_stats["mean_discounted_loss"]
    rl_vs_zero = (zero_loss - rl_loss) / zero_loss * 100.0
    rl_gap_to_opt = (rl_loss - opt_loss) / opt_loss * 100.0

    lines: list[str] = []
    lines.append("# Phase 5 PPO Baseline Summary")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append("Train a continuous-action PPO agent on the exact same LQ benchmark solved analytically in Phase 4.")
    lines.append("")
    lines.append("## PPO Configuration")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    for key, value in training_result["config"].items():
        lines.append(f"| {key} | {value} |")
    lines.append("")

    lines.append("## Evaluation Comparison")
    lines.append("")
    lines.append("| Policy | Mean discounted loss | Std. discounted loss | Mean reward |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Zero policy | {zero_loss:.6f} | {zero_stats['std_discounted_loss']:.6f} | {zero_stats['mean_reward']:.6f} |")
    lines.append(f"| Riccati optimal | {opt_loss:.6f} | {optimal_stats['std_discounted_loss']:.6f} | {optimal_stats['mean_reward']:.6f} |")
    lines.append(f"| PPO baseline | {rl_loss:.6f} | {rl_stats['std_discounted_loss']:.6f} | {rl_stats['mean_reward']:.6f} |")
    lines.append("")

    lines.append("## PPO Relative Performance")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Improvement over zero policy (%) | {rl_vs_zero:.6f} |")
    lines.append(f"| Distance above Riccati optimum (%) | {rl_gap_to_opt:.6f} |")
    lines.append("")

    lines.append("## Final Training Log Snapshot")
    lines.append("")
    lines.append(training_log.tail(10).round(6).to_markdown(index=False))
    lines.append("")

    lines.append("## Riccati Benchmark Reference")
    lines.append("")
    eig_modulus = np.abs(riccati_solution.closed_loop_eigenvalues)
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Largest closed-loop eigenvalue modulus | {eig_modulus.max():.6f} |")
    lines.append(f"| Feedback matrix K | {np.round(riccati_solution.K, 6).tolist()} |")
    lines.append("")

    traj_df = pd.DataFrame(rl_stats["first_trajectory"]).head(12).round(6)
    lines.append("## PPO First Evaluation Trajectory")
    lines.append("")
    lines.append(traj_df.to_markdown(index=False))
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- This is a baseline PPO run, not the final RL result of the thesis.")
    lines.append("- Phase 6 will compare PPO and Riccati solutions more systematically.")
    lines.append("- ANN environment tuning remains a separate parallel task and does not affect this benchmark PPO result.")
    lines.append("")

    (OUTPUT_DIR / "ppo_baseline_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_config = LQBenchmarkConfig.from_json(BENCHMARK_CONFIG_PATH)
    ppo_config = PPOConfig.from_json(PPO_CONFIG_PATH)
    model = LQBenchmarkModel(benchmark_config)
    env = LQBenchmarkEnv(model, BenchmarkEnvConfig(horizon=60, seed=ppo_config.seed))

    trainer = PPOTrainer(env, ppo_config)
    training_result = trainer.train()

    riccati_solution = solve_discounted_lq_riccati(model)
    optimal_policy = build_optimal_linear_policy(riccati_solution)

    eval_env = LQBenchmarkEnv(model, BenchmarkEnvConfig(horizon=60, seed=123))
    zero_stats = evaluate_policy(eval_env, zero_gap_policy, episodes=48, gamma=ppo_config.gamma, seed=20_000)
    optimal_stats = evaluate_policy(eval_env, optimal_policy, episodes=48, gamma=ppo_config.gamma, seed=20_000)
    rl_stats = trainer.evaluate(48, seed=20_000)

    torch.save(training_result["policy_state_dict"], OUTPUT_DIR / "ppo_policy_state_dict.pt")
    (OUTPUT_DIR / "ppo_training_result.json").write_text(
        json.dumps(
            {
                "config": training_result["config"],
                "eval_stats": training_result["eval_stats"],
                "zero_stats": zero_stats,
                "optimal_stats": optimal_stats,
                "rl_stats": rl_stats,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_summary(training_result, zero_stats, optimal_stats, rl_stats, riccati_solution)


if __name__ == "__main__":
    main()
