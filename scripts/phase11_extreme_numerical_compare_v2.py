from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monetary_rl.evaluation import evaluate_policy, fit_linear_policy_response, simulate_with_common_shocks
from monetary_rl.phase11_extreme_specs_v2 import NEW_ENV_SPECS_V2, make_model
from monetary_rl.solvers.finite_horizon_dp import (
    FiniteHorizonDPConfig,
    solve_finite_horizon_dp,
    three_point_normal_quadrature,
)

MATRIX_DIR = ROOT / "outputs" / "phase11" / "extreme_matrix_v2"
OUTPUT_DIR = ROOT / "outputs" / "phase11" / "extreme_numerical_compare_v2"


def affine_policy(coeff_row: pd.Series):
    intercept = float(coeff_row["intercept"])
    inflation_coeff = float(coeff_row["inflation_gap"])
    output_coeff = float(coeff_row["output_gap"])
    lagged_coeff = float(coeff_row["lagged_policy_rate_gap"])

    def policy(state: np.ndarray, t: int) -> float:
        del t
        return intercept + inflation_coeff * float(state[0]) + output_coeff * float(state[1]) + lagged_coeff * float(state[2])

    return policy


def load_phase11_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_rl = pd.read_csv(MATRIX_DIR / "raw_rl_results.csv")
    coeffs = pd.read_csv(MATRIX_DIR / "policy_coefficients.csv")
    baselines = pd.read_csv(MATRIX_DIR / "baseline_results.csv")
    return raw_rl, coeffs, baselines


def build_policies(case_id: str, coeffs: pd.DataFrame, raw_rl: pd.DataFrame):
    case_rl = raw_rl.loc[raw_rl["env_id"] == case_id].copy()
    grouped = (
        case_rl.groupby(["algo", "seed"], as_index=False)["mean_discounted_loss"]
        .mean()
        .sort_values("mean_discounted_loss")
        .reset_index(drop=True)
    )
    best_rl = grouped.iloc[0]
    best_rl_coeff = coeffs.loc[
        (coeffs["env_id"] == case_id)
        & (coeffs["algo"] == best_rl["algo"])
        & (coeffs["seed"] == best_rl["seed"])
    ].iloc[0]

    riccati_coeff = coeffs.loc[(coeffs["env_id"] == case_id) & (coeffs["policy"] == "riccati_reference")].iloc[0]
    linear_search_coeff = coeffs.loc[(coeffs["env_id"] == case_id) & (coeffs["policy"] == "linear_policy_search")].iloc[0]
    return {
        "best_rl": best_rl,
        "best_rl_policy": affine_policy(best_rl_coeff),
        "riccati_policy": affine_policy(riccati_coeff),
        "linear_search_policy": affine_policy(linear_search_coeff),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", action="append", dest="env_ids")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_rl, coeffs, _ = load_phase11_tables()

    summary_rows = []
    coeff_rows = []
    shock_nodes_2d, shock_weights = three_point_normal_quadrature(2)
    shock_nodes = np.column_stack([shock_nodes_2d[:, 0], shock_nodes_2d[:, 1], np.zeros(len(shock_nodes_2d))])

    from monetary_rl.envs import BenchmarkEnvConfig, LQBenchmarkEnv

    env_specs = NEW_ENV_SPECS_V2
    if args.env_ids:
        wanted = set(args.env_ids)
        env_specs = [spec for spec in NEW_ENV_SPECS_V2 if spec["env_id"] in wanted]

    for idx, case_spec in enumerate(env_specs):
        case_id = case_spec["env_id"]
        case_dir = OUTPUT_DIR / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        config, model = make_model(case_spec)
        policies = build_policies(case_id, coeffs, raw_rl)
        env = LQBenchmarkEnv(model, BenchmarkEnvConfig(**case_spec["env_kwargs"]))

        dp_config = FiniteHorizonDPConfig(
            horizon=case_spec["env_kwargs"]["horizon"],
            discount_factor=config.discount_factor,
            action_low=case_spec["env_kwargs"]["action_low"],
            action_high=case_spec["env_kwargs"]["action_high"],
            action_points=case_spec["dp_kwargs"]["action_points"],
            state_low=case_spec["dp_kwargs"]["state_low"],
            state_high=case_spec["dp_kwargs"]["state_high"],
            state_points=case_spec["dp_kwargs"]["state_points"],
            state_abs_limit=case_spec["env_kwargs"]["state_abs_limit"],
            terminal_penalty=case_spec["env_kwargs"]["terminal_penalty"],
        )
        start = time.perf_counter()
        dp_solution = solve_finite_horizon_dp(model, dp_config, shock_nodes, shock_weights)
        solve_seconds = time.perf_counter() - start
        dp_policy = dp_solution.policy()

        eval_seed = 9800
        results = {
            "finite_horizon_dp": evaluate_policy(env, dp_policy, 32, config.discount_factor, eval_seed),
            "riccati_reference": evaluate_policy(LQBenchmarkEnv(model, BenchmarkEnvConfig(**case_spec["env_kwargs"])), policies["riccati_policy"], 32, config.discount_factor, eval_seed),
            "linear_policy_search": evaluate_policy(LQBenchmarkEnv(model, BenchmarkEnvConfig(**case_spec["env_kwargs"])), policies["linear_search_policy"], 32, config.discount_factor, eval_seed),
            "best_rl_surrogate": evaluate_policy(LQBenchmarkEnv(model, BenchmarkEnvConfig(**case_spec["env_kwargs"])), policies["best_rl_policy"], 32, config.discount_factor, eval_seed),
        }

        comparison_rows = []
        for name, stats in results.items():
            comparison_rows.append(
                {
                    "env_id": case_id,
                    "policy_name": name,
                    "mean_discounted_loss": stats["mean_discounted_loss"],
                    "std_discounted_loss": stats["std_discounted_loss"],
                    "mean_abs_action": stats["mean_abs_action"],
                    "clip_rate": stats["clip_rate"],
                    "explosion_rate": stats["explosion_rate"],
                    "solve_seconds": solve_seconds if name == "finite_horizon_dp" else np.nan,
                }
            )
        comparison_df = pd.DataFrame(comparison_rows)
        dp_loss = float(comparison_df.loc[comparison_df["policy_name"] == "finite_horizon_dp", "mean_discounted_loss"].iloc[0])
        comparison_df["gap_vs_dp_pct"] = (comparison_df["mean_discounted_loss"] / dp_loss - 1.0) * 100.0
        comparison_df.to_csv(case_dir / "comparison.csv", index=False)

        coeff_rows.append(
            {
                "env_id": case_id,
                **fit_linear_policy_response(
                    "finite_horizon_dp",
                    dp_policy,
                    case_spec["env_kwargs"]["initial_state_low"],
                    case_spec["env_kwargs"]["initial_state_high"],
                ),
            }
        )

        rng = np.random.default_rng(20282400 + idx)
        common_df = simulate_with_common_shocks(
            model=model,
            policy_map={
                "finite_horizon_dp": dp_policy,
                "riccati_reference": policies["riccati_policy"],
                "best_rl_surrogate": policies["best_rl_policy"],
            },
            initial_state=np.asarray(case_spec["common_initial_state"], dtype=float),
            shocks=rng.standard_normal((case_spec["env_kwargs"]["horizon"], model.state_dim)),
            action_low=case_spec["env_kwargs"]["action_low"],
            action_high=case_spec["env_kwargs"]["action_high"],
        )
        common_df.to_csv(case_dir / "common_shock_trajectories.csv", index=False)
        (case_dir / "run_metadata.json").write_text(
            json.dumps({"solve_seconds": solve_seconds, "dp_config": case_spec["dp_kwargs"]}, indent=2),
            encoding="utf-8",
        )

        summary_rows.append(
            {
                "env_id": case_id,
                "group": case_spec["group"],
                "tier": case_spec["tier"],
                "dp_mean_discounted_loss": dp_loss,
                "riccati_re_eval_mean_discounted_loss": results["riccati_reference"]["mean_discounted_loss"],
                "linear_search_re_eval_mean_discounted_loss": results["linear_policy_search"]["mean_discounted_loss"],
                "best_rl_algo": policies["best_rl"]["algo"],
                "best_rl_seed": int(policies["best_rl"]["seed"]),
                "best_rl_surrogate_re_eval_mean_discounted_loss": results["best_rl_surrogate"]["mean_discounted_loss"],
                "dp_improvement_vs_riccati_pct": (results["riccati_reference"]["mean_discounted_loss"] / dp_loss - 1.0) * 100.0,
                "dp_improvement_vs_best_rl_surrogate_pct": (results["best_rl_surrogate"]["mean_discounted_loss"] / dp_loss - 1.0) * 100.0,
                "solve_seconds": solve_seconds,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    pd.DataFrame(coeff_rows).to_csv(OUTPUT_DIR / "policy_coefficients.csv", index=False)

    lines = []
    lines.append("# Phase 11 v2 数值求解对照")
    lines.append("")
    lines.append("## 总表")
    lines.append("")
    lines.append(summary_df.round(6).to_markdown(index=False))
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append("- 本表只覆盖 `phase11 v2` 六个新增环境。")
    lines.append("- 若 `dp_improvement_vs_riccati_pct` 为正，则说明传统 benchmark Riccati 外推已被环境内数值解压过。")
    lines.append("- 若 `dp_improvement_vs_best_rl_surrogate_pct` 为正，则说明数值最优仍优于 RL 线性 surrogate。")
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
