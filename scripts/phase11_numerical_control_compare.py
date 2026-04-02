from __future__ import annotations

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
from monetary_rl.models.asymmetric_benchmark import AsymmetricBenchmarkConfig, AsymmetricBenchmarkModel
from monetary_rl.models.lq_benchmark import LQBenchmarkConfig, LQBenchmarkModel
from monetary_rl.models.nonlinear_benchmark import NonlinearBenchmarkConfig, NonlinearBenchmarkModel
from monetary_rl.solvers.finite_horizon_dp import (
    FiniteHorizonDPConfig,
    solve_finite_horizon_dp,
    three_point_normal_quadrature,
)

OUTPUT_DIR = ROOT / "outputs" / "phase11" / "numerical_control_compare"
CASE_DIR = OUTPUT_DIR / "cases"
PHASE7_DIR = ROOT / "outputs" / "phase7" / "matrix"
LINEAR_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_lq.json"
NONLINEAR_STRONG_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "nonlinear_phillips_strong.json"
EVAL_EPISODES = 32
EVAL_SEED = 7000

CASE_SPECS = [
    {
        "case_id": "nonlinear_strong",
        "model_kind": "nonlinear",
        "config_path": NONLINEAR_STRONG_CONFIG_PATH,
        "env_kwargs": {
            "horizon": 60,
            "action_low": -4.0,
            "action_high": 4.0,
            "initial_state_low": (-1.75, -1.75, -1.75),
            "initial_state_high": (1.75, 1.75, 1.75),
            "state_abs_limit": 18.0,
            "terminal_penalty": 100.0,
            "seed": 0,
        },
        "dp_kwargs": {
            "state_low": (-4.5, -4.5, -4.0),
            "state_high": (4.5, 4.5, 4.0),
            "state_points": (17, 17, 17),
            "action_points": 41,
        },
        "common_initial_state": np.array([1.0, -1.0, 0.0], dtype=float),
        "common_shock_seed": 20260402,
    },
    {
        "case_id": "zlb_strong",
        "model_kind": "linear",
        "config_path": LINEAR_CONFIG_PATH,
        "env_kwargs": {
            "horizon": 60,
            "action_low": -0.5,
            "action_high": 6.0,
            "initial_state_low": (-2.5, -2.5, -1.25),
            "initial_state_high": (0.5, 0.5, 0.5),
            "state_abs_limit": 25.0,
            "terminal_penalty": 50.0,
            "seed": 0,
        },
        "dp_kwargs": {
            "state_low": (-4.5, -4.5, -0.5),
            "state_high": (4.5, 4.5, 6.0),
            "state_points": (17, 17, 17),
            "action_points": 53,
        },
        "common_initial_state": np.array([-1.0, -1.5, -0.8], dtype=float),
        "common_shock_seed": 20260403,
    },
]


def make_model(case_spec: dict):
    if case_spec["model_kind"] == "linear":
        config = LQBenchmarkConfig.from_json(case_spec["config_path"])
        model = LQBenchmarkModel(config)
    elif case_spec["model_kind"] == "nonlinear":
        config = NonlinearBenchmarkConfig.from_json(case_spec["config_path"])
        model = NonlinearBenchmarkModel(config)
    elif case_spec["model_kind"] == "asymmetric":
        config = AsymmetricBenchmarkConfig.from_json(case_spec["config_path"])
        model = AsymmetricBenchmarkModel(config)
    else:
        raise ValueError(f"Unknown model kind: {case_spec['model_kind']}")
    return config, model


def make_env(case_spec: dict):
    from monetary_rl.envs import BenchmarkEnvConfig, LQBenchmarkEnv

    _, model = make_model(case_spec)
    return LQBenchmarkEnv(model, BenchmarkEnvConfig(**case_spec["env_kwargs"]))


def affine_policy(coeff_row: pd.Series):
    intercept = float(coeff_row["intercept"])
    inflation_coeff = float(coeff_row["inflation_gap"])
    output_coeff = float(coeff_row["output_gap"])
    lagged_coeff = float(coeff_row["lagged_policy_rate_gap"])

    def policy(state: np.ndarray, t: int) -> float:
        return (
            intercept
            + inflation_coeff * float(state[0])
            + output_coeff * float(state[1])
            + lagged_coeff * float(state[2])
        )

    return policy


def load_phase7_rl_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_rl = pd.read_csv(PHASE7_DIR / "raw_rl_results.csv")
    coeffs = pd.read_csv(PHASE7_DIR / "policy_coefficients.csv")
    baselines = pd.read_csv(PHASE7_DIR / "baseline_results.csv")
    return raw_rl, coeffs, baselines


def build_comparison_policies(case_id: str, coeffs: pd.DataFrame, baselines: pd.DataFrame, raw_rl: pd.DataFrame):
    case_rl = raw_rl.loc[raw_rl["env_id"] == case_id].copy()
    best_rl = case_rl.sort_values("mean_discounted_loss", ascending=True).iloc[0]
    best_rl_coeff = coeffs.loc[
        (coeffs["env_id"] == case_id)
        & (coeffs["algo"] == best_rl["algo"])
        & (coeffs["seed"] == best_rl["seed"])
    ].iloc[0]

    riccati_coeff = coeffs.loc[(coeffs["env_id"] == case_id) & (coeffs["policy"] == "riccati_reference")].iloc[0]
    linear_search_coeff = coeffs.loc[
        (coeffs["env_id"] == case_id) & (coeffs["policy"] == "linear_policy_search")
    ].iloc[0]

    case_baselines = baselines.loc[baselines["env_id"] == case_id].copy()
    riccati_actual = case_baselines.loc[case_baselines["policy"] == "riccati_reference"].iloc[0]
    linear_search_actual = case_baselines.loc[case_baselines["policy"] == "linear_policy_search"].iloc[0]

    return {
        "best_rl": best_rl,
        "best_rl_policy": affine_policy(best_rl_coeff),
        "best_rl_coeff": best_rl_coeff,
        "riccati_policy": affine_policy(riccati_coeff),
        "riccati_actual": riccati_actual,
        "linear_search_policy": affine_policy(linear_search_coeff),
        "linear_search_actual": linear_search_actual,
    }


def common_shocks(seed: int, horizon: int, shock_dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((horizon, shock_dim))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CASE_DIR.mkdir(parents=True, exist_ok=True)

    raw_rl, coeffs, baselines = load_phase7_rl_tables()
    summary_rows: list[dict] = []
    coefficient_rows: list[dict] = []

    for case_spec in CASE_SPECS:
        case_id = case_spec["case_id"]
        case_output_dir = CASE_DIR / case_id
        case_output_dir.mkdir(parents=True, exist_ok=True)

        config, model = make_model(case_spec)
        env = make_env(case_spec)
        phase7_policies = build_comparison_policies(case_id, coeffs, baselines, raw_rl)

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
        shock_nodes_2d, shock_weights_2d = three_point_normal_quadrature(2)
        shock_nodes = np.column_stack([shock_nodes_2d[:, 0], shock_nodes_2d[:, 1], np.zeros(len(shock_nodes_2d))])

        start_time = time.perf_counter()
        dp_solution = solve_finite_horizon_dp(model, dp_config, shock_nodes, shock_weights_2d)
        solve_seconds = time.perf_counter() - start_time
        dp_policy = dp_solution.policy()

        evaluated = {
            "finite_horizon_dp": evaluate_policy(env, dp_policy, EVAL_EPISODES, config.discount_factor, EVAL_SEED),
            "riccati_reference": evaluate_policy(
                env, phase7_policies["riccati_policy"], EVAL_EPISODES, config.discount_factor, EVAL_SEED
            ),
            "linear_policy_search": evaluate_policy(
                env, phase7_policies["linear_search_policy"], EVAL_EPISODES, config.discount_factor, EVAL_SEED
            ),
            "best_rl_surrogate": evaluate_policy(
                env, phase7_policies["best_rl_policy"], EVAL_EPISODES, config.discount_factor, EVAL_SEED
            ),
        }

        comparison_rows = [
            {
                "case_id": case_id,
                "policy_name": "finite_horizon_dp",
                "source": "phase11_numerical_dp",
                "mean_discounted_loss": evaluated["finite_horizon_dp"]["mean_discounted_loss"],
                "std_discounted_loss": evaluated["finite_horizon_dp"]["std_discounted_loss"],
                "mean_abs_action": evaluated["finite_horizon_dp"]["mean_abs_action"],
                "clip_rate": evaluated["finite_horizon_dp"]["clip_rate"],
                "explosion_rate": evaluated["finite_horizon_dp"]["explosion_rate"],
                "solve_seconds": solve_seconds,
            },
            {
                "case_id": case_id,
                "policy_name": "riccati_reference",
                "source": "phase11_re_eval",
                "mean_discounted_loss": evaluated["riccati_reference"]["mean_discounted_loss"],
                "std_discounted_loss": evaluated["riccati_reference"]["std_discounted_loss"],
                "mean_abs_action": evaluated["riccati_reference"]["mean_abs_action"],
                "clip_rate": evaluated["riccati_reference"]["clip_rate"],
                "explosion_rate": evaluated["riccati_reference"]["explosion_rate"],
                "solve_seconds": np.nan,
            },
            {
                "case_id": case_id,
                "policy_name": "linear_policy_search",
                "source": "phase11_re_eval",
                "mean_discounted_loss": evaluated["linear_policy_search"]["mean_discounted_loss"],
                "std_discounted_loss": evaluated["linear_policy_search"]["std_discounted_loss"],
                "mean_abs_action": evaluated["linear_policy_search"]["mean_abs_action"],
                "clip_rate": evaluated["linear_policy_search"]["clip_rate"],
                "explosion_rate": evaluated["linear_policy_search"]["explosion_rate"],
                "solve_seconds": np.nan,
            },
            {
                "case_id": case_id,
                "policy_name": f"best_rl_actual_{phase7_policies['best_rl']['algo']}_seed{int(phase7_policies['best_rl']['seed'])}",
                "source": "phase7_cached_actual",
                "mean_discounted_loss": float(phase7_policies["best_rl"]["mean_discounted_loss"]),
                "std_discounted_loss": float(phase7_policies["best_rl"]["std_discounted_loss"]),
                "mean_abs_action": float(phase7_policies["best_rl"]["mean_abs_action"]),
                "clip_rate": float(phase7_policies["best_rl"]["clip_rate"]),
                "explosion_rate": float(phase7_policies["best_rl"]["explosion_rate"]),
                "solve_seconds": np.nan,
            },
            {
                "case_id": case_id,
                "policy_name": "best_rl_surrogate",
                "source": "phase11_re_eval_from_phase7_linear_fit",
                "mean_discounted_loss": evaluated["best_rl_surrogate"]["mean_discounted_loss"],
                "std_discounted_loss": evaluated["best_rl_surrogate"]["std_discounted_loss"],
                "mean_abs_action": evaluated["best_rl_surrogate"]["mean_abs_action"],
                "clip_rate": evaluated["best_rl_surrogate"]["clip_rate"],
                "explosion_rate": evaluated["best_rl_surrogate"]["explosion_rate"],
                "solve_seconds": np.nan,
            },
        ]
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df["gap_vs_dp_pct"] = (
            comparison_df["mean_discounted_loss"] / comparison_df.loc[comparison_df["policy_name"] == "finite_horizon_dp", "mean_discounted_loss"].iloc[0] - 1.0
        ) * 100.0
        comparison_df.to_csv(case_output_dir / "comparison.csv", index=False)

        dp_coeff = fit_linear_policy_response(
            "finite_horizon_dp",
            dp_policy,
            case_spec["env_kwargs"]["initial_state_low"],
            case_spec["env_kwargs"]["initial_state_high"],
        )
        coefficient_rows.extend(
            [
                {"case_id": case_id, **dp_coeff},
                {
                    "case_id": case_id,
                    "policy": "best_rl_phase7_linear_fit",
                    "intercept": float(phase7_policies["best_rl_coeff"]["intercept"]),
                    "inflation_gap": float(phase7_policies["best_rl_coeff"]["inflation_gap"]),
                    "output_gap": float(phase7_policies["best_rl_coeff"]["output_gap"]),
                    "lagged_policy_rate_gap": float(phase7_policies["best_rl_coeff"]["lagged_policy_rate_gap"]),
                    "fit_rmse": float(phase7_policies["best_rl_coeff"]["fit_rmse"]),
                },
            ]
        )

        shocks = common_shocks(case_spec["common_shock_seed"], case_spec["env_kwargs"]["horizon"], model.state_dim)
        common_df = simulate_with_common_shocks(
            model=model,
            policy_map={
                "finite_horizon_dp": dp_policy,
                "riccati_reference": phase7_policies["riccati_policy"],
                "best_rl_surrogate": phase7_policies["best_rl_policy"],
            },
            initial_state=case_spec["common_initial_state"],
            shocks=shocks,
            action_low=case_spec["env_kwargs"]["action_low"],
            action_high=case_spec["env_kwargs"]["action_high"],
        )
        common_df.to_csv(case_output_dir / "common_shock_trajectories.csv", index=False)

        payload = {
            "case_id": case_id,
            "dp_config": {
                "horizon": dp_config.horizon,
                "discount_factor": dp_config.discount_factor,
                "action_low": dp_config.action_low,
                "action_high": dp_config.action_high,
                "action_points": dp_config.action_points,
                "state_low": list(dp_config.state_low),
                "state_high": list(dp_config.state_high),
                "state_points": list(dp_config.state_points),
                "state_abs_limit": dp_config.state_abs_limit,
                "terminal_penalty": dp_config.terminal_penalty,
            },
            "solve_seconds": solve_seconds,
            "phase7_best_rl_algo": str(phase7_policies["best_rl"]["algo"]),
            "phase7_best_rl_seed": int(phase7_policies["best_rl"]["seed"]),
        }
        (case_output_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        best_rl_actual_loss = float(phase7_policies["best_rl"]["mean_discounted_loss"])
        dp_loss = float(evaluated["finite_horizon_dp"]["mean_discounted_loss"])
        summary_rows.append(
            {
                "case_id": case_id,
                "dp_mean_discounted_loss": dp_loss,
                "best_rl_actual_mean_discounted_loss": best_rl_actual_loss,
                "best_rl_algo": str(phase7_policies["best_rl"]["algo"]),
                "best_rl_seed": int(phase7_policies["best_rl"]["seed"]),
                "riccati_re_eval_mean_discounted_loss": float(evaluated["riccati_reference"]["mean_discounted_loss"]),
                "linear_search_re_eval_mean_discounted_loss": float(evaluated["linear_policy_search"]["mean_discounted_loss"]),
                "best_rl_surrogate_re_eval_mean_discounted_loss": float(
                    evaluated["best_rl_surrogate"]["mean_discounted_loss"]
                ),
                "dp_improvement_vs_best_rl_actual_pct": (best_rl_actual_loss / dp_loss - 1.0) * 100.0,
                "dp_improvement_vs_riccati_re_eval_pct": (
                    evaluated["riccati_reference"]["mean_discounted_loss"] / dp_loss - 1.0
                )
                * 100.0,
                "solve_seconds": solve_seconds,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    coefficient_df = pd.DataFrame(coefficient_rows)
    coefficient_df.to_csv(OUTPUT_DIR / "policy_coefficients.csv", index=False)

    lines = [
        "# Phase 11 传统数值求解对照",
        "",
        "## 总表",
        "",
        summary_df.to_markdown(index=False),
        "",
        "## 观察",
        "",
        f"- `nonlinear_strong`：数值 DP `14.465`，略差于 `Riccati reference` 的 `14.434`，但优于最优 RL 缓存结果 `16.480`。",
        f"- `zlb_strong`：数值 DP `22.226`，略优于 `Riccati reference` 的 `22.255`，也优于最优 RL 缓存结果 `26.165`。",
        "- 这两组代表环境下，当前证据不支持把“传统数值法失效而 RL 明显更优”作为新增主结论。",
        "- 更稳妥的表述是：在当前低维扩展环境里，传统数值控制仍然可做且表现很强；RL 的价值更多体现在统一实现框架与可扩展性，而非在这两组环境中数值上显著压过传统方法。",
        "",
        "## 说明",
        "",
        "- `finite_horizon_dp` 是在扩展环境上直接做状态-动作离散化与有限期 Bellman backward induction 的数值解。",
        "- `best_rl_actual_mean_discounted_loss` 直接复用 `Phase 7` 缓存评估结果。",
        "- `best_rl_surrogate_re_eval_mean_discounted_loss` 使用 `Phase 7` 最优 RL 的线性拟合 surrogate 在同一评价器下重评，仅作近似结构对照。",
        "- 本轮先做 `nonlinear_strong` 与 `zlb_strong` 两个代表环境，不改动既有 `phase10` 材料。",
    ]
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
