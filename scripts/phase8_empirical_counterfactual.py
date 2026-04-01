from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monetary_rl.envs import EmpiricalEnvConfig, EmpiricalSVAREnv
from monetary_rl.evaluation import evaluate_policy
from monetary_rl.experiment_utils import build_taylor_gap_policy, load_taylor_rule
from monetary_rl.models import EmpiricalSVARConfig, EmpiricalSVARModel, LQBenchmarkConfig, LQBenchmarkModel
from monetary_rl.solvers import build_optimal_linear_policy, solve_discounted_lq_riccati


DATA_PATH = ROOT / "data" / "processed" / "macro_quarterly_sample_1987Q3_2007Q2.csv"
PHASE2_DIR = ROOT / "outputs" / "phase2"
PHASE6_DIR = ROOT / "outputs" / "phase6"
PHASE7_DIR = ROOT / "outputs" / "phase7" / "matrix"
LINEAR_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_lq.json"
OUTPUT_DIR = ROOT / "outputs" / "phase8"
PLOTS_DIR = OUTPUT_DIR / "plots"

HISTORICAL_HORIZON = 77
STOCHASTIC_HORIZON = 80
STOCHASTIC_EPISODES = 256
STOCHASTIC_SEED = 20260401


def add_lags(df: pd.DataFrame, columns: list[str], max_lag: int) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        for lag in range(1, max_lag + 1):
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_empirical_model() -> tuple[EmpiricalSVARModel, pd.DataFrame, np.ndarray]:
    benchmark_config = LQBenchmarkConfig.from_json(LINEAR_CONFIG_PATH)
    macro = pd.read_csv(DATA_PATH, parse_dates=["date"])
    env_df = add_lags(macro, ["inflation", "output_gap", "policy_rate"], max_lag=2).dropna().reset_index(drop=True)

    svar_output = load_json(PHASE2_DIR / "svar_output_gap.json")
    svar_inflation = load_json(PHASE2_DIR / "svar_inflation.json")
    output_fit = pd.read_csv(PHASE2_DIR / "svar_output_gap_fitted.csv")
    inflation_fit = pd.read_csv(PHASE2_DIR / "svar_inflation_fitted.csv")

    if len(env_df) != len(output_fit) or len(env_df) != len(inflation_fit):
        raise ValueError("Phase 2 fitted outputs and lagged empirical frame are misaligned.")

    config = EmpiricalSVARConfig(
        name="empirical_recursive_svar",
        inflation_target=benchmark_config.inflation_target,
        neutral_rate=benchmark_config.neutral_rate,
        discount_factor=benchmark_config.discount_factor,
        loss_weights=benchmark_config.loss_weights,
        output_gap_coefficients={k: float(v) for k, v in svar_output["coefficients"].items()},
        inflation_coefficients={k: float(v) for k, v in svar_inflation["coefficients"].items()},
        sample_start=str(env_df["quarter"].iloc[0]),
        sample_end=str(env_df["quarter"].iloc[-1]),
        action_low=-2.0,
        action_high=8.0,
    )
    model = EmpiricalSVARModel(config)
    shocks = np.column_stack([output_fit["output_gap_resid"].to_numpy(), inflation_fit["inflation_resid"].to_numpy()])
    return model, env_df, shocks


def make_initial_states(env_df: pd.DataFrame) -> np.ndarray:
    return env_df[["inflation", "inflation_lag1", "output_gap", "output_gap_lag1", "policy_rate_lag1"]].to_numpy(dtype=float)


def build_linear_policy(name: str, intercept: float, inflation_gap: float, output_gap: float, lagged_policy_rate_gap: float):
    def policy(state: np.ndarray, t: int) -> float:
        del t
        pi_gap, output_gap_now, lagged_rate_gap = np.asarray(state, dtype=float)
        return (
            intercept
            + inflation_gap * pi_gap
            + output_gap * output_gap_now
            + lagged_policy_rate_gap * lagged_rate_gap
        )

    policy.__name__ = name
    return policy


def load_phase6_linear_policy() -> tuple[callable, dict]:
    coeff_df = pd.read_csv(PHASE6_DIR / "policy_coefficients.csv")
    row = coeff_df.loc[coeff_df["policy"] == "linear_policy_search"].iloc[0]
    payload = {
        "policy_name": "linear_policy_search_transfer",
        "rule_type": "linear",
        "source": "phase6",
        "intercept": float(row["intercept"]),
        "inflation_gap": float(row["inflation_gap"]),
        "output_gap": float(row["output_gap"]),
        "lagged_policy_rate_gap": float(row["lagged_policy_rate_gap"]),
        "fit_rmse": float(row["fit_rmse"]),
        "note": "Phase 6 benchmark linear policy search coefficients",
    }
    return build_linear_policy(
        payload["policy_name"],
        payload["intercept"],
        payload["inflation_gap"],
        payload["output_gap"],
        payload["lagged_policy_rate_gap"],
    ), payload


def load_best_benchmark_rl_policies() -> tuple[dict[str, callable], list[dict]]:
    raw_df = pd.read_csv(PHASE7_DIR / "raw_rl_results.csv")
    coeff_df = pd.read_csv(PHASE7_DIR / "policy_coefficients.csv")
    policies: dict[str, callable] = {}
    registry_rows: list[dict] = []

    for algo in ["ppo", "td3", "sac"]:
        best_row = raw_df[(raw_df["env_id"] == "benchmark") & (raw_df["algo"] == algo)].sort_values("mean_discounted_loss").iloc[0]
        coeff_row = coeff_df[
            (coeff_df["env_id"] == "benchmark")
            & (coeff_df["algo"] == algo)
            & (coeff_df["seed"] == best_row["seed"])
        ].iloc[0]
        policy_name = f"{algo}_benchmark_transfer"
        registry = {
            "policy_name": policy_name,
            "rule_type": "linear_surrogate",
            "source": "phase7_benchmark_best_seed",
            "algo": algo,
            "seed": int(best_row["seed"]),
            "intercept": float(coeff_row["intercept"]),
            "inflation_gap": float(coeff_row["inflation_gap"]),
            "output_gap": float(coeff_row["output_gap"]),
            "lagged_policy_rate_gap": float(coeff_row["lagged_policy_rate_gap"]),
            "fit_rmse": float(coeff_row["fit_rmse"]),
            "benchmark_mean_discounted_loss": float(best_row["mean_discounted_loss"]),
            "note": "Transferred benchmark-trained RL rule represented by saved linear surrogate",
        }
        policies[policy_name] = build_linear_policy(
            policy_name,
            registry["intercept"],
            registry["inflation_gap"],
            registry["output_gap"],
            registry["lagged_policy_rate_gap"],
        )
        registry_rows.append(registry)

    return policies, registry_rows


def build_policy_registry(model: EmpiricalSVARModel, env_df: pd.DataFrame) -> tuple[dict[str, callable], pd.DataFrame]:
    benchmark_config = LQBenchmarkConfig.from_json(LINEAR_CONFIG_PATH)
    taylor_rule = load_taylor_rule(PHASE2_DIR / "taylor_rule.json")
    taylor_policy, taylor_intercept = build_taylor_gap_policy(taylor_rule, benchmark_config)

    linear_model = LQBenchmarkModel(benchmark_config)
    riccati_solution = solve_discounted_lq_riccati(linear_model)
    riccati_policy = build_optimal_linear_policy(riccati_solution)

    actual_gaps = env_df["policy_rate"].to_numpy(dtype=float) - model.config.neutral_rate

    def historical_actual_policy(state: np.ndarray, t: int) -> float:
        del state
        idx = min(t, len(actual_gaps) - 1)
        return float(actual_gaps[idx])

    policies: dict[str, callable] = {
        "historical_actual_policy": historical_actual_policy,
        "empirical_taylor_rule": taylor_policy,
        "riccati_reference": riccati_policy,
    }
    registry_rows = [
        {
            "policy_name": "historical_actual_policy",
            "rule_type": "historical_path",
            "source": "data",
            "intercept": np.nan,
            "inflation_gap": np.nan,
            "output_gap": np.nan,
            "lagged_policy_rate_gap": np.nan,
            "fit_rmse": np.nan,
            "note": "Observed policy-rate path in the sample",
        },
        {
            "policy_name": "empirical_taylor_rule",
            "rule_type": "estimated_rule",
            "source": "phase2",
            "intercept": float(taylor_intercept),
            "inflation_gap": float(taylor_rule["phi_pi"]),
            "output_gap": float(taylor_rule["phi_x"]),
            "lagged_policy_rate_gap": float(taylor_rule["phi_i"]),
            "fit_rmse": 0.0,
            "note": "Estimated Taylor rule converted to gap form",
        },
        {
            "policy_name": "riccati_reference",
            "rule_type": "theory_reference",
            "source": "phase4",
            "intercept": 0.0,
            "inflation_gap": float(riccati_solution.K[0, 0]),
            "output_gap": float(riccati_solution.K[0, 1]),
            "lagged_policy_rate_gap": float(riccati_solution.K[0, 2]),
            "fit_rmse": 0.0,
            "note": "Theoretical Riccati benchmark rule",
        },
    ]

    linear_policy, linear_registry = load_phase6_linear_policy()
    policies[linear_registry["policy_name"]] = linear_policy
    registry_rows.append(linear_registry)

    rl_policies, rl_registry_rows = load_best_benchmark_rl_policies()
    policies.update(rl_policies)
    registry_rows.extend(rl_registry_rows)
    return policies, pd.DataFrame(registry_rows)


def simulate_historical_counterfactual(
    model: EmpiricalSVARModel,
    policy_map: dict[str, callable],
    initial_state: np.ndarray,
    action_dates: pd.Series,
    state_dates: pd.Series,
    shocks: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict] = []
    for policy_name, policy_fn in policy_map.items():
        full_state = np.asarray(initial_state, dtype=float).copy()
        for t, action_date in enumerate(action_dates):
            obs = model.observe(full_state)
            raw_action = float(policy_fn(obs.copy(), t))
            action_gap = float(np.clip(raw_action, model.config.action_low, model.config.action_high))
            action_level = model.action_to_level(action_gap)
            loss = float(model.stage_loss(full_state, action_gap))
            inflation_gap, output_gap, lagged_rate_gap = obs
            rows.append(
                {
                    "policy_name": policy_name,
                    "action_date": pd.Timestamp(action_date),
                    "state_date": pd.Timestamp(state_dates.iloc[t]),
                    "inflation": float(full_state[0]),
                    "inflation_gap": float(inflation_gap),
                    "output_gap": float(output_gap),
                    "lagged_policy_rate": float(full_state[4]),
                    "lagged_policy_rate_gap": float(lagged_rate_gap),
                    "policy_rate": action_level,
                    "policy_rate_gap": action_gap,
                    "rate_change": action_level - float(full_state[4]),
                    "loss": loss,
                    "discounted_loss": loss * (model.config.discount_factor ** t),
                }
            )
            full_state = model.state_transition(full_state, action_gap, shocks[t])

        final_obs = model.observe(full_state)
        rows.append(
            {
                "policy_name": policy_name,
                "action_date": pd.NaT,
                "state_date": pd.Timestamp(state_dates.iloc[-1]),
                "inflation": float(full_state[0]),
                "inflation_gap": float(final_obs[0]),
                "output_gap": float(final_obs[1]),
                "lagged_policy_rate": float(full_state[4]),
                "lagged_policy_rate_gap": float(final_obs[2]),
                "policy_rate": np.nan,
                "policy_rate_gap": np.nan,
                "rate_change": np.nan,
                "loss": np.nan,
                "discounted_loss": np.nan,
            }
        )
    return pd.DataFrame(rows)


def historical_summary(path_df: pd.DataFrame) -> pd.DataFrame:
    action_df = path_df.dropna(subset=["policy_rate"]).copy()
    summary = (
        action_df.groupby("policy_name")
        .agg(
            total_discounted_loss=("discounted_loss", "sum"),
            mean_period_loss=("loss", "mean"),
            mean_sq_inflation_gap=("inflation_gap", lambda s: float(np.mean(np.square(s)))),
            mean_sq_output_gap=("output_gap", lambda s: float(np.mean(np.square(s)))),
            mean_sq_rate_change=("rate_change", lambda s: float(np.mean(np.square(s)))),
            mean_policy_rate=("policy_rate", "mean"),
            std_policy_rate=("policy_rate", "std"),
        )
        .reset_index()
    )
    baseline_actual = float(summary.loc[summary["policy_name"] == "historical_actual_policy", "total_discounted_loss"].iloc[0])
    baseline_taylor = float(summary.loc[summary["policy_name"] == "empirical_taylor_rule", "total_discounted_loss"].iloc[0])
    summary["improvement_vs_actual_pct"] = (baseline_actual - summary["total_discounted_loss"]) / baseline_actual * 100.0
    summary["improvement_vs_taylor_pct"] = (baseline_taylor - summary["total_discounted_loss"]) / baseline_taylor * 100.0
    return summary.sort_values("total_discounted_loss").reset_index(drop=True)


def evaluate_stochastic_policies(
    model: EmpiricalSVARModel,
    initial_states: np.ndarray,
    shock_pool: np.ndarray,
    policies: dict[str, callable],
) -> pd.DataFrame:
    eval_rows: list[dict] = []
    for idx, (policy_name, policy_fn) in enumerate(policies.items()):
        env = EmpiricalSVAREnv(
            model=model,
            initial_states=initial_states,
            shock_pool=shock_pool,
            config=EmpiricalEnvConfig(
                horizon=STOCHASTIC_HORIZON,
                action_low=model.config.action_low,
                action_high=model.config.action_high,
                seed=STOCHASTIC_SEED + idx,
            ),
        )
        stats = evaluate_policy(
            env=env,
            policy_fn=policy_fn,
            episodes=STOCHASTIC_EPISODES,
            gamma=model.config.discount_factor,
            seed=STOCHASTIC_SEED + 100 * (idx + 1),
        )
        eval_rows.append(
            {
                "policy_name": policy_name,
                "mean_discounted_loss": stats["mean_discounted_loss"],
                "std_discounted_loss": stats["std_discounted_loss"],
                "mean_reward": stats["mean_reward"],
                "mean_abs_action": stats["mean_abs_action"],
                "clip_rate": stats["clip_rate"],
                "explosion_rate": stats["explosion_rate"],
            }
        )
    return pd.DataFrame(eval_rows).sort_values("mean_discounted_loss").reset_index(drop=True)


def make_plots(path_df: pd.DataFrame, historical_df: pd.DataFrame, stochastic_df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    core_policies = [
        "historical_actual_policy",
        "empirical_taylor_rule",
        "riccati_reference",
        "sac_benchmark_transfer",
    ]
    core_path_df = path_df[path_df["policy_name"].isin(core_policies)].copy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for policy_name in core_policies:
        subset = core_path_df[core_path_df["policy_name"] == policy_name]
        axes[0].plot(subset["state_date"], subset["inflation"], label=policy_name)
        axes[1].plot(subset["state_date"], subset["output_gap"], label=policy_name)
        axes[2].plot(subset.dropna(subset=["policy_rate"])["action_date"], subset.dropna(subset=["policy_rate"])["policy_rate"], label=policy_name)
    axes[0].axhline(2.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Phase 8 Historical vs Counterfactual Paths: Inflation")
    axes[1].set_title("Phase 8 Historical vs Counterfactual Paths: Output Gap")
    axes[2].set_title("Phase 8 Historical vs Counterfactual Paths: Policy Rate")
    for ax in axes:
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase8_historical_paths_core.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(historical_df["policy_name"], historical_df["total_discounted_loss"])
    ax.set_title("Phase 8 Historical Counterfactual Welfare")
    ax.set_ylabel("discounted loss")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase8_historical_welfare.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = historical_df[["policy_name", "mean_sq_inflation_gap", "mean_sq_output_gap", "mean_sq_rate_change"]].set_index("policy_name")
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("Phase 8 Welfare Decomposition")
    ax.set_ylabel("mean squared metric")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase8_historical_decomposition.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(stochastic_df["policy_name"], stochastic_df["mean_discounted_loss"], yerr=stochastic_df["std_discounted_loss"])
    ax.set_title("Phase 8 Stochastic Evaluation: Feedback Rules")
    ax.set_ylabel("mean discounted loss")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase8_stochastic_welfare.png", dpi=200)
    plt.close(fig)


def write_summary(
    registry_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    stochastic_df: pd.DataFrame,
    ann_gate_df: pd.DataFrame,
    reproduction_error: float,
) -> None:
    lines: list[str] = []
    lines.append("# Phase 8 经验 SVAR 反事实总结")
    lines.append("")
    lines.append("## 1. 任务完成情况")
    lines.append("")
    lines.append("| 项目 | 结果 |")
    lines.append("|---|---|")
    lines.append("| 主环境 | `SVAR` |")
    lines.append("| 已比较对象 | `historical_actual_policy`、`empirical_taylor_rule`、`riccati_reference`、`linear_policy_search_transfer`、`ppo/td3/sac_benchmark_transfer` |")
    lines.append("| 历史反事实 | 已完成 |")
    lines.append("| 长期随机评估 | 已完成（仅反馈规则） |")
    lines.append("| ANN | 未进入主结果，保留到 `Phase 9` 门槛判断 |")
    lines.append("")

    lines.append("## 2. 历史反事实主表")
    lines.append("")
    lines.append(historical_df.round(6).to_markdown(index=False))
    lines.append("")

    lines.append("## 3. 长期随机评估")
    lines.append("")
    lines.append(stochastic_df.round(6).to_markdown(index=False))
    lines.append("")

    lines.append("## 4. 规则登记表")
    lines.append("")
    lines.append(registry_df.round(6).to_markdown(index=False))
    lines.append("")

    lines.append("## 5. Phase 9 门槛判断")
    lines.append("")
    lines.append(ann_gate_df.to_markdown(index=False))
    lines.append("")

    lines.append("## 6. 说明")
    lines.append("")
    lines.append(f"- 历史实际政策在长期随机评估中未纳入，因为它是样本路径而不是固定反馈规则。")
    lines.append(f"- benchmark 训练得到的 RL 规则在 `Phase 8` 中以已保存的线性 surrogate 形式迁入经验环境，而不是在 `SVAR` 环境中重新训练。")
    lines.append(f"- 历史实际政策复现实验的最大绝对误差为 `{reproduction_error:.10f}`，说明 recovered shocks 与递归 SVAR 转移实现是对齐的。")
    lines.append(f"- 写作中必须明确 `Lucas critique`：经验转移固定不变只是方法近似。")
    lines.append("")

    (OUTPUT_DIR / "phase8_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    model, env_df, shocks_all = build_empirical_model()
    initial_states = make_initial_states(env_df.iloc[:-1].copy())
    initial_state = initial_states[0].copy()
    action_dates = env_df["date"].iloc[:-1].reset_index(drop=True)
    state_dates = env_df["date"].iloc[: len(action_dates) + 1].reset_index(drop=True)
    historical_shocks = shocks_all[1:].copy()

    policy_map, registry_df = build_policy_registry(model, env_df.iloc[:-1].copy())

    historical_path_df = simulate_historical_counterfactual(
        model=model,
        policy_map=policy_map,
        initial_state=initial_state,
        action_dates=action_dates,
        state_dates=state_dates,
        shocks=historical_shocks,
    )
    historical_df = historical_summary(historical_path_df)

    feedback_policy_names = [
        "empirical_taylor_rule",
        "riccati_reference",
        "linear_policy_search_transfer",
        "ppo_benchmark_transfer",
        "td3_benchmark_transfer",
        "sac_benchmark_transfer",
    ]
    stochastic_df = evaluate_stochastic_policies(
        model=model,
        initial_states=initial_states,
        shock_pool=historical_shocks,
        policies={name: policy_map[name] for name in feedback_policy_names},
    )

    actual_subset = historical_path_df[(historical_path_df["policy_name"] == "historical_actual_policy") & historical_path_df["policy_rate"].notna()].copy()
    actual_future = env_df.iloc[1 : len(actual_subset) + 1].reset_index(drop=True)
    reproduction_error = max(
        float(np.max(np.abs(actual_subset["inflation"].to_numpy() - env_df.iloc[:-1]["inflation"].to_numpy()))),
        float(np.max(np.abs(actual_subset["output_gap"].to_numpy() - env_df.iloc[:-1]["output_gap"].to_numpy()))),
    )
    next_state_df = historical_path_df[(historical_path_df["policy_name"] == "historical_actual_policy") & historical_path_df["policy_rate"].isna().eq(False)].copy()
    if len(actual_future) > 0:
        simulated_next = simulate_historical_counterfactual(
            model=model,
            policy_map={"historical_actual_policy": policy_map["historical_actual_policy"]},
            initial_state=initial_state,
            action_dates=action_dates,
            state_dates=state_dates,
            shocks=historical_shocks,
        )
        simulated_future = simulated_next[(simulated_next["policy_name"] == "historical_actual_policy") & simulated_next["state_date"].isin(state_dates.iloc[1:])]
        reproduction_error = max(
            reproduction_error,
            float(np.max(np.abs(simulated_future["inflation"].to_numpy() - actual_future["inflation"].to_numpy()))),
            float(np.max(np.abs(simulated_future["output_gap"].to_numpy() - actual_future["output_gap"].to_numpy()))),
        )

    ann_gate_df = pd.DataFrame(
        [
            {
                "module": "ann_phase9_gate",
                "status": "not_passed_yet",
                "reason": "inflation equation still underperforms SVAR in Phase 2 summary",
                "next_step": "keep ANN as Phase 9 supplementary module after SVAR main results",
            },
            {
                "module": "dsge_phase9_gate",
                "status": "deferred",
                "reason": "model uncertainty extension is intentionally placed in Phase 9",
                "next_step": "prepare transferable linear rule registry and coefficient table",
            },
        ]
    )

    registry_df.to_csv(OUTPUT_DIR / "policy_registry.csv", index=False)
    historical_path_df.to_csv(OUTPUT_DIR / "historical_counterfactual_paths.csv", index=False)
    historical_df.to_csv(OUTPUT_DIR / "historical_welfare_summary.csv", index=False)
    stochastic_df.to_csv(OUTPUT_DIR / "stochastic_welfare_summary.csv", index=False)
    ann_gate_df.to_csv(OUTPUT_DIR / "phase9_gate_summary.csv", index=False)
    make_plots(historical_path_df, historical_df, stochastic_df)
    write_summary(registry_df, historical_df, stochastic_df, ann_gate_df, reproduction_error)


if __name__ == "__main__":
    main()
