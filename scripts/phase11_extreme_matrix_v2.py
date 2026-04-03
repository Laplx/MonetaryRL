from __future__ import annotations

import argparse
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
from monetary_rl.phase11_extreme_specs_v2 import (
    COMMON_SHOCK_HORIZON,
    EVAL_EPISODES,
    LINEAR_CONFIG_PATH,
    NEW_ENV_SPECS_V2,
    OFFPOLICY_OVERRIDES,
    PPO_CONFIG_PATH,
    PPO_OVERRIDES,
    SAC_CONFIG_PATH,
    SEEDS,
    TAYLOR_RULE_PATH,
    TD3_CONFIG_PATH,
    TIER_ORDER_V2,
    make_model,
)
from monetary_rl.solvers import build_optimal_linear_policy, solve_discounted_lq_riccati


OUTPUT_DIR = ROOT / "outputs" / "phase11" / "extreme_matrix_v2"
CACHE_DIR = OUTPUT_DIR / "cache"


def cache_file(spec: dict, algo: str, seed: int) -> Path:
    return CACHE_DIR / f"{spec['env_id']}__{algo}__seed{seed}.json"


def baseline_cache_file(spec: dict, name: str) -> Path:
    return CACHE_DIR / f"{spec['env_id']}__baseline__{name}.json"


def run_rl_one(spec: dict, algo: str, seed: int) -> dict:
    path = cache_file(spec, algo, seed)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    _, model = make_model(spec)
    env = LQBenchmarkEnv(model, BenchmarkEnvConfig(**spec["env_kwargs"]))
    eval_env = LQBenchmarkEnv(model, BenchmarkEnvConfig(**spec["env_kwargs"]))

    if algo == "ppo":
        result, policy_fn = run_ppo(env, PPO_CONFIG_PATH, EVAL_EPISODES, seed=seed, **PPO_OVERRIDES)
    elif algo == "sac":
        result, policy_fn = run_sac(env, SAC_CONFIG_PATH, EVAL_EPISODES, seed=seed, **OFFPOLICY_OVERRIDES)
    elif algo == "td3":
        result, policy_fn = run_td3(env, TD3_CONFIG_PATH, EVAL_EPISODES, seed=seed, **OFFPOLICY_OVERRIDES)
    else:
        raise ValueError(algo)

    row = policy_row(algo, eval_env, policy_fn, EVAL_EPISODES, 10_000 + seed * 100)
    coeff = fit_linear_policy_response(algo, policy_fn, eval_env.config.initial_state_low, eval_env.config.initial_state_high)
    training_log = training_log_frame(result, algo, seed).to_dict(orient="records")
    rng = np.random.default_rng(20271405 + seed)
    traj = simulate_with_common_shocks(
        model,
        {algo: policy_fn},
        initial_state=np.asarray(spec["common_initial_state"], dtype=float),
        shocks=rng.standard_normal((COMMON_SHOCK_HORIZON, model.state_dim)),
        action_low=eval_env.config.action_low,
        action_high=eval_env.config.action_high,
    ).to_dict(orient="records")

    payload = {
        "env_id": spec["env_id"],
        "group": spec["group"],
        "tier": spec["tier"],
        "algo": algo,
        "seed": seed,
        "row": row,
        "coeff": coeff,
        "training_log": training_log,
        "trajectory": traj,
        "config": result["config"],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def run_baselines_for_env(spec: dict, riccati_policy, riccati_k: np.ndarray) -> list[dict]:
    results = []
    config, model = make_model(spec)
    env = LQBenchmarkEnv(model, BenchmarkEnvConfig(**spec["env_kwargs"]))
    taylor_rule = load_taylor_rule(TAYLOR_RULE_PATH)
    taylor_policy, taylor_intercept = build_taylor_gap_policy(taylor_rule, config)

    baseline_specs = [
        ("zero_policy", zero_gap_policy),
        ("riccati_reference", riccati_policy),
        ("empirical_taylor", taylor_policy),
    ]

    linear_cache = baseline_cache_file(spec, "linear_policy_search")
    if linear_cache.exists():
        linear_payload = json.loads(linear_cache.read_text(encoding="utf-8"))
    else:
        linear_theta, linear_result, linear_policy = run_linear_search(
            LQBenchmarkEnv(model, BenchmarkEnvConfig(**spec["env_kwargs"])),
            EVAL_EPISODES,
            seed=123,
        )
        linear_row = policy_row("linear_policy_search", env, linear_policy, EVAL_EPISODES, 20_000)
        linear_coeff = {
            "policy": "linear_policy_search",
            "intercept": 0.0,
            "inflation_gap": float(linear_theta[0]),
            "output_gap": float(linear_theta[1]),
            "lagged_policy_rate_gap": float(linear_theta[2]),
            "fit_rmse": 0.0,
        }
        rng = np.random.default_rng(20271500)
        linear_traj = simulate_with_common_shocks(
            model,
            {"linear_policy_search": linear_policy},
            initial_state=np.asarray(spec["common_initial_state"], dtype=float),
            shocks=rng.standard_normal((COMMON_SHOCK_HORIZON, model.state_dim)),
            action_low=env.config.action_low,
            action_high=env.config.action_high,
        ).to_dict(orient="records")
        linear_payload = {
            "env_id": spec["env_id"],
            "group": spec["group"],
            "tier": spec["tier"],
            "row": linear_row,
            "coeff": linear_coeff,
            "trajectory": linear_traj,
            "config": linear_result["config"],
        }
        linear_cache.write_text(json.dumps(linear_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    results.append(linear_payload)

    seed_map = {"zero_policy": 1, "riccati_reference": 2, "empirical_taylor": 3}
    for name, policy_fn in baseline_specs:
        path = baseline_cache_file(spec, name)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            row = policy_row(name, env, policy_fn, EVAL_EPISODES, 20_000)
            if name == "riccati_reference":
                coeff = {
                    "policy": name,
                    "intercept": 0.0,
                    "inflation_gap": float(riccati_k[0, 0]),
                    "output_gap": float(riccati_k[0, 1]),
                    "lagged_policy_rate_gap": float(riccati_k[0, 2]),
                    "fit_rmse": 0.0,
                }
            elif name == "empirical_taylor":
                coeff = {
                    "policy": name,
                    "intercept": taylor_intercept,
                    "inflation_gap": taylor_rule["phi_pi"],
                    "output_gap": taylor_rule["phi_x"],
                    "lagged_policy_rate_gap": taylor_rule["phi_i"],
                    "fit_rmse": 0.0,
                }
            else:
                coeff = {
                    "policy": name,
                    "intercept": 0.0,
                    "inflation_gap": 0.0,
                    "output_gap": 0.0,
                    "lagged_policy_rate_gap": 0.0,
                    "fit_rmse": 0.0,
                }
            rng = np.random.default_rng(20271600 + seed_map[name])
            traj = simulate_with_common_shocks(
                model,
                {name: policy_fn},
                initial_state=np.asarray(spec["common_initial_state"], dtype=float),
                shocks=rng.standard_normal((COMMON_SHOCK_HORIZON, model.state_dim)),
                action_low=env.config.action_low,
                action_high=env.config.action_high,
            ).to_dict(orient="records")
            payload = {
                "env_id": spec["env_id"],
                "group": spec["group"],
                "tier": spec["tier"],
                "row": row,
                "coeff": coeff,
                "trajectory": traj,
            }
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        results.append(payload)
    return results


def load_all_payloads() -> tuple[list[dict], list[dict]]:
    rl_payloads = []
    baseline_payloads = []
    for path in CACHE_DIR.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "__seed" in path.stem:
            rl_payloads.append(payload)
        else:
            baseline_payloads.append(payload)
    return rl_payloads, baseline_payloads


def build_summary_tables(rl_payloads: list[dict], baseline_payloads: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rl_rows = []
    rl_coeff_rows = []
    training_rows = []
    for payload in rl_payloads:
        row = payload["row"].copy()
        row.update(
            {
                "env_id": payload["env_id"],
                "group": payload["group"],
                "tier": payload["tier"],
                "algo": payload["algo"],
                "seed": payload["seed"],
            }
        )
        rl_rows.append(row)
        coeff = payload["coeff"].copy()
        coeff.update(
            {
                "env_id": payload["env_id"],
                "group": payload["group"],
                "tier": payload["tier"],
                "algo": payload["algo"],
                "seed": payload["seed"],
            }
        )
        rl_coeff_rows.append(coeff)
        for record in payload["training_log"]:
            row_record = record.copy()
            row_record.update({"env_id": payload["env_id"], "group": payload["group"], "tier": payload["tier"]})
            training_rows.append(row_record)

    baseline_rows = []
    baseline_coeff_rows = []
    for payload in baseline_payloads:
        row = payload["row"].copy()
        row.update({"env_id": payload["env_id"], "group": payload["group"], "tier": payload["tier"]})
        baseline_rows.append(row)
        coeff = payload["coeff"].copy()
        coeff.update({"env_id": payload["env_id"], "group": payload["group"], "tier": payload["tier"]})
        baseline_coeff_rows.append(coeff)

    rl_df = pd.DataFrame(rl_rows).sort_values(["group", "tier", "algo", "seed"]).reset_index(drop=True)
    baseline_df = pd.DataFrame(baseline_rows).sort_values(["group", "tier", "policy"]).reset_index(drop=True)
    coeff_df = pd.concat([pd.DataFrame(rl_coeff_rows), pd.DataFrame(baseline_coeff_rows)], ignore_index=True, sort=False)
    training_df = pd.DataFrame(training_rows)
    return rl_df, baseline_df, coeff_df, training_df


def write_summaries(rl_df: pd.DataFrame, baseline_df: pd.DataFrame, coeff_df: pd.DataFrame, training_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rl_df.to_csv(OUTPUT_DIR / "raw_rl_results.csv", index=False)
    baseline_df.to_csv(OUTPUT_DIR / "baseline_results.csv", index=False)
    coeff_df.to_csv(OUTPUT_DIR / "policy_coefficients.csv", index=False)
    training_df.to_csv(OUTPUT_DIR / "training_logs.csv", index=False)

    rl_summary = (
        rl_df.groupby(["env_id", "group", "tier", "algo"], as_index=False)
        .agg(
            mean_discounted_loss=("mean_discounted_loss", "mean"),
            std_discounted_loss=("mean_discounted_loss", "std"),
            median_discounted_loss=("mean_discounted_loss", "median"),
            mean_reward=("mean_reward", "mean"),
            mean_clip_rate=("clip_rate", "mean"),
            mean_explosion_rate=("explosion_rate", "mean"),
        )
        .fillna(0.0)
    )
    rl_summary["best_loss_in_env"] = rl_summary.groupby("env_id")["mean_discounted_loss"].transform("min")
    rl_summary["loss_gap_vs_best_rl_pct"] = (rl_summary["mean_discounted_loss"] - rl_summary["best_loss_in_env"]) / rl_summary["best_loss_in_env"] * 100.0
    tier_cat = pd.CategoricalDtype(categories=TIER_ORDER_V2, ordered=True)
    rl_summary["tier"] = rl_summary["tier"].astype(tier_cat)
    rl_summary = rl_summary.sort_values(["group", "tier", "algo"]).reset_index(drop=True)
    rl_summary.to_csv(OUTPUT_DIR / "rl_summary.csv", index=False)

    baseline_summary = baseline_df[
        ["env_id", "group", "tier", "policy", "mean_discounted_loss", "std_discounted_loss", "mean_reward", "clip_rate", "explosion_rate"]
    ].rename(columns={"policy": "policy_name"})
    rl_policy_summary = rl_summary.rename(columns={"algo": "policy_name", "mean_clip_rate": "clip_rate", "mean_explosion_rate": "explosion_rate"})[
        ["env_id", "group", "tier", "policy_name", "mean_discounted_loss", "std_discounted_loss", "mean_reward", "clip_rate", "explosion_rate"]
    ]
    all_policy_df = pd.concat([baseline_summary, rl_policy_summary], ignore_index=True, sort=False)
    all_policy_df["tier"] = all_policy_df["tier"].astype(tier_cat)
    all_policy_df = all_policy_df.sort_values(["group", "tier", "policy_name"]).reset_index(drop=True)
    all_policy_df.to_csv(OUTPUT_DIR / "all_policy_summary.csv", index=False)

    new_env_ids = [spec["env_id"] for spec in NEW_ENV_SPECS_V2]
    riccati_new = all_policy_df[
        all_policy_df["env_id"].isin(new_env_ids) & (all_policy_df["policy_name"] == "riccati_reference")
    ][["env_id", "group", "tier", "mean_discounted_loss"]].rename(columns={"mean_discounted_loss": "riccati_loss"})
    env_best = all_policy_df[all_policy_df["env_id"].isin(new_env_ids)].groupby("env_id", as_index=False)["mean_discounted_loss"].min().rename(columns={"mean_discounted_loss": "best_policy_loss"})
    riccati_gap = riccati_new.merge(env_best, on="env_id", how="left")
    riccati_gap["riccati_gap_pct"] = (riccati_gap["riccati_loss"] / riccati_gap["best_policy_loss"] - 1.0) * 100.0
    riccati_gap.to_csv(OUTPUT_DIR / "new_tiers_riccati_gap.csv", index=False)

    best_rl_by_env = (
        rl_summary.sort_values(["env_id", "mean_discounted_loss"])
        .groupby("env_id", as_index=False)
        .first()[["env_id", "algo", "mean_discounted_loss"]]
        .rename(columns={"algo": "best_rl_algo", "mean_discounted_loss": "best_rl_loss"})
    )
    riccati_vs_rl = riccati_new.merge(best_rl_by_env, on="env_id", how="left")
    riccati_vs_rl["rl_beats_riccati"] = riccati_vs_rl["best_rl_loss"] < riccati_vs_rl["riccati_loss"]
    riccati_vs_rl["riccati_gap_vs_best_rl_pct"] = (riccati_vs_rl["riccati_loss"] / riccati_vs_rl["best_rl_loss"] - 1.0) * 100.0
    riccati_vs_rl.to_csv(OUTPUT_DIR / "riccati_vs_best_rl.csv", index=False)

    lines = []
    lines.append("# Phase 11 v2 极端扩展矩阵")
    lines.append("")
    lines.append("## 范围")
    lines.append("")
    lines.append("| 维度 | 内容 |")
    lines.append("|---|---|")
    lines.append("| v2 环境 | `nonlinear` 保留一档并新增更高曲率一档；`zlb/asymmetric` 两档均替换为更强结构扭曲 |")
    lines.append("| RL 算法 | `PPO`、`TD3`、`SAC` |")
    lines.append(f"| Seeds | `{SEEDS}` |")
    lines.append("")
    lines.append("## v2 RL 汇总")
    lines.append("")
    lines.append(rl_summary[["env_id", "algo", "mean_discounted_loss", "std_discounted_loss", "loss_gap_vs_best_rl_pct"]].round(6).to_markdown(index=False))
    lines.append("")
    lines.append("## Riccati 相对最优策略差距")
    lines.append("")
    lines.append(riccati_gap.round(6).to_markdown(index=False))
    lines.append("")
    lines.append("## Riccati 对比最优 RL")
    lines.append("")
    lines.append(riccati_vs_rl.round(6).to_markdown(index=False))
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append("- 本批次不改 `phase10`，全部输出独立放在 `phase11/extreme_matrix_v2`。")
    lines.append("- `zlb` 通过 state-contingent trap 机制放大衰退与通缩。")
    lines.append("- `asymmetric` 通过阈值型高阶尾部惩罚扭曲传统二次损失。")
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", action="append", dest="env_ids")
    parser.add_argument("--algo", action="append", dest="algos")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_config = LQBenchmarkConfig.from_json(LINEAR_CONFIG_PATH)
    benchmark_model = LQBenchmarkModel(benchmark_config)
    riccati_solution = solve_discounted_lq_riccati(benchmark_model)
    riccati_policy = build_optimal_linear_policy(riccati_solution)
    riccati_k = riccati_solution.K

    env_specs = NEW_ENV_SPECS_V2
    if args.env_ids:
        wanted = set(args.env_ids)
        env_specs = [spec for spec in NEW_ENV_SPECS_V2 if spec["env_id"] in wanted]
    algos = args.algos or ["ppo", "sac", "td3"]

    for spec in env_specs:
        run_baselines_for_env(spec, riccati_policy, riccati_k)
        for algo in algos:
            for seed in SEEDS:
                run_rl_one(spec, algo, seed)

    rl_payloads, baseline_payloads = load_all_payloads()
    rl_df, baseline_df, coeff_df, training_df = build_summary_tables(rl_payloads, baseline_payloads)
    write_summaries(rl_df, baseline_df, coeff_df, training_df)


if __name__ == "__main__":
    main()
