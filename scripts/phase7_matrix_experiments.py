from __future__ import annotations

import argparse
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
from monetary_rl.models import (
    AsymmetricBenchmarkConfig,
    AsymmetricBenchmarkModel,
    LQBenchmarkConfig,
    LQBenchmarkModel,
    NonlinearBenchmarkConfig,
    NonlinearBenchmarkModel,
)
from monetary_rl.solvers import build_optimal_linear_policy, solve_discounted_lq_riccati


OUTPUT_DIR = ROOT / "outputs" / "phase7" / "matrix"
CACHE_DIR = OUTPUT_DIR / "cache"
PLOTS_DIR = OUTPUT_DIR / "plots"

LINEAR_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_lq.json"
PPO_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_ppo_tuned.json"
SAC_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_sac.json"
TD3_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_td3.json"
TAYLOR_RULE_PATH = ROOT / "outputs" / "phase2" / "taylor_rule.json"

SEEDS = [7, 29, 43]
EVAL_EPISODES = 32
COMMON_SHOCK_HORIZON = 20


ENV_SPECS = [
    {
        "env_id": "benchmark",
        "group": "benchmark",
        "tier": "baseline",
        "model_kind": "linear",
        "config_path": str(LINEAR_CONFIG_PATH),
        "env_kwargs": {
            "horizon": 60,
            "action_low": -6.0,
            "action_high": 6.0,
            "initial_state_low": (-2.0, -2.0, -2.0),
            "initial_state_high": (2.0, 2.0, 2.0),
            "state_abs_limit": 25.0,
            "terminal_penalty": 50.0,
            "seed": 0,
        },
        "common_initial_state": [1.0, -1.0, 0.0],
    },
    {
        "env_id": "nonlinear_mild",
        "group": "nonlinear",
        "tier": "mild",
        "model_kind": "nonlinear",
        "config_path": str(ROOT / "src" / "monetary_rl" / "config" / "nonlinear_phillips_mild.json"),
        "env_kwargs": {
            "horizon": 60,
            "action_low": -5.0,
            "action_high": 5.0,
            "initial_state_low": (-1.25, -1.25, -1.25),
            "initial_state_high": (1.25, 1.25, 1.25),
            "state_abs_limit": 20.0,
            "terminal_penalty": 75.0,
            "seed": 0,
        },
        "common_initial_state": [1.0, -1.0, 0.0],
    },
    {
        "env_id": "nonlinear_medium",
        "group": "nonlinear",
        "tier": "medium",
        "model_kind": "nonlinear",
        "config_path": str(ROOT / "src" / "monetary_rl" / "config" / "nonlinear_phillips_medium.json"),
        "env_kwargs": {
            "horizon": 60,
            "action_low": -5.0,
            "action_high": 5.0,
            "initial_state_low": (-1.5, -1.5, -1.5),
            "initial_state_high": (1.5, 1.5, 1.5),
            "state_abs_limit": 20.0,
            "terminal_penalty": 75.0,
            "seed": 0,
        },
        "common_initial_state": [1.0, -1.0, 0.0],
    },
    {
        "env_id": "nonlinear_strong",
        "group": "nonlinear",
        "tier": "strong",
        "model_kind": "nonlinear",
        "config_path": str(ROOT / "src" / "monetary_rl" / "config" / "nonlinear_phillips_strong.json"),
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
        "common_initial_state": [1.0, -1.0, 0.0],
    },
    {
        "env_id": "zlb_mild",
        "group": "zlb",
        "tier": "mild",
        "model_kind": "linear",
        "config_path": str(LINEAR_CONFIG_PATH),
        "env_kwargs": {
            "horizon": 60,
            "action_low": -2.0,
            "action_high": 6.0,
            "initial_state_low": (-2.0, -2.0, -2.0),
            "initial_state_high": (1.0, 1.0, 1.0),
            "state_abs_limit": 25.0,
            "terminal_penalty": 50.0,
            "seed": 0,
        },
        "common_initial_state": [-0.5, -1.0, -1.2],
    },
    {
        "env_id": "zlb_medium",
        "group": "zlb",
        "tier": "medium",
        "model_kind": "linear",
        "config_path": str(LINEAR_CONFIG_PATH),
        "env_kwargs": {
            "horizon": 60,
            "action_low": -1.0,
            "action_high": 6.0,
            "initial_state_low": (-2.25, -2.25, -1.5),
            "initial_state_high": (0.75, 0.75, 0.75),
            "state_abs_limit": 25.0,
            "terminal_penalty": 50.0,
            "seed": 0,
        },
        "common_initial_state": [-0.75, -1.25, -1.0],
    },
    {
        "env_id": "zlb_strong",
        "group": "zlb",
        "tier": "strong",
        "model_kind": "linear",
        "config_path": str(LINEAR_CONFIG_PATH),
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
        "common_initial_state": [-1.0, -1.5, -0.8],
    },
    {
        "env_id": "asymmetric_mild",
        "group": "asymmetric",
        "tier": "mild",
        "model_kind": "asymmetric",
        "config_path": str(ROOT / "src" / "monetary_rl" / "config" / "asymmetric_loss_mild.json"),
        "env_kwargs": {
            "horizon": 60,
            "action_low": -6.0,
            "action_high": 6.0,
            "initial_state_low": (-1.5, -2.0, -2.0),
            "initial_state_high": (2.0, 1.5, 2.0),
            "state_abs_limit": 25.0,
            "terminal_penalty": 50.0,
            "seed": 0,
        },
        "common_initial_state": [1.0, -1.0, 0.0],
    },
    {
        "env_id": "asymmetric_medium",
        "group": "asymmetric",
        "tier": "medium",
        "model_kind": "asymmetric",
        "config_path": str(ROOT / "src" / "monetary_rl" / "config" / "asymmetric_loss_medium.json"),
        "env_kwargs": {
            "horizon": 60,
            "action_low": -6.0,
            "action_high": 6.0,
            "initial_state_low": (-1.5, -2.0, -2.0),
            "initial_state_high": (2.0, 1.5, 2.0),
            "state_abs_limit": 25.0,
            "terminal_penalty": 50.0,
            "seed": 0,
        },
        "common_initial_state": [1.0, -1.0, 0.0],
    },
    {
        "env_id": "asymmetric_strong",
        "group": "asymmetric",
        "tier": "strong",
        "model_kind": "asymmetric",
        "config_path": str(ROOT / "src" / "monetary_rl" / "config" / "asymmetric_loss_strong.json"),
        "env_kwargs": {
            "horizon": 60,
            "action_low": -6.0,
            "action_high": 6.0,
            "initial_state_low": (-1.5, -2.0, -2.0),
            "initial_state_high": (2.0, 1.5, 2.0),
            "state_abs_limit": 25.0,
            "terminal_penalty": 50.0,
            "seed": 0,
        },
        "common_initial_state": [1.0, -1.0, 0.0],
    },
]


PPO_OVERRIDES = {
    "total_updates": 120,
    "rollout_steps": 512,
    "train_epochs": 4,
    "eval_interval": 20,
}

OFFPOLICY_OVERRIDES = {
    "total_steps": 9000,
    "eval_interval": 3000,
}


def make_model(spec: dict):
    if spec["model_kind"] == "linear":
        config = LQBenchmarkConfig.from_json(spec["config_path"])
        model = LQBenchmarkModel(config)
    elif spec["model_kind"] == "nonlinear":
        config = NonlinearBenchmarkConfig.from_json(spec["config_path"])
        model = NonlinearBenchmarkModel(config)
    elif spec["model_kind"] == "asymmetric":
        config = AsymmetricBenchmarkConfig.from_json(spec["config_path"])
        model = AsymmetricBenchmarkModel(config)
    else:
        raise ValueError(f"Unknown model kind: {spec['model_kind']}")
    return config, model


def make_env(spec: dict):
    _, model = make_model(spec)
    env = LQBenchmarkEnv(model, BenchmarkEnvConfig(**spec["env_kwargs"]))
    return env


def riccati_reference_policy() -> tuple[callable, np.ndarray]:
    linear_config = LQBenchmarkConfig.from_json(LINEAR_CONFIG_PATH)
    linear_model = LQBenchmarkModel(linear_config)
    solution = solve_discounted_lq_riccati(linear_model)
    return build_optimal_linear_policy(solution), solution.K


def cache_file(spec: dict, algo: str, seed: int) -> Path:
    return CACHE_DIR / f"{spec['env_id']}__{algo}__seed{seed}.json"


def baseline_cache_file(spec: dict, name: str) -> Path:
    return CACHE_DIR / f"{spec['env_id']}__baseline__{name}.json"


def run_rl_one(spec: dict, algo: str, seed: int) -> dict:
    path = cache_file(spec, algo, seed)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    config, model = make_model(spec)
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
    rng = np.random.default_rng(20260405 + seed)
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
        linear_theta, linear_result, linear_policy = run_linear_search(LQBenchmarkEnv(model, BenchmarkEnvConfig(**spec["env_kwargs"])), EVAL_EPISODES, seed=123)
        linear_row = policy_row("linear_policy_search", env, linear_policy, EVAL_EPISODES, 20_000)
        linear_coeff = {
            "policy": "linear_policy_search",
            "intercept": 0.0,
            "inflation_gap": float(linear_theta[0]),
            "output_gap": float(linear_theta[1]),
            "lagged_policy_rate_gap": float(linear_theta[2]),
            "fit_rmse": 0.0,
        }
        rng = np.random.default_rng(20260500)
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
            rng = np.random.default_rng(20260600 + hash(name) % 100)
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
            record = record.copy()
            record.update({"env_id": payload["env_id"], "group": payload["group"], "tier": payload["tier"]})
            training_rows.append(record)

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
    rl_summary.to_csv(OUTPUT_DIR / "rl_summary.csv", index=False)

    baseline_summary = baseline_df[
        [
            "env_id",
            "group",
            "tier",
            "policy",
            "mean_discounted_loss",
            "std_discounted_loss",
            "mean_reward",
            "clip_rate",
            "explosion_rate",
        ]
    ].rename(columns={"policy": "policy_name"})
    rl_policy_summary = rl_summary.rename(columns={"algo": "policy_name", "mean_clip_rate": "clip_rate", "mean_explosion_rate": "explosion_rate"})[
        ["env_id", "group", "tier", "policy_name", "mean_discounted_loss", "std_discounted_loss", "mean_reward", "clip_rate", "explosion_rate"]
    ]
    all_policy_df = pd.concat([baseline_summary, rl_policy_summary], ignore_index=True, sort=False)
    all_policy_df.to_csv(OUTPUT_DIR / "all_policy_summary.csv", index=False)

    # Master markdown summary.
    lines: list[str] = []
    lines.append("# Phase 7 Matrix Summary")
    lines.append("")
    lines.append("## Matrix Scope")
    lines.append("")
    lines.append("| Dimension | Value |")
    lines.append("|---|---|")
    lines.append("| Environments | 1 benchmark + 3 nonlinear + 3 ZLB/ELB-tightness + 3 asymmetric-loss = 10 |")
    lines.append("| RL algorithms | PPO, TD3, SAC |")
    lines.append(f"| Seeds per algorithm-environment pair | {SEEDS} |")
    lines.append("")

    lines.append("## RL Summary Table")
    lines.append("")
    lines.append(
        rl_summary[
            [
                "env_id",
                "algo",
                "mean_discounted_loss",
                "std_discounted_loss",
                "mean_reward",
                "mean_clip_rate",
                "mean_explosion_rate",
                "loss_gap_vs_best_rl_pct",
            ]
        ].round(6).to_markdown(index=False)
    )
    lines.append("")

    benchmark_table = all_policy_df[all_policy_df["env_id"] == "benchmark"].copy()
    lines.append("## Benchmark Including Reference Rules")
    lines.append("")
    lines.append(benchmark_table.round(6).to_markdown(index=False))
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- `zlb_*` tiers should be read as progressively tighter effective lower-bound environments, implemented through reduced policy-rate room and more recessionary initial-state support.")
    lines.append("- `nonlinear_*` tiers increase the strength of the nonlinear Phillips distortion.")
    lines.append("- `asymmetric_*` tiers increase the extra penalty on upside inflation and downside output gaps.")
    lines.append("- RL summary statistics are averaged across seeds; benchmark reference rules are single deterministic baselines.")
    lines.append("")
    (OUTPUT_DIR / "phase7_matrix_summary.md").write_text("\n".join(lines), encoding="utf-8")


def make_plots(rl_df: pd.DataFrame, baseline_df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    rl_summary = pd.read_csv(OUTPUT_DIR / "rl_summary.csv")

    pivot = rl_summary.pivot(index="env_id", columns="algo", values="mean_discounted_loss")
    plt.figure(figsize=(8, 6))
    im = plt.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(im, label="Mean discounted loss")
    plt.title("RL Mean Discounted Loss by Environment")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rl_mean_loss_heatmap.png", dpi=200)
    plt.close()

    gap_pivot = rl_summary.pivot(index="env_id", columns="algo", values="loss_gap_vs_best_rl_pct")
    plt.figure(figsize=(8, 6))
    im = plt.imshow(gap_pivot.values, aspect="auto", cmap="magma")
    plt.xticks(range(len(gap_pivot.columns)), gap_pivot.columns)
    plt.yticks(range(len(gap_pivot.index)), gap_pivot.index)
    plt.colorbar(im, label="Loss gap vs best RL (%)")
    plt.title("RL Relative Gap vs Best RL Policy")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rl_loss_gap_heatmap.png", dpi=200)
    plt.close()

    for group in ["benchmark", "nonlinear", "zlb", "asymmetric"]:
        subset = rl_df[rl_df["group"] == group]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = []
        data = []
        for env_id in subset["env_id"].drop_duplicates():
            for algo in ["ppo", "td3", "sac"]:
                values = subset[(subset["env_id"] == env_id) & (subset["algo"] == algo)]["mean_discounted_loss"].to_numpy()
                labels.append(f"{env_id}\n{algo}")
                data.append(values)
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.set_title(f"{group.capitalize()} Loss Distribution Across Seeds")
        ax.set_ylabel("Mean discounted loss")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{group}_seed_boxplots.png", dpi=200)
        plt.close(fig)

    curve_fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, group in zip(axes, ["nonlinear", "zlb", "asymmetric"]):
        subset = rl_summary[rl_summary["group"] == group].copy()
        tier_order = {"mild": 1, "medium": 2, "strong": 3}
        subset["tier_order"] = subset["tier"].map(tier_order)
        for algo in ["ppo", "td3", "sac"]:
            algo_sub = subset[subset["algo"] == algo].sort_values("tier_order")
            ax.plot(algo_sub["tier_order"], algo_sub["mean_discounted_loss"], marker="o", label=algo)
        ax.set_title(group.capitalize())
        ax.set_xticks([1, 2, 3], ["mild", "medium", "strong"])
        ax.set_ylabel("Mean discounted loss")
    axes[0].legend()
    curve_fig.tight_layout()
    curve_fig.savefig(PLOTS_DIR / "distortion_strength_curves.png", dpi=200)
    plt.close(curve_fig)

    benchmark_all = pd.concat(
        [
            baseline_df[baseline_df["env_id"] == "benchmark"][["policy", "mean_discounted_loss"]],
            rl_summary[rl_summary["env_id"] == "benchmark"][["algo", "mean_discounted_loss"]].rename(columns={"algo": "policy"}),
        ],
        ignore_index=True,
    ).sort_values("mean_discounted_loss")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(benchmark_all["policy"], benchmark_all["mean_discounted_loss"])
    ax.set_title("Benchmark: All Policies")
    ax.set_ylabel("Mean discounted loss")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "benchmark_all_policies.png", dpi=200)
    plt.close(fig)

    win_counts = rl_summary.loc[rl_summary.groupby("env_id")["mean_discounted_loss"].idxmin(), "algo"].value_counts().reindex(["ppo", "td3", "sac"]).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(win_counts.index, win_counts.values)
    ax.set_title("Best RL Algorithm Count Across Environments")
    ax.set_ylabel("Number of environments won")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "algorithm_win_counts.png", dpi=200)
    plt.close(fig)

    coeff_df = pd.read_csv(OUTPUT_DIR / "policy_coefficients.csv")
    rl_coeff = coeff_df[coeff_df["algo"].isin(["ppo", "td3", "sac"])].copy()
    best_rows = []
    for env_id in rl_summary["env_id"].unique():
        env_best = rl_summary[rl_summary["env_id"] == env_id].sort_values("mean_discounted_loss").iloc[0]
        best_seed = rl_df[(rl_df["env_id"] == env_id) & (rl_df["algo"] == env_best["algo"])].sort_values("mean_discounted_loss").iloc[0]["seed"]
        row = rl_coeff[(rl_coeff["env_id"] == env_id) & (rl_coeff["algo"] == env_best["algo"]) & (rl_coeff["seed"] == best_seed)]
        if not row.empty:
            best_rows.append(row.iloc[0])
    if best_rows:
        best_coeff = pd.DataFrame(best_rows).set_index("env_id")[["inflation_gap", "output_gap", "lagged_policy_rate_gap"]]
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(best_coeff.values, aspect="auto", cmap="coolwarm")
        ax.set_xticks(range(best_coeff.shape[1]), best_coeff.columns)
        ax.set_yticks(range(best_coeff.shape[0]), best_coeff.index)
        ax.set_title("Approximate Coefficients of Best RL Policy by Environment")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "best_rl_coefficients_heatmap.png", dpi=200)
        plt.close(fig)

    zlb_subset = pd.concat(
        [
            baseline_df[baseline_df["group"] == "zlb"][["env_id", "policy", "clip_rate"]].rename(columns={"policy": "algo", "clip_rate": "mean_clip_rate"}),
            rl_summary[rl_summary["group"] == "zlb"][["env_id", "algo", "mean_clip_rate"]],
        ],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = []
    vals = []
    for env_id in ["zlb_mild", "zlb_medium", "zlb_strong"]:
        for algo in ["ppo", "td3", "sac", "riccati_reference", "empirical_taylor", "linear_policy_search"]:
            row = zlb_subset[(zlb_subset["env_id"] == env_id) & (zlb_subset["algo"] == algo)]
            if row.empty:
                continue
            x_labels.append(f"{env_id}\n{algo}")
            vals.append(float(row["mean_clip_rate"].iloc[0]))
    ax.bar(range(len(vals)), vals)
    ax.set_xticks(range(len(vals)), x_labels, rotation=45, ha="right")
    ax.set_ylabel("Clip / lower-bound hit rate")
    ax.set_title("ZLB / ELB Tightness: Binding Frequency")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "zlb_clip_rates.png", dpi=200)
    plt.close(fig)


def filter_specs(group: str) -> list[dict]:
    if group == "all":
        return ENV_SPECS
    return [spec for spec in ENV_SPECS if spec["group"] == group]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["all", "benchmark", "nonlinear", "zlb", "asymmetric"], default="all")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    riccati_policy, riccati_k = riccati_reference_policy()

    for spec in filter_specs(args.group):
        run_baselines_for_env(spec, riccati_policy, riccati_k)
        for algo in ["ppo", "td3", "sac"]:
            for seed in SEEDS:
                run_rl_one(spec, algo, seed)

    rl_payloads, baseline_payloads = load_all_payloads()
    rl_df, baseline_df, coeff_df, training_df = build_summary_tables(rl_payloads, baseline_payloads)
    write_summaries(rl_df, baseline_df, coeff_df, training_df)
    make_plots(rl_df, baseline_df)


if __name__ == "__main__":
    main()
