from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import phase10_counterfactual_eval as cfe
import phase10_revealed_policy_eval as rpe
from monetary_rl.evaluation import evaluate_policy, fit_linear_policy_response
from monetary_rl.envs import BenchmarkEnvConfig, LQBenchmarkEnv
from monetary_rl.experiment_utils import build_taylor_gap_policy, load_taylor_rule, run_ppo, run_sac, run_td3, training_log_frame
from monetary_rl.models import LQBenchmarkConfig, LQBenchmarkModel
from monetary_rl.phase10_utils import (
    EmpiricalContext,
    build_ann_context,
    build_svar_context,
    historical_summary,
    make_empirical_env,
    simulate_historical_counterfactual,
)
from monetary_rl.phase11_extreme_specs_v2 import (
    LINEAR_CONFIG_PATH,
    NEW_ENV_SPECS_V2,
    OFFPOLICY_OVERRIDES,
    PPO_CONFIG_PATH as PHASE11_PPO_CONFIG_PATH,
    PPO_OVERRIDES,
    SAC_CONFIG_PATH as PHASE11_SAC_CONFIG_PATH,
    TAYLOR_RULE_PATH,
    TD3_CONFIG_PATH as PHASE11_TD3_CONFIG_PATH,
    make_model,
)
from monetary_rl.solvers import build_optimal_linear_policy, solve_discounted_lq_riccati


OUTPUT_DIR = ROOT / "outputs" / "phase15"
FIG_DIR = OUTPUT_DIR / "figures"
RETRAIN_DIR = OUTPUT_DIR / "retrained"
PHASE11_RETRAIN_DIR = OUTPUT_DIR / "phase11_retrained"

STOCHASTIC_HORIZON = 120
STOCHASTIC_EPISODES = 96
EMPIRICAL_RETRAIN_SEED = 43
PHASE11_RETRAIN_SEED = 43

BENCHMARK_CONFIG = json.loads((ROOT / "src" / "monetary_rl" / "config" / "benchmark_lq.json").read_text(encoding="utf-8"))
ARTIFICIAL_BASE = {k: float(v) for k, v in BENCHMARK_CONFIG["loss_weights"].items()}
REVEALED_BASE = rpe.revealed_loss_weights()

BETA_SCENARIOS = [
    {"scenario_id": "beta_097", "scenario_group": "beta", "beta": 0.97},
    {"scenario_id": "beta_099", "scenario_group": "beta", "beta": 0.99},
    {"scenario_id": "beta_0995", "scenario_group": "beta", "beta": 0.995},
]

ARTIFICIAL_LAMBDA_SCENARIOS = [
    {"scenario_id": "lambda_art_output_low", "scenario_group": "lambda", "beta": 0.99, "weights": {"inflation": 1.0, "output_gap": 0.25, "rate_smoothing": 0.1}},
    {"scenario_id": "lambda_art_output_high", "scenario_group": "lambda", "beta": 0.99, "weights": {"inflation": 1.0, "output_gap": 1.0, "rate_smoothing": 0.1}},
    {"scenario_id": "lambda_art_smooth_low", "scenario_group": "lambda", "beta": 0.99, "weights": {"inflation": 1.0, "output_gap": 0.5, "rate_smoothing": 0.05}},
    {"scenario_id": "lambda_art_smooth_high", "scenario_group": "lambda", "beta": 0.99, "weights": {"inflation": 1.0, "output_gap": 0.5, "rate_smoothing": 0.2}},
]

REVEALED_LAMBDA_SCENARIOS = [
    {
        "scenario_id": "lambda_rev_output_down",
        "scenario_group": "lambda",
        "beta": 0.99,
        "weights": {"inflation": REVEALED_BASE["inflation"], "output_gap": REVEALED_BASE["output_gap"] * 0.5, "rate_smoothing": REVEALED_BASE["rate_smoothing"]},
    },
    {
        "scenario_id": "lambda_rev_output_up",
        "scenario_group": "lambda",
        "beta": 0.99,
        "weights": {"inflation": REVEALED_BASE["inflation"], "output_gap": REVEALED_BASE["output_gap"] * 1.5, "rate_smoothing": REVEALED_BASE["rate_smoothing"]},
    },
    {
        "scenario_id": "lambda_rev_smooth_down",
        "scenario_group": "lambda",
        "beta": 0.99,
        "weights": {"inflation": REVEALED_BASE["inflation"], "output_gap": REVEALED_BASE["output_gap"], "rate_smoothing": REVEALED_BASE["rate_smoothing"] * 0.5},
    },
    {
        "scenario_id": "lambda_rev_smooth_up",
        "scenario_group": "lambda",
        "beta": 0.99,
        "weights": {"inflation": REVEALED_BASE["inflation"], "output_gap": REVEALED_BASE["output_gap"], "rate_smoothing": REVEALED_BASE["rate_smoothing"] * 1.5},
    },
]

EMPIRICAL_RETRAIN_SCENARIOS = {
    "artificial": [
        {"scenario_id": "beta_097", "beta": 0.97, "weights": ARTIFICIAL_BASE},
        {"scenario_id": "lambda_art_output_high", "beta": 0.99, "weights": {"inflation": 1.0, "output_gap": 1.0, "rate_smoothing": 0.1}},
    ],
    "revealed": [
        {"scenario_id": "beta_097", "beta": 0.97, "weights": REVEALED_BASE},
        {"scenario_id": "lambda_rev_smooth_down", "beta": 0.99, "weights": {"inflation": REVEALED_BASE["inflation"], "output_gap": REVEALED_BASE["output_gap"], "rate_smoothing": REVEALED_BASE["rate_smoothing"] * 0.5}},
    ],
}

PHASE11_CASES_FOR_RETRAIN = ["nonlinear_hyper", "zlb_trap_extreme", "asymmetric_threshold_extreme"]
PHASE11_RETRAIN_SCENARIOS = [
    {"scenario_id": "beta_097", "beta": 0.97, "weights": ARTIFICIAL_BASE},
    {"scenario_id": "lambda_art_smooth_high", "beta": 0.99, "weights": {"inflation": 1.0, "output_gap": 0.5, "rate_smoothing": 0.2}},
]


def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, FIG_DIR, RETRAIN_DIR, PHASE11_RETRAIN_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def policy_from_theta(theta: np.ndarray):
    theta = np.asarray(theta, dtype=float)

    def policy(state: np.ndarray, t: int) -> float:
        del t
        return float(theta[0] + theta[1:] @ np.asarray(state, dtype=float))

    return policy


def load_ann_search_policies() -> tuple[dict[str, Any], pd.DataFrame]:
    seed_df = pd.read_csv(ROOT / "outputs" / "phase14" / "ann_search_seed_summary.csv")
    rows = []
    policies = {}
    for label, training_env in [("ann_affine_search_artificial", "ann"), ("ann_affine_search_revealed", "ann_revealed")]:
        row = seed_df.loc[seed_df["search_label"] == label].sort_values("best_mean_discounted_loss").iloc[0]
        theta = np.array([float(row["intercept"]), float(row["inflation_gap"]), float(row["output_gap"]), float(row["lagged_policy_rate_gap"])], dtype=float)
        policies[label] = policy_from_theta(theta)
        rows.append(
            {
                "policy_name": label,
                "rule_family": "ann_native_numerical_search",
                "source_env": "ann",
                "training_env": training_env,
                "policy_parameterization": "affine_linear_search",
                "algo": "numerical_search",
            }
        )
    return policies, pd.DataFrame(rows)


def policy_metadata(unified_registry: pd.DataFrame, manual_meta: pd.DataFrame | None = None) -> pd.DataFrame:
    keep_cols = ["policy_name", "rule_family", "source_env", "training_env", "policy_parameterization", "algo"]
    frames = [unified_registry[keep_cols].copy()]
    if manual_meta is not None and not manual_meta.empty:
        frames.append(manual_meta[keep_cols].copy())
    return pd.concat(frames, ignore_index=True).drop_duplicates("policy_name", keep="last")


def clone_context(context: EmpiricalContext, beta: float, loss_weights: dict[str, float], suffix: str) -> EmpiricalContext:
    cloned_model = deepcopy(context.model)
    cloned_model.config.discount_factor = float(beta)
    cloned_model.config.loss_weights = {k: float(v) for k, v in loss_weights.items()}
    return EmpiricalContext(
        name=f"{context.name}_{suffix}",
        model=cloned_model,
        env_df=context.env_df.copy(),
        initial_states=context.initial_states.copy(),
        shock_pool=context.shock_pool.copy(),
        action_dates=context.action_dates.copy(),
        state_dates=context.state_dates.copy(),
        observation_low=context.observation_low,
        observation_high=context.observation_high,
    )


def path_moments(path_df: pd.DataFrame) -> pd.DataFrame:
    work = path_df.dropna(subset=["loss"]).copy()
    return (
        work.groupby("policy_name", as_index=False)
        .agg(
            var_inflation_gap=("inflation_gap", "var"),
            std_inflation_gap=("inflation_gap", "std"),
            var_output_gap=("output_gap", "var"),
            std_output_gap=("output_gap", "std"),
            var_policy_rate=("policy_rate", "var"),
            std_policy_rate=("policy_rate", "std"),
            var_rate_change=("rate_change", "var"),
            std_rate_change=("rate_change", "std"),
        )
    )


def stochastic_moments(context: EmpiricalContext, policy_map: dict[str, Any], include_historical: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    loss_rows: list[dict[str, Any]] = []
    moment_rows: list[dict[str, Any]] = []
    for idx, (policy_name, policy_fn) in enumerate(policy_map.items()):
        if (not include_historical) and policy_name == "historical_actual_policy":
            continue
        env = make_empirical_env(context, horizon=STOCHASTIC_HORIZON, seed=20260415 + idx)
        trajectories: list[dict[str, float]] = []
        discounted_losses: list[float] = []
        rewards: list[float] = []
        clip_count = 0
        explosion_count = 0
        step_count = 0
        for ep in range(STOCHASTIC_EPISODES):
            state = np.asarray(env.reset(seed=304000 + idx * 100 + ep), dtype=float)
            done = False
            discount = 1.0
            total_reward = 0.0
            discounted_loss = 0.0
            t = 0
            while not done:
                prev_rate = float(context.model.action_to_level(state[2]))
                raw_action = float(policy_fn(state.copy(), t))
                next_state, reward, done, info = env.step(raw_action)
                current_rate = float(info["policy_rate"])
                trajectories.append(
                    {
                        "policy_name": policy_name,
                        "episode": ep,
                        "period": t,
                        "inflation_gap": float(state[0]),
                        "output_gap": float(state[1]),
                        "policy_rate": current_rate,
                        "rate_change": current_rate - prev_rate,
                        "loss": float(info["loss"]),
                    }
                )
                total_reward += reward
                discounted_loss += (-reward) * discount
                discount *= context.model.config.discount_factor
                clip_count += int(abs(float(info["raw_action"]) - float(info["action"])) > 1e-8)
                explosion_count += int(bool(info.get("exploded", False)))
                step_count += 1
                state = np.asarray(next_state, dtype=float)
                t += 1
            rewards.append(total_reward)
            discounted_losses.append(discounted_loss)
        traj_df = pd.DataFrame(trajectories)
        loss_rows.append(
            {
                "policy_name": policy_name,
                "mean_discounted_loss": float(np.mean(discounted_losses)),
                "std_discounted_loss": float(np.std(discounted_losses, ddof=1)) if len(discounted_losses) > 1 else 0.0,
                "mean_reward": float(np.mean(rewards)),
                "mean_abs_action": float(traj_df["policy_rate"].sub(context.model.config.neutral_rate).abs().mean()),
                "clip_rate": clip_count / step_count if step_count else 0.0,
                "explosion_rate": explosion_count / STOCHASTIC_EPISODES,
            }
        )
        moment_rows.append(
            {
                "policy_name": policy_name,
                "var_inflation_gap": float(traj_df["inflation_gap"].var()),
                "std_inflation_gap": float(traj_df["inflation_gap"].std()),
                "var_output_gap": float(traj_df["output_gap"].var()),
                "std_output_gap": float(traj_df["output_gap"].std()),
                "var_policy_rate": float(traj_df["policy_rate"].var()),
                "std_policy_rate": float(traj_df["policy_rate"].std()),
                "var_rate_change": float(traj_df["rate_change"].var()),
                "std_rate_change": float(traj_df["rate_change"].std()),
            }
        )
    return pd.DataFrame(loss_rows), pd.DataFrame(moment_rows)


def evaluate_empirical_scenario(
    base_context: EmpiricalContext,
    env_name: str,
    loss_function: str,
    scenario_id: str,
    scenario_group: str,
    beta: float,
    weights: dict[str, float],
    unified_registry: pd.DataFrame,
    meta_df: pd.DataFrame,
    extra_policies: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenario_context = clone_context(base_context, beta, weights, f"{loss_function}_{scenario_id}")
    policy_map = cfe.build_policy_map(scenario_context, unified_registry)
    if extra_policies:
        policy_map.update(extra_policies)

    historical_paths = simulate_historical_counterfactual(
        model=scenario_context.model,
        policy_map=policy_map,
        initial_state=scenario_context.initial_states[0],
        action_dates=scenario_context.action_dates,
        state_dates=scenario_context.state_dates,
        shocks=scenario_context.shock_pool[:-1],
    )
    historical_df = historical_summary(historical_paths).merge(path_moments(historical_paths), on="policy_name", how="left")
    historical_df = historical_df.merge(meta_df, on="policy_name", how="left")
    historical_df.insert(0, "evaluation_env", env_name)
    historical_df.insert(1, "loss_function", loss_function)
    historical_df.insert(2, "evaluation_type", "historical_counterfactual")
    historical_df.insert(3, "scenario_group", scenario_group)
    historical_df.insert(4, "scenario_id", scenario_id)
    historical_df.insert(5, "beta", beta)
    historical_df["lambda_inflation"] = weights["inflation"]
    historical_df["lambda_output_gap"] = weights["output_gap"]
    historical_df["lambda_rate_smoothing"] = weights["rate_smoothing"]

    stoch_summary, stoch_mom = stochastic_moments(scenario_context, policy_map)
    stochastic_df = stoch_summary.merge(stoch_mom, on="policy_name", how="left").merge(meta_df, on="policy_name", how="left")
    stochastic_df.insert(0, "evaluation_env", env_name)
    stochastic_df.insert(1, "loss_function", loss_function)
    stochastic_df.insert(2, "evaluation_type", "stochastic_long_run")
    stochastic_df.insert(3, "scenario_group", scenario_group)
    stochastic_df.insert(4, "scenario_id", scenario_id)
    stochastic_df.insert(5, "beta", beta)
    stochastic_df["lambda_inflation"] = weights["inflation"]
    stochastic_df["lambda_output_gap"] = weights["output_gap"]
    stochastic_df["lambda_rate_smoothing"] = weights["rate_smoothing"]

    historical_paths.insert(0, "evaluation_env", env_name)
    historical_paths.insert(1, "loss_function", loss_function)
    historical_paths.insert(2, "scenario_group", scenario_group)
    historical_paths.insert(3, "scenario_id", scenario_id)
    historical_paths.insert(4, "beta", beta)
    historical_paths["lambda_inflation"] = weights["inflation"]
    historical_paths["lambda_output_gap"] = weights["output_gap"]
    historical_paths["lambda_rate_smoothing"] = weights["rate_smoothing"]
    return historical_paths, historical_df, stochastic_df


def empirical_scenarios_for_loss(loss_function: str) -> list[dict[str, Any]]:
    base_weights = ARTIFICIAL_BASE if loss_function == "artificial" else REVEALED_BASE
    lambda_scenarios = ARTIFICIAL_LAMBDA_SCENARIOS if loss_function == "artificial" else REVEALED_LAMBDA_SCENARIOS
    scenarios: list[dict[str, Any]] = []
    for item in BETA_SCENARIOS:
        scenarios.append({"scenario_id": item["scenario_id"], "scenario_group": item["scenario_group"], "beta": item["beta"], "weights": base_weights})
    scenarios.extend(lambda_scenarios)
    return scenarios


def top_non_rl_loss(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["is_rl"] = work["algo"].isin(["ppo", "td3", "sac"])
    grouped = []
    group_cols = ["evaluation_env", "loss_function", "evaluation_type", "scenario_group", "scenario_id"]
    for keys, sub in work.groupby(group_cols):
        metric_col = "total_discounted_loss" if keys[2] == "historical_counterfactual" else "mean_discounted_loss"
        best_rl = sub.loc[sub["is_rl"]].sort_values(metric_col).head(1)
        best_non_rl = sub.loc[~sub["is_rl"]].sort_values(metric_col).head(1)
        if best_rl.empty or best_non_rl.empty:
            continue
        rl_row = best_rl.iloc[0]
        base_row = best_non_rl.iloc[0]
        grouped.append(
            {
                "evaluation_env": keys[0],
                "loss_function": keys[1],
                "evaluation_type": keys[2],
                "scenario_group": keys[3],
                "scenario_id": keys[4],
                "best_rl_policy": rl_row["policy_name"],
                "best_rl_loss": rl_row[metric_col],
                "best_baseline_policy": base_row["policy_name"],
                "best_baseline_loss": base_row[metric_col],
                "rl_advantage_pct": (base_row[metric_col] - rl_row[metric_col]) / base_row[metric_col] * 100.0,
            }
        )
    return pd.DataFrame(grouped)


def make_empirical_heatmap(compare_df: pd.DataFrame) -> None:
    if compare_df.empty:
        return
    targets = [("artificial", "historical_counterfactual"), ("artificial", "stochastic_long_run"), ("revealed", "historical_counterfactual"), ("revealed", "stochastic_long_run")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, (loss_function, evaluation_type) in zip(axes.flat, targets):
        sub = compare_df[(compare_df["loss_function"] == loss_function) & (compare_df["evaluation_type"] == evaluation_type)].copy()
        if sub.empty:
            ax.axis("off")
            continue
        pivot = sub.pivot(index="scenario_id", columns="evaluation_env", values="rl_advantage_pct").reindex(columns=["svar", "ann"])
        im = ax.imshow(pivot.to_numpy(dtype=float), cmap="RdYlGn", aspect="auto")
        ax.set_title(f"{loss_function} / {evaluation_type}")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), pivot.index)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.iloc[i, j]:.1f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Phase 15: RL relative to best non-RL baseline (%)", fontsize=14)
    fig.savefig(FIG_DIR / "phase15_empirical_rank_heatmap.png", dpi=180)
    plt.close(fig)


def make_empirical_contexts() -> dict[str, EmpiricalContext]:
    return {"svar": build_svar_context(ROOT), "ann": build_ann_context(ROOT)}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def retrain_empirical_bundle(
    context: EmpiricalContext,
    env_name: str,
    loss_function: str,
    scenario_id: str,
    beta: float,
    weights: dict[str, float],
    unified_registry: pd.DataFrame,
    meta_df: pd.DataFrame,
    extra_policies: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario_context = clone_context(context, beta, weights, f"retrain_{loss_function}_{scenario_id}")
    out_dir = RETRAIN_DIR / env_name / loss_function / scenario_id
    out_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = out_dir / "configs"
    checkpoints_dir = out_dir / "checkpoints"
    configs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    trained_policies: dict[str, Any] = {}
    registry_rows: list[dict[str, Any]] = []
    coeff_rows: list[dict[str, Any]] = []
    log_frames: list[pd.DataFrame] = []

    algo_runs = [
        ("ppo", run_ppo, ROOT / "src" / "monetary_rl" / "config" / "benchmark_ppo_tuned.json", {"gamma": beta, "linear_policy": True}),
        ("td3", run_td3, ROOT / "src" / "monetary_rl" / "config" / "benchmark_td3.json", {"gamma": beta}),
        ("sac", run_sac, ROOT / "src" / "monetary_rl" / "config" / "benchmark_sac.json", {"gamma": beta}),
    ]

    for algo, runner, config_path, overrides in algo_runs:
        policy_name = f"{algo}_{env_name}_{loss_function}_{scenario_id}_phase15"
        env = make_empirical_env(scenario_context, seed=EMPIRICAL_RETRAIN_SEED)
        result, policy_fn = runner(env, config_path, eval_episodes=16, seed=EMPIRICAL_RETRAIN_SEED, **overrides)
        trained_policies[policy_name] = policy_fn

        checkpoint_path = checkpoints_dir / f"{policy_name}.pt"
        config_output_path = configs_dir / f"{policy_name}.json"
        if algo == "ppo":
            torch.save(result["policy_state_dict"], checkpoint_path)
        else:
            torch.save(result["actor_state_dict"], checkpoint_path)
        save_json(config_output_path, result["config"])

        eval_env = make_empirical_env(scenario_context, seed=EMPIRICAL_RETRAIN_SEED + 501)
        eval_stats = evaluate_policy(eval_env, policy_fn, episodes=32, gamma=beta, seed=EMPIRICAL_RETRAIN_SEED + 20_000)
        coeff = fit_linear_policy_response(policy_name, policy_fn, scenario_context.observation_low, scenario_context.observation_high, grid_points=7)
        coeff.update(
            {
                "training_env": env_name,
                "loss_function": loss_function,
                "scenario_id": scenario_id,
                "algo": algo,
                "seed": EMPIRICAL_RETRAIN_SEED,
                "policy_parameterization": "linear_policy" if algo == "ppo" else "standard_nonlinear",
            }
        )
        coeff_rows.append(coeff)

        log_df = training_log_frame(result, algo, EMPIRICAL_RETRAIN_SEED)
        log_df.insert(0, "policy_name", policy_name)
        log_df.insert(1, "training_env", env_name)
        log_df.insert(2, "loss_function", loss_function)
        log_df.insert(3, "scenario_id", scenario_id)
        log_frames.append(log_df)
        registry_rows.append(
            {
                "policy_name": policy_name,
                "rule_family": "phase15_retrained",
                "source_env": env_name,
                "training_env": env_name,
                "callable_type": "checkpoint",
                "algo": algo,
                "seed": EMPIRICAL_RETRAIN_SEED,
                "policy_parameterization": "linear_policy" if algo == "ppo" else "standard_nonlinear",
                "checkpoint_path": str(checkpoint_path),
                "config_path": str(config_output_path),
                "intercept": coeff["intercept"],
                "inflation_gap": coeff["inflation_gap"],
                "output_gap": coeff["output_gap"],
                "lagged_policy_rate_gap": coeff["lagged_policy_rate_gap"],
                "fit_rmse": coeff["fit_rmse"],
                "mean_discounted_loss": eval_stats["mean_discounted_loss"],
                "std_discounted_loss": eval_stats["std_discounted_loss"],
                "mean_reward": eval_stats["mean_reward"],
                "clip_rate": eval_stats["clip_rate"],
                "explosion_rate": eval_stats["explosion_rate"],
                "note": f"Phase15 retrained {algo.upper()} policy in {env_name} under {loss_function}/{scenario_id}.",
            }
        )

    registry_df = pd.DataFrame(registry_rows)
    coeff_df = pd.DataFrame(coeff_rows)
    log_df = pd.concat(log_frames, ignore_index=True, sort=False)
    registry_df.to_csv(out_dir / "policy_registry.csv", index=False)
    coeff_df.to_csv(out_dir / "policy_coefficients.csv", index=False)
    log_df.to_csv(out_dir / "training_logs.csv", index=False)

    base_policy_map = cfe.build_policy_map(scenario_context, unified_registry)
    if extra_policies:
        base_policy_map.update(extra_policies)
    base_policy_map.update(trained_policies)
    retrain_meta = pd.DataFrame(
        [
            {
                "policy_name": row["policy_name"],
                "rule_family": "phase15_retrained",
                "source_env": env_name,
                "training_env": env_name,
                "policy_parameterization": row["policy_parameterization"],
                "algo": row["algo"],
            }
            for row in registry_rows
        ]
    )
    local_meta = pd.concat([meta_df, retrain_meta], ignore_index=True).drop_duplicates("policy_name", keep="last")
    _, hist_df, stoch_df = evaluate_empirical_scenario(
        base_context=context,
        env_name=env_name,
        loss_function=loss_function,
        scenario_id=scenario_id,
        scenario_group="retrain",
        beta=beta,
        weights=weights,
        unified_registry=unified_registry,
        meta_df=local_meta,
        extra_policies={**(extra_policies or {}), **trained_policies},
    )
    combined = pd.concat([hist_df, stoch_df], ignore_index=True, sort=False)
    combined.to_csv(out_dir / "evaluation_matrix.csv", index=False)
    return registry_df, combined


def make_modified_phase11_model(spec: dict[str, Any], beta: float, weights: dict[str, float]):
    config, model = make_model(spec)
    config = deepcopy(config)
    config.discount_factor = float(beta)
    config.loss_weights = {k: float(v) for k, v in weights.items()}
    model.config.discount_factor = float(beta)
    model.config.loss_weights = {k: float(v) for k, v in weights.items()}
    return config, model


def affine_policy_from_row(row: pd.Series | dict[str, Any]):
    payload = row if isinstance(row, dict) else row.to_dict()
    theta = np.array([float(payload["intercept"]), float(payload["inflation_gap"]), float(payload["output_gap"]), float(payload["lagged_policy_rate_gap"])])
    return policy_from_theta(theta)


def evaluate_phase11_robustness() -> pd.DataFrame:
    output_path = OUTPUT_DIR / "phase11_beta_lambda_matrix.csv"
    if output_path.exists():
        return pd.read_csv(output_path)
    rl_df = pd.read_csv(ROOT / "outputs" / "phase11" / "extreme_matrix_v2" / "raw_rl_results.csv")
    coeff_df = pd.read_csv(ROOT / "outputs" / "phase11" / "extreme_matrix_v2" / "policy_coefficients.csv")
    benchmark_config = LQBenchmarkConfig.from_json(LINEAR_CONFIG_PATH)
    taylor_rule = load_taylor_rule(TAYLOR_RULE_PATH)
    scenario_rows: list[dict[str, Any]] = []
    scenarios = [{"scenario_id": item["scenario_id"], "scenario_group": item["scenario_group"], "beta": item["beta"], "weights": ARTIFICIAL_BASE} for item in BETA_SCENARIOS] + ARTIFICIAL_LAMBDA_SCENARIOS
    for spec in NEW_ENV_SPECS_V2:
        grouped = (
            rl_df.loc[rl_df["env_id"] == spec["env_id"]]
            .groupby(["algo", "seed"], as_index=False)["mean_discounted_loss"]
            .mean()
            .sort_values("mean_discounted_loss")
            .reset_index(drop=True)
        )
        best_rl = grouped.iloc[0]
        best_rl_coeff = coeff_df.loc[(coeff_df["env_id"] == spec["env_id"]) & (coeff_df["algo"] == best_rl["algo"]) & (coeff_df["seed"] == best_rl["seed"])].iloc[0]
        linear_search_coeff = coeff_df.loc[(coeff_df["env_id"] == spec["env_id"]) & (coeff_df["policy"] == "linear_policy_search")].iloc[0]
        for scenario in scenarios:
            beta = float(scenario["beta"])
            weights = scenario["weights"]
            modified_benchmark = deepcopy(benchmark_config)
            modified_benchmark.discount_factor = beta
            modified_benchmark.loss_weights = {k: float(v) for k, v in weights.items()}
            benchmark_model = LQBenchmarkModel(modified_benchmark)
            riccati_solution = solve_discounted_lq_riccati(benchmark_model)
            riccati_policy = build_optimal_linear_policy(riccati_solution)
            taylor_policy, _ = build_taylor_gap_policy(taylor_rule, modified_benchmark)
            _, model = make_modified_phase11_model(spec, beta, weights)
            env = LQBenchmarkEnv(model, BenchmarkEnvConfig(**spec["env_kwargs"]))
            policies = {
                "best_rl_affine": affine_policy_from_row(best_rl_coeff),
                "linear_policy_search": affine_policy_from_row(linear_search_coeff),
                "riccati_reference_recomputed": riccati_policy,
                "empirical_taylor_rule": taylor_policy,
            }
            for idx, (policy_name, policy_fn) in enumerate(policies.items()):
                stats = evaluate_policy(env, policy_fn, episodes=48, gamma=beta, seed=42000 + idx)
                scenario_rows.append(
                    {
                        "env_id": spec["env_id"],
                        "group": spec["group"],
                        "tier": spec["tier"],
                        "scenario_group": scenario["scenario_group"],
                        "scenario_id": scenario["scenario_id"],
                        "beta": beta,
                        "lambda_inflation": weights["inflation"],
                        "lambda_output_gap": weights["output_gap"],
                        "lambda_rate_smoothing": weights["rate_smoothing"],
                        "policy_name": policy_name,
                        "algo": "rl" if policy_name == "best_rl_affine" else "baseline",
                        **stats,
                    }
                )
    phase11_df = pd.DataFrame(scenario_rows).sort_values(["env_id", "scenario_id", "mean_discounted_loss"]).reset_index(drop=True)
    phase11_df.to_csv(output_path, index=False)
    return phase11_df


def retrain_phase11_selected() -> pd.DataFrame:
    output_path = OUTPUT_DIR / "phase11_selected_retrain_summary.csv"
    if output_path.exists():
        return pd.read_csv(output_path)
    rl_df = pd.read_csv(ROOT / "outputs" / "phase11" / "extreme_matrix_v2" / "raw_rl_results.csv")
    rows: list[dict[str, Any]] = []
    for case_id in PHASE11_CASES_FOR_RETRAIN:
        spec = next(spec for spec in NEW_ENV_SPECS_V2 if spec["env_id"] == case_id)
        best_algo = (
            rl_df.loc[rl_df["env_id"] == case_id]
            .groupby(["algo", "seed"], as_index=False)["mean_discounted_loss"]
            .mean()
            .sort_values("mean_discounted_loss")
            .iloc[0]["algo"]
        )
        for scenario in PHASE11_RETRAIN_SCENARIOS:
            beta = float(scenario["beta"])
            weights = scenario["weights"]
            _, model = make_modified_phase11_model(spec, beta, weights)
            env = LQBenchmarkEnv(model, BenchmarkEnvConfig(**spec["env_kwargs"]))
            out_dir = PHASE11_RETRAIN_DIR / case_id / scenario["scenario_id"]
            out_dir.mkdir(parents=True, exist_ok=True)
            if best_algo == "ppo":
                result, policy_fn = run_ppo(env, PHASE11_PPO_CONFIG_PATH, eval_episodes=48, seed=PHASE11_RETRAIN_SEED, gamma=beta, **PPO_OVERRIDES)
                checkpoint = result["policy_state_dict"]
            elif best_algo == "sac":
                result, policy_fn = run_sac(env, PHASE11_SAC_CONFIG_PATH, eval_episodes=48, seed=PHASE11_RETRAIN_SEED, gamma=beta, **OFFPOLICY_OVERRIDES)
                checkpoint = result["actor_state_dict"]
            else:
                result, policy_fn = run_td3(env, PHASE11_TD3_CONFIG_PATH, eval_episodes=48, seed=PHASE11_RETRAIN_SEED, gamma=beta, **OFFPOLICY_OVERRIDES)
                checkpoint = result["actor_state_dict"]
            torch.save(checkpoint, out_dir / f"{best_algo}_{case_id}_{scenario['scenario_id']}.pt")
            save_json(out_dir / f"{best_algo}_{case_id}_{scenario['scenario_id']}.json", result["config"])
            training_log_frame(result, best_algo, PHASE11_RETRAIN_SEED).to_csv(out_dir / "training_log.csv", index=False)
            stats = evaluate_policy(env, policy_fn, episodes=48, gamma=beta, seed=52000)
            stats = {k: v for k, v in stats.items() if k not in {"first_trajectory", "episode_rewards", "discounted_losses"}}
            rows.append(
                {
                    "env_id": case_id,
                    "scenario_id": scenario["scenario_id"],
                    "beta": beta,
                    "lambda_inflation": weights["inflation"],
                    "lambda_output_gap": weights["output_gap"],
                    "lambda_rate_smoothing": weights["rate_smoothing"],
                    "algo": best_algo,
                    **stats,
                }
            )
    retrain_df = pd.DataFrame(rows)
    retrain_df.to_csv(output_path, index=False)
    return retrain_df


def make_phase11_heatmap(phase11_df: pd.DataFrame) -> None:
    compare = []
    for keys, sub in phase11_df.groupby(["env_id", "scenario_id"]):
        best_rl = sub.loc[sub["policy_name"] == "best_rl_affine"].iloc[0]
        riccati = sub.loc[sub["policy_name"] == "riccati_reference_recomputed"].iloc[0]
        compare.append(
            {
                "env_id": keys[0],
                "scenario_id": keys[1],
                "advantage_pct": (riccati["mean_discounted_loss"] - best_rl["mean_discounted_loss"]) / riccati["mean_discounted_loss"] * 100.0,
            }
        )
    cmp_df = pd.DataFrame(compare)
    if cmp_df.empty:
        return
    pivot = cmp_df.pivot(index="scenario_id", columns="env_id", values="advantage_pct")
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Phase 15: Best RL vs recomputed Riccati in phase11 cases (%)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(FIG_DIR / "phase15_phase11_riccati_heatmap.png", dpi=180)
    plt.close(fig)


def write_summary(empirical_compare: pd.DataFrame, empirical_retrain: pd.DataFrame, phase11_df: pd.DataFrame, phase11_retrain: pd.DataFrame) -> None:
    lines = ["# Phase 15 beta / lambda 稳健性", ""]
    if not empirical_compare.empty:
        lines.extend(
            [
                "## 经验环境：最佳 RL 相对最佳非 RL 基线",
                "",
                empirical_compare.sort_values(["loss_function", "evaluation_type", "evaluation_env", "scenario_id"]).to_markdown(index=False),
                "",
            ]
        )
        summary_counts = (
            empirical_compare.assign(rl_wins=empirical_compare["rl_advantage_pct"] > 0)
            .groupby(["loss_function", "evaluation_type", "evaluation_env"], as_index=False)["rl_wins"]
            .agg(["sum", "count"])
            .reset_index()
        )
        lines.extend(["## 经验环境：RL 胜出次数", "", summary_counts.to_markdown(index=False), ""])
    if not empirical_retrain.empty:
        lines.extend(["## 经验环境：少量重训检查", "", empirical_retrain.to_markdown(index=False), ""])
    if not phase11_df.empty:
        phase11_cmp = []
        for keys, sub in phase11_df.groupby(["env_id", "scenario_id"]):
            best_rl = sub.loc[sub["policy_name"] == "best_rl_affine"].iloc[0]
            riccati = sub.loc[sub["policy_name"] == "riccati_reference_recomputed"].iloc[0]
            phase11_cmp.append(
                {
                    "env_id": keys[0],
                    "scenario_id": keys[1],
                    "best_rl_loss": best_rl["mean_discounted_loss"],
                    "riccati_loss": riccati["mean_discounted_loss"],
                    "rl_advantage_pct": (riccati["mean_discounted_loss"] - best_rl["mean_discounted_loss"]) / riccati["mean_discounted_loss"] * 100.0,
                }
            )
        lines.extend(["## phase11 六环境：最佳 RL 相对 Riccati 外推", "", pd.DataFrame(phase11_cmp).sort_values(["env_id", "scenario_id"]).to_markdown(index=False), ""])
    if not phase11_retrain.empty:
        keep_cols = [col for col in ["env_id", "scenario_id", "algo", "mean_discounted_loss", "std_discounted_loss", "mean_abs_action", "clip_rate", "explosion_rate"] if col in phase11_retrain.columns]
        lines.extend(["## phase11 代表环境：少量重训检查", "", phase11_retrain[keep_cols].to_markdown(index=False), ""])
    lines.extend(
        [
            "## 文件",
            "",
            "- `empirical_beta_lambda_full_matrix.csv`：SVAR / ANN 全部重评估矩阵",
            "- `empirical_rank_compare.csv`：最佳 RL 相对最佳非 RL 基线比较",
            "- `empirical_retrain_summary.csv`：经验环境少量重训比较",
            "- `phase11_beta_lambda_matrix.csv`：phase11 六环境稳健性表",
            "- `phase11_selected_retrain_summary.csv`：phase11 代表环境少量重训",
            "- `figures/phase15_empirical_rank_heatmap.png`：经验环境稳健性热图",
            "- `figures/phase15_phase11_riccati_heatmap.png`：phase11 相对 Riccati 热图",
        ]
    )
    (OUTPUT_DIR / "phase15_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    empirical_matrix_path = OUTPUT_DIR / "empirical_beta_lambda_full_matrix.csv"
    empirical_compare_path = OUTPUT_DIR / "empirical_rank_compare.csv"
    empirical_retrain_path = OUTPUT_DIR / "empirical_retrain_summary.csv"

    if empirical_matrix_path.exists() and empirical_compare_path.exists() and empirical_retrain_path.exists():
        empirical_matrix = pd.read_csv(empirical_matrix_path)
        empirical_compare = pd.read_csv(empirical_compare_path)
        empirical_retrain_df = pd.read_csv(empirical_retrain_path)
        if not (FIG_DIR / "phase15_empirical_rank_heatmap.png").exists():
            make_empirical_heatmap(empirical_compare)
    else:
        unified_registry = cfe.load_unified_registry()
        ann_search_policies, ann_search_meta = load_ann_search_policies()
        meta_df = policy_metadata(unified_registry, ann_search_meta)
        contexts = make_empirical_contexts()

        historical_frames: list[pd.DataFrame] = []
        matrix_frames: list[pd.DataFrame] = []
        for env_name, base_context in contexts.items():
            for loss_function in ["artificial", "revealed"]:
                extra_policies = ann_search_policies if env_name == "ann" else None
                for scenario in empirical_scenarios_for_loss(loss_function):
                    historical_paths, historical_df, stochastic_df = evaluate_empirical_scenario(
                        base_context=base_context,
                        env_name=env_name,
                        loss_function=loss_function,
                        scenario_id=scenario["scenario_id"],
                        scenario_group=scenario["scenario_group"],
                        beta=float(scenario["beta"]),
                        weights=scenario["weights"],
                        unified_registry=unified_registry,
                        meta_df=meta_df,
                        extra_policies=extra_policies,
                    )
                    historical_frames.append(historical_paths)
                    matrix_frames.extend([historical_df, stochastic_df])

        historical_all = pd.concat(historical_frames, ignore_index=True, sort=False)
        empirical_matrix = pd.concat(matrix_frames, ignore_index=True, sort=False)
        historical_all.to_csv(OUTPUT_DIR / "empirical_historical_paths.csv", index=False)
        empirical_matrix.to_csv(empirical_matrix_path, index=False)

        empirical_compare = top_non_rl_loss(empirical_matrix)
        empirical_compare.to_csv(empirical_compare_path, index=False)
        make_empirical_heatmap(empirical_compare)

        empirical_retrain_rows: list[dict[str, Any]] = []
        for env_name, base_context in contexts.items():
            extra_policies = ann_search_policies if env_name == "ann" else None
            for loss_function, scenarios in EMPIRICAL_RETRAIN_SCENARIOS.items():
                for scenario in scenarios:
                    registry_df, eval_df = retrain_empirical_bundle(
                        context=base_context,
                        env_name=env_name,
                        loss_function=loss_function,
                        scenario_id=scenario["scenario_id"],
                        beta=float(scenario["beta"]),
                        weights=scenario["weights"],
                        unified_registry=unified_registry,
                        meta_df=meta_df,
                        extra_policies=extra_policies,
                    )
                    for _, row in registry_df.iterrows():
                        policy_name = row["policy_name"]
                        hist_sub = eval_df[(eval_df["policy_name"] == policy_name) & (eval_df["evaluation_type"] == "historical_counterfactual")]
                        stoch_sub = eval_df[(eval_df["policy_name"] == policy_name) & (eval_df["evaluation_type"] == "stochastic_long_run")]
                        empirical_retrain_rows.append(
                            {
                                "evaluation_env": env_name,
                                "loss_function": loss_function,
                                "scenario_id": scenario["scenario_id"],
                                "policy_name": policy_name,
                                "algo": row["algo"],
                                "historical_loss": float(hist_sub["total_discounted_loss"].iloc[0]),
                                "stochastic_loss": float(stoch_sub["mean_discounted_loss"].iloc[0]),
                            }
                        )
        empirical_retrain_df = pd.DataFrame(empirical_retrain_rows)
        empirical_retrain_df.to_csv(empirical_retrain_path, index=False)

    phase11_df = evaluate_phase11_robustness()
    make_phase11_heatmap(phase11_df)
    phase11_retrain_df = retrain_phase11_selected()
    write_summary(empirical_compare, empirical_retrain_df, phase11_df, phase11_retrain_df)


if __name__ == "__main__":
    main()
