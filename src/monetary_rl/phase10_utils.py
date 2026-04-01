from __future__ import annotations

import json
import pickle
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from monetary_rl.agents.ppo import PPOConfig, PPOTrainer
from monetary_rl.agents.sac import SACConfig, SACTrainer
from monetary_rl.agents.td3 import TD3Config, TD3Trainer
from monetary_rl.envs.empirical_env import EmpiricalEnvConfig, EmpiricalSVAREnv
from monetary_rl.evaluation import evaluate_policy, fit_linear_policy_response
from monetary_rl.experiment_utils import (
    build_taylor_gap_policy,
    load_taylor_rule,
    policy_row,
    run_ppo,
    run_sac,
    run_td3,
    training_log_frame,
)
from monetary_rl.models.empirical_ann import EmpiricalANNConfig, EmpiricalANNModel
from monetary_rl.models.empirical_svar import EmpiricalSVARConfig, EmpiricalSVARModel
from monetary_rl.models.lq_benchmark import LQBenchmarkConfig, LQBenchmarkModel
from monetary_rl.solvers.riccati import build_optimal_linear_policy, solve_discounted_lq_riccati


@dataclass
class EmpiricalContext:
    name: str
    model: Any
    env_df: pd.DataFrame
    initial_states: np.ndarray
    shock_pool: np.ndarray
    action_dates: pd.Series
    state_dates: pd.Series
    observation_low: tuple[float, float, float]
    observation_high: tuple[float, float, float]


def clone_context_with_loss_weights(
    context: EmpiricalContext,
    loss_weights: dict[str, float],
    name_suffix: str,
) -> EmpiricalContext:
    cloned_model = deepcopy(context.model)
    cloned_model.config.loss_weights = {k: float(v) for k, v in loss_weights.items()}
    return EmpiricalContext(
        name=f"{context.name}_{name_suffix}",
        model=cloned_model,
        env_df=context.env_df.copy(),
        initial_states=context.initial_states.copy(),
        shock_pool=context.shock_pool.copy(),
        action_dates=context.action_dates.copy(),
        state_dates=context.state_dates.copy(),
        observation_low=context.observation_low,
        observation_high=context.observation_high,
    )


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def add_lags(frame: pd.DataFrame, columns: list[str], max_lag: int) -> pd.DataFrame:
    out = frame.copy()
    for lag in range(1, max_lag + 1):
        for col in columns:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def build_linear_policy(
    name: str,
    intercept: float,
    inflation_gap: float,
    output_gap_coef: float,
    lagged_policy_rate_gap: float,
):
    def policy(state: np.ndarray, t: int) -> float:
        del t
        pi_gap, output_gap_now, lagged_rate_gap = np.asarray(state, dtype=float)
        return (
            intercept
            + inflation_gap * pi_gap
            + output_gap_coef * output_gap_now
            + lagged_policy_rate_gap * lagged_rate_gap
        )

    policy.__name__ = name
    return policy


def observed_state_support(model: Any, initial_states: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    observed = np.asarray([model.observe(state) for state in initial_states], dtype=float)
    low = np.quantile(observed, 0.05, axis=0)
    high = np.quantile(observed, 0.95, axis=0)
    too_close = np.isclose(low, high)
    low = np.where(too_close, low - 1.0, low)
    high = np.where(too_close, high + 1.0, high)
    return tuple(float(v) for v in low), tuple(float(v) for v in high)


def build_svar_context(root: str | Path) -> EmpiricalContext:
    root_path = Path(root)
    config_path = root_path / "src" / "monetary_rl" / "config" / "benchmark_lq.json"
    data_path = root_path / "data" / "processed" / "macro_quarterly_sample_1987Q3_2007Q2.csv"
    phase2_dir = root_path / "outputs" / "phase2"

    benchmark_config = LQBenchmarkConfig.from_json(config_path)
    macro = pd.read_csv(data_path, parse_dates=["date"])
    env_df = add_lags(macro, ["inflation", "output_gap", "policy_rate"], max_lag=2).dropna().reset_index(drop=True)

    svar_output = load_json(phase2_dir / "svar_output_gap.json")
    svar_inflation = load_json(phase2_dir / "svar_inflation.json")
    output_fit = pd.read_csv(phase2_dir / "svar_output_gap_fitted.csv")
    inflation_fit = pd.read_csv(phase2_dir / "svar_inflation_fitted.csv")
    if len(env_df) != len(output_fit) or len(env_df) != len(inflation_fit):
        raise ValueError("Phase 2 fitted outputs and lagged empirical frame are misaligned.")

    model = EmpiricalSVARModel(
        EmpiricalSVARConfig(
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
    )
    initial_states = env_df[
        ["inflation", "inflation_lag1", "output_gap", "output_gap_lag1", "policy_rate_lag1"]
    ].to_numpy(dtype=float)
    shock_pool = np.column_stack(
        [output_fit["output_gap_resid"].to_numpy(dtype=float), inflation_fit["inflation_resid"].to_numpy(dtype=float)]
    )
    observation_low, observation_high = observed_state_support(model, initial_states)
    action_dates = env_df["date"].iloc[:-1].reset_index(drop=True)
    state_dates = env_df["date"].iloc[: len(action_dates) + 1].reset_index(drop=True)
    return EmpiricalContext(
        name="svar",
        model=model,
        env_df=env_df,
        initial_states=initial_states,
        shock_pool=shock_pool,
        action_dates=action_dates,
        state_dates=state_dates,
        observation_low=observation_low,
        observation_high=observation_high,
    )


def build_ann_context(root: str | Path) -> EmpiricalContext:
    root_path = Path(root)
    config_path = root_path / "src" / "monetary_rl" / "config" / "benchmark_lq.json"
    data_path = root_path / "data" / "processed" / "macro_quarterly_sample_1987Q3_2007Q2.csv"
    phase9_dir = root_path / "outputs" / "phase9"
    models_dir = phase9_dir / "models"

    benchmark_config = LQBenchmarkConfig.from_json(config_path)
    output_payload = load_json(phase9_dir / "ann_output_gap_tuned.json")
    inflation_payload = load_json(phase9_dir / "ann_inflation_tuned.json")
    output_fit_df = pd.read_csv(phase9_dir / "ann_output_gap_tuned_fitted.csv")
    inflation_fit_df = pd.read_csv(phase9_dir / "ann_inflation_tuned_fitted.csv")
    max_lag = int(max(output_payload["state_max_lag"], inflation_payload["state_max_lag"]))

    macro = pd.read_csv(data_path, parse_dates=["date"])
    env_df = add_lags(macro, ["inflation", "output_gap", "policy_rate"], max_lag=max_lag).dropna().reset_index(drop=True)
    with (models_dir / "ann_output_gap_pipeline.pkl").open("rb") as f:
        output_pipeline = pickle.load(f)
    with (models_dir / "ann_inflation_pipeline.pkl").open("rb") as f:
        inflation_pipeline = pickle.load(f)

    model = EmpiricalANNModel(
        EmpiricalANNConfig(
            name="empirical_ann_tuned",
            inflation_target=benchmark_config.inflation_target,
            neutral_rate=benchmark_config.neutral_rate,
            discount_factor=benchmark_config.discount_factor,
            loss_weights=benchmark_config.loss_weights,
            output_regressors=list(output_payload["regressors"]),
            inflation_regressors=list(inflation_payload["regressors"]),
            state_max_lag=max_lag,
            sample_start=str(env_df["quarter"].iloc[0]),
            sample_end=str(env_df["quarter"].iloc[-1]),
            action_low=-2.0,
            action_high=8.0,
            output_spec=output_payload,
            inflation_spec=inflation_payload,
        ),
        output_pipeline=output_pipeline,
        inflation_pipeline=inflation_pipeline,
    )
    cols = ["inflation", "inflation_lag1", "output_gap", "output_gap_lag1", "policy_rate_lag1"]
    for lag in range(2, max_lag):
        cols.extend([f"inflation_lag{lag}", f"output_gap_lag{lag}", f"policy_rate_lag{lag}"])
    initial_states = env_df[cols].to_numpy(dtype=float)
    shock_pool = np.column_stack(
        [
            output_fit_df["output_gap_resid"].to_numpy(dtype=float),
            inflation_fit_df["inflation_resid"].to_numpy(dtype=float),
        ]
    )
    observation_low, observation_high = observed_state_support(model, initial_states)
    action_dates = env_df["date"].iloc[:-1].reset_index(drop=True)
    state_dates = env_df["date"].iloc[: len(action_dates) + 1].reset_index(drop=True)
    return EmpiricalContext(
        name="ann",
        model=model,
        env_df=env_df,
        initial_states=initial_states,
        shock_pool=shock_pool,
        action_dates=action_dates,
        state_dates=state_dates,
        observation_low=observation_low,
        observation_high=observation_high,
    )


def make_empirical_env(context: EmpiricalContext, horizon: int = 80, seed: int = 0) -> EmpiricalSVAREnv:
    return EmpiricalSVAREnv(
        model=context.model,
        initial_states=context.initial_states,
        shock_pool=context.shock_pool,
        config=EmpiricalEnvConfig(
            horizon=horizon,
            action_low=context.model.config.action_low,
            action_high=context.model.config.action_high,
            seed=seed,
        ),
    )


def base_policy_registry(root: str | Path, context: EmpiricalContext) -> tuple[pd.DataFrame, dict[str, Any]]:
    root_path = Path(root)
    registry_df = pd.read_csv(root_path / "outputs" / "phase8" / "policy_registry.csv")
    registry_df = registry_df.copy()
    registry_df["callable_type"] = np.where(registry_df["intercept"].notna(), "linear", "historical")
    registry_df["rule_family"] = registry_df["policy_name"].map(
        {
            "historical_actual_policy": "historical_actual",
            "empirical_taylor_rule": "empirical_rule",
            "riccati_reference": "theory_reference",
            "linear_policy_search_transfer": "benchmark_transfer",
            "ppo_benchmark_transfer": "benchmark_transfer",
            "td3_benchmark_transfer": "benchmark_transfer",
            "sac_benchmark_transfer": "benchmark_transfer",
        }
    )
    registry_df["source_env"] = np.where(
        registry_df["rule_family"] == "benchmark_transfer",
        "benchmark",
        registry_df["rule_family"].fillna("benchmark"),
    )
    policy_map: dict[str, Any] = {}
    actual_gaps = context.env_df["policy_rate"].to_numpy(dtype=float) - context.model.config.neutral_rate

    def historical_actual_policy(state: np.ndarray, t: int) -> float:
        del state
        idx = min(t, len(actual_gaps) - 1)
        return float(actual_gaps[idx])

    policy_map["historical_actual_policy"] = historical_actual_policy
    for row in registry_df.to_dict("records"):
        if row["policy_name"] == "historical_actual_policy":
            continue
        if not np.isfinite(row.get("intercept", np.nan)):
            continue
        policy_map[row["policy_name"]] = build_linear_policy(
            row["policy_name"],
            float(row["intercept"]),
            float(row["inflation_gap"]),
            float(row["output_gap"]),
            float(row["lagged_policy_rate_gap"]),
        )
    return registry_df, policy_map


def training_support_payload(context: EmpiricalContext) -> dict[str, Any]:
    shock_cov = np.cov(context.shock_pool.T, ddof=1)
    return {
        "environment": context.name,
        "sample_start": str(context.env_df["quarter"].iloc[0]),
        "sample_end": str(context.env_df["quarter"].iloc[-1]),
        "state_observation_low": list(context.observation_low),
        "state_observation_high": list(context.observation_high),
        "initial_state_count": int(len(context.initial_states)),
        "shock_pool_size": int(len(context.shock_pool)),
        "shock_covariance": np.asarray(shock_cov, dtype=float).round(6).tolist(),
        "action_low": float(context.model.config.action_low),
        "action_high": float(context.model.config.action_high),
    }


def _save_checkpoint(result: dict[str, Any], algo: str, checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    key = "policy_state_dict" if algo == "ppo" else "actor_state_dict"
    torch.save(result[key], checkpoint_path)


def _save_config(result: dict[str, Any], config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(result["config"], indent=2, ensure_ascii=False), encoding="utf-8")


def train_policy_bundle(
    context: EmpiricalContext,
    output_dir: str | Path,
    ppo_config_path: str | Path,
    sac_config_path: str | Path,
    td3_config_path: str | Path,
    seed: int = 43,
    ppo_linear_policy: bool = True,
    ppo_policy_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "evaluation_support.json", training_support_payload(context))

    runs = [
        ("ppo", ppo_config_path, {"linear_policy": ppo_linear_policy}, ppo_policy_name or f"ppo_{context.name}_direct"),
        ("td3", td3_config_path, {}, f"td3_{context.name}_direct"),
        ("sac", sac_config_path, {}, f"sac_{context.name}_direct"),
    ]

    registry_rows: list[dict[str, Any]] = []
    coeff_rows: list[dict[str, Any]] = []
    train_logs: list[pd.DataFrame] = []
    eval_rows: list[dict[str, Any]] = []

    for algo, config_path, overrides, policy_name in runs:
        env = make_empirical_env(context, seed=seed)
        if algo == "ppo":
            result, policy_fn = run_ppo(env, config_path, eval_episodes=16, seed=seed, **overrides)
            log_df = training_log_frame(result, algo, seed)
            parameterization = "linear_policy" if ppo_linear_policy else "nonlinear_policy"
        elif algo == "td3":
            result, policy_fn = run_td3(env, config_path, eval_episodes=16, seed=seed)
            log_df = training_log_frame(result, algo, seed)
            parameterization = "standard_nonlinear"
        else:
            result, policy_fn = run_sac(env, config_path, eval_episodes=16, seed=seed)
            log_df = training_log_frame(result, algo, seed)
            parameterization = "standard_nonlinear"

        log_df.insert(0, "training_env", context.name)
        log_df.insert(0, "policy_name", policy_name)
        train_logs.append(log_df)

        checkpoint_path = out_dir / "checkpoints" / f"{policy_name}.pt"
        config_output_path = out_dir / "configs" / f"{policy_name}.json"
        _save_checkpoint(result, algo, checkpoint_path)
        _save_config(result, config_output_path)

        eval_env = make_empirical_env(context, seed=seed + 500)
        eval_stats = policy_row(policy_name, eval_env, policy_fn, episodes=32, seed=seed + 10_000)
        coeff_row = fit_linear_policy_response(
            policy_name,
            policy_fn,
            context.observation_low,
            context.observation_high,
            grid_points=7,
        )
        coeff_row.update(
            {
                "training_env": context.name,
                "algo": algo,
                "seed": seed,
                "policy_parameterization": parameterization,
            }
        )
        coeff_rows.append(coeff_row)

        registry_rows.append(
            {
                "policy_name": policy_name,
                "rule_family": f"{context.name}_direct",
                "source_env": context.name,
                "training_env": context.name,
                "callable_type": "checkpoint",
                "algo": algo,
                "seed": seed,
                "policy_parameterization": parameterization,
                "checkpoint_path": str(checkpoint_path),
                "config_path": str(config_output_path),
                "intercept": coeff_row["intercept"],
                "inflation_gap": coeff_row["inflation_gap"],
                "output_gap": coeff_row["output_gap"],
                "lagged_policy_rate_gap": coeff_row["lagged_policy_rate_gap"],
                "fit_rmse": coeff_row["fit_rmse"],
                "mean_discounted_loss": eval_stats["mean_discounted_loss"],
                "std_discounted_loss": eval_stats["std_discounted_loss"],
                "mean_reward": eval_stats["mean_reward"],
                "clip_rate": eval_stats["clip_rate"],
                "explosion_rate": eval_stats["explosion_rate"],
                "note": f"Direct-trained {algo.upper()} policy in the {context.name.upper()} empirical environment.",
            }
        )
        eval_rows.append({"policy": policy_name, "training_env": context.name, **eval_stats})

    registry_df = pd.DataFrame(registry_rows).sort_values("mean_discounted_loss").reset_index(drop=True)
    coeff_df = pd.DataFrame(coeff_rows).sort_values(["algo", "seed"]).reset_index(drop=True)
    log_df = pd.concat(train_logs, ignore_index=True, sort=False)
    eval_df = pd.DataFrame(eval_rows).sort_values("mean_discounted_loss").reset_index(drop=True)

    registry_df.to_csv(out_dir / "policy_registry.csv", index=False)
    coeff_df.to_csv(out_dir / "policy_coefficients.csv", index=False)
    log_df.to_csv(out_dir / "training_logs.csv", index=False)
    eval_df.to_csv(out_dir / "training_evaluation.csv", index=False)
    return registry_df, coeff_df, log_df


def load_checkpoint_policy(row: pd.Series | dict[str, Any], context: EmpiricalContext):
    row_dict = row if isinstance(row, dict) else row.to_dict()
    env = make_empirical_env(context, seed=int(row_dict.get("seed", 0)))
    algo = str(row_dict["algo"])
    config_payload = load_json(row_dict["config_path"])
    checkpoint = torch.load(row_dict["checkpoint_path"], map_location="cpu", weights_only=True)

    if algo == "ppo":
        config = PPOConfig(**config_payload)
        trainer = PPOTrainer(env, config)
        trainer.model.load_state_dict(checkpoint)

        def policy(state: np.ndarray, t: int) -> float:
            del t
            return trainer._deterministic_action(state)

        return policy

    if algo == "td3":
        config = TD3Config(**config_payload)
        trainer = TD3Trainer(env, config)
        trainer.actor.load_state_dict(checkpoint)

        def policy(state: np.ndarray, t: int) -> float:
            del t
            return trainer._deterministic_action(state)

        return policy

    config = SACConfig(**config_payload)
    trainer = SACTrainer(env, config)
    trainer.actor.load_state_dict(checkpoint)

    def policy(state: np.ndarray, t: int) -> float:
        del t
        return trainer._deterministic_action(state)

    return policy


def simulate_historical_counterfactual(
    model: Any,
    policy_map: dict[str, Any],
    initial_state: np.ndarray,
    action_dates: pd.Series,
    state_dates: pd.Series,
    shocks: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
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
                    "discounted_loss": loss * (model.config.discount_factor**t),
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
    work = path_df.dropna(subset=["loss"]).copy()
    summary = (
        work.groupby("policy_name", as_index=False)
        .agg(
            total_discounted_loss=("discounted_loss", "sum"),
            mean_period_loss=("loss", "mean"),
            mean_sq_inflation_gap=("inflation_gap", lambda s: float(np.mean(np.square(s)))),
            mean_sq_output_gap=("output_gap", lambda s: float(np.mean(np.square(s)))),
            mean_sq_rate_change=("rate_change", lambda s: float(np.mean(np.square(s)))),
            mean_policy_rate=("policy_rate", "mean"),
            std_policy_rate=("policy_rate", "std"),
        )
        .sort_values("total_discounted_loss")
        .reset_index(drop=True)
    )
    actual_loss = float(summary.loc[summary["policy_name"] == "historical_actual_policy", "total_discounted_loss"].iloc[0])
    taylor_loss = float(summary.loc[summary["policy_name"] == "empirical_taylor_rule", "total_discounted_loss"].iloc[0])
    summary["improvement_vs_actual_pct"] = (actual_loss - summary["total_discounted_loss"]) / actual_loss * 100.0
    summary["improvement_vs_taylor_pct"] = (taylor_loss - summary["total_discounted_loss"]) / taylor_loss * 100.0
    return summary


def stochastic_policy_summary(
    context: EmpiricalContext,
    policies: dict[str, Any],
    horizon: int = 120,
    episodes: int = 96,
    seed: int = 20260401,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, (policy_name, policy_fn) in enumerate(policies.items()):
        env = make_empirical_env(context, horizon=horizon, seed=seed + idx)
        stats = evaluate_policy(
            env=env,
            policy_fn=policy_fn,
            episodes=episodes,
            gamma=context.model.config.discount_factor,
            seed=seed + 100 * (idx + 1),
        )
        rows.append(
            {
                "policy_name": policy_name,
                "evaluation_env": context.name,
                "mean_discounted_loss": stats["mean_discounted_loss"],
                "std_discounted_loss": stats["std_discounted_loss"],
                "mean_reward": stats["mean_reward"],
                "mean_abs_action": stats["mean_abs_action"],
                "clip_rate": stats["clip_rate"],
                "explosion_rate": stats["explosion_rate"],
            }
        )
    return pd.DataFrame(rows).sort_values("mean_discounted_loss").reset_index(drop=True)
