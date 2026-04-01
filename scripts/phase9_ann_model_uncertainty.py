from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monetary_rl.envs import BenchmarkEnvConfig, EmpiricalEnvConfig, EmpiricalSVAREnv, LQBenchmarkEnv
from monetary_rl.evaluation import evaluate_policy
from monetary_rl.experiment_utils import build_taylor_gap_policy, load_taylor_rule
from monetary_rl.models import (
    AsymmetricBenchmarkConfig,
    AsymmetricBenchmarkModel,
    EmpiricalANNConfig,
    EmpiricalANNModel,
    EmpiricalSVARConfig,
    EmpiricalSVARModel,
    LQBenchmarkConfig,
    LQBenchmarkModel,
    NonlinearBenchmarkConfig,
    NonlinearBenchmarkModel,
)
from monetary_rl.solvers import build_optimal_linear_policy, solve_discounted_lq_riccati


DATA_PATH = ROOT / "data" / "processed" / "macro_quarterly_sample_1987Q3_2007Q2.csv"
PHASE2_DIR = ROOT / "outputs" / "phase2"
PHASE7_DIR = ROOT / "outputs" / "phase7" / "matrix"
PHASE8_DIR = ROOT / "outputs" / "phase8"
LINEAR_CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_lq.json"
OUTPUT_DIR = ROOT / "outputs" / "phase9"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"

ANN_STOCHASTIC_HORIZON = 80
ANN_STOCHASTIC_EPISODES = 192
ANN_STOCHASTIC_SEED = 29_000
LOCAL_ENV_EPISODES = 64
LOCAL_ENV_SEED = 41_000

OUTPUT_BASE_REGS = ["output_gap_lag1", "inflation_lag1", "policy_rate_lag1", "policy_rate_lag2"]
OUTPUT_EXTRA_REGS = OUTPUT_BASE_REGS + ["output_gap_lag2", "inflation_lag2", "policy_rate_lag3"]
INFLATION_BASE_REGS = [
    "output_gap",
    "output_gap_lag1",
    "output_gap_lag2",
    "inflation_lag1",
    "inflation_lag2",
    "policy_rate_lag1",
]
INFLATION_EXTRA_REGS = INFLATION_BASE_REGS + ["output_gap_lag3", "inflation_lag3", "policy_rate_lag2"]

LOCAL_ENV_SPECS = [
    {
        "env_id": "benchmark",
        "group": "benchmark",
        "tier": "baseline",
        "model_kind": "linear",
        "config_path": LINEAR_CONFIG_PATH,
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
    },
    {
        "env_id": "nonlinear_mild",
        "group": "nonlinear",
        "tier": "mild",
        "model_kind": "nonlinear",
        "config_path": ROOT / "src" / "monetary_rl" / "config" / "nonlinear_phillips_mild.json",
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
    },
    {
        "env_id": "nonlinear_medium",
        "group": "nonlinear",
        "tier": "medium",
        "model_kind": "nonlinear",
        "config_path": ROOT / "src" / "monetary_rl" / "config" / "nonlinear_phillips_medium.json",
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
    },
    {
        "env_id": "nonlinear_strong",
        "group": "nonlinear",
        "tier": "strong",
        "model_kind": "nonlinear",
        "config_path": ROOT / "src" / "monetary_rl" / "config" / "nonlinear_phillips_strong.json",
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
    },
    {
        "env_id": "zlb_mild",
        "group": "zlb",
        "tier": "mild",
        "model_kind": "linear",
        "config_path": LINEAR_CONFIG_PATH,
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
    },
    {
        "env_id": "zlb_medium",
        "group": "zlb",
        "tier": "medium",
        "model_kind": "linear",
        "config_path": LINEAR_CONFIG_PATH,
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
    },
    {
        "env_id": "zlb_strong",
        "group": "zlb",
        "tier": "strong",
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
    },
    {
        "env_id": "asymmetric_mild",
        "group": "asymmetric",
        "tier": "mild",
        "model_kind": "asymmetric",
        "config_path": ROOT / "src" / "monetary_rl" / "config" / "asymmetric_loss_mild.json",
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
    },
    {
        "env_id": "asymmetric_medium",
        "group": "asymmetric",
        "tier": "medium",
        "model_kind": "asymmetric",
        "config_path": ROOT / "src" / "monetary_rl" / "config" / "asymmetric_loss_medium.json",
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
    },
    {
        "env_id": "asymmetric_strong",
        "group": "asymmetric",
        "tier": "strong",
        "model_kind": "asymmetric",
        "config_path": ROOT / "src" / "monetary_rl" / "config" / "asymmetric_loss_strong.json",
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
    },
]


def add_lags(df: pd.DataFrame, columns: list[str], max_lag: int) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        for lag in range(1, max_lag + 1):
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def time_split_indices(n: int, train_frac: float = 0.7, val_frac: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(np.floor(n * train_frac))
    val_end = int(np.floor(n * (train_frac + val_frac)))
    idx = np.arange(n)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


def build_pipeline(spec: dict[str, Any]) -> Pipeline:
    mlp_kwargs = {
        "hidden_layer_sizes": spec["hidden_layer_sizes"],
        "activation": spec["activation"],
        "solver": spec["solver"],
        "alpha": spec["alpha"],
        "random_state": spec["seed"],
        "max_iter": spec["max_iter"],
    }
    if spec["solver"] == "adam":
        mlp_kwargs["learning_rate_init"] = spec["learning_rate_init"]
    return Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(**mlp_kwargs))])


def candidate_specs(stage_name: str) -> list[dict[str, Any]]:
    if stage_name == "stage1_same_input_structure":
        hidden_sizes = [(2,), (3,), (4,), (3, 2)]
        return [
            {
                "stage_name": stage_name,
                "hidden_layer_sizes": hidden,
                "activation": "tanh",
                "solver": "lbfgs",
                "alpha": 1e-4,
                "seed": seed,
                "max_iter": 2500,
                "learning_rate_init": 1e-3,
            }
            for hidden in hidden_sizes
            for seed in range(3)
        ]
    hidden_sizes = [(2,), (3,), (4,), (3, 2)]
    rows: list[dict[str, Any]] = []
    for hidden in hidden_sizes:
        for activation in ["tanh", "relu"]:
            for solver in ["lbfgs", "adam"]:
                for alpha in [1e-4, 1e-3]:
                    for seed in range(2):
                        learning_rates = [1e-3]
                        for lr in learning_rates:
                            rows.append(
                                {
                                    "stage_name": stage_name,
                                    "hidden_layer_sizes": hidden,
                                    "activation": activation,
                                    "solver": solver,
                                    "alpha": alpha,
                                    "seed": seed,
                                    "max_iter": 2500,
                                    "learning_rate_init": lr,
                                }
                            )
    return rows


def stage_priority(stage_name: str) -> int:
    return {
        "stage1_same_input_structure": 1,
        "stage2_same_input_training": 2,
        "stage3_extra_lag": 3,
    }[stage_name]


def run_search_stage(
    equation_name: str,
    df: pd.DataFrame,
    target: str,
    regressors: list[str],
    feature_set: str,
    stage_name: str,
) -> pd.DataFrame:
    model_df = df[["date", "quarter", target] + regressors].dropna().reset_index(drop=True)
    X = model_df[regressors].to_numpy(dtype=float)
    y = model_df[target].to_numpy(dtype=float)
    train_idx, val_idx, test_idx = time_split_indices(len(model_df))

    rows: list[dict[str, Any]] = []
    for spec in candidate_specs(stage_name):
        pipeline = build_pipeline(spec)
        pipeline.fit(X[train_idx], y[train_idx])
        train_pred = pipeline.predict(X[train_idx])
        val_pred = pipeline.predict(X[val_idx])
        test_pred = pipeline.predict(X[test_idx])
        full_pred = pipeline.predict(X)
        rows.append(
            {
                "equation": equation_name,
                "target": target,
                "feature_set": feature_set,
                "stage_name": stage_name,
                "stage_priority": stage_priority(stage_name),
                "nobs": len(model_df),
                "regressor_count": len(regressors),
                "regressors": "|".join(regressors),
                "hidden_layer_sizes": str(spec["hidden_layer_sizes"]),
                "activation": spec["activation"],
                "solver": spec["solver"],
                "alpha": spec["alpha"],
                "learning_rate_init": spec["learning_rate_init"],
                "seed": spec["seed"],
                "train_mse": float(mean_squared_error(y[train_idx], train_pred)),
                "val_mse": float(mean_squared_error(y[val_idx], val_pred)),
                "test_mse": float(mean_squared_error(y[test_idx], test_pred)),
                "full_sample_mse": float(mean_squared_error(y, full_pred)),
                "residual_std_full_sample": float(np.std(y - full_pred, ddof=1)),
                "n_iter": int(pipeline.named_steps["mlp"].n_iter_),
            }
        )
    return pd.DataFrame(rows).sort_values(["val_mse", "test_mse", "full_sample_mse", "train_mse"]).reset_index(drop=True)


def choose_best_spec(search_df: pd.DataFrame) -> dict[str, Any]:
    row = search_df.sort_values(["val_mse", "test_mse", "full_sample_mse", "stage_priority"]).iloc[0]
    return row.to_dict()


def required_max_lag(regressors: list[str]) -> int:
    max_lag = 2
    for reg in regressors:
        if "lag" in reg:
            max_lag = max(max_lag, int(reg.split("lag")[-1]))
    return max_lag


def refit_best_ann(
    df: pd.DataFrame,
    target: str,
    regressors: list[str],
    spec: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, Pipeline]:
    model_df = df[["date", "quarter", target] + regressors].dropna().reset_index(drop=True)
    X = model_df[regressors].to_numpy(dtype=float)
    y = model_df[target].to_numpy(dtype=float)
    train_idx, val_idx, test_idx = time_split_indices(len(model_df))
    hidden = tuple(int(x.strip()) for x in spec["hidden_layer_sizes"].strip("()").split(",") if x.strip())
    pipeline = build_pipeline(
        {
            "hidden_layer_sizes": hidden,
            "activation": spec["activation"],
            "solver": spec["solver"],
            "alpha": float(spec["alpha"]),
            "seed": int(spec["seed"]),
            "max_iter": 5000,
            "learning_rate_init": float(spec["learning_rate_init"]),
        }
    )
    pipeline.fit(X[train_idx], y[train_idx])
    preds = pipeline.predict(X)
    residuals = y - preds
    result = {
        "name": spec["equation"],
        "feature_set": spec["feature_set"],
        "stage_name": spec["stage_name"],
        "regressors": regressors,
        "hidden_layer_sizes": spec["hidden_layer_sizes"],
        "activation": spec["activation"],
        "solver": spec["solver"],
        "alpha": float(spec["alpha"]),
        "learning_rate_init": float(spec["learning_rate_init"]),
        "random_state": int(spec["seed"]),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "train_mse": float(mean_squared_error(y[train_idx], pipeline.predict(X[train_idx]))),
        "val_mse": float(mean_squared_error(y[val_idx], pipeline.predict(X[val_idx]))),
        "test_mse": float(mean_squared_error(y[test_idx], pipeline.predict(X[test_idx]))),
        "full_sample_mse": float(mean_squared_error(y, preds)),
        "residual_std_full_sample": float(np.std(residuals, ddof=1)),
        "n_iter": int(pipeline.named_steps["mlp"].n_iter_),
        "nobs": int(len(model_df)),
    }
    fitted = model_df.copy()
    fitted[f"{target}_fitted"] = preds
    fitted[f"{target}_resid"] = residuals
    return result, fitted, pipeline


def fit_historical_proxy(path_df: pd.DataFrame) -> dict[str, Any]:
    subset = path_df[(path_df["policy_name"] == "historical_actual_policy") & path_df["policy_rate_gap"].notna()].copy()
    X = np.column_stack(
        [
            np.ones(len(subset)),
            subset["inflation_gap"].to_numpy(dtype=float),
            subset["output_gap"].to_numpy(dtype=float),
            subset["lagged_policy_rate_gap"].to_numpy(dtype=float),
        ]
    )
    y = subset["policy_rate_gap"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    return {
        "policy_name": "historical_actual_proxy",
        "rule_type": "linear_proxy",
        "source": "phase8_actual_path_fit",
        "intercept": float(beta[0]),
        "inflation_gap": float(beta[1]),
        "output_gap": float(beta[2]),
        "lagged_policy_rate_gap": float(beta[3]),
        "fit_rmse": float(np.sqrt(np.mean((y - y_hat) ** 2))),
        "note": "Linear proxy fitted to historical actual policy path in empirical sample",
    }


def build_linear_policy(intercept: float, inflation_gap: float, output_gap: float, lagged_rate_gap: float):
    def policy(state: np.ndarray, t: int) -> float:
        del t
        obs = np.asarray(state, dtype=float).reshape(3)
        return float(intercept + inflation_gap * obs[0] + output_gap * obs[1] + lagged_rate_gap * obs[2])

    return policy


def build_feedback_policy_registry() -> tuple[pd.DataFrame, dict[str, Any]]:
    registry_df = pd.read_csv(PHASE8_DIR / "policy_registry.csv")
    proxy_row = fit_historical_proxy(pd.read_csv(PHASE8_DIR / "historical_counterfactual_paths.csv"))
    registry_df = pd.concat([registry_df, pd.DataFrame([proxy_row])], ignore_index=True)

    policy_map: dict[str, Any] = {}
    for row in registry_df.to_dict("records"):
        if not np.isfinite(row.get("intercept", np.nan)):
            continue
        policy_map[row["policy_name"]] = build_linear_policy(
            float(row["intercept"]),
            float(row["inflation_gap"]),
            float(row["output_gap"]),
            float(row["lagged_policy_rate_gap"]),
        )
    return registry_df, policy_map


def make_initial_states(env_df: pd.DataFrame, max_lag: int) -> np.ndarray:
    cols = ["inflation", "inflation_lag1", "output_gap", "output_gap_lag1", "policy_rate_lag1"]
    for lag in range(2, max_lag):
        cols.extend([f"inflation_lag{lag}", f"output_gap_lag{lag}", f"policy_rate_lag{lag}"])
    return env_df[cols].to_numpy(dtype=float)


def build_empirical_svar_model() -> tuple[EmpiricalSVARModel, pd.DataFrame, np.ndarray]:
    benchmark_config = LQBenchmarkConfig.from_json(LINEAR_CONFIG_PATH)
    macro = pd.read_csv(DATA_PATH, parse_dates=["date"])
    env_df = add_lags(macro, ["inflation", "output_gap", "policy_rate"], max_lag=2).dropna().reset_index(drop=True)
    svar_output = load_json(PHASE2_DIR / "svar_output_gap.json")
    svar_inflation = load_json(PHASE2_DIR / "svar_inflation.json")
    output_fit = pd.read_csv(PHASE2_DIR / "svar_output_gap_fitted.csv")
    inflation_fit = pd.read_csv(PHASE2_DIR / "svar_inflation_fitted.csv")
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


def tune_ann_models() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    macro = pd.read_csv(DATA_PATH, parse_dates=["date"])
    env_df_lag2 = add_lags(macro, ["inflation", "output_gap", "policy_rate"], max_lag=2).dropna().reset_index(drop=True)
    env_df_lag3 = add_lags(macro, ["inflation", "output_gap", "policy_rate"], max_lag=3).dropna().reset_index(drop=True)

    output_search = pd.concat(
        [
            run_search_stage("ann_output_gap_tuned", env_df_lag2, "output_gap", OUTPUT_BASE_REGS, "same_input", "stage1_same_input_structure"),
            run_search_stage("ann_output_gap_tuned", env_df_lag2, "output_gap", OUTPUT_BASE_REGS, "same_input", "stage2_same_input_training"),
            run_search_stage("ann_output_gap_tuned", env_df_lag3, "output_gap", OUTPUT_EXTRA_REGS, "extra_lag", "stage3_extra_lag"),
        ],
        ignore_index=True,
    )
    inflation_search = pd.concat(
        [
            run_search_stage("ann_inflation_tuned", env_df_lag2, "inflation", INFLATION_BASE_REGS, "same_input", "stage1_same_input_structure"),
            run_search_stage("ann_inflation_tuned", env_df_lag2, "inflation", INFLATION_BASE_REGS, "same_input", "stage2_same_input_training"),
            run_search_stage("ann_inflation_tuned", env_df_lag3, "inflation", INFLATION_EXTRA_REGS, "extra_lag", "stage3_extra_lag"),
        ],
        ignore_index=True,
    )

    output_best = choose_best_spec(output_search)
    inflation_best = choose_best_spec(inflation_search)
    output_regs = OUTPUT_EXTRA_REGS if output_best["feature_set"] == "extra_lag" else OUTPUT_BASE_REGS
    inflation_regs = INFLATION_EXTRA_REGS if inflation_best["feature_set"] == "extra_lag" else INFLATION_BASE_REGS
    common_max_lag = max(required_max_lag(output_regs), required_max_lag(inflation_regs))
    common_env_df = add_lags(macro, ["inflation", "output_gap", "policy_rate"], max_lag=common_max_lag).dropna().reset_index(drop=True)

    output_result, output_fit_df, output_pipeline = refit_best_ann(common_env_df, "output_gap", output_regs, output_best)
    inflation_result, inflation_fit_df, inflation_pipeline = refit_best_ann(common_env_df, "inflation", inflation_regs, inflation_best)
    output_payload = {**output_result, "state_max_lag": common_max_lag}
    inflation_payload = {**inflation_result, "state_max_lag": common_max_lag}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with (MODELS_DIR / "ann_output_gap_pipeline.pkl").open("wb") as f:
        pickle.dump(output_pipeline, f)
    with (MODELS_DIR / "ann_inflation_pipeline.pkl").open("wb") as f:
        pickle.dump(inflation_pipeline, f)

    write_json(OUTPUT_DIR / "ann_output_gap_tuned.json", output_payload)
    write_json(OUTPUT_DIR / "ann_inflation_tuned.json", inflation_payload)
    output_search.to_csv(OUTPUT_DIR / "ann_output_search.csv", index=False)
    inflation_search.to_csv(OUTPUT_DIR / "ann_inflation_search.csv", index=False)
    output_fit_df.to_csv(OUTPUT_DIR / "ann_output_gap_tuned_fitted.csv", index=False)
    inflation_fit_df.to_csv(OUTPUT_DIR / "ann_inflation_tuned_fitted.csv", index=False)
    return output_payload, inflation_payload, common_env_df, output_fit_df, inflation_fit_df


def build_empirical_ann_model(
    output_payload: dict[str, Any],
    inflation_payload: dict[str, Any],
    env_df: pd.DataFrame,
    output_fit_df: pd.DataFrame,
    inflation_fit_df: pd.DataFrame,
) -> tuple[EmpiricalANNModel, np.ndarray]:
    benchmark_config = LQBenchmarkConfig.from_json(LINEAR_CONFIG_PATH)
    with (MODELS_DIR / "ann_output_gap_pipeline.pkl").open("rb") as f:
        output_pipeline = pickle.load(f)
    with (MODELS_DIR / "ann_inflation_pipeline.pkl").open("rb") as f:
        inflation_pipeline = pickle.load(f)

    config = EmpiricalANNConfig(
        name="empirical_ann_tuned",
        inflation_target=benchmark_config.inflation_target,
        neutral_rate=benchmark_config.neutral_rate,
        discount_factor=benchmark_config.discount_factor,
        loss_weights=benchmark_config.loss_weights,
        output_regressors=list(output_payload["regressors"]),
        inflation_regressors=list(inflation_payload["regressors"]),
        state_max_lag=int(output_payload["state_max_lag"]),
        sample_start=str(env_df["quarter"].iloc[0]),
        sample_end=str(env_df["quarter"].iloc[-1]),
        action_low=-2.0,
        action_high=8.0,
        output_spec=output_payload,
        inflation_spec=inflation_payload,
    )
    model = EmpiricalANNModel(config, output_pipeline=output_pipeline, inflation_pipeline=inflation_pipeline)
    shocks = np.column_stack(
        [
            output_fit_df["output_gap_resid"].to_numpy(dtype=float),
            inflation_fit_df["inflation_resid"].to_numpy(dtype=float),
        ]
    )
    return model, shocks


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
            rows.append(
                {
                    "policy_name": policy_name,
                    "action_date": pd.Timestamp(action_date),
                    "state_date": pd.Timestamp(state_dates.iloc[t]),
                    "inflation": float(full_state[0]),
                    "inflation_gap": float(obs[0]),
                    "output_gap": float(obs[1]),
                    "lagged_policy_rate": float(full_state[4]),
                    "lagged_policy_rate_gap": float(obs[2]),
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


def evaluate_empirical_feedback_policies(
    model: Any,
    initial_states: np.ndarray,
    shock_pool: np.ndarray,
    policies: dict[str, Any],
    horizon: int,
    episodes: int,
    seed_base: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, (policy_name, policy_fn) in enumerate(policies.items()):
        env = EmpiricalSVAREnv(
            model=model,
            initial_states=initial_states,
            shock_pool=shock_pool,
            config=EmpiricalEnvConfig(
                horizon=horizon,
                action_low=model.config.action_low,
                action_high=model.config.action_high,
                seed=seed_base + idx,
            ),
        )
        stats = evaluate_policy(env, policy_fn, episodes, model.config.discount_factor, seed_base + 100 * (idx + 1))
        rows.append(
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
    return pd.DataFrame(rows).sort_values("mean_discounted_loss").reset_index(drop=True)


def make_local_env(spec: dict[str, Any]):
    if spec["model_kind"] == "linear":
        model = LQBenchmarkModel(LQBenchmarkConfig.from_json(spec["config_path"]))
    elif spec["model_kind"] == "nonlinear":
        model = NonlinearBenchmarkModel(NonlinearBenchmarkConfig.from_json(spec["config_path"]))
    elif spec["model_kind"] == "asymmetric":
        model = AsymmetricBenchmarkModel(AsymmetricBenchmarkConfig.from_json(spec["config_path"]))
    else:
        raise ValueError(f"Unknown model kind: {spec['model_kind']}")
    return LQBenchmarkEnv(model, BenchmarkEnvConfig(**spec["env_kwargs"]))


def evaluate_local_model_uncertainty(
    feedback_registry_df: pd.DataFrame,
    feedback_policy_map: dict[str, Any],
    svar_model: EmpiricalSVARModel,
    svar_initial_states: np.ndarray,
    svar_shock_pool: np.ndarray,
    ann_model: EmpiricalANNModel,
    ann_initial_states: np.ndarray,
    ann_shock_pool: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for spec in LOCAL_ENV_SPECS:
        env = make_local_env(spec)
        for idx, row in enumerate(feedback_registry_df.to_dict("records")):
            stats = evaluate_policy(
                env,
                feedback_policy_map[row["policy_name"]],
                LOCAL_ENV_EPISODES,
                env.model.config.discount_factor,
                LOCAL_ENV_SEED + 1000 * len(rows) + idx,
            )
            rows.append(
                {
                    "env_id": spec["env_id"],
                    "group": spec["group"],
                    "tier": spec["tier"],
                    "policy_name": row["policy_name"],
                    "mean_discounted_loss": stats["mean_discounted_loss"],
                    "std_discounted_loss": stats["std_discounted_loss"],
                    "mean_reward": stats["mean_reward"],
                    "mean_abs_action": stats["mean_abs_action"],
                    "clip_rate": stats["clip_rate"],
                    "explosion_rate": stats["explosion_rate"],
                }
            )

    empirical_specs = [
        ("empirical_svar", "empirical", "svar", svar_model, svar_initial_states, svar_shock_pool),
        ("empirical_ann", "empirical", "ann", ann_model, ann_initial_states, ann_shock_pool),
    ]
    for env_id, group, tier, model, initial_states, shock_pool in empirical_specs:
        for idx, row in enumerate(feedback_registry_df.to_dict("records")):
            env = EmpiricalSVAREnv(
                model=model,
                initial_states=initial_states,
                shock_pool=shock_pool,
                config=EmpiricalEnvConfig(
                    horizon=ANN_STOCHASTIC_HORIZON,
                    action_low=model.config.action_low,
                    action_high=model.config.action_high,
                    seed=LOCAL_ENV_SEED + idx,
                ),
            )
            stats = evaluate_policy(
                env,
                feedback_policy_map[row["policy_name"]],
                ANN_STOCHASTIC_EPISODES,
                model.config.discount_factor,
                LOCAL_ENV_SEED + 50_000 + 100 * (idx + 1),
            )
            rows.append(
                {
                    "env_id": env_id,
                    "group": group,
                    "tier": tier,
                    "policy_name": row["policy_name"],
                    "mean_discounted_loss": stats["mean_discounted_loss"],
                    "std_discounted_loss": stats["std_discounted_loss"],
                    "mean_reward": stats["mean_reward"],
                    "mean_abs_action": stats["mean_abs_action"],
                    "clip_rate": stats["clip_rate"],
                    "explosion_rate": stats["explosion_rate"],
                }
            )

    detail_df = pd.DataFrame(rows)
    detail_df["rank_within_env"] = detail_df.groupby("env_id")["mean_discounted_loss"].rank(method="dense")
    best_by_env = detail_df.groupby("env_id")["mean_discounted_loss"].transform("min")
    detail_df["loss_gap_vs_best_pct"] = (detail_df["mean_discounted_loss"] - best_by_env) / best_by_env * 100.0
    aggregate_df = (
        detail_df.groupby("policy_name")
        .agg(
            mean_rank=("rank_within_env", "mean"),
            median_rank=("rank_within_env", "median"),
            win_count=("rank_within_env", lambda s: int(np.sum(np.isclose(s, 1.0)))),
            top2_count=("rank_within_env", lambda s: int(np.sum(s <= 2.0))),
            mean_gap_vs_best_pct=("loss_gap_vs_best_pct", "mean"),
            median_gap_vs_best_pct=("loss_gap_vs_best_pct", "median"),
            max_gap_vs_best_pct=("loss_gap_vs_best_pct", "max"),
            mean_clip_rate=("clip_rate", "mean"),
            mean_explosion_rate=("explosion_rate", "mean"),
        )
        .reset_index()
        .sort_values(["mean_rank", "mean_gap_vs_best_pct", "median_gap_vs_best_pct"])
        .reset_index(drop=True)
    )
    return detail_df.sort_values(["env_id", "rank_within_env"]).reset_index(drop=True), aggregate_df


def make_fit_comparison(output_payload: dict[str, Any], inflation_payload: dict[str, Any], ann_reproduction_error: float, ann_stochastic_df: pd.DataFrame) -> pd.DataFrame:
    phase2_output = load_json(PHASE2_DIR / "ann_output_gap.json")
    phase2_inflation = load_json(PHASE2_DIR / "ann_inflation.json")
    svar_output = load_json(PHASE2_DIR / "svar_output_gap.json")
    svar_inflation = load_json(PHASE2_DIR / "svar_inflation.json")
    rows = [
        {
            "equation": "output_gap",
            "svar_mse": float(svar_output["rmse"]) ** 2,
            "phase2_ann_mse": float(phase2_output["full_sample_mse"]),
            "phase9_tuned_ann_mse": float(output_payload["full_sample_mse"]),
            "improvement_vs_phase2_pct": (float(phase2_output["full_sample_mse"]) - float(output_payload["full_sample_mse"])) / float(phase2_output["full_sample_mse"]) * 100.0,
            "improvement_vs_svar_pct": ((float(svar_output["rmse"]) ** 2) - float(output_payload["full_sample_mse"])) / (float(svar_output["rmse"]) ** 2) * 100.0,
        },
        {
            "equation": "inflation",
            "svar_mse": float(svar_inflation["rmse"]) ** 2,
            "phase2_ann_mse": float(phase2_inflation["full_sample_mse"]),
            "phase9_tuned_ann_mse": float(inflation_payload["full_sample_mse"]),
            "improvement_vs_phase2_pct": (float(phase2_inflation["full_sample_mse"]) - float(inflation_payload["full_sample_mse"])) / float(phase2_inflation["full_sample_mse"]) * 100.0,
            "improvement_vs_svar_pct": ((float(svar_inflation["rmse"]) ** 2) - float(inflation_payload["full_sample_mse"])) / (float(svar_inflation["rmse"]) ** 2) * 100.0,
        },
    ]
    fit_df = pd.DataFrame(rows)
    fit_df["inflation_gate"] = fit_df["equation"].eq("inflation") & (fit_df["phase9_tuned_ann_mse"] <= fit_df["svar_mse"] * 1.10)
    fit_df["output_gate"] = fit_df["equation"].eq("output_gap") & (
        (fit_df["phase9_tuned_ann_mse"] <= fit_df["phase2_ann_mse"] * 1.10)
        & (fit_df["phase9_tuned_ann_mse"] <= fit_df["svar_mse"])
    )
    fit_df["dynamic_gate_reference"] = ann_reproduction_error
    fit_df["ann_best_feedback_loss"] = float(ann_stochastic_df["mean_discounted_loss"].min())
    return fit_df


def make_phase9_gate(fit_df: pd.DataFrame, ann_reproduction_error: float, ann_stochastic_df: pd.DataFrame) -> pd.DataFrame:
    inflation_ok = bool(fit_df.loc[fit_df["equation"] == "inflation", "inflation_gate"].iloc[0])
    output_ok = bool(fit_df.loc[fit_df["equation"] == "output_gap", "output_gate"].iloc[0])
    dynamic_ok = ann_reproduction_error < 1e-5 and float(ann_stochastic_df["explosion_rate"].max()) == 0.0
    ann_status = "passed_as_supplementary" if (inflation_ok and output_ok and dynamic_ok) else "supplementary_only"
    return pd.DataFrame(
        [
            {
                "module": "ann_phase9_module",
                "status": ann_status,
                "reason": "inflation gate, output gate, and dynamic stability are jointly evaluated under Phase 9 rules",
                "next_step": "use ANN as supplementary empirical environment result" if ann_status == "passed_as_supplementary" else "report ANN fit comparison and limitations, but keep SVAR as empirical main result",
            },
            {
                "module": "local_model_uncertainty_module",
                "status": "completed",
                "reason": "frozen policy registry has been evaluated on benchmark, extension, and empirical local environments",
                "next_step": "use aggregate robustness table and gap distribution figure in writing",
            },
            {
                "module": "external_dsge_source",
                "status": "not_available_locally",
                "reason": "no external MMB/DSGE model files are present in the repository",
                "next_step": "if needed later, plug user-supplied DSGE/MMB models into the frozen registry evaluator",
            },
        ]
    )


def make_plots(
    fit_df: pd.DataFrame,
    ann_path_df: pd.DataFrame,
    ann_hist_df: pd.DataFrame,
    local_uncertainty_df: pd.DataFrame,
    local_aggregate_df: pd.DataFrame,
) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = fit_df.melt(
        id_vars=["equation"],
        value_vars=["svar_mse", "phase2_ann_mse", "phase9_tuned_ann_mse"],
        var_name="model_type",
        value_name="mse",
    )
    for idx, model_type in enumerate(["svar_mse", "phase2_ann_mse", "phase9_tuned_ann_mse"]):
        subset = plot_df[plot_df["model_type"] == model_type]
        ax.bar(np.arange(len(subset)) + 0.25 * idx - 0.25, subset["mse"], width=0.25, label=model_type)
    ax.set_xticks(np.arange(len(fit_df)))
    ax.set_xticklabels(fit_df["equation"])
    ax.set_ylabel("MSE")
    ax.set_title("Phase 9 ANN Fit Comparison")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase9_ann_fit_comparison.png", dpi=200)
    plt.close(fig)

    core_policies = ["historical_actual_policy", "empirical_taylor_rule", "riccati_reference", "sac_benchmark_transfer"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for policy_name in core_policies:
        subset = ann_path_df[ann_path_df["policy_name"] == policy_name]
        axes[0].plot(subset["state_date"], subset["inflation"], label=policy_name)
        axes[1].plot(subset["state_date"], subset["output_gap"], label=policy_name)
        rate_subset = subset.dropna(subset=["policy_rate"])
        axes[2].plot(rate_subset["action_date"], rate_subset["policy_rate"], label=policy_name)
    axes[0].axhline(2.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Phase 9 ANN Historical vs Counterfactual Paths: Inflation")
    axes[1].set_title("Phase 9 ANN Historical vs Counterfactual Paths: Output Gap")
    axes[2].set_title("Phase 9 ANN Historical vs Counterfactual Paths: Policy Rate")
    for ax in axes:
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase9_ann_historical_paths_core.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ann_hist_df["policy_name"], ann_hist_df["total_discounted_loss"])
    ax.set_title("Phase 9 ANN Historical Counterfactual Welfare")
    ax.set_ylabel("discounted loss")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase9_ann_historical_welfare.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(local_aggregate_df["policy_name"], local_aggregate_df["mean_gap_vs_best_pct"])
    ax.set_title("Phase 9 Local Model Uncertainty: Mean Gap vs Best")
    ax.set_ylabel("mean loss gap vs best (%)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase9_local_model_uncertainty_gaps.png", dpi=200)
    plt.close(fig)

    pivot = local_uncertainty_df.pivot(index="env_id", columns="policy_name", values="loss_gap_vs_best_pct").sort_index().reindex(columns=local_aggregate_df["policy_name"])
    fig, ax = plt.subplots(figsize=(12, 6))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Phase 9 Local Model Uncertainty Heatmap")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="loss gap vs best (%)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "phase9_local_model_uncertainty_heatmap.png", dpi=200)
    plt.close(fig)


def write_summary(
    output_payload: dict[str, Any],
    inflation_payload: dict[str, Any],
    fit_df: pd.DataFrame,
    ann_hist_df: pd.DataFrame,
    ann_stochastic_df: pd.DataFrame,
    local_aggregate_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    ann_reproduction_error: float,
) -> None:
    lines: list[str] = []
    lines.append("# Phase 9 ANN 补充与 Model Uncertainty 总结")
    lines.append("")
    lines.append("## 1. 任务完成情况")
    lines.append("")
    lines.append("| 项目 | 结果 |")
    lines.append("|---|---|")
    lines.append("| ANN 调优 | 已完成 |")
    lines.append("| ANN 补充反事实 | 已完成 |")
    lines.append("| local model uncertainty | 已完成 |")
    lines.append("| 外部 DSGE/MMB 模型 | 仓库中暂无现成文件，已保留规则接口 |")
    lines.append("")
    lines.append("## 2. ANN 调优选型")
    lines.append("")
    lines.append("| equation | feature_set | stage_name | hidden_layer_sizes | activation | solver | alpha | learning_rate_init | full_sample_mse |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    lines.append(f"| output_gap | {output_payload['feature_set']} | {output_payload['stage_name']} | {output_payload['hidden_layer_sizes']} | {output_payload['activation']} | {output_payload['solver']} | {output_payload['alpha']:.6f} | {output_payload['learning_rate_init']:.6f} | {output_payload['full_sample_mse']:.6f} |")
    lines.append(f"| inflation | {inflation_payload['feature_set']} | {inflation_payload['stage_name']} | {inflation_payload['hidden_layer_sizes']} | {inflation_payload['activation']} | {inflation_payload['solver']} | {inflation_payload['alpha']:.6f} | {inflation_payload['learning_rate_init']:.6f} | {inflation_payload['full_sample_mse']:.6f} |")
    lines.append("")
    lines.append("## 3. ANN 拟合门槛表")
    lines.append("")
    lines.append(fit_df.to_markdown(index=False))
    lines.append("")
    lines.append("## 4. ANN 历史反事实主表")
    lines.append("")
    lines.append(ann_hist_df.to_markdown(index=False))
    lines.append("")
    lines.append("## 5. ANN 长期随机评估")
    lines.append("")
    lines.append(ann_stochastic_df.to_markdown(index=False))
    lines.append("")
    lines.append("## 6. Local Model Uncertainty 汇总")
    lines.append("")
    lines.append(local_aggregate_df.to_markdown(index=False))
    lines.append("")
    lines.append("## 7. Phase 9 模块状态")
    lines.append("")
    lines.append(gate_df.to_markdown(index=False))
    lines.append("")
    lines.append("## 8. 说明")
    lines.append("")
    lines.append(f"- ANN 历史实际政策复现实验最大绝对误差为 `{ann_reproduction_error:.10f}`。")
    lines.append("- `Phase 9` 的 model uncertainty 模块使用仓库内可执行的本地结构模型族：benchmark、nonlinear、ZLB/ELB-tightness、asymmetric、empirical SVAR、empirical ANN。")
    lines.append("- 由于仓库中没有现成 `MMB` / 外部 DSGE 模型文件，本轮未机械复刻 11 个外部 DSGE；规则 registry 与评估接口已经为用户后续提供模型后继续扩展预留。")
    lines.append("- 写作中仍须明确 `Lucas critique`：经验环境中的固定转移仅为近似，Phase 9 的跨模型比较只能部分缓解该边界。")
    (OUTPUT_DIR / "phase9_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    output_payload, inflation_payload, ann_env_df, output_fit_df, inflation_fit_df = tune_ann_models()
    ann_model, ann_shocks_all = build_empirical_ann_model(output_payload, inflation_payload, ann_env_df, output_fit_df, inflation_fit_df)
    ann_initial_states = make_initial_states(ann_env_df.iloc[:-1].copy(), int(output_payload["state_max_lag"]))
    ann_initial_state = ann_initial_states[0].copy()
    ann_action_dates = ann_env_df["date"].iloc[:-1].reset_index(drop=True)
    ann_state_dates = ann_env_df["date"].iloc[: len(ann_action_dates) + 1].reset_index(drop=True)
    ann_historical_shocks = ann_shocks_all[1:].copy()

    registry_df, feedback_policy_map = build_feedback_policy_registry()
    feedback_policy_names = [
        "empirical_taylor_rule",
        "riccati_reference",
        "linear_policy_search_transfer",
        "ppo_benchmark_transfer",
        "td3_benchmark_transfer",
        "sac_benchmark_transfer",
    ]
    ann_actual_gaps = ann_env_df["policy_rate"].iloc[:-1].to_numpy(dtype=float) - ann_model.config.neutral_rate

    def ann_historical_actual_policy(state: np.ndarray, t: int) -> float:
        del state
        return float(ann_actual_gaps[min(t, len(ann_actual_gaps) - 1)])

    ann_policy_map = {"historical_actual_policy": ann_historical_actual_policy}
    ann_policy_map.update({name: feedback_policy_map[name] for name in feedback_policy_names})
    ann_path_df = simulate_historical_counterfactual(
        model=ann_model,
        policy_map=ann_policy_map,
        initial_state=ann_initial_state,
        action_dates=ann_action_dates,
        state_dates=ann_state_dates,
        shocks=ann_historical_shocks,
    )
    ann_hist_df = historical_summary(ann_path_df)
    ann_stochastic_df = evaluate_empirical_feedback_policies(
        model=ann_model,
        initial_states=ann_initial_states,
        shock_pool=ann_historical_shocks,
        policies={name: feedback_policy_map[name] for name in feedback_policy_names},
        horizon=ANN_STOCHASTIC_HORIZON,
        episodes=ANN_STOCHASTIC_EPISODES,
        seed_base=ANN_STOCHASTIC_SEED,
    )
    ann_hist_actual = ann_path_df[(ann_path_df["policy_name"] == "historical_actual_policy") & ann_path_df["policy_rate"].notna()].reset_index(drop=True)
    ann_reproduction_error = max(
        float(np.max(np.abs(ann_hist_actual["inflation"].to_numpy() - ann_env_df.iloc[:-1]["inflation"].to_numpy()))),
        float(np.max(np.abs(ann_hist_actual["output_gap"].to_numpy() - ann_env_df.iloc[:-1]["output_gap"].to_numpy()))),
    )
    fit_df = make_fit_comparison(output_payload, inflation_payload, ann_reproduction_error, ann_stochastic_df)

    svar_model, svar_env_df, svar_shocks_all = build_empirical_svar_model()
    svar_initial_states = make_initial_states(svar_env_df.iloc[:-1].copy(), 2)
    svar_historical_shocks = svar_shocks_all[1:].copy()
    feedback_registry_df = registry_df[registry_df["policy_name"] != "historical_actual_policy"].reset_index(drop=True)
    local_uncertainty_df, local_aggregate_df = evaluate_local_model_uncertainty(
        feedback_registry_df=feedback_registry_df,
        feedback_policy_map=feedback_policy_map,
        svar_model=svar_model,
        svar_initial_states=svar_initial_states,
        svar_shock_pool=svar_historical_shocks,
        ann_model=ann_model,
        ann_initial_states=ann_initial_states,
        ann_shock_pool=ann_historical_shocks,
    )
    gate_df = make_phase9_gate(fit_df, ann_reproduction_error, ann_stochastic_df)

    fit_df.to_csv(OUTPUT_DIR / "ann_fit_comparison.csv", index=False)
    registry_df.to_csv(OUTPUT_DIR / "phase9_policy_registry.csv", index=False)
    ann_path_df.to_csv(OUTPUT_DIR / "ann_historical_counterfactual_paths.csv", index=False)
    ann_hist_df.to_csv(OUTPUT_DIR / "ann_historical_welfare_summary.csv", index=False)
    ann_stochastic_df.to_csv(OUTPUT_DIR / "ann_stochastic_welfare_summary.csv", index=False)
    local_uncertainty_df.to_csv(OUTPUT_DIR / "local_model_uncertainty_results.csv", index=False)
    local_aggregate_df.to_csv(OUTPUT_DIR / "local_model_uncertainty_aggregate.csv", index=False)
    gate_df.to_csv(OUTPUT_DIR / "phase9_gate_summary.csv", index=False)

    make_plots(fit_df, ann_path_df, ann_hist_df, local_uncertainty_df, local_aggregate_df)
    write_summary(output_payload, inflation_payload, fit_df, ann_hist_df, ann_stochastic_df, local_aggregate_df, gate_df, ann_reproduction_error)


if __name__ == "__main__":
    main()
