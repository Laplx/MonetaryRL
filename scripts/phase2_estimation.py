from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs" / "phase2"

SAMPLE_START = "1987-07-01"  # 1987:Q3
SAMPLE_END = "2007-04-01"    # 2007:Q2
INFLATION_TARGET = 2.0
RATE_TARGET = 2.0


@dataclass
class OLSResult:
    name: str
    nobs: int
    r2: float
    adj_r2: float
    mse: float
    rmse: float
    coefficients: dict[str, float]
    pvalues: dict[str, float]
    stderr: dict[str, float]
    residual_std: float


@dataclass
class MLPResult:
    name: str
    hidden_units: int
    train_size: int
    val_size: int
    test_size: int
    train_mse: float
    val_mse: float
    test_mse: float
    full_sample_mse: float
    residual_std_full_sample: float
    n_iter: int
    random_state: int


def load_series(name: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / f"{name}.csv")
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.rename(columns={"observation_date": "date", name: "value"})
    return df


def quarter_average_fedfunds(df: pd.DataFrame) -> pd.DataFrame:
    monthly = df.copy()
    monthly["quarter"] = monthly["date"].dt.to_period("Q")
    quarterly = monthly.groupby("quarter", as_index=False)["value"].mean()
    quarterly["date"] = quarterly["quarter"].dt.start_time
    return quarterly[["date", "value"]]


def build_macro_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    gdpdef = load_series("GDPDEF").rename(columns={"value": "gdpdef"})
    gdpc1 = load_series("GDPC1").rename(columns={"value": "real_gdp"})
    gdppot = load_series("GDPPOT").rename(columns={"value": "potential_gdp"})
    fedfunds = quarter_average_fedfunds(load_series("FEDFUNDS")).rename(columns={"value": "policy_rate"})

    macro = gdpdef.merge(gdpc1, on="date", how="inner")
    macro = macro.merge(gdppot, on="date", how="inner")
    macro = macro.merge(fedfunds, on="date", how="inner")
    macro = macro.sort_values("date").reset_index(drop=True)

    macro["inflation"] = 100.0 * (np.log(macro["gdpdef"]) - np.log(macro["gdpdef"].shift(4)))
    macro["output_gap"] = 100.0 * (np.log(macro["real_gdp"]) - np.log(macro["potential_gdp"]))
    macro["inflation_gap"] = macro["inflation"] - INFLATION_TARGET
    macro["policy_rate_gap"] = macro["policy_rate"] - RATE_TARGET
    macro["quarter"] = macro["date"].dt.to_period("Q").astype(str)

    macro_full = macro.dropna().reset_index(drop=True)
    sample = macro_full[(macro_full["date"] >= SAMPLE_START) & (macro_full["date"] <= SAMPLE_END)].copy()
    sample = sample.reset_index(drop=True)
    return macro_full, sample


def add_lags(df: pd.DataFrame, columns: Iterable[str], max_lag: int) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        for lag in range(1, max_lag + 1):
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def fit_ols(name: str, df: pd.DataFrame, target: str, regressors: list[str]) -> tuple[OLSResult, pd.DataFrame]:
    model_df = df[[target] + regressors].dropna().copy()
    X = sm.add_constant(model_df[regressors], has_constant="add")
    y = model_df[target]
    fit = sm.OLS(y, X).fit()
    preds = fit.predict(X)
    mse = float(mean_squared_error(y, preds))
    result = OLSResult(
        name=name,
        nobs=int(fit.nobs),
        r2=float(fit.rsquared),
        adj_r2=float(fit.rsquared_adj),
        mse=mse,
        rmse=float(np.sqrt(mse)),
        coefficients={k: float(v) for k, v in fit.params.items()},
        pvalues={k: float(v) for k, v in fit.pvalues.items()},
        stderr={k: float(v) for k, v in fit.bse.items()},
        residual_std=float(np.std(fit.resid, ddof=1)),
    )
    fitted = model_df.copy()
    fitted[f"{target}_fitted"] = preds
    fitted[f"{target}_resid"] = y - preds
    return result, fitted


def time_split_indices(n: int, train_frac: float = 0.7, val_frac: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(np.floor(n * train_frac))
    val_end = int(np.floor(n * (train_frac + val_frac)))
    idx = np.arange(n)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


def fit_mlp_equation(
    name: str,
    df: pd.DataFrame,
    target: str,
    regressors: list[str],
    hidden_grid: Iterable[int] = range(2, 11),
    seed_grid: Iterable[int] = range(5),
) -> tuple[MLPResult, pd.DataFrame]:
    model_df = df[[target] + regressors].dropna().copy()
    X = model_df[regressors].to_numpy()
    y = model_df[target].to_numpy()
    train_idx, val_idx, test_idx = time_split_indices(len(model_df))

    best_pipeline = None
    best_result = None

    for hidden in hidden_grid:
        for seed in seed_grid:
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "mlp",
                        MLPRegressor(
                            hidden_layer_sizes=(hidden,),
                            activation="tanh",
                            solver="lbfgs",
                            random_state=seed,
                            max_iter=5000,
                        ),
                    ),
                ]
            )
            pipeline.fit(X[train_idx], y[train_idx])
            train_mse = mean_squared_error(y[train_idx], pipeline.predict(X[train_idx]))
            val_mse = mean_squared_error(y[val_idx], pipeline.predict(X[val_idx]))
            test_mse = mean_squared_error(y[test_idx], pipeline.predict(X[test_idx]))
            result = MLPResult(
                name=name,
                hidden_units=hidden,
                train_size=len(train_idx),
                val_size=len(val_idx),
                test_size=len(test_idx),
                train_mse=float(train_mse),
                val_mse=float(val_mse),
                test_mse=float(test_mse),
                full_sample_mse=0.0,
                residual_std_full_sample=0.0,
                n_iter=int(pipeline.named_steps["mlp"].n_iter_),
                random_state=seed,
            )
            if best_result is None or result.val_mse < best_result.val_mse:
                best_pipeline = pipeline
                best_result = result

    assert best_pipeline is not None and best_result is not None
    preds = best_pipeline.predict(X)
    residuals = y - preds
    best_result.full_sample_mse = float(mean_squared_error(y, preds))
    best_result.residual_std_full_sample = float(np.std(residuals, ddof=1))

    fitted = model_df.copy()
    fitted[f"{target}_fitted"] = preds
    fitted[f"{target}_resid"] = residuals
    return best_result, fitted


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown(
    macro_sample: pd.DataFrame,
    svar_output: OLSResult,
    svar_inflation: OLSResult,
    taylor_rule: OLSResult,
    ann_output: MLPResult,
    ann_inflation: MLPResult,
) -> None:
    lines: list[str] = []
    lines.append("# Phase 2 Estimation Summary")
    lines.append("")
    lines.append("## Sample")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Start | {macro_sample['quarter'].iloc[0]} |")
    lines.append(f"| End | {macro_sample['quarter'].iloc[-1]} |")
    lines.append(f"| Observations | {len(macro_sample)} |")
    lines.append("")

    lines.append("## Variable Summary")
    lines.append("")
    summary = macro_sample[["inflation", "output_gap", "policy_rate"]].agg(["mean", "std", "min", "max"]).round(4)
    lines.append(summary.to_markdown())
    lines.append("")

    lines.append("## Environment Fit Comparison")
    lines.append("")
    fit_df = pd.DataFrame(
        {
            "SVAR_MSE": {
                "output_gap": svar_output.mse,
                "inflation": svar_inflation.mse,
            },
            "ANN_MSE": {
                "output_gap": ann_output.full_sample_mse,
                "inflation": ann_inflation.full_sample_mse,
            },
        }
    )
    fit_df["ANN_improvement_pct"] = ((fit_df["SVAR_MSE"] - fit_df["ANN_MSE"]) / fit_df["SVAR_MSE"] * 100.0).round(2)
    lines.append(fit_df.round(6).to_markdown())
    lines.append("")

    for title, result in [
        ("SVAR Output Gap Equation", svar_output),
        ("SVAR Inflation Equation", svar_inflation),
        ("Taylor Rule Regression", taylor_rule),
    ]:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Item | Value |")
        lines.append("|---|---|")
        lines.append(f"| nobs | {result.nobs} |")
        lines.append(f"| R2 | {result.r2:.4f} |")
        lines.append(f"| Adj. R2 | {result.adj_r2:.4f} |")
        lines.append(f"| RMSE | {result.rmse:.4f} |")
        lines.append("")
        coef_df = pd.DataFrame(
            {
                "coef": pd.Series(result.coefficients),
                "stderr": pd.Series(result.stderr),
                "pvalue": pd.Series(result.pvalues),
            }
        ).round(6)
        lines.append(coef_df.to_markdown())
        lines.append("")

    for title, result in [
        ("ANN Output Gap Equation", ann_output),
        ("ANN Inflation Equation", ann_inflation),
    ]:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Item | Value |")
        lines.append("|---|---|")
        lines.append(f"| Hidden units | {result.hidden_units} |")
        lines.append(f"| Random state | {result.random_state} |")
        lines.append(f"| Train MSE | {result.train_mse:.6f} |")
        lines.append(f"| Validation MSE | {result.val_mse:.6f} |")
        lines.append(f"| Test MSE | {result.test_mse:.6f} |")
        lines.append(f"| Full-sample MSE | {result.full_sample_mse:.6f} |")
        lines.append(f"| Residual std (full sample) | {result.residual_std_full_sample:.6f} |")
        lines.append("")

    long_run_phi_pi = taylor_rule.coefficients["inflation"] / (1.0 - taylor_rule.coefficients["policy_rate_lag1"])
    long_run_phi_x = taylor_rule.coefficients["output_gap"] / (1.0 - taylor_rule.coefficients["policy_rate_lag1"])
    lines.append("## Taylor Rule Long-run Responses")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Long-run inflation response | {long_run_phi_pi:.4f} |")
    lines.append(f"| Long-run output-gap response | {long_run_phi_x:.4f} |")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Inflation is measured as four-quarter log growth of the GDP deflator.")
    lines.append("- Output gap is measured as 100 * (log real GDP - log potential GDP).")
    lines.append("- The empirical linear environment follows Hinterlang's recursive SVAR structure.")
    lines.append("- The empirical ANN environment uses the same input sets as the linear environment for a fair comparison.")
    lines.append("- The empirical SVAR environment is not identical to the theoretical 3-state LQ benchmark and will require state augmentation in implementation.")
    lines.append("")

    (OUTPUT_DIR / "phase2_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    macro_full, macro_sample = build_macro_dataset()
    macro_full.to_csv(PROCESSED_DIR / "macro_quarterly_full.csv", index=False)
    macro_sample.to_csv(PROCESSED_DIR / "macro_quarterly_sample_1987Q3_2007Q2.csv", index=False)

    # Build lagged frame for environment estimation.
    env = add_lags(macro_sample, ["inflation", "output_gap", "policy_rate"], max_lag=2)

    output_regs = ["output_gap_lag1", "inflation_lag1", "policy_rate_lag1", "policy_rate_lag2"]
    inflation_regs = [
        "output_gap",
        "output_gap_lag1",
        "output_gap_lag2",
        "inflation_lag1",
        "inflation_lag2",
        "policy_rate_lag1",
    ]
    taylor_regs = ["inflation", "output_gap", "policy_rate_lag1"]

    svar_output, output_fit = fit_ols("svar_output_gap", env, "output_gap", output_regs)
    svar_inflation, inflation_fit = fit_ols("svar_inflation", env, "inflation", inflation_regs)
    taylor_rule, taylor_fit = fit_ols("taylor_rule", env, "policy_rate", taylor_regs)

    ann_output, ann_output_fit = fit_mlp_equation("ann_output_gap", env, "output_gap", output_regs)
    ann_inflation, ann_inflation_fit = fit_mlp_equation("ann_inflation", env, "inflation", inflation_regs)

    output_fit.to_csv(OUTPUT_DIR / "svar_output_gap_fitted.csv", index=False)
    inflation_fit.to_csv(OUTPUT_DIR / "svar_inflation_fitted.csv", index=False)
    taylor_fit.to_csv(OUTPUT_DIR / "taylor_rule_fitted.csv", index=False)
    ann_output_fit.to_csv(OUTPUT_DIR / "ann_output_gap_fitted.csv", index=False)
    ann_inflation_fit.to_csv(OUTPUT_DIR / "ann_inflation_fitted.csv", index=False)

    write_json(OUTPUT_DIR / "svar_output_gap.json", asdict(svar_output))
    write_json(OUTPUT_DIR / "svar_inflation.json", asdict(svar_inflation))
    write_json(OUTPUT_DIR / "taylor_rule.json", asdict(taylor_rule))
    write_json(OUTPUT_DIR / "ann_output_gap.json", asdict(ann_output))
    write_json(OUTPUT_DIR / "ann_inflation.json", asdict(ann_inflation))

    write_markdown(macro_sample, svar_output, svar_inflation, taylor_rule, ann_output, ann_inflation)


if __name__ == "__main__":
    main()
