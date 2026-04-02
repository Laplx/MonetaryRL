from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "phase10" / "case_inventory"
COUNTERFACTUAL_DIR = ROOT / "outputs" / "phase10" / "counterfactual_eval"
REVEALED_WELFARE_DIR = ROOT / "outputs" / "phase10" / "revealed_welfare"
REVEALED_POLICY_EVAL_DIR = ROOT / "outputs" / "phase10" / "revealed_policy_eval"
EXTERNAL_DIR = ROOT / "outputs" / "phase10" / "external_model_robustness"
REVEALED_TRAIN_DIR = ROOT / "outputs" / "phase10" / "revealed_policy_training"


def _load_policy_meta() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in [
        COUNTERFACTUAL_DIR / "unified_policy_registry.csv",
        REVEALED_TRAIN_DIR / "policy_registry.csv",
    ]:
        if path.exists():
            df = pd.read_csv(path)
            keep = [
                col
                for col in [
                    "policy_name",
                    "rule_family",
                    "source_env",
                    "training_env",
                    "policy_parameterization",
                    "algo",
                ]
                if col in df.columns
            ]
            frames.append(df[keep].copy())
    if not frames:
        return pd.DataFrame(columns=["policy_name"])
    meta = pd.concat(frames, ignore_index=True, sort=False)
    return meta.drop_duplicates(subset=["policy_name"], keep="last").reset_index(drop=True)


def _attach_meta(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if "policy_name" not in df.columns:
        return df.copy()
    out = df.merge(meta, on="policy_name", how="left", suffixes=("", "_meta"))
    for col in ["rule_family", "source_env", "training_env", "policy_parameterization", "algo"]:
        alt = f"{col}_meta"
        if alt in out.columns:
            out[col] = out[col].where(out[col].notna(), out[alt])
            out = out.drop(columns=[alt])
    return out


def _base_record(row: pd.Series, source_file: str, case_group: str, evaluation_type: str, loss_function: str, environment: str) -> dict[str, object]:
    return {
        "case_group": case_group,
        "evaluation_type": evaluation_type,
        "loss_function": loss_function,
        "environment": environment,
        "policy_name": row.get("policy_name", ""),
        "rule_family": row.get("rule_family", ""),
        "source_env": row.get("source_env", ""),
        "training_env": row.get("training_env", ""),
        "policy_parameterization": row.get("policy_parameterization", ""),
        "algo": row.get("algo", ""),
        "source_file": source_file,
    }


def _normalize_summary(
    path: Path,
    *,
    case_group: str,
    evaluation_type: str,
    loss_function: str,
    environment_col: str,
    metric_col: str,
    secondary_col: str | None = None,
    comparison_cols: list[str] | None = None,
    meta: pd.DataFrame,
) -> list[dict[str, object]]:
    df = _attach_meta(pd.read_csv(path), meta)
    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        rec = _base_record(
            row,
            source_file=str(path.relative_to(ROOT)),
            case_group=case_group,
            evaluation_type=evaluation_type,
            loss_function=loss_function,
            environment=str(row[environment_col]),
        )
        rec["result_metric"] = metric_col
        rec["result_value"] = float(row[metric_col]) if pd.notna(row[metric_col]) else np.nan
        rec["secondary_metric"] = secondary_col or ""
        rec["secondary_value"] = float(row[secondary_col]) if secondary_col and pd.notna(row.get(secondary_col)) else np.nan
        rec["comparison_metric"] = ""
        rec["comparison_value"] = np.nan
        if comparison_cols:
            for col in comparison_cols:
                if col in row.index and pd.notna(row[col]):
                    rec["comparison_metric"] = col
                    rec["comparison_value"] = float(row[col])
                    break
        rec["solver_status"] = row.get("solver_status", "")
        records.append(rec)
    return records


def _normalize_external(path: Path, meta: pd.DataFrame) -> list[dict[str, object]]:
    df = _attach_meta(pd.read_csv(path), meta)
    records: list[dict[str, object]] = []
    comparison_map = {
        "pyfrbus": ("improvement_vs_pyfrbus_baseline_pct", "improvement_vs_pyfrbus_baseline_revealed_pct"),
        "US_SW07": ("improvement_vs_US_SW07_baseline_pct", "improvement_vs_US_SW07_baseline_revealed_pct"),
        "US_CCTW10": ("improvement_vs_US_CCTW10_baseline_pct", "improvement_vs_US_CCTW10_baseline_revealed_pct"),
        "US_KS15": ("improvement_vs_US_KS15_baseline_pct", "improvement_vs_US_KS15_baseline_revealed_pct"),
        "NK_CW09": ("improvement_vs_NK_CW09_baseline_pct", "improvement_vs_NK_CW09_baseline_revealed_pct"),
    }
    for _, row in df.iterrows():
        env = str(row["model_id"])
        artificial_cmp, revealed_cmp = comparison_map.get(env, ("", ""))
        base = _base_record(
            row,
            source_file=str(path.relative_to(ROOT)),
            case_group="external_models",
            evaluation_type="external_model_eval",
            loss_function="artificial",
            environment=env,
        )
        base["result_metric"] = "total_discounted_loss"
        base["result_value"] = float(row["total_discounted_loss"]) if pd.notna(row["total_discounted_loss"]) else np.nan
        base["secondary_metric"] = "mean_period_loss"
        base["secondary_value"] = float(row["mean_period_loss"]) if pd.notna(row["mean_period_loss"]) else np.nan
        base["comparison_metric"] = artificial_cmp
        base["comparison_value"] = float(row[artificial_cmp]) if artificial_cmp and pd.notna(row.get(artificial_cmp)) else np.nan
        base["solver_status"] = row.get("solver_status", "")
        records.append(base)

        revealed = dict(base)
        revealed["loss_function"] = "revealed"
        revealed["result_metric"] = "total_discounted_revealed_loss"
        revealed["result_value"] = (
            float(row["total_discounted_revealed_loss"]) if pd.notna(row["total_discounted_revealed_loss"]) else np.nan
        )
        revealed["secondary_metric"] = "mean_period_revealed_loss"
        revealed["secondary_value"] = (
            float(row["mean_period_revealed_loss"]) if pd.notna(row["mean_period_revealed_loss"]) else np.nan
        )
        revealed["comparison_metric"] = revealed_cmp
        revealed["comparison_value"] = float(row[revealed_cmp]) if revealed_cmp and pd.notna(row.get(revealed_cmp)) else np.nan
        records.append(revealed)
    return records


def build_inventory() -> pd.DataFrame:
    meta = _load_policy_meta()
    records: list[dict[str, object]] = []

    records += _normalize_summary(
        COUNTERFACTUAL_DIR / "svar_historical_summary.csv",
        case_group="empirical_unified",
        evaluation_type="historical_shock",
        loss_function="artificial",
        environment_col="evaluation_env",
        metric_col="total_discounted_loss",
        secondary_col="mean_period_loss",
        comparison_cols=["improvement_vs_actual_pct", "improvement_vs_taylor_pct"],
        meta=meta,
    )
    records += _normalize_summary(
        COUNTERFACTUAL_DIR / "ann_historical_summary.csv",
        case_group="empirical_unified",
        evaluation_type="historical_shock",
        loss_function="artificial",
        environment_col="evaluation_env",
        metric_col="total_discounted_loss",
        secondary_col="mean_period_loss",
        comparison_cols=["improvement_vs_actual_pct", "improvement_vs_taylor_pct"],
        meta=meta,
    )
    records += _normalize_summary(
        COUNTERFACTUAL_DIR / "svar_stochastic_summary.csv",
        case_group="empirical_unified",
        evaluation_type="long_run_stochastic",
        loss_function="artificial",
        environment_col="evaluation_env",
        metric_col="mean_discounted_loss",
        secondary_col="std_discounted_loss",
        meta=meta,
    )
    records += _normalize_summary(
        COUNTERFACTUAL_DIR / "ann_stochastic_summary.csv",
        case_group="empirical_unified",
        evaluation_type="long_run_stochastic",
        loss_function="artificial",
        environment_col="evaluation_env",
        metric_col="mean_discounted_loss",
        secondary_col="std_discounted_loss",
        meta=meta,
    )
    records += _normalize_summary(
        COUNTERFACTUAL_DIR / "cross_transfer_summary.csv",
        case_group="empirical_unified",
        evaluation_type="cross_transfer",
        loss_function="artificial",
        environment_col="evaluation_env",
        metric_col="mean_discounted_loss",
        secondary_col="std_discounted_loss",
        meta=meta,
    )

    records += _normalize_summary(
        REVEALED_WELFARE_DIR / "revealed_historical_summary.csv",
        case_group="empirical_unified",
        evaluation_type="historical_shock",
        loss_function="revealed",
        environment_col="evaluation_env",
        metric_col="total_discounted_revealed_loss",
        secondary_col="mean_period_revealed_loss",
        meta=meta,
    )
    records += _normalize_summary(
        REVEALED_WELFARE_DIR / "revealed_stochastic_summary.csv",
        case_group="empirical_unified",
        evaluation_type="long_run_stochastic",
        loss_function="revealed",
        environment_col="evaluation_env",
        metric_col="mean_discounted_revealed_loss",
        secondary_col="std_discounted_revealed_loss",
        meta=meta,
    )

    records += _normalize_summary(
        REVEALED_POLICY_EVAL_DIR / "svar_historical_summary.csv",
        case_group="revealed_trained_rules",
        evaluation_type="historical_shock",
        loss_function="revealed",
        environment_col="evaluation_env",
        metric_col="total_discounted_loss",
        secondary_col="mean_period_loss",
        comparison_cols=["improvement_vs_actual_pct", "improvement_vs_taylor_pct"],
        meta=meta,
    )
    records += _normalize_summary(
        REVEALED_POLICY_EVAL_DIR / "ann_historical_summary.csv",
        case_group="revealed_trained_rules",
        evaluation_type="historical_shock",
        loss_function="revealed",
        environment_col="evaluation_env",
        metric_col="total_discounted_loss",
        secondary_col="mean_period_loss",
        comparison_cols=["improvement_vs_actual_pct", "improvement_vs_taylor_pct"],
        meta=meta,
    )
    records += _normalize_summary(
        REVEALED_POLICY_EVAL_DIR / "svar_stochastic_summary.csv",
        case_group="revealed_trained_rules",
        evaluation_type="long_run_stochastic",
        loss_function="revealed",
        environment_col="evaluation_env",
        metric_col="mean_discounted_loss",
        secondary_col="std_discounted_loss",
        meta=meta,
    )
    records += _normalize_summary(
        REVEALED_POLICY_EVAL_DIR / "ann_stochastic_summary.csv",
        case_group="revealed_trained_rules",
        evaluation_type="long_run_stochastic",
        loss_function="revealed",
        environment_col="evaluation_env",
        metric_col="mean_discounted_loss",
        secondary_col="std_discounted_loss",
        meta=meta,
    )
    records += _normalize_summary(
        REVEALED_POLICY_EVAL_DIR / "cross_transfer_summary.csv",
        case_group="revealed_trained_rules",
        evaluation_type="cross_transfer",
        loss_function="revealed",
        environment_col="evaluation_env",
        metric_col="mean_discounted_loss",
        secondary_col="std_discounted_loss",
        meta=meta,
    )

    records += _normalize_external(EXTERNAL_DIR / "all_external_summary.csv", meta)
    inventory = pd.DataFrame(records)
    sort_cols = ["case_group", "evaluation_type", "loss_function", "environment", "rule_family", "policy_name"]
    inventory = inventory.sort_values(sort_cols).reset_index(drop=True)
    inventory.insert(0, "case_id", [f"case_{idx+1:04d}" for idx in range(len(inventory))])
    return inventory


def write_outputs(inventory: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "phase10_case_inventory.csv"
    md_path = OUTPUT_DIR / "phase10_case_inventory_summary.md"
    inventory.to_csv(csv_path, index=False)

    counts = (
        inventory.groupby(["case_group", "evaluation_type", "loss_function"], dropna=False)
        .size()
        .reset_index(name="case_count")
        .sort_values(["case_group", "evaluation_type", "loss_function"])
        .reset_index(drop=True)
    )
    env_counts = (
        inventory.groupby(["environment", "evaluation_type", "loss_function"], dropna=False)
        .size()
        .reset_index(name="case_count")
        .sort_values(["environment", "evaluation_type", "loss_function"])
        .reset_index(drop=True)
    )
    lines = [
        "# Phase 10 Case Inventory",
        "",
        f"- Total cases: `{len(inventory)}`",
        f"- CSV: `{csv_path.relative_to(ROOT)}`",
        "",
        "## Count By Group",
        "",
        counts.to_markdown(index=False),
        "",
        "## Count By Environment",
        "",
        env_counts.to_markdown(index=False),
        "",
        "## Fields",
        "",
        "| 字段 | 含义 |",
        "|---|---|",
        "| `environment` | 经验环境或外部模型名 |",
        "| `policy_name` | 规则名 |",
        "| `loss_function` | `artificial` 或 `revealed` |",
        "| `evaluation_type` | `historical_shock` / `long_run_stochastic` / `cross_transfer` / `external_model_eval` |",
        "| `result_metric` | 主结果指标名 |",
        "| `result_value` | 主结果指标值 |",
        "| `comparison_metric` | 相对 baseline 或历史政策的比较列 |",
        "| `comparison_value` | 比较值 |",
        "",
        "## Preview",
        "",
        inventory.head(20).to_markdown(index=False),
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    inventory = build_inventory()
    write_outputs(inventory)


if __name__ == "__main__":
    main()
