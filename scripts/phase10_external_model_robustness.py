from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

EXTERNAL_ROOT = ROOT / "external_models"
PYFRBUS_ROOT = EXTERNAL_ROOT / "frbus_extracted" / "pyfrbus"
if str(PYFRBUS_ROOT) not in sys.path:
    sys.path.insert(0, str(PYFRBUS_ROOT))

from monetary_rl.phase10_utils import (
    build_ann_context,
    build_linear_policy,
    build_svar_context,
    load_checkpoint_policy,
)

from pyfrbus.frbus import Frbus
from pyfrbus.load_data import load_data


OUTPUT_DIR = ROOT / "outputs" / "phase10" / "external_model_robustness"
COUNTERFACTUAL_DIR = ROOT / "outputs" / "phase10" / "counterfactual_eval"
REVEALED_DIR = ROOT / "outputs" / "phase10" / "revealed_welfare"
REVEALED_TRAIN_ROOT = ROOT / "outputs" / "phase10" / "revealed_policy_training"
PPO_VARIANT_DIR = ROOT / "outputs" / "phase10" / "ppo_policy_variants"
MMB_ROOT = EXTERNAL_ROOT / "mmb_extracted" / "mmb-rep-master"
PYFRBUS_MODEL = PYFRBUS_ROOT / "models" / "model.xml"
PYFRBUS_DATA = PYFRBUS_ROOT / "data" / "LONGBASE.TXT"

PRIORITY_MODELS = ["pyfrbus", "US_FRB03", "US_SW07", "US_CCTW10", "US_CPS10", "US_KS15", "US_RA07"]
FALLBACK_MODELS = ["NK_CW09", "NK_CFP10", "NK_GLSV07", "NK_GK13"]
BENCHMARK_CONFIG = json.loads((ROOT / "src" / "monetary_rl" / "config" / "benchmark_lq.json").read_text(encoding="utf-8"))
REVEALED_WEIGHTS = json.loads((REVEALED_DIR / "revealed_weights.json").read_text(encoding="utf-8"))


def pick_preferred_rules() -> pd.DataFrame:
    unified_registry = pd.read_csv(COUNTERFACTUAL_DIR / "unified_policy_registry.csv")
    svar_hist = pd.read_csv(COUNTERFACTUAL_DIR / "svar_historical_summary.csv")
    ann_hist = pd.read_csv(COUNTERFACTUAL_DIR / "ann_historical_summary.csv")
    combined = pd.concat([svar_hist, ann_hist], ignore_index=True)
    merged = combined.merge(
        unified_registry[["policy_name", "rule_family", "source_env", "policy_parameterization"]],
        on="policy_name",
        how="left",
    )
    merged = merged[
        merged["rule_family"].isin(
            [
                "benchmark_transfer",
                "svar_direct",
                "ann_direct",
                "svar_revealed_direct",
                "ann_revealed_direct",
            ]
        )
    ]
    preferred = (
        merged.sort_values("total_discounted_loss")
        .groupby(["evaluation_env", "rule_family"], as_index=False)
        .first()[
            [
                "evaluation_env",
                "rule_family",
                "policy_name",
                "source_env",
                "policy_parameterization",
                "total_discounted_loss",
            ]
        ]
    )
    return preferred


def locate_model_file(model_id: str) -> tuple[str, Path | None]:
    if model_id == "pyfrbus":
        return ("pyfrbus", PYFRBUS_MODEL if PYFRBUS_MODEL.exists() else None)
    model_dir = MMB_ROOT / model_id
    if not model_dir.exists():
        return ("dynare_mmb", None)
    mod_files = sorted(model_dir.rglob("*.mod"))
    if mod_files:
        return ("dynare_mmb", mod_files[0])
    m_files = sorted(model_dir.rglob("*.m"))
    return ("dynare_mmb", m_files[0] if m_files else None)


def candidate_tokens(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {"inflation": "", "output_gap": "", "policy_rate": ""}
    tokens = sorted(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,40}", text.lower())))

    def shortlist(patterns: list[str]) -> str:
        matches = [token for token in tokens if any(pattern in token for pattern in patterns)]
        return ", ".join(matches[:8])

    return {
        "inflation": shortlist(["infl", "pinf", "pi", "pic"]),
        "output_gap": shortlist(["outputgap", "ygap", "gap", "xgap"]),
        "policy_rate": shortlist(["rff", "interest", "rate", "ffr", "funds", "int"]),
    }


def build_model_inventory() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for priority_group, model_ids in [("priority", PRIORITY_MODELS), ("fallback", FALLBACK_MODELS)]:
        for model_id in model_ids:
            runner_family, file_path = locate_model_file(model_id)
            candidates = candidate_tokens(file_path) if file_path is not None else {"inflation": "", "output_gap": "", "policy_rate": ""}
            rows.append(
                {
                    "model_id": model_id,
                    "priority_group": priority_group,
                    "runner_family": runner_family,
                    "available_locally": bool(file_path is not None),
                    "source_path": str(file_path) if file_path is not None else "",
                    "interface_status": "mapping_stub_ready" if file_path is not None else "missing_locally",
                    "candidate_inflation_vars": candidates["inflation"],
                    "candidate_output_gap_vars": candidates["output_gap"],
                    "candidate_policy_rate_vars": candidates["policy_rate"],
                }
            )
    return pd.DataFrame(rows)


def build_policy_callable_map() -> dict[str, object]:
    unified_registry = pd.read_csv(COUNTERFACTUAL_DIR / "unified_policy_registry.csv")
    contexts = {"svar": build_svar_context(ROOT), "ann": build_ann_context(ROOT)}
    policies: dict[str, object] = {}
    for row in unified_registry.to_dict("records"):
        name = row["policy_name"]
        callable_type = row["callable_type"]
        if callable_type == "linear":
            policies[name] = build_linear_policy(
                name,
                float(row["intercept"]),
                float(row["inflation_gap"]),
                float(row["output_gap"]),
                float(row["lagged_policy_rate_gap"]),
            )
        elif callable_type == "checkpoint":
            policies[name] = load_checkpoint_policy(row, contexts[str(row["training_env"])])
    return policies


def build_revealed_policy_callable_map() -> dict[str, object]:
    policies: dict[str, object] = {}
    for env_name in ["svar", "ann"]:
        registry_path = ROOT / "outputs" / "phase10" / "revealed_policy_training" / env_name / "policy_registry.csv"
        if not registry_path.exists():
            continue
        registry_df = pd.read_csv(registry_path)
        context = build_svar_context(ROOT) if env_name == "svar" else build_ann_context(ROOT)
        for row in registry_df.to_dict("records"):
            policies[row["policy_name"]] = load_checkpoint_policy(row, context)
    return policies


def _registry_policy_names(path: Path, *, parameterization: str | None = None) -> list[str]:
    if not path.exists():
        return []
    registry = pd.read_csv(path)
    if parameterization is not None and "policy_parameterization" in registry.columns:
        registry = registry.loc[registry["policy_parameterization"] == parameterization]
    return registry["policy_name"].astype(str).tolist()


def pyfrbus_policy_names(preferred_rules: pd.DataFrame) -> list[str]:
    ordered = list(preferred_rules["policy_name"].tolist())
    ordered.extend(_registry_policy_names(REVEALED_TRAIN_ROOT / "policy_registry.csv"))
    ordered.extend(_registry_policy_names(PPO_VARIANT_DIR / "policy_registry.csv", parameterization="nonlinear_policy"))
    ordered.extend(["empirical_taylor_rule", "riccati_reference"])
    unique = list(dict.fromkeys(ordered))
    return unique


def baseline_loss_frame(sim: pd.DataFrame, start: pd.Period, end: pd.Period) -> pd.DataFrame:
    window = sim.loc[start:end, ["rff", "picxfe", "pitarg", "xgap"]].copy()
    window["inflation_gap"] = window["picxfe"] - window["pitarg"]
    window["lagged_rff"] = sim["rff"].shift(1).loc[start:end]
    window["rate_change"] = window["rff"] - window["lagged_rff"]
    return window


def welfare_summary(window: pd.DataFrame, policy_name: str) -> dict[str, float | str]:
    discount = float(BENCHMARK_CONFIG["discount_factor"])
    weights = BENCHMARK_CONFIG["loss_weights"]
    revealed_weights = {
        "inflation": float(REVEALED_WEIGHTS["inflation_weight"]),
        "output_gap": float(REVEALED_WEIGHTS["output_gap_weight"]),
        "rate_smoothing": float(REVEALED_WEIGHTS["rate_smoothing_weight"]),
    }
    per_loss = (
        weights["inflation"] * window["inflation_gap"] ** 2
        + weights["output_gap"] * window["xgap"] ** 2
        + weights["rate_smoothing"] * window["rate_change"] ** 2
    )
    per_revealed_loss = (
        revealed_weights["inflation"] * window["inflation_gap"] ** 2
        + revealed_weights["output_gap"] * window["xgap"] ** 2
        + revealed_weights["rate_smoothing"] * window["rate_change"] ** 2
    )
    discounted = per_loss.to_numpy(dtype=float) * (discount ** np.arange(len(window)))
    discounted_revealed = per_revealed_loss.to_numpy(dtype=float) * (discount ** np.arange(len(window)))
    return {
        "policy_name": policy_name,
        "model_id": "pyfrbus",
        "total_discounted_loss": float(discounted.sum()),
        "mean_period_loss": float(per_loss.mean()),
        "total_discounted_revealed_loss": float(discounted_revealed.sum()),
        "mean_period_revealed_loss": float(per_revealed_loss.mean()),
        "mean_sq_inflation_gap": float(np.mean(np.square(window["inflation_gap"]))),
        "mean_sq_output_gap": float(np.mean(np.square(window["xgap"]))),
        "mean_sq_rate_change": float(np.mean(np.square(window["rate_change"]))),
        "mean_policy_rate": float(window["rff"].mean()),
        "std_policy_rate": float(window["rff"].std(ddof=1)),
    }


def run_pyfrbus_fixed_point(policy_name: str, policy_fn, start: str = "2040Q1", end: str = "2045Q4") -> tuple[pd.DataFrame, dict[str, object]]:
    data = load_data(str(PYFRBUS_DATA))
    frbus = Frbus(str(PYFRBUS_MODEL))
    frbus.exogenize(["rff"])
    start_p = pd.Period(start, freq="Q")
    end_p = pd.Period(end, freq="Q")
    periods = pd.period_range(start_p, end_p, freq="Q")

    data.loc[start_p:end_p, "dfpdbt"] = 0.0
    data.loc[start_p:end_p, "dfpsrp"] = 1.0
    base = frbus.init_trac(start_p, end_p, data)
    sim = frbus.solve(start_p, end_p, base, options={"newton": "newton", "single_block": True})
    current_path = sim.loc[periods, "rff"].to_numpy(dtype=float)

    converged = False
    clip_count = 0
    iterations = 0
    for iterations in range(1, 21):
        prev_rate = float(sim.loc[start_p - 1, "rff"])
        implied_path = []
        local_clips = 0
        for period in periods:
            inflation_gap = float(sim.loc[period, "picxfe"] - sim.loc[period, "pitarg"])
            output_gap = float(sim.loc[period, "xgap"])
            state = np.array([inflation_gap, output_gap, prev_rate - 2.0], dtype=float)
            raw_gap = float(policy_fn(state, 0))
            clipped_gap = float(np.clip(raw_gap, -2.0, 8.0))
            local_clips += int(abs(raw_gap - clipped_gap) > 1e-8)
            policy_rate = 2.0 + clipped_gap
            implied_path.append(policy_rate)
            prev_rate = policy_rate
        implied_arr = np.asarray(implied_path, dtype=float)
        gap = float(np.max(np.abs(implied_arr - current_path)))
        damped_path = 0.7 * implied_arr + 0.3 * current_path
        trial = base.copy()
        trial.loc[periods, "rff"] = damped_path
        sim = frbus.solve(start_p, end_p, trial, options={"newton": "newton", "single_block": True})
        current_path = sim.loc[periods, "rff"].to_numpy(dtype=float)
        clip_count = local_clips
        if gap < 1e-4:
            converged = True
            break

    window = baseline_loss_frame(sim, start_p, end_p)
    window = window.reset_index().rename(columns={"OBS": "period", "index": "period"})
    window.insert(0, "policy_name", policy_name)
    meta = {
        "policy_name": policy_name,
        "model_id": "pyfrbus",
        "converged": converged,
        "iterations": iterations,
        "max_fixed_point_gap": float(np.max(np.abs((window["rff"].to_numpy(dtype=float)) - current_path))),
        "clip_rate": clip_count / len(periods) if len(periods) else 0.0,
        "start": str(start_p),
        "end": str(end_p),
    }
    return window, meta


def run_pyfrbus_bundle(preferred_rules: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy_map = build_policy_callable_map()
    policy_map.update(build_revealed_policy_callable_map())
    policy_names = pyfrbus_policy_names(preferred_rules)
    policy_names = list(dict.fromkeys([name for name in policy_names if name in policy_map]))
    path_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    meta_rows: list[dict[str, object]] = []

    # Baseline projection under FRB/US packaged path
    data = load_data(str(PYFRBUS_DATA))
    frbus = Frbus(str(PYFRBUS_MODEL))
    start_p = pd.Period("2040Q1", freq="Q")
    end_p = pd.Period("2045Q4", freq="Q")
    data.loc[start_p:end_p, "dfpdbt"] = 0.0
    data.loc[start_p:end_p, "dfpsrp"] = 1.0
    base = frbus.init_trac(start_p, end_p, data)
    baseline = frbus.solve(start_p, end_p, base, options={"newton": "newton", "single_block": True})
    baseline_window = baseline_loss_frame(baseline, start_p, end_p).reset_index().rename(columns={"OBS": "period", "index": "period"})
    baseline_window.insert(0, "policy_name", "pyfrbus_baseline")
    path_frames.append(baseline_window)
    summary_rows.append(welfare_summary(baseline_loss_frame(baseline, start_p, end_p), "pyfrbus_baseline"))
    meta_rows.append({"policy_name": "pyfrbus_baseline", "model_id": "pyfrbus", "converged": True, "iterations": 0, "max_fixed_point_gap": 0.0, "clip_rate": 0.0, "start": str(start_p), "end": str(end_p)})

    for name in policy_names:
        window, meta = run_pyfrbus_fixed_point(name, policy_map[name])
        path_frames.append(window)
        summary_rows.append(welfare_summary(window.set_index("period"), name))
        meta_rows.append(meta)

    summary_df = pd.DataFrame(summary_rows).sort_values("total_discounted_loss").reset_index(drop=True)
    meta_df = pd.DataFrame(meta_rows).sort_values("policy_name").reset_index(drop=True)
    path_df = pd.concat(path_frames, ignore_index=True)
    baseline_loss = float(summary_df.loc[summary_df["policy_name"] == "pyfrbus_baseline", "total_discounted_loss"].iloc[0])
    summary_df["improvement_vs_pyfrbus_baseline_pct"] = (baseline_loss - summary_df["total_discounted_loss"]) / baseline_loss * 100.0
    baseline_revealed_loss = float(
        summary_df.loc[summary_df["policy_name"] == "pyfrbus_baseline", "total_discounted_revealed_loss"].iloc[0]
    )
    summary_df["improvement_vs_pyfrbus_baseline_revealed_pct"] = (
        (baseline_revealed_loss - summary_df["total_discounted_revealed_loss"]) / baseline_revealed_loss * 100.0
    )
    return summary_df, meta_df, path_df


def detect_runtime_status() -> pd.DataFrame:
    rows = []
    dynare_m = Path(r"C:\dynare\7.0\matlab\preprocessor64\dynare_m.exe")
    rows.append(
        {
            "component": "dynare_preprocessor",
            "status": "available" if dynare_m.exists() else "missing",
            "detail": str(dynare_m) if dynare_m.exists() else "not found",
        }
    )
    rows.append(
        {
            "component": "matlab_runtime_for_mmb",
            "status": "license_check_failed",
            "detail": "Current shell still reports MATLAB license error -9/57; MMB numerical solve remains blocked.",
        }
    )
    return pd.DataFrame(rows)


def write_phase10_summary(
    preferred_rules: pd.DataFrame,
    inventory_df: pd.DataFrame,
    pyfrbus_summary: pd.DataFrame,
    pyfrbus_meta: pd.DataFrame,
    runtime_status: pd.DataFrame,
) -> None:
    svar_direct = pd.read_csv(ROOT / "outputs" / "phase10" / "svar_direct" / "policy_registry.csv")
    ann_direct = pd.read_csv(ROOT / "outputs" / "phase10" / "ann_direct" / "policy_registry.csv")
    ppo_variants = pd.read_csv(ROOT / "outputs" / "phase10" / "ppo_policy_variants" / "policy_registry.csv")
    cross_df = pd.read_csv(COUNTERFACTUAL_DIR / "cross_transfer_summary.csv")
    revealed_weights = pd.DataFrame([json.loads((REVEALED_DIR / "revealed_weights.json").read_text(encoding="utf-8"))])

    lines = [
        "# Phase 10 Summary",
        "",
        "## Direct-Trained Empirical RL",
        "",
        pd.concat([svar_direct, ann_direct], ignore_index=True)[
            [
                "policy_name",
                "training_env",
                "algo",
                "policy_parameterization",
                "mean_discounted_loss",
                "clip_rate",
                "explosion_rate",
            ]
        ]
        .round(6)
        .to_markdown(index=False),
        "",
        "## PPO Variants",
        "",
        ppo_variants[
            [
                "policy_name",
                "training_env",
                "policy_parameterization",
                "mean_discounted_loss",
                "clip_rate",
                "explosion_rate",
            ]
        ]
        .round(6)
        .to_markdown(index=False),
        "",
        "## Preferred Rule Bundle For External Interface",
        "",
        preferred_rules.round(6).to_markdown(index=False),
        "",
        "## Cross-Transfer Snapshot",
        "",
        cross_df[
            [
                "policy_name",
                "rule_family",
                "source_env",
                "evaluation_env",
                "mean_discounted_loss",
                "clip_rate",
                "explosion_rate",
            ]
        ]
        .round(6)
        .to_markdown(index=False),
        "",
        "## Revealed Welfare Weights",
        "",
        revealed_weights.round(6).to_markdown(index=False),
        "",
        "## PyFRBUS External Results",
        "",
        pyfrbus_summary.round(6).to_markdown(index=False),
        "",
        "## External Runtime Status",
        "",
        runtime_status.to_markdown(index=False),
        "",
        "## External Model Inventory",
        "",
        inventory_df[
            [
                "model_id",
                "priority_group",
                "runner_family",
                "available_locally",
                "interface_status",
                "candidate_inflation_vars",
                "candidate_output_gap_vars",
                "candidate_policy_rate_vars",
            ]
        ]
        .to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- `pyfrbus` 已实际跑通；`Dynare/MMB` 批次由 `phase10_external_mmb_eval.py` 单独维护并汇总。",
        "- `Phase 8/9` 仍是 benchmark-transfer baseline，本轮新增的核心是 `SVAR direct` 与 `ANN direct` 的外部接口延伸。",
        "- Lucas critique 仍是经验环境与外部模型迁移时必须明确的边界。",
    ]
    (ROOT / "outputs" / "phase10" / "phase10_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preferred_rules = pick_preferred_rules()
    inventory_df = build_model_inventory()
    runtime_status = detect_runtime_status()

    preferred_rules.to_csv(OUTPUT_DIR / "preferred_rule_bundle.csv", index=False)
    inventory_df.to_csv(OUTPUT_DIR / "external_model_inventory.csv", index=False)
    runtime_status.to_csv(OUTPUT_DIR / "runtime_status.csv", index=False)

    pyfrbus_summary, pyfrbus_meta, pyfrbus_paths = run_pyfrbus_bundle(preferred_rules)
    pyfrbus_summary.to_csv(OUTPUT_DIR / "pyfrbus_summary.csv", index=False)
    pyfrbus_meta.to_csv(OUTPUT_DIR / "pyfrbus_meta.csv", index=False)
    pyfrbus_paths.to_csv(OUTPUT_DIR / "pyfrbus_paths.csv", index=False)

    lines = [
        "# Phase 10 External Model Robustness Interface",
        "",
        "## Preferred Rules",
        "",
        preferred_rules.round(6).to_markdown(index=False),
        "",
        "## PyFRBUS Results",
        "",
        pyfrbus_summary.round(6).to_markdown(index=False),
        "",
        "## PyFRBUS Fixed-Point Status",
        "",
        pyfrbus_meta.to_markdown(index=False),
        "",
        "## Runtime Status",
        "",
        runtime_status.to_markdown(index=False),
        "",
        "## Model Inventory",
        "",
        inventory_df[
            [
                "model_id",
                "priority_group",
                "runner_family",
                "available_locally",
                "source_path",
                "interface_status",
                "candidate_inflation_vars",
                "candidate_output_gap_vars",
                "candidate_policy_rate_vars",
            ]
        ]
        .to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- 本文件汇报 `pyfrbus` 闭环固定点评估；`Dynare/MMB` 批次见 `mmb_summary.csv` 与 `all_external_summary.csv`。",
        "- `pyfrbus` 中使用的状态映射是 `picxfe - pitarg`、`xgap`、`rff(-1) - 2`，并对规则路径做固定点迭代。",
    ]
    (OUTPUT_DIR / "external_model_robustness_summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_phase10_summary(preferred_rules, inventory_df, pyfrbus_summary, pyfrbus_meta, runtime_status)


if __name__ == "__main__":
    main()
