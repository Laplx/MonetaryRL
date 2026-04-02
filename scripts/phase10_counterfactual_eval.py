from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monetary_rl.phase10_utils import (
    base_policy_registry,
    build_ann_context,
    build_linear_policy,
    build_svar_context,
    historical_summary,
    load_checkpoint_policy,
    simulate_historical_counterfactual,
    stochastic_policy_summary,
)


OUTPUT_DIR = ROOT / "outputs" / "phase10" / "counterfactual_eval"


def load_unified_registry() -> pd.DataFrame:
    base_df = pd.read_csv(ROOT / "outputs" / "phase8" / "policy_registry.csv").copy()
    base_df["callable_type"] = base_df["intercept"].notna().map({True: "linear", False: "historical"})
    base_df["rule_family"] = base_df["policy_name"].map(
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
    base_df["source_env"] = base_df["rule_family"].map(
        {
            "benchmark_transfer": "benchmark",
            "empirical_rule": "svar",
            "theory_reference": "benchmark",
            "historical_actual": "historical",
        }
    ).fillna("benchmark")
    base_df["training_env"] = base_df["source_env"]
    base_df["policy_parameterization"] = base_df["policy_name"].map(
        {
            "ppo_benchmark_transfer": "linear_surrogate",
            "td3_benchmark_transfer": "linear_surrogate",
            "sac_benchmark_transfer": "linear_surrogate",
        }
    ).fillna("fixed_rule")
    base_df["algo"] = base_df.get("algo")
    base_df["seed"] = base_df.get("seed")
    base_df["checkpoint_path"] = ""
    base_df["config_path"] = ""

    frames = [
        base_df,
        pd.read_csv(ROOT / "outputs" / "phase10" / "svar_direct" / "policy_registry.csv"),
        pd.read_csv(ROOT / "outputs" / "phase10" / "ann_direct" / "policy_registry.csv"),
    ]
    ppo_variant_path = ROOT / "outputs" / "phase10" / "ppo_policy_variants" / "policy_registry.csv"
    if ppo_variant_path.exists():
        ppo_variants = pd.read_csv(ppo_variant_path)
        if "policy_parameterization" in ppo_variants.columns:
            ppo_variants = ppo_variants.loc[ppo_variants["policy_parameterization"] == "nonlinear_policy"].copy()
        frames.append(ppo_variants)

    revealed_training_path = ROOT / "outputs" / "phase10" / "revealed_policy_training" / "policy_registry.csv"
    if revealed_training_path.exists():
        frames.append(pd.read_csv(revealed_training_path))

    columns = sorted(set().union(*(set(frame.columns) for frame in frames)))
    unified = pd.concat([frame.reindex(columns=columns) for frame in frames], ignore_index=True)
    return unified.drop_duplicates(subset=["policy_name"], keep="last").reset_index(drop=True)


def build_policy_map(context, unified_registry: pd.DataFrame) -> dict[str, object]:
    _, base_map = base_policy_registry(ROOT, context)
    policies: dict[str, object] = {}
    for row in unified_registry.to_dict("records"):
        name = row["policy_name"]
        if row["callable_type"] == "historical":
            policies[name] = base_map[name]
        elif row["callable_type"] == "linear":
            policies[name] = build_linear_policy(
                name,
                float(row["intercept"]),
                float(row["inflation_gap"]),
                float(row["output_gap"]),
                float(row["lagged_policy_rate_gap"]),
            )
        elif row["callable_type"] == "checkpoint":
            policies[name] = load_checkpoint_policy(row, context)
        else:
            raise ValueError(f"Unsupported callable type: {row['callable_type']}")
    return policies


def run_for_context(context, unified_registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    policies = build_policy_map(context, unified_registry)
    historical_paths = simulate_historical_counterfactual(
        model=context.model,
        policy_map=policies,
        initial_state=context.initial_states[0],
        action_dates=context.action_dates,
        state_dates=context.state_dates,
        shocks=context.shock_pool[:-1],
    )
    historical_paths.insert(0, "evaluation_env", context.name)
    historical_df = historical_summary(historical_paths)
    historical_df.insert(0, "evaluation_env", context.name)

    feedback_policies = {name: fn for name, fn in policies.items() if name != "historical_actual_policy"}
    stochastic_df = stochastic_policy_summary(context, feedback_policies)
    return historical_paths, historical_df, stochastic_df


def make_cross_transfer_summary(stochastic_df: pd.DataFrame, unified_registry: pd.DataFrame) -> pd.DataFrame:
    merged = stochastic_df.merge(
        unified_registry[
            [
                "policy_name",
                "rule_family",
                "source_env",
                "training_env",
                "policy_parameterization",
            ]
        ],
        on="policy_name",
        how="left",
    )
    mask = merged["rule_family"].isin(["benchmark_transfer", "svar_direct", "ann_direct"]) & (
        merged["source_env"] != merged["evaluation_env"]
    )
    cross_df = merged.loc[mask].copy()
    return cross_df.sort_values(["evaluation_env", "mean_discounted_loss"]).reset_index(drop=True)


def write_summary(
    unified_registry: pd.DataFrame,
    svar_historical: pd.DataFrame,
    ann_historical: pd.DataFrame,
    svar_stochastic: pd.DataFrame,
    ann_stochastic: pd.DataFrame,
    cross_df: pd.DataFrame,
) -> None:
    lines = [
        "# Phase 10 Unified Counterfactual Evaluation",
        "",
        "## Unified Policy Registry",
        "",
        unified_registry[
            [
                "policy_name",
                "rule_family",
                "source_env",
                "policy_parameterization",
                "callable_type",
            ]
        ]
        .sort_values(["rule_family", "policy_name"])
        .to_markdown(index=False),
        "",
        "## SVAR Historical Counterfactual",
        "",
        svar_historical.round(6).to_markdown(index=False),
        "",
        "## ANN Historical Counterfactual",
        "",
        ann_historical.round(6).to_markdown(index=False),
        "",
        "## SVAR Long-Run Stochastic Evaluation",
        "",
        svar_stochastic.round(6).to_markdown(index=False),
        "",
        "## ANN Long-Run Stochastic Evaluation",
        "",
        ann_stochastic.round(6).to_markdown(index=False),
        "",
        "## Cross-Transfer Summary",
        "",
        cross_df[
            [
                "policy_name",
                "rule_family",
                "source_env",
                "evaluation_env",
                "policy_parameterization",
                "mean_discounted_loss",
                "std_discounted_loss",
                "clip_rate",
                "explosion_rate",
            ]
        ]
        .round(6)
        .to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- `Phase 8/9` remains the benchmark-transfer baseline; this file adds the direct-trained empirical rules under the same evaluator.",
        "- `benchmark transfer` and empirical direct-trained rules are kept distinct in all tables.",
        "- Lucas critique still applies because both empirical environments hold reduced-form transitions fixed under policy changes.",
    ]
    (OUTPUT_DIR / "counterfactual_eval_summary.md").write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unified_registry = load_unified_registry().sort_values(["rule_family", "policy_name"]).reset_index(drop=True)
    unified_registry.to_csv(OUTPUT_DIR / "unified_policy_registry.csv", index=False)

    svar_context = build_svar_context(ROOT)
    ann_context = build_ann_context(ROOT)

    svar_paths, svar_historical, svar_stochastic = run_for_context(svar_context, unified_registry)
    ann_paths, ann_historical, ann_stochastic = run_for_context(ann_context, unified_registry)

    svar_paths.to_csv(OUTPUT_DIR / "svar_historical_paths.csv", index=False)
    ann_paths.to_csv(OUTPUT_DIR / "ann_historical_paths.csv", index=False)
    svar_historical.to_csv(OUTPUT_DIR / "svar_historical_summary.csv", index=False)
    ann_historical.to_csv(OUTPUT_DIR / "ann_historical_summary.csv", index=False)
    svar_stochastic.to_csv(OUTPUT_DIR / "svar_stochastic_summary.csv", index=False)
    ann_stochastic.to_csv(OUTPUT_DIR / "ann_stochastic_summary.csv", index=False)

    stochastic_all = pd.concat([svar_stochastic, ann_stochastic], ignore_index=True)
    cross_df = make_cross_transfer_summary(stochastic_all, unified_registry)
    cross_df.to_csv(OUTPUT_DIR / "cross_transfer_summary.csv", index=False)

    write_summary(unified_registry, svar_historical, ann_historical, svar_stochastic, ann_stochastic, cross_df)


if __name__ == "__main__":
    main()
