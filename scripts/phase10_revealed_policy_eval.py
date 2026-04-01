from __future__ import annotations

import json
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
    build_svar_context,
    clone_context_with_loss_weights,
    historical_summary,
    load_checkpoint_policy,
    simulate_historical_counterfactual,
    stochastic_policy_summary,
)


REVEALED_DIR = ROOT / "outputs" / "phase10" / "revealed_welfare"
TRAIN_DIR = ROOT / "outputs" / "phase10" / "revealed_policy_training"
OUTPUT_DIR = ROOT / "outputs" / "phase10" / "revealed_policy_eval"


def revealed_loss_weights() -> dict[str, float]:
    payload = json.loads((REVEALED_DIR / "revealed_weights.json").read_text(encoding="utf-8"))
    return {
        "inflation": float(payload["inflation_weight"]),
        "output_gap": float(payload["output_gap_weight"]),
        "rate_smoothing": float(payload["rate_smoothing_weight"]),
    }


def build_revealed_policy_map(context, env_name: str) -> dict[str, object]:
    _, base_map = base_policy_registry(ROOT, context)
    registry_df = pd.read_csv(TRAIN_DIR / env_name / "policy_registry.csv")
    policy_map: dict[str, object] = {
        "historical_actual_policy": base_map["historical_actual_policy"],
        "empirical_taylor_rule": base_map["empirical_taylor_rule"],
    }
    for row in registry_df.to_dict("records"):
        policy_map[row["policy_name"]] = load_checkpoint_policy(row, context)
    return policy_map


def run_context(env_name: str, context) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy_map = build_revealed_policy_map(context, env_name)
    historical_paths = simulate_historical_counterfactual(
        model=context.model,
        policy_map=policy_map,
        initial_state=context.initial_states[0],
        action_dates=context.action_dates,
        state_dates=context.state_dates,
        shocks=context.shock_pool[:-1],
    )
    historical_paths.insert(0, "evaluation_env", env_name)
    historical_df = historical_summary(historical_paths)
    historical_df.insert(0, "evaluation_env", env_name)
    stochastic_df = stochastic_policy_summary(context, policy_map)
    return historical_paths, historical_df, stochastic_df


def cross_transfer(stochastic_all: pd.DataFrame) -> pd.DataFrame:
    df = stochastic_all.copy()
    df["source_env"] = df["policy_name"].str.extract(r"^(?:ppo|td3|sac)_(svar|ann)_revealed_direct$")
    cross = df[df["source_env"].notna() & (df["source_env"] != df["evaluation_env"])].copy()
    cross["rule_family"] = cross["source_env"] + "_revealed_direct"
    return cross.sort_values(["evaluation_env", "mean_discounted_loss"]).reset_index(drop=True)


def write_summary(svar_hist: pd.DataFrame, ann_hist: pd.DataFrame, svar_stoch: pd.DataFrame, ann_stoch: pd.DataFrame, cross_df: pd.DataFrame) -> None:
    lines = [
        "# Phase 10 Revealed-Welfare Policy Evaluation",
        "",
        "## SVAR Historical Counterfactual",
        "",
        svar_hist.round(6).to_markdown(index=False),
        "",
        "## ANN Historical Counterfactual",
        "",
        ann_hist.round(6).to_markdown(index=False),
        "",
        "## SVAR Long-Run Stochastic",
        "",
        svar_stoch.round(6).to_markdown(index=False),
        "",
        "## ANN Long-Run Stochastic",
        "",
        ann_stoch.round(6).to_markdown(index=False),
        "",
        "## Cross Transfer",
        "",
        cross_df.round(6).to_markdown(index=False),
    ]
    (OUTPUT_DIR / "revealed_policy_eval_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weights = revealed_loss_weights()
    svar_context = clone_context_with_loss_weights(build_svar_context(ROOT), weights, "revealed")
    ann_context = clone_context_with_loss_weights(build_ann_context(ROOT), weights, "revealed")

    svar_paths, svar_hist, svar_stoch = run_context("svar", svar_context)
    ann_paths, ann_hist, ann_stoch = run_context("ann", ann_context)

    svar_paths.to_csv(OUTPUT_DIR / "svar_historical_paths.csv", index=False)
    ann_paths.to_csv(OUTPUT_DIR / "ann_historical_paths.csv", index=False)
    svar_hist.to_csv(OUTPUT_DIR / "svar_historical_summary.csv", index=False)
    ann_hist.to_csv(OUTPUT_DIR / "ann_historical_summary.csv", index=False)
    svar_stoch.to_csv(OUTPUT_DIR / "svar_stochastic_summary.csv", index=False)
    ann_stoch.to_csv(OUTPUT_DIR / "ann_stochastic_summary.csv", index=False)

    stochastic_all = pd.concat([svar_stoch, ann_stoch], ignore_index=True)
    cross_df = cross_transfer(stochastic_all)
    cross_df.to_csv(OUTPUT_DIR / "cross_transfer_summary.csv", index=False)
    write_summary(svar_hist, ann_hist, svar_stoch, ann_stoch, cross_df)


if __name__ == "__main__":
    main()
