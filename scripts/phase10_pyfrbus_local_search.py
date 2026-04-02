from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monetary_rl.phase10_utils import build_linear_policy
from phase10_external_model_robustness import run_pyfrbus_fixed_point, welfare_summary


OUTPUT_DIR = ROOT / "outputs" / "phase10" / "external_model_robustness"
COUNTERFACTUAL_DIR = ROOT / "outputs" / "phase10" / "counterfactual_eval"
REVEALED_TRAIN_DIR = ROOT / "outputs" / "phase10" / "revealed_policy_training"

BASE_POLICY_NAMES = [
    "empirical_taylor_rule",
    "sac_svar_revealed_direct",
]
SCALE_GRID = [0.7, 0.85, 1.15, 1.3]
INTERCEPT_OFFSETS = [-0.5, -0.25, 0.25, 0.5]


def load_registry() -> pd.DataFrame:
    frames = [pd.read_csv(COUNTERFACTUAL_DIR / "unified_policy_registry.csv")]
    revealed_path = REVEALED_TRAIN_DIR / "policy_registry.csv"
    if revealed_path.exists():
        frames.append(pd.read_csv(revealed_path))
    registry = pd.concat(frames, ignore_index=True, sort=False)
    registry = registry.drop_duplicates(subset=["policy_name"], keep="last").reset_index(drop=True)
    return registry


def candidate_specs(row: pd.Series) -> list[dict[str, float | str]]:
    base = {
        "policy_name": str(row["policy_name"]),
        "parent_policy": str(row["policy_name"]),
        "intercept": float(row.get("intercept", 0.0) or 0.0),
        "inflation_gap": float(row.get("inflation_gap", 0.0) or 0.0),
        "output_gap": float(row.get("output_gap", 0.0) or 0.0),
        "lagged_policy_rate_gap": float(row.get("lagged_policy_rate_gap", 0.0) or 0.0),
    }
    candidates = [base]
    for offset in INTERCEPT_OFFSETS:
        candidate = base.copy()
        candidate["policy_name"] = f"{base['parent_policy']}_int_{offset:+.2f}".replace(".", "p")
        candidate["intercept"] = float(base["intercept"]) + offset
        candidates.append(candidate)
    for field in ["inflation_gap", "output_gap", "lagged_policy_rate_gap"]:
        for scale in SCALE_GRID:
            candidate = base.copy()
            candidate["policy_name"] = f"{base['parent_policy']}_{field}_x{scale:.2f}".replace(".", "p")
            candidate[field] = float(base[field]) * scale
            candidates.append(candidate)
    for scale in SCALE_GRID:
        candidate = base.copy()
        candidate["policy_name"] = f"{base['parent_policy']}_all_x{scale:.2f}".replace(".", "p")
        candidate["inflation_gap"] = float(base["inflation_gap"]) * scale
        candidate["output_gap"] = float(base["output_gap"]) * scale
        candidate["lagged_policy_rate_gap"] = float(base["lagged_policy_rate_gap"]) * scale
        candidates.append(candidate)
    return candidates


def evaluate_candidate(spec: dict[str, float | str]) -> dict[str, object]:
    policy_fn = build_linear_policy(
        str(spec["policy_name"]),
        float(spec["intercept"]),
        float(spec["inflation_gap"]),
        float(spec["output_gap"]),
        float(spec["lagged_policy_rate_gap"]),
    )
    window, meta = run_pyfrbus_fixed_point(str(spec["policy_name"]), policy_fn)
    summary = welfare_summary(window.set_index("period"), str(spec["policy_name"]))
    summary.update(
        {
            "parent_policy": spec["parent_policy"],
            "intercept": spec["intercept"],
            "inflation_gap": spec["inflation_gap"],
            "output_gap": spec["output_gap"],
            "lagged_policy_rate_gap": spec["lagged_policy_rate_gap"],
            "converged": meta["converged"],
            "iterations": meta["iterations"],
            "clip_rate": meta["clip_rate"],
            "max_fixed_point_gap": meta["max_fixed_point_gap"],
        }
    )
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    rows = registry.loc[registry["policy_name"].isin(BASE_POLICY_NAMES)].copy()
    results: list[dict[str, object]] = []
    for row in rows.to_dict("records"):
        for candidate in candidate_specs(pd.Series(row)):
            results.append(evaluate_candidate(candidate))
    result_df = pd.DataFrame(results).sort_values("total_discounted_loss").reset_index(drop=True)
    result_df.to_csv(OUTPUT_DIR / "pyfrbus_local_search.csv", index=False)


if __name__ == "__main__":
    main()
