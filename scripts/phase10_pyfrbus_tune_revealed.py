from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from phase10_pyfrbus_native_utils import ROOT, evaluate_linear_params, load_reference_row, summarize_results_md


OUTPUT_DIR = ROOT / "outputs" / "phase10" / "pyfrbus_native" / "a_tuning"
REFERENCE_POLICY = "sac_svar_revealed_direct"
BASELINE_REVEALED_LOSS = float(
    pd.read_csv(ROOT / "outputs" / "phase10" / "external_model_robustness" / "pyfrbus_summary.csv")
    .loc[lambda x: x["policy_name"] == "pyfrbus_baseline", "total_discounted_revealed_loss"]
    .iloc[0]
)


def coarse_candidates() -> list[tuple[str, float, float, float, float]]:
    rows: list[tuple[str, float, float, float, float]] = []
    base = load_reference_row(REFERENCE_POLICY)
    rows.append(
        (
            "reference_sac_svar_revealed_direct",
            float(base["intercept"]),
            float(base["inflation_gap"]),
            float(base["output_gap"]),
            float(base["lagged_policy_rate_gap"]),
        )
    )
    rows.extend(
        [
            ("const_0p80", 0.80, 0.0, 0.0, 0.0),
            ("const_1p00", 1.00, 0.0, 0.0, 0.0),
            ("const_1p20", 1.20, 0.0, 0.0, 0.0),
            ("const_1p00_lag0p01", 1.00, 0.0, 0.0, 0.01),
            ("const_0p995_lag0p01", 0.995, 0.0, 0.0, 0.01),
            ("const_1p005", 1.005, 0.0, 0.0, 0.0),
        ]
    )
    return rows


def fine_candidates() -> list[tuple[str, float, float, float, float]]:
    rows: list[tuple[str, float, float, float, float]] = []
    idx = 0
    for intercept, inflation_gap, output_gap, lagged_rate_gap in itertools.product(
        [0.995, 1.000, 1.005],
        [-0.002, 0.0, 0.002],
        [-0.002, 0.0, 0.002],
        [0.0, 0.01],
    ):
        rows.append((f"fine_{idx:02d}", intercept, inflation_gap, output_gap, lagged_rate_gap))
        idx += 1
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = coarse_candidates() + fine_candidates()
    evaluations: list[dict[str, object]] = []

    for name, intercept, inflation_gap, output_gap, lagged_rate_gap in candidates:
        summary = evaluate_linear_params(name, intercept, inflation_gap, output_gap, lagged_rate_gap)
        summary["revealed_improvement_vs_baseline_pct"] = (
            (BASELINE_REVEALED_LOSS - float(summary["total_discounted_revealed_loss"])) / BASELINE_REVEALED_LOSS * 100.0
        )
        evaluations.append(summary)

    result_df = pd.DataFrame(evaluations).sort_values("total_discounted_revealed_loss").reset_index(drop=True)
    result_df.insert(0, "rank", np.arange(1, len(result_df) + 1))
    result_df.to_csv(OUTPUT_DIR / "linear_tuning_results.csv", index=False)

    best = result_df.iloc[0].to_dict()
    pd.DataFrame([best]).to_csv(OUTPUT_DIR / "best_linear_rule.csv", index=False)

    preview = result_df[
        [
            "rank",
            "policy_name",
            "total_discounted_revealed_loss",
            "revealed_improvement_vs_baseline_pct",
            "total_discounted_loss",
            "intercept",
            "inflation_coeff",
            "output_coeff",
            "lagged_rate_coeff",
        ]
    ].head(20)
    summarize_results_md(
        OUTPUT_DIR / "tuning_summary.md",
        "PyFRBUS Revealed-Loss Local Tuning",
        [
            ("Top Results", preview.round(6)),
            (
                "Reference",
                pd.DataFrame([best]).round(6),
            ),
        ],
    )


if __name__ == "__main__":
    main()
