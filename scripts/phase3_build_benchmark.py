from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monetary_rl.models import LQBenchmarkConfig, LQBenchmarkModel


OUTPUT_DIR = ROOT / "outputs" / "phase3"
CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_lq.json"
PHASE2_SUMMARY_PATH = ROOT / "outputs" / "phase2" / "phase2_summary.md"


def zero_gap_policy(state: np.ndarray, t: int) -> float:
    del t
    return 0.0


def write_summary(config: LQBenchmarkConfig, model: LQBenchmarkModel, simulation: dict[str, np.ndarray]) -> None:
    q, n, r = model.qnr_matrices()
    states = simulation["states"]
    actions = simulation["actions"]
    losses = simulation["losses"]

    lines: list[str] = []
    lines.append("# Phase 3 Benchmark Summary")
    lines.append("")
    lines.append("## Benchmark Role")
    lines.append("")
    lines.append("This benchmark is a stylized 3-state LQ model. It is not the empirical SVAR environment itself; instead, it is the theoretical benchmark that matches the Phase 0/1 specification and is calibrated to realistic magnitudes informed by Phase 2.")
    lines.append("")
    lines.append("## Core Calibration")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Name | {config.name} |")
    lines.append(f"| Discount factor | {config.discount_factor:.4f} |")
    lines.append(f"| Inflation target | {config.inflation_target:.4f} |")
    lines.append(f"| Neutral rate | {config.neutral_rate:.4f} |")
    lines.append(f"| Loss weight: inflation | {config.loss_weights['inflation']:.4f} |")
    lines.append(f"| Loss weight: output gap | {config.loss_weights['output_gap']:.4f} |")
    lines.append(f"| Loss weight: rate smoothing | {config.loss_weights['rate_smoothing']:.4f} |")
    lines.append("")

    lines.append("## State Transition")
    lines.append("")
    lines.append("State vector:")
    lines.append("")
    lines.append("$$")
    lines.append("s_t = [\\tilde{\\pi}_t, x_t, \\tilde{i}_{t-1}]^\\top")
    lines.append("$$")
    lines.append("")
    lines.append("Transition:")
    lines.append("")
    lines.append("$$")
    lines.append("s_{t+1} = A s_t + B a_t + \\Sigma \\varepsilon_{t+1}")
    lines.append("$$")
    lines.append("")
    lines.append("### A Matrix")
    lines.append("")
    lines.append(pd.DataFrame(config.A, columns=config.state_names, index=config.state_names).round(4).to_markdown())
    lines.append("")
    lines.append("### B Matrix")
    lines.append("")
    lines.append(pd.DataFrame(config.B, columns=[config.action_name], index=config.state_names).round(4).to_markdown())
    lines.append("")
    lines.append("### Sigma Matrix")
    lines.append("")
    lines.append(pd.DataFrame(config.Sigma, columns=config.state_names, index=config.state_names).round(4).to_markdown())
    lines.append("")

    lines.append("## LQ Loss Representation")
    lines.append("")
    lines.append("Single-period loss:")
    lines.append("")
    lines.append("$$")
    lines.append("\\ell_t = \\lambda_\\pi \\tilde{\\pi}_t^2 + \\lambda_x x_t^2 + \\lambda_i(\\tilde{i}_t-\\tilde{i}_{t-1})^2")
    lines.append("$$")
    lines.append("")
    lines.append("### Q Matrix")
    lines.append("")
    lines.append(pd.DataFrame(q, columns=config.state_names, index=config.state_names).round(4).to_markdown())
    lines.append("")
    lines.append("### N Matrix")
    lines.append("")
    lines.append(pd.DataFrame(n, columns=[config.action_name], index=config.state_names).round(4).to_markdown())
    lines.append("")
    lines.append("### R Matrix")
    lines.append("")
    lines.append(pd.DataFrame(r, columns=[config.action_name], index=[config.action_name]).round(4).to_markdown())
    lines.append("")

    lines.append("## Example Simulation")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append("| Policy | zero-gap policy: $a_t = 0$ |")
    lines.append("| Initial state | $[1.0, -1.0, 0.0]^\\top$ |")
    lines.append(f"| Horizon | {len(actions)} |")
    lines.append(f"| Total discounted loss | {simulation['total_discounted_loss'][0]:.6f} |")
    lines.append("")

    path_df = pd.DataFrame(
        {
            "period": np.arange(states.shape[0]),
            "inflation_gap": states[:, 0],
            "output_gap": states[:, 1],
            "lagged_policy_rate_gap": states[:, 2],
        }
    ).round(6)
    lines.append(path_df.head(10).to_markdown(index=False))
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Shock scales are anchored to Phase 2 empirical SVAR residual standard deviations.")
    lines.append("- Dynamic coefficients are stylized and chosen to provide a stable, interpretable, 3-state LQ benchmark.")
    lines.append("- This benchmark is intentionally simpler than the empirical SVAR environment, which uses additional lags.")
    lines.append("- ANN environment tuning remains a parallel task and does not block this benchmark track.")
    lines.append("")

    (OUTPUT_DIR / "benchmark_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = LQBenchmarkConfig.from_json(CONFIG_PATH)
    model = LQBenchmarkModel(config)

    initial_state = np.array([1.0, -1.0, 0.0], dtype=float)
    simulation = model.simulate(initial_state, zero_gap_policy, horizon=40, seed=42)

    sim_df = pd.DataFrame(
        simulation["states"],
        columns=config.state_names,
    )
    sim_df.insert(0, "period", np.arange(sim_df.shape[0]))
    sim_df["action"] = np.append(simulation["actions"], np.nan)
    sim_df["loss"] = np.append(simulation["losses"], np.nan)
    sim_df.to_csv(OUTPUT_DIR / "benchmark_zero_policy_simulation.csv", index=False)

    summary = {
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "discount_factor": config.discount_factor,
        "loss_weights": config.loss_weights,
        "A": config.A.tolist(),
        "B": config.B.tolist(),
        "Sigma": config.Sigma.tolist(),
        "initial_state": initial_state.tolist(),
        "example_total_discounted_loss": float(simulation["total_discounted_loss"][0]),
        "phase2_summary_present": PHASE2_SUMMARY_PATH.exists(),
    }
    (OUTPUT_DIR / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    write_summary(config, model, simulation)


if __name__ == "__main__":
    main()
