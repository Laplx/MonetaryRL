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
from monetary_rl.solvers import build_optimal_linear_policy, solve_discounted_lq_riccati


OUTPUT_DIR = ROOT / "outputs" / "phase4"
CONFIG_PATH = ROOT / "src" / "monetary_rl" / "config" / "benchmark_lq.json"


def zero_gap_policy(state: np.ndarray, t: int) -> float:
    del state, t
    return 0.0


def format_matrix(matrix: np.ndarray, row_names: list[str], col_names: list[str]) -> str:
    return pd.DataFrame(matrix, index=row_names, columns=col_names).round(6).to_markdown()


def write_summary(
    config: LQBenchmarkConfig,
    model: LQBenchmarkModel,
    solution,
    zero_sim: dict[str, np.ndarray],
    optimal_sim: dict[str, np.ndarray],
) -> None:
    q, n, r = model.qnr_matrices()
    eig_modulus = np.abs(solution.closed_loop_eigenvalues)
    stable = bool(np.all(eig_modulus < 1.0))

    zero_loss = float(zero_sim["total_discounted_loss"][0])
    optimal_loss = float(optimal_sim["total_discounted_loss"][0])
    improvement_pct = (zero_loss - optimal_loss) / zero_loss * 100.0

    lines: list[str] = []
    lines.append("# Phase 4 LQ Solution Summary")
    lines.append("")
    lines.append("## Core Result")
    lines.append("")
    lines.append("The Phase 3 benchmark now has an infinite-horizon discounted LQ solution. The optimal policy is a linear state-feedback rule obtained from the generalized discounted discrete algebraic Riccati equation.")
    lines.append("")
    lines.append("## Problem")
    lines.append("")
    lines.append("$$")
    lines.append("V(s)=\\min_a \\left\\{ \\ell(s,a)+\\beta \\mathbb{E}[V(s')\\mid s,a] \\right\\}")
    lines.append("$$")
    lines.append("")
    lines.append("$$")
    lines.append("\\ell(s_t,a_t)=s_t^\\top Q s_t + 2 s_t^\\top N a_t + a_t^\\top R a_t")
    lines.append("$$")
    lines.append("")
    lines.append("## Q / N / R")
    lines.append("")
    lines.append("### Q")
    lines.append("")
    lines.append(format_matrix(q, config.state_names, config.state_names))
    lines.append("")
    lines.append("### N")
    lines.append("")
    lines.append(format_matrix(n, config.state_names, [config.action_name]))
    lines.append("")
    lines.append("### R")
    lines.append("")
    lines.append(format_matrix(r, [config.action_name], [config.action_name]))
    lines.append("")

    lines.append("## Riccati Solution")
    lines.append("")
    lines.append("### P Matrix")
    lines.append("")
    lines.append(format_matrix(solution.P, config.state_names, config.state_names))
    lines.append("")
    lines.append("### Feedback Matrix F in $a_t=-F s_t$")
    lines.append("")
    lines.append(format_matrix(solution.F, [config.action_name], config.state_names))
    lines.append("")
    lines.append("### Equivalent Policy Matrix K in $a_t=K s_t$")
    lines.append("")
    lines.append(format_matrix(solution.K, [config.action_name], config.state_names))
    lines.append("")

    lines.append("## Closed-loop Stability")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(f"| Stable (all eigenvalue moduli < 1) | {stable} |")
    lines.append(f"| Largest eigenvalue modulus | {eig_modulus.max():.6f} |")
    lines.append(f"| Value-function constant | {solution.value_constant:.6f} |")
    lines.append("")
    eig_df = pd.DataFrame(
        {
            "eigenvalue_real": np.real(solution.closed_loop_eigenvalues),
            "eigenvalue_imag": np.imag(solution.closed_loop_eigenvalues),
            "modulus": eig_modulus,
        }
    ).round(6)
    lines.append(eig_df.to_markdown(index=False))
    lines.append("")

    lines.append("## Stationary Covariance")
    lines.append("")
    lines.append(format_matrix(solution.stationary_covariance, config.state_names, config.state_names))
    lines.append("")

    lines.append("## Example Simulation Comparison")
    lines.append("")
    lines.append("| Item | Zero policy | Optimal policy |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Total discounted loss | {zero_loss:.6f} | {optimal_loss:.6f} |")
    lines.append(f"| Improvement (%) | 0.000000 | {improvement_pct:.6f} |")
    lines.append("")

    comparison_df = pd.DataFrame(
        {
            "period": np.arange(10),
            "zero_inflation_gap": zero_sim["states"][:10, 0],
            "opt_inflation_gap": optimal_sim["states"][:10, 0],
            "zero_output_gap": zero_sim["states"][:10, 1],
            "opt_output_gap": optimal_sim["states"][:10, 1],
            "zero_action": np.append(zero_sim["actions"][:9], np.nan),
            "opt_action": np.append(optimal_sim["actions"][:9], np.nan),
        }
    ).round(6)
    lines.append(comparison_df.to_markdown(index=False))
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- The optimal policy reacts positively to inflation and output-gap pressures because the policy matrix $K$ maps adverse states into corrective interest-rate moves.")
    lines.append("- The lagged-rate state enters with a positive coefficient in $K$, which implies policy smoothing through the state-augmentation channel.")
    lines.append("- This is the theoretical benchmark against which RL will be judged in Phase 5-6.")
    lines.append("")

    (OUTPUT_DIR / "lq_solution_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = LQBenchmarkConfig.from_json(CONFIG_PATH)
    model = LQBenchmarkModel(config)
    solution = solve_discounted_lq_riccati(model)
    optimal_policy = build_optimal_linear_policy(solution)

    initial_state = np.array([1.0, -1.0, 0.0], dtype=float)
    zero_sim = model.simulate(initial_state, zero_gap_policy, horizon=40, seed=42)
    optimal_sim = model.simulate(initial_state, optimal_policy, horizon=40, seed=42)

    zero_df = pd.DataFrame(zero_sim["states"], columns=config.state_names)
    zero_df.insert(0, "period", np.arange(len(zero_df)))
    zero_df["action"] = np.append(zero_sim["actions"], np.nan)
    zero_df["loss"] = np.append(zero_sim["losses"], np.nan)
    zero_df.to_csv(OUTPUT_DIR / "zero_policy_simulation.csv", index=False)

    opt_df = pd.DataFrame(optimal_sim["states"], columns=config.state_names)
    opt_df.insert(0, "period", np.arange(len(opt_df)))
    opt_df["action"] = np.append(optimal_sim["actions"], np.nan)
    opt_df["loss"] = np.append(optimal_sim["losses"], np.nan)
    opt_df.to_csv(OUTPUT_DIR / "optimal_policy_simulation.csv", index=False)

    summary_json = {
        "P": solution.P.tolist(),
        "F": solution.F.tolist(),
        "K": solution.K.tolist(),
        "closed_loop_A": solution.closed_loop_A.tolist(),
        "closed_loop_eigenvalues": [
            {"real": float(np.real(v)), "imag": float(np.imag(v)), "modulus": float(np.abs(v))}
            for v in solution.closed_loop_eigenvalues
        ],
        "stationary_covariance": solution.stationary_covariance.tolist(),
        "value_constant": float(solution.value_constant),
        "zero_policy_discounted_loss": float(zero_sim["total_discounted_loss"][0]),
        "optimal_policy_discounted_loss": float(optimal_sim["total_discounted_loss"][0]),
    }
    (OUTPUT_DIR / "lq_solution_summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")

    write_summary(config, model, solution, zero_sim, optimal_sim)


if __name__ == "__main__":
    main()
