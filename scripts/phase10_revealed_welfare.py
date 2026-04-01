from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from monetary_rl.phase10_utils import (
    base_policy_registry,
    build_ann_context,
    build_linear_policy,
    build_svar_context,
    load_checkpoint_policy,
    make_empirical_env,
    training_support_payload,
)
from monetary_rl.solvers.riccati import solve_discounted_lq_riccati


OUTPUT_DIR = ROOT / "outputs" / "phase10" / "revealed_welfare"
COUNTERFACTUAL_DIR = ROOT / "outputs" / "phase10" / "counterfactual_eval"
PHASE2_DIR = ROOT / "outputs" / "phase2"


@dataclass
class ProxyConfig:
    discount_factor: float
    loss_weights: dict[str, float]


class ObservedStateLQProxy:
    def __init__(self, A: np.ndarray, B: np.ndarray, Sigma: np.ndarray, config: ProxyConfig) -> None:
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.config = config

    @property
    def state_dim(self) -> int:
        return 3

    @property
    def action_dim(self) -> int:
        return 1

    def qnr_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weights = self.config.loss_weights
        q = np.diag([weights["inflation"], weights["output_gap"], weights["rate_smoothing"]])
        n = np.array([[0.0], [0.0], [-weights["rate_smoothing"]]])
        r = np.array([[weights["rate_smoothing"]]])
        return q, n, r


def load_taylor_target() -> dict[str, float]:
    return json.loads((PHASE2_DIR / "taylor_rule.json").read_text(encoding="utf-8"))


def estimate_observed_state_proxy(context) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.asarray([context.model.observe(state) for state in context.initial_states], dtype=float)
    current = observed[:-1]
    next_obs = observed[1:]
    action_gap = context.env_df["policy_rate"].iloc[:-1].to_numpy(dtype=float) - context.model.config.neutral_rate

    design = np.column_stack([current, action_gap])
    beta_infl, *_ = np.linalg.lstsq(design, next_obs[:, 0], rcond=None)
    beta_output, *_ = np.linalg.lstsq(design, next_obs[:, 1], rcond=None)
    residuals = np.column_stack(
        [
            next_obs[:, 0] - design @ beta_infl,
            next_obs[:, 1] - design @ beta_output,
            np.zeros(len(design), dtype=float),
        ]
    )
    cov = np.cov(residuals.T, ddof=1)
    cov = cov + np.eye(3) * 1e-10
    sigma = np.linalg.cholesky(cov)

    A = np.array(
        [
            beta_infl[:3],
            beta_output[:3],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    B = np.array([[beta_infl[3]], [beta_output[3]], [1.0]], dtype=float)
    return A, B, sigma


def solve_revealed_weights(context) -> tuple[dict[str, float], pd.DataFrame]:
    taylor = load_taylor_target()
    coeffs = taylor["coefficients"]
    target = np.array(
        [float(coeffs["inflation"]), float(coeffs["output_gap"]), float(coeffs["policy_rate_lag1"])],
        dtype=float,
    )
    A, B, sigma = estimate_observed_state_proxy(context)
    grid_rows: list[dict[str, float]] = []

    def objective(log_weights: np.ndarray) -> float:
        output_weight = float(np.exp(log_weights[0]))
        rate_weight = float(np.exp(log_weights[1]))
        proxy = ObservedStateLQProxy(
            A=A,
            B=B,
            Sigma=sigma,
            config=ProxyConfig(
                discount_factor=context.model.config.discount_factor,
                loss_weights={"inflation": 1.0, "output_gap": output_weight, "rate_smoothing": rate_weight},
            ),
        )
        try:
            solution = solve_discounted_lq_riccati(proxy)
            implied = solution.K.reshape(-1)
            gap = float(np.mean(((implied - target) / np.maximum(np.abs(target), 0.1)) ** 2))
            eig_penalty = max(float(np.max(np.abs(solution.closed_loop_eigenvalues))) - 0.999, 0.0) ** 2 * 1e4
            return gap + eig_penalty
        except Exception:
            return 1e9

    output_grid = np.exp(np.linspace(np.log(1e-3), np.log(5.0), 21))
    rate_grid = np.exp(np.linspace(np.log(1e-3), np.log(5.0), 21))
    for output_weight in output_grid:
        for rate_weight in rate_grid:
            value = objective(np.log([output_weight, rate_weight]))
            grid_rows.append(
                {
                    "output_weight": float(output_weight),
                    "rate_smoothing_weight": float(rate_weight),
                    "objective": float(value),
                }
            )
    grid_df = pd.DataFrame(grid_rows).sort_values("objective").reset_index(drop=True)
    start = grid_df.iloc[0]
    result = optimize.minimize(
        objective,
        x0=np.log([float(start["output_weight"]), float(start["rate_smoothing_weight"])]),
        method="L-BFGS-B",
        bounds=[(-8.0, 3.0), (-8.0, 3.0)],
    )
    output_weight = float(np.exp(result.x[0]))
    rate_weight = float(np.exp(result.x[1]))
    proxy = ObservedStateLQProxy(
        A=A,
        B=B,
        Sigma=sigma,
        config=ProxyConfig(
            discount_factor=context.model.config.discount_factor,
            loss_weights={"inflation": 1.0, "output_gap": output_weight, "rate_smoothing": rate_weight},
        ),
    )
    solution = solve_discounted_lq_riccati(proxy)
    payload = {
        "inflation_weight": 1.0,
        "output_gap_weight": output_weight,
        "rate_smoothing_weight": rate_weight,
        "objective": float(result.fun),
        "success": bool(result.success),
        "implied_phi_pi": float(solution.K[0, 0]),
        "implied_phi_x": float(solution.K[0, 1]),
        "implied_phi_i": float(solution.K[0, 2]),
        "target_phi_pi": target[0],
        "target_phi_x": target[1],
        "target_phi_i": target[2],
    }
    return payload, grid_df


def revealed_loss_series(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    work = df.dropna(subset=["policy_rate"]).copy()
    work["period"] = work.groupby(["evaluation_env", "policy_name"]).cumcount()
    work["revealed_loss"] = (
        work["inflation_gap"] ** 2
        + weights["output_gap_weight"] * work["output_gap"] ** 2
        + weights["rate_smoothing_weight"] * work["rate_change"] ** 2
    )
    work["revealed_discounted_loss"] = work["revealed_loss"] * (0.99 ** work["period"])
    return work


def revealed_summary(path_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        path_df.groupby(["evaluation_env", "policy_name"], as_index=False)
        .agg(
            total_discounted_revealed_loss=("revealed_discounted_loss", "sum"),
            mean_period_revealed_loss=("revealed_loss", "mean"),
        )
        .sort_values(["evaluation_env", "total_discounted_revealed_loss"])
        .reset_index(drop=True)
    )
    return summary


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
        else:
            policies[name] = load_checkpoint_policy(row, context)
    return policies


def evaluate_revealed_stochastic(context, unified_registry: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    policies = build_policy_map(context, unified_registry)
    rows: list[dict[str, float]] = []
    for idx, (policy_name, policy_fn) in enumerate(policies.items()):
        if policy_name == "historical_actual_policy":
            continue
        env = make_empirical_env(context, horizon=120, seed=20260401 + idx)
        episode_losses: list[float] = []
        for ep in range(96):
            state = env.reset(seed=20260501 + ep + idx * 100)
            done = False
            discount = 1.0
            total_loss = 0.0
            while not done:
                action = float(policy_fn(state.copy(), 0))
                next_state, reward, done, info = env.step(action)
                del reward
                total_loss += (
                    float(state[0]) ** 2
                    + weights["output_gap_weight"] * float(state[1]) ** 2
                    + weights["rate_smoothing_weight"] * (float(info["action"]) - float(state[2])) ** 2
                ) * discount
                discount *= context.model.config.discount_factor
                state = next_state
            episode_losses.append(total_loss)
        rows.append(
            {
                "evaluation_env": context.name,
                "policy_name": policy_name,
                "mean_discounted_revealed_loss": float(np.mean(episode_losses)),
                "std_discounted_revealed_loss": float(np.std(episode_losses, ddof=1)) if len(episode_losses) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["evaluation_env", "mean_discounted_revealed_loss"]).reset_index(drop=True)


def ranking_change_table(baseline_df: pd.DataFrame, revealed_df: pd.DataFrame, baseline_col: str, revealed_col: str) -> pd.DataFrame:
    base = baseline_df.copy()
    rev = revealed_df.copy()
    base["baseline_rank"] = base.groupby("evaluation_env")[baseline_col].rank(method="dense")
    rev["revealed_rank"] = rev.groupby("evaluation_env")[revealed_col].rank(method="dense")
    merged = base[["evaluation_env", "policy_name", "baseline_rank"]].merge(
        rev[["evaluation_env", "policy_name", "revealed_rank"]],
        on=["evaluation_env", "policy_name"],
        how="inner",
    )
    merged["rank_change"] = merged["baseline_rank"] - merged["revealed_rank"]
    return merged.sort_values(["evaluation_env", "revealed_rank", "policy_name"]).reset_index(drop=True)


def write_summary(
    weights: dict[str, float],
    grid_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    stochastic_df: pd.DataFrame,
    rank_change_historical: pd.DataFrame,
    rank_change_stochastic: pd.DataFrame,
) -> None:
    lines = [
        "# Phase 10 Revealed Welfare Summary",
        "",
        "## Revealed Weights",
        "",
        pd.DataFrame([weights]).round(6).to_markdown(index=False),
        "",
        "## Top Objective Grid Points",
        "",
        grid_df.head(10).round(6).to_markdown(index=False),
        "",
        "## Historical Re-Scoring",
        "",
        historical_df.round(6).to_markdown(index=False),
        "",
        "## Stochastic Re-Scoring",
        "",
        stochastic_df.round(6).to_markdown(index=False),
        "",
        "## Historical Rank Change",
        "",
        rank_change_historical.to_markdown(index=False),
        "",
        "## Stochastic Rank Change",
        "",
        rank_change_stochastic.to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- Inflation weight is normalized to `1`; only output-gap and rate-smoothing weights are revealed.",
        "- This is a report-only welfare metric based on `SVAR + empirical Taylor` and does not replace the main training objective.",
        "- Lucas critique still applies because the revealed metric is inferred under a fixed reduced-form transition.",
    ]
    (OUTPUT_DIR / "revealed_welfare_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    svar_context = build_svar_context(ROOT)
    ann_context = build_ann_context(ROOT)
    weights, grid_df = solve_revealed_weights(svar_context)
    (OUTPUT_DIR / "revealed_weights.json").write_text(json.dumps(weights, indent=2, ensure_ascii=False), encoding="utf-8")
    grid_df.to_csv(OUTPUT_DIR / "revealed_weight_grid.csv", index=False)
    (OUTPUT_DIR / "revealed_support.json").write_text(
        json.dumps({"svar_support": training_support_payload(svar_context)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    unified_registry = pd.read_csv(COUNTERFACTUAL_DIR / "unified_policy_registry.csv")
    history_paths = pd.concat(
        [
            pd.read_csv(COUNTERFACTUAL_DIR / "svar_historical_paths.csv"),
            pd.read_csv(COUNTERFACTUAL_DIR / "ann_historical_paths.csv"),
        ],
        ignore_index=True,
    )
    revealed_history_paths = revealed_loss_series(history_paths, weights)
    revealed_history_paths.to_csv(OUTPUT_DIR / "revealed_historical_paths.csv", index=False)
    historical_df = revealed_summary(revealed_history_paths)
    historical_df.to_csv(OUTPUT_DIR / "revealed_historical_summary.csv", index=False)

    revealed_stochastic = pd.concat(
        [
            evaluate_revealed_stochastic(svar_context, unified_registry, weights),
            evaluate_revealed_stochastic(ann_context, unified_registry, weights),
        ],
        ignore_index=True,
    )
    revealed_stochastic.to_csv(OUTPUT_DIR / "revealed_stochastic_summary.csv", index=False)

    baseline_historical = pd.concat(
        [
            pd.read_csv(COUNTERFACTUAL_DIR / "svar_historical_summary.csv"),
            pd.read_csv(COUNTERFACTUAL_DIR / "ann_historical_summary.csv"),
        ],
        ignore_index=True,
    )
    baseline_stochastic = pd.concat(
        [
            pd.read_csv(COUNTERFACTUAL_DIR / "svar_stochastic_summary.csv"),
            pd.read_csv(COUNTERFACTUAL_DIR / "ann_stochastic_summary.csv"),
        ],
        ignore_index=True,
    )
    rank_change_historical = ranking_change_table(
        baseline_historical,
        historical_df,
        baseline_col="total_discounted_loss",
        revealed_col="total_discounted_revealed_loss",
    )
    rank_change_stochastic = ranking_change_table(
        baseline_stochastic,
        revealed_stochastic,
        baseline_col="mean_discounted_loss",
        revealed_col="mean_discounted_revealed_loss",
    )
    rank_change_historical.to_csv(OUTPUT_DIR / "historical_rank_change.csv", index=False)
    rank_change_stochastic.to_csv(OUTPUT_DIR / "stochastic_rank_change.csv", index=False)
    write_summary(weights, grid_df, historical_df, revealed_stochastic, rank_change_historical, rank_change_stochastic)


if __name__ == "__main__":
    main()
