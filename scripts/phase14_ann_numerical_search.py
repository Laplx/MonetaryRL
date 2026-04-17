from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import phase10_counterfactual_eval as cfe
import phase10_revealed_policy_eval as rpe
from monetary_rl.phase10_utils import (
    build_ann_context,
    clone_context_with_loss_weights,
    historical_summary,
    make_empirical_env,
    simulate_historical_counterfactual,
)

OUTPUT_DIR = ROOT / "outputs" / "phase14"
FIGURES_DIR = OUTPUT_DIR / "figures"

SEARCH_SEEDS = [7, 43, 99]
SEARCH_HORIZON = 80
STOCHASTIC_HORIZON = 120
STOCHASTIC_EPISODES = 96


@dataclass
class AffineSearchConfig:
    iterations: int = 24
    population_size: int = 48
    elite_frac: float = 0.125
    episodes_per_candidate: int = 4
    eval_episodes: int = 96
    init_std: float = 1.0
    min_std: float = 0.05
    seed: int = 0


class AffinePolicySearch:
    def __init__(self, env: Any, config: AffineSearchConfig) -> None:
        self.env = env
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        sample_state = np.asarray(self.env.reset(seed=config.seed), dtype=float)
        self.state_dim = sample_state.shape[0] + 1
        self.mean = np.zeros(self.state_dim, dtype=float)
        self.std = np.full(self.state_dim, config.init_std, dtype=float)

    @staticmethod
    def action(theta: np.ndarray, state: np.ndarray) -> float:
        return float(theta[0] + theta[1:] @ np.asarray(state, dtype=float))

    def _evaluate_theta(self, theta: np.ndarray, episodes: int, seed: int) -> dict[str, float]:
        discounted_losses: list[float] = []
        rewards: list[float] = []
        clip_count = 0
        step_count = 0
        abs_action_sum = 0.0
        for ep in range(episodes):
            state = np.asarray(self.env.reset(seed=seed + ep), dtype=float)
            done = False
            discount = 1.0
            discounted_loss = 0.0
            total_reward = 0.0
            while not done:
                raw_action = self.action(theta, state)
                next_state, reward, done, info = self.env.step(raw_action)
                total_reward += reward
                discounted_loss += (-reward) * discount
                discount *= self.env.model.config.discount_factor
                clip_count += int(abs(float(info["raw_action"]) - float(info["action"])) > 1e-8)
                abs_action_sum += abs(float(info["action"]))
                step_count += 1
                state = np.asarray(next_state, dtype=float)
            discounted_losses.append(discounted_loss)
            rewards.append(total_reward)
        return {
            "mean_discounted_loss": float(np.mean(discounted_losses)),
            "std_discounted_loss": float(np.std(discounted_losses, ddof=1)) if len(discounted_losses) > 1 else 0.0,
            "mean_reward": float(np.mean(rewards)),
            "clip_rate": clip_count / step_count if step_count else 0.0,
            "mean_abs_action": abs_action_sum / step_count if step_count else 0.0,
        }

    def train(self) -> dict[str, Any]:
        elite_count = max(1, int(round(self.config.population_size * self.config.elite_frac)))
        history: list[dict[str, float]] = []
        best_theta = self.mean.copy()
        best_stats = self._evaluate_theta(best_theta, self.config.episodes_per_candidate, seed=100_000)
        for iteration in range(self.config.iterations):
            population = self.rng.normal(loc=self.mean, scale=self.std, size=(self.config.population_size, self.state_dim))
            scores = np.zeros(self.config.population_size, dtype=float)
            for idx, theta in enumerate(population):
                stats = self._evaluate_theta(theta, self.config.episodes_per_candidate, seed=iteration * 10_000 + idx * 100)
                scores[idx] = stats["mean_discounted_loss"]
                if stats["mean_discounted_loss"] < best_stats["mean_discounted_loss"]:
                    best_theta = theta.copy()
                    best_stats = stats
            elite_idx = np.argsort(scores)[:elite_count]
            elite = population[elite_idx]
            self.mean = elite.mean(axis=0)
            self.std = np.maximum(elite.std(axis=0), self.config.min_std)
            history.append(
                {
                    "iteration": iteration,
                    "best_population_loss": float(scores[elite_idx[0]]),
                    "mean_population_loss": float(scores.mean()),
                    "elite_mean_loss": float(scores[elite_idx].mean()),
                    "mean_norm": float(np.linalg.norm(self.mean)),
                    "std_norm": float(np.linalg.norm(self.std)),
                }
            )
        final_stats = self._evaluate_theta(best_theta, self.config.eval_episodes, seed=900_000)
        return {
            "config": asdict(self.config),
            "history": history,
            "best_theta": best_theta.tolist(),
            "best_stats": final_stats,
        }


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def policy_from_theta(theta: np.ndarray):
    theta = np.asarray(theta, dtype=float)

    def policy(state: np.ndarray, t: int) -> float:
        del t
        return float(theta[0] + theta[1:] @ np.asarray(state, dtype=float))

    return policy


def search_context(context: Any, label: str) -> tuple[dict[str, Any], pd.DataFrame]:
    summary_path = OUTPUT_DIR / "ann_search_seed_summary.csv"
    if summary_path.exists():
        cached_df = pd.read_csv(summary_path)
        cached_df = cached_df.loc[cached_df["search_label"] == label].copy()
        if len(cached_df) >= len(SEARCH_SEEDS):
            cached_df = cached_df.sort_values("best_mean_discounted_loss").reset_index(drop=True)
            best_row = cached_df.iloc[0]
            theta = np.array(
                [
                    float(best_row["intercept"]),
                    float(best_row["inflation_gap"]),
                    float(best_row["output_gap"]),
                    float(best_row["lagged_policy_rate_gap"]),
                ],
                dtype=float,
            )
            best_payload = {
                "label": label,
                "seed": int(best_row["seed"]),
                "theta": theta,
                "policy": policy_from_theta(theta),
                "result": {"best_stats": {"mean_discounted_loss": float(best_row["best_mean_discounted_loss"])}},
                "wall_seconds": float(best_row["wall_seconds"]),
                "total_env_steps": int(best_row["total_env_steps"]),
            }
            return best_payload, cached_df

    rows: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_loss = float("inf")
    for seed in SEARCH_SEEDS:
        cfg = AffineSearchConfig(seed=seed)
        env = make_empirical_env(context, horizon=SEARCH_HORIZON, seed=seed)
        search = AffinePolicySearch(env, cfg)
        started = time.perf_counter()
        result = search.train()
        wall_seconds = time.perf_counter() - started
        train_env_steps = cfg.iterations * cfg.population_size * cfg.episodes_per_candidate * SEARCH_HORIZON
        eval_env_steps = cfg.eval_episodes * SEARCH_HORIZON
        theta = np.asarray(result["best_theta"], dtype=float)
        row = {
            "search_label": label,
            "seed": seed,
            "wall_seconds": wall_seconds,
            "train_env_steps": train_env_steps,
            "eval_env_steps": eval_env_steps,
            "total_env_steps": train_env_steps + eval_env_steps,
            "best_mean_discounted_loss": float(result["best_stats"]["mean_discounted_loss"]),
            "best_std_discounted_loss": float(result["best_stats"]["std_discounted_loss"]),
            "best_mean_reward": float(result["best_stats"]["mean_reward"]),
            "best_clip_rate": float(result["best_stats"]["clip_rate"]),
            "intercept": float(theta[0]),
            "inflation_gap": float(theta[1]),
            "output_gap": float(theta[2]),
            "lagged_policy_rate_gap": float(theta[3]),
        }
        rows.append(row)
        if row["best_mean_discounted_loss"] < best_loss:
            best_loss = row["best_mean_discounted_loss"]
            best_payload = {
                "label": label,
                "seed": seed,
                "theta": theta,
                "policy": policy_from_theta(theta),
                "result": result,
                "wall_seconds": wall_seconds,
                "total_env_steps": train_env_steps + eval_env_steps,
            }
    assert best_payload is not None
    return best_payload, pd.DataFrame(rows).sort_values("best_mean_discounted_loss").reset_index(drop=True)


def path_moments(path_df: pd.DataFrame) -> pd.DataFrame:
    work = path_df.dropna(subset=["loss"]).copy()
    return (
        work.groupby("policy_name", as_index=False)
        .agg(
            var_inflation_gap=("inflation_gap", "var"),
            std_inflation_gap=("inflation_gap", "std"),
            var_output_gap=("output_gap", "var"),
            std_output_gap=("output_gap", "std"),
            var_policy_rate=("policy_rate", "var"),
            std_policy_rate=("policy_rate", "std"),
            var_rate_change=("rate_change", "var"),
            std_rate_change=("rate_change", "std"),
        )
    )


def stochastic_moments(context: Any, policy_map: dict[str, Any], include_historical: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    loss_rows: list[dict[str, Any]] = []
    moment_rows: list[dict[str, Any]] = []
    for idx, (policy_name, policy_fn) in enumerate(policy_map.items()):
        if (not include_historical) and policy_name == "historical_actual_policy":
            continue
        env = make_empirical_env(context, horizon=STOCHASTIC_HORIZON, seed=20260401 + idx)
        trajectories: list[dict[str, float]] = []
        discounted_losses: list[float] = []
        rewards: list[float] = []
        clip_count = 0
        explosion_count = 0
        step_count = 0
        for ep in range(STOCHASTIC_EPISODES):
            state = np.asarray(env.reset(seed=303000 + idx * 100 + ep), dtype=float)
            done = False
            discount = 1.0
            total_reward = 0.0
            discounted_loss = 0.0
            t = 0
            while not done:
                prev_rate = float(context.model.action_to_level(state[2]))
                raw_action = float(policy_fn(state.copy(), t))
                next_state, reward, done, info = env.step(raw_action)
                current_rate = float(info["policy_rate"])
                trajectories.append(
                    {
                        "policy_name": policy_name,
                        "episode": ep,
                        "period": t,
                        "inflation_gap": float(state[0]),
                        "output_gap": float(state[1]),
                        "policy_rate": current_rate,
                        "rate_change": current_rate - prev_rate,
                        "loss": float(info["loss"]),
                    }
                )
                total_reward += reward
                discounted_loss += (-reward) * discount
                discount *= context.model.config.discount_factor
                clip_count += int(abs(float(info["raw_action"]) - float(info["action"])) > 1e-8)
                explosion_count += int(bool(info.get("exploded", False)))
                step_count += 1
                state = np.asarray(next_state, dtype=float)
                t += 1
            rewards.append(total_reward)
            discounted_losses.append(discounted_loss)
        traj_df = pd.DataFrame(trajectories)
        loss_rows.append(
            {
                "policy_name": policy_name,
                "mean_discounted_loss": float(np.mean(discounted_losses)),
                "std_discounted_loss": float(np.std(discounted_losses, ddof=1)) if len(discounted_losses) > 1 else 0.0,
                "mean_reward": float(np.mean(rewards)),
                "mean_abs_action": float(traj_df["policy_rate"].sub(context.model.config.neutral_rate).abs().mean()),
                "clip_rate": clip_count / step_count if step_count else 0.0,
                "explosion_rate": explosion_count / STOCHASTIC_EPISODES,
            }
        )
        moment_rows.append(
            {
                "policy_name": policy_name,
                "var_inflation_gap": float(traj_df["inflation_gap"].var()),
                "std_inflation_gap": float(traj_df["inflation_gap"].std()),
                "var_output_gap": float(traj_df["output_gap"].var()),
                "std_output_gap": float(traj_df["output_gap"].std()),
                "var_policy_rate": float(traj_df["policy_rate"].var()),
                "std_policy_rate": float(traj_df["policy_rate"].std()),
                "var_rate_change": float(traj_df["rate_change"].var()),
                "std_rate_change": float(traj_df["rate_change"].std()),
            }
        )
    return pd.DataFrame(loss_rows), pd.DataFrame(moment_rows)


def policy_metadata(unified_registry: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "policy_name",
        "rule_family",
        "source_env",
        "training_env",
        "policy_parameterization",
        "algo",
    ]
    meta = unified_registry[keep_cols].copy()
    manual = pd.DataFrame(
        [
            {
                "policy_name": "ann_affine_search_artificial",
                "rule_family": "ann_native_numerical_search",
                "source_env": "ann",
                "training_env": "ann",
                "policy_parameterization": "affine_linear_search",
                "algo": "numerical_search",
            },
            {
                "policy_name": "ann_affine_search_revealed",
                "rule_family": "ann_native_numerical_search",
                "source_env": "ann",
                "training_env": "ann_revealed",
                "policy_parameterization": "affine_linear_search",
                "algo": "numerical_search",
            },
        ]
    )
    return pd.concat([meta, manual], ignore_index=True).drop_duplicates("policy_name", keep="last")


def summarize_context(
    context: Any,
    policy_map: dict[str, Any],
    loss_function: str,
    meta_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    historical_paths = simulate_historical_counterfactual(
        model=context.model,
        policy_map=policy_map,
        initial_state=context.initial_states[0],
        action_dates=context.action_dates,
        state_dates=context.state_dates,
        shocks=context.shock_pool[:-1],
    )
    historical_summary_df = historical_summary(historical_paths)
    historical_moments_df = path_moments(historical_paths)
    historical_df = historical_summary_df.merge(historical_moments_df, on="policy_name", how="left")
    historical_df = historical_df.merge(meta_df, on="policy_name", how="left")
    historical_df.insert(0, "loss_function", loss_function)
    historical_df.insert(1, "evaluation_type", "historical_counterfactual")
    historical_df.insert(2, "evaluation_env", "ann")

    stochastic_summary_df, stochastic_moments_df = stochastic_moments(context, policy_map)
    stochastic_df = stochastic_summary_df.merge(stochastic_moments_df, on="policy_name", how="left")
    stochastic_df = stochastic_df.merge(meta_df, on="policy_name", how="left")
    stochastic_df.insert(0, "loss_function", loss_function)
    stochastic_df.insert(1, "evaluation_type", "stochastic_long_run")
    stochastic_df.insert(2, "evaluation_env", "ann")

    historical_paths.insert(0, "loss_function", loss_function)
    historical_paths.insert(1, "evaluation_env", "ann")
    return historical_paths, historical_df, stochastic_df


def compute_budget_rows(best_art: dict[str, Any], best_rev: dict[str, Any], unified_registry: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "policy_name": "ann_affine_search_artificial",
            "training_context": "ann_artificial",
            "method_family": "numerical_search",
            "training_env_steps": best_art["total_env_steps"],
            "wall_seconds": best_art["wall_seconds"],
        },
        {
            "policy_name": "ann_affine_search_revealed",
            "training_context": "ann_revealed",
            "method_family": "numerical_search",
            "training_env_steps": best_rev["total_env_steps"],
            "wall_seconds": best_rev["wall_seconds"],
        },
    ]
    for policy_name in [
        "ppo_ann_direct",
        "td3_ann_direct",
        "sac_ann_direct",
        "ppo_ann_direct_nonlinear",
        "ppo_ann_revealed_direct",
        "td3_ann_revealed_direct",
        "sac_ann_revealed_direct",
    ]:
        row = unified_registry.loc[unified_registry["policy_name"] == policy_name].iloc[0]
        cfg = json.loads((ROOT / row["config_path"]).read_text(encoding="utf-8"))
        if row["algo"] == "ppo":
            steps = int(cfg["total_updates"]) * int(cfg["rollout_steps"])
        else:
            steps = int(cfg["total_steps"])
        rows.append(
            {
                "policy_name": policy_name,
                "training_context": row.get("training_env", ""),
                "method_family": str(row["algo"]).upper(),
                "training_env_steps": steps,
                "wall_seconds": math.nan,
            }
        )
    return pd.DataFrame(rows)


def standardize_legacy_matrix() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    mappings = [
        ("phase9_legacy", "artificial", "historical_counterfactual", ROOT / "outputs/phase9/ann_historical_welfare_summary.csv"),
        ("phase9_legacy", "artificial", "stochastic_long_run", ROOT / "outputs/phase9/ann_stochastic_welfare_summary.csv"),
        ("phase10", "artificial", "historical_counterfactual", ROOT / "outputs/phase10/counterfactual_eval/ann_historical_summary.csv"),
        ("phase10", "artificial", "stochastic_long_run", ROOT / "outputs/phase10/counterfactual_eval/ann_stochastic_summary.csv"),
        ("phase10", "revealed", "historical_counterfactual", ROOT / "outputs/phase10/revealed_policy_eval/ann_historical_summary.csv"),
        ("phase10", "revealed", "stochastic_long_run", ROOT / "outputs/phase10/revealed_policy_eval/ann_stochastic_summary.csv"),
    ]
    for source_phase, loss_function, evaluation_type, path in mappings:
        df = pd.read_csv(path)
        df.insert(0, "source_phase", source_phase)
        df.insert(1, "loss_function", loss_function)
        df.insert(2, "evaluation_type", evaluation_type)
        df["evaluation_env"] = "ann"
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def highlight_tables(full_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    art_hist = full_matrix[
        (full_matrix["loss_function"] == "artificial") & (full_matrix["evaluation_type"] == "historical_counterfactual")
    ].copy()
    art_stoch = full_matrix[
        (full_matrix["loss_function"] == "artificial") & (full_matrix["evaluation_type"] == "stochastic_long_run")
    ].copy()
    rev_hist = full_matrix[
        (full_matrix["loss_function"] == "revealed") & (full_matrix["evaluation_type"] == "historical_counterfactual")
    ].copy()
    rev_stoch = full_matrix[
        (full_matrix["loss_function"] == "revealed") & (full_matrix["evaluation_type"] == "stochastic_long_run")
    ].copy()
    highlight = pd.concat(
        [
            art_hist.sort_values("total_discounted_loss").head(8),
            art_stoch.sort_values("mean_discounted_loss").head(8),
            rev_hist.sort_values("total_discounted_loss").head(8),
            rev_stoch.sort_values("mean_discounted_loss").head(8),
        ],
        ignore_index=True,
    )
    search_compare = pd.concat(
        [
            art_hist[art_hist["policy_name"].isin(["ann_affine_search_artificial", "ppo_ann_direct", "td3_ann_direct", "sac_ann_direct", "empirical_taylor_rule", "riccati_reference", "linear_policy_search_transfer"])],
            art_stoch[art_stoch["policy_name"].isin(["ann_affine_search_artificial", "ppo_ann_direct", "td3_ann_direct", "sac_ann_direct", "empirical_taylor_rule", "riccati_reference", "linear_policy_search_transfer"])],
            rev_hist[rev_hist["policy_name"].isin(["ann_affine_search_revealed", "td3_ann_revealed_direct", "sac_ann_revealed_direct", "empirical_taylor_rule"])],
            rev_stoch[rev_stoch["policy_name"].isin(["ann_affine_search_revealed", "td3_ann_revealed_direct", "sac_ann_revealed_direct", "empirical_taylor_rule"])],
        ],
        ignore_index=True,
    )
    return highlight, search_compare


def make_figures(compare_df: pd.DataFrame, budget_df: pd.DataFrame) -> None:
    plt.style.use("default")
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    mapping = [
        ("artificial", "historical_counterfactual", "total_discounted_loss", "Artificial / Historical"),
        ("artificial", "stochastic_long_run", "mean_discounted_loss", "Artificial / Stochastic"),
        ("revealed", "historical_counterfactual", "total_discounted_loss", "Revealed / Historical"),
        ("revealed", "stochastic_long_run", "mean_discounted_loss", "Revealed / Stochastic"),
    ]
    for ax, (loss_function, evaluation_type, metric, title) in zip(axes.flatten(), mapping, strict=True):
        sub = compare_df[
            (compare_df["loss_function"] == loss_function) & (compare_df["evaluation_type"] == evaluation_type)
        ].copy()
        sub = sub.sort_values(metric)
        ax.bar(sub["policy_name"], sub[metric], color="#4e79a7")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ann_numerical_search_vs_rl_matrix.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    plot_df = budget_df.dropna(subset=["training_env_steps"]).copy()
    ax.bar(plot_df["policy_name"], plot_df["training_env_steps"], color="#59a14f")
    ax.set_yscale("log")
    ax.set_ylabel("Training interaction budget (log scale)")
    ax.set_title("ANN environment: numerical search vs RL training budgets")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ann_numerical_search_compute_cost.png", dpi=220)
    plt.close(fig)


def write_summary(best_art: dict[str, Any], best_rev: dict[str, Any], search_compare: pd.DataFrame, budget_df: pd.DataFrame) -> None:
    art_hist = search_compare[
        (search_compare["loss_function"] == "artificial") & (search_compare["evaluation_type"] == "historical_counterfactual")
    ].sort_values("total_discounted_loss")
    art_stoch = search_compare[
        (search_compare["loss_function"] == "artificial") & (search_compare["evaluation_type"] == "stochastic_long_run")
    ].sort_values("mean_discounted_loss")
    rev_hist = search_compare[
        (search_compare["loss_function"] == "revealed") & (search_compare["evaluation_type"] == "historical_counterfactual")
    ].sort_values("total_discounted_loss")
    rev_stoch = search_compare[
        (search_compare["loss_function"] == "revealed") & (search_compare["evaluation_type"] == "stochastic_long_run")
    ].sort_values("mean_discounted_loss")
    content = "\n".join(
        [
            "# Phase 14 ANN-native 数值搜索比较",
            "",
            "## 新增数值搜索政策",
            "",
            f"- `ann_affine_search_artificial`：best seed = `{best_art['seed']}`，wall time = `{best_art['wall_seconds']:.2f}s`，train env steps = `{best_art['total_env_steps']}`",
            f"- `ann_affine_search_revealed`：best seed = `{best_rev['seed']}`，wall time = `{best_rev['wall_seconds']:.2f}s`，train env steps = `{best_rev['total_env_steps']}`",
            "",
            "## 关键结果",
            "",
            "### Artificial loss / Historical",
            "",
            art_hist.head(6)[["policy_name", "total_discounted_loss", "std_inflation_gap", "std_output_gap", "std_rate_change"]].to_markdown(index=False),
            "",
            "### Artificial loss / Stochastic",
            "",
            art_stoch.head(6)[["policy_name", "mean_discounted_loss", "std_inflation_gap", "std_output_gap", "std_rate_change"]].to_markdown(index=False),
            "",
            "### Revealed loss / Historical",
            "",
            rev_hist.head(6)[["policy_name", "total_discounted_loss", "std_inflation_gap", "std_output_gap", "std_rate_change"]].to_markdown(index=False),
            "",
            "### Revealed loss / Stochastic",
            "",
            rev_stoch.head(6)[["policy_name", "mean_discounted_loss", "std_inflation_gap", "std_output_gap", "std_rate_change"]].to_markdown(index=False),
            "",
            "## 计算开销对比",
            "",
            budget_df.to_markdown(index=False),
            "",
            "## 文件",
            "",
            "- `ann_full_matrix.csv`：phase14 最终 ANN 全矩阵",
            "- `ann_master_matrix.csv`：整合 phase9 / phase10 / phase14 的 ANN 写作用主表",
            "- `ann_search_seed_summary.csv`：数值搜索三 seed 汇总",
            "- `figures/ann_numerical_search_vs_rl_matrix.png`：ANN 数值搜索与 RL 主比较图",
            "- `figures/ann_numerical_search_compute_cost.png`：计算开销对比图",
        ]
    )
    (OUTPUT_DIR / "phase14_summary.md").write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    unified_registry = cfe.load_unified_registry()
    ann_context = build_ann_context(ROOT)
    weights = rpe.revealed_loss_weights()
    ann_revealed_context = clone_context_with_loss_weights(build_ann_context(ROOT), weights, "revealed")

    best_art, art_seed_df = search_context(ann_context, "ann_affine_search_artificial")
    best_rev, rev_seed_df = search_context(ann_revealed_context, "ann_affine_search_revealed")
    search_seed_df = pd.concat([art_seed_df, rev_seed_df], ignore_index=True)
    search_seed_df.to_csv(OUTPUT_DIR / "ann_search_seed_summary.csv", index=False)

    meta_df = policy_metadata(unified_registry)

    artificial_policy_map = cfe.build_policy_map(ann_context, unified_registry)
    artificial_policy_map["ann_affine_search_artificial"] = best_art["policy"]
    artificial_policy_map["ann_affine_search_revealed"] = best_rev["policy"]
    historical_paths_art, historical_art, stochastic_art = summarize_context(
        ann_context,
        artificial_policy_map,
        loss_function="artificial",
        meta_df=meta_df,
    )

    revealed_policy_map = cfe.build_policy_map(ann_revealed_context, unified_registry)
    revealed_policy_map["ann_affine_search_artificial"] = best_art["policy"]
    revealed_policy_map["ann_affine_search_revealed"] = best_rev["policy"]
    historical_paths_rev, historical_rev, stochastic_rev = summarize_context(
        ann_revealed_context,
        revealed_policy_map,
        loss_function="revealed",
        meta_df=meta_df,
    )

    historical_paths = pd.concat([historical_paths_art, historical_paths_rev], ignore_index=True)
    historical_paths.to_csv(OUTPUT_DIR / "ann_historical_paths_phase14.csv", index=False)

    full_matrix = pd.concat([historical_art, stochastic_art, historical_rev, stochastic_rev], ignore_index=True)
    full_matrix.to_csv(OUTPUT_DIR / "ann_full_matrix.csv", index=False)

    legacy_df = standardize_legacy_matrix()
    master_matrix = pd.concat([legacy_df, full_matrix.assign(source_phase="phase14_final")], ignore_index=True, sort=False)
    master_matrix.to_csv(OUTPUT_DIR / "ann_master_matrix.csv", index=False)

    budget_df = compute_budget_rows(best_art, best_rev, unified_registry)
    budget_df.to_csv(OUTPUT_DIR / "ann_compute_budget_comparison.csv", index=False)

    highlight_df, search_compare = highlight_tables(full_matrix)
    highlight_df.to_csv(OUTPUT_DIR / "ann_highlight_table.csv", index=False)
    search_compare.to_csv(OUTPUT_DIR / "ann_search_vs_rl_comparison.csv", index=False)

    make_figures(search_compare, budget_df)
    write_summary(best_art, best_rev, search_compare, budget_df)


if __name__ == "__main__":
    main()
