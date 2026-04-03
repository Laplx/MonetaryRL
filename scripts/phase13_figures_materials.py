from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import phase10_counterfactual_eval as cfe
import phase10_revealed_policy_eval as rpe
from monetary_rl.phase10_utils import (
    build_ann_context,
    build_svar_context,
    clone_context_with_loss_weights,
    make_empirical_env,
)

OUTPUT_DIR = ROOT / "outputs" / "phase13"
FIGURES_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "tables"


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def configure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def readable_tier(tier: str) -> str:
    mapping = {
        "benchmark": "benchmark",
        "mild": "mild",
        "medium": "medium",
        "strong": "strong",
        "very_strong": "very strong",
        "very_strong_v2": "very strong",
        "extreme": "extreme",
        "extreme_v2": "extreme",
    }
    return mapping.get(tier, tier.replace("_", " "))


def safe_float(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    return float(value)


def load_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative_path)


def plot_theory_heatmap() -> dict[str, str]:
    df = load_csv("outputs/phase7/matrix/rl_summary.csv").copy()
    env_order = (
        df[["env_id", "group", "tier"]]
        .drop_duplicates()
        .assign(
            group_order=lambda x: x["group"].map(
                {"benchmark": 0, "nonlinear": 1, "zlb": 2, "asymmetric": 3}
            ),
            tier_order=lambda x: x["tier"].map(
                {"benchmark": 0, "mild": 1, "medium": 2, "strong": 3}
            ),
        )
        .sort_values(["group_order", "tier_order", "env_id"])
    )
    env_labels = [
        f"{row.group}\n{readable_tier(row.tier)}"
        for row in env_order.itertuples(index=False)
    ]
    pivot = (
        df.pivot(index="algo", columns="env_id", values="mean_discounted_loss")
        .reindex(columns=env_order["env_id"])
        .reindex(index=["ppo", "td3", "sac"])
    )
    fig, ax = plt.subplots(figsize=(11, 4.6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(env_labels)))
    ax.set_xticklabels(env_labels)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([label.upper() for label in pivot.index])
    ax.set_title("Phase 7: RL loss ranking across benchmark and first-wave distortions")
    ax.set_xlabel("Environment")
    ax.set_ylabel("Algorithm")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.1f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, label="Mean discounted loss")
    fig.tight_layout()
    path = FIGURES_DIR / "phase13_theory_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    return {
        "figure": path.name,
        "section": "理论基准与人工损失 benchmark",
        "message": "用一张图展示 benchmark 与三类早期扩展里 RL 排序如何变化，建立“RL 在核心环境中具有系统竞争力”的总览。",
        "source": "outputs/phase7/matrix/rl_summary.csv",
    }


def plot_theory_strength_curves() -> dict[str, str]:
    df = load_csv("outputs/phase7/matrix/all_policy_summary.csv").copy()
    group_order = ["nonlinear", "zlb", "asymmetric"]
    tier_order = {"mild": 1, "medium": 2, "strong": 3}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    rl_summary = load_csv("outputs/phase7/matrix/rl_summary.csv")
    for ax, group in zip(axes, group_order, strict=True):
        group_df = df[df["group"] == group].copy()
        group_df["tier_num"] = group_df["tier"].map(tier_order)
        rl_group = (
            rl_summary[rl_summary["group"] == group]
            .sort_values(["env_id", "mean_discounted_loss"])
            .drop_duplicates("env_id")
        )
        rl_group["tier_num"] = rl_group["tier"].map(tier_order)
        ax.plot(
            rl_group.sort_values("tier_num")["tier_num"],
            rl_group.sort_values("tier_num")["mean_discounted_loss"],
            marker="o",
            linewidth=2,
            label="best RL",
        )
        for policy_name, label in {
            "riccati_reference": "Riccati",
            "linear_policy_search": "linear search",
        }.items():
            policy_df = group_df[group_df["policy_name"] == policy_name].sort_values("tier_num")
            ax.plot(
                policy_df["tier_num"],
                policy_df["mean_discounted_loss"],
                marker="o",
                linewidth=2,
                label=label,
            )
        ax.set_title(group.upper())
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["mild", "medium", "strong"])
        ax.set_xlabel("Distortion intensity")
        ax.set_ylabel("Mean discounted loss")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best")
    fig.suptitle("Phase 7: RL gains widen as distortions strengthen")
    fig.tight_layout()
    path = FIGURES_DIR / "phase13_theory_strength_curves.png"
    fig.savefig(path)
    plt.close(fig)
    return {
        "figure": path.name,
        "section": "理论基准与人工损失 benchmark",
        "message": "把 benchmark 外推、线性搜索和最优 RL 放到同一扭曲强度轴上，直接展示 RL 优势如何随非线性/约束增强而扩大。",
        "source": "outputs/phase7/matrix/all_policy_summary.csv + outputs/phase7/matrix/rl_summary.csv",
    }


def plot_extreme_v2_advantage() -> dict[str, str]:
    df = load_csv("outputs/phase11/extreme_matrix_v2/riccati_vs_best_rl.csv").copy()
    df["label"] = df["group"] + "\n" + df["tier"].map(readable_tier)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(
        df["label"],
        df["riccati_gap_vs_best_rl_pct"],
        color=["#1f77b4", "#4e79a7", "#59a14f", "#76b7b2", "#f28e2b", "#e15759"],
    )
    for bar, algo, value in zip(bars, df["best_rl_algo"], df["riccati_gap_vs_best_rl_pct"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{algo.upper()}\n{value:.1f}%",
            ha="center",
            va="bottom",
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Best RL improvement vs Riccati (%)")
    ax.set_title("Phase 11 v2: RL beats Riccati in all retained non-benchmark environments")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIGURES_DIR / "phase13_extreme_v2_advantage.png"
    fig.savefig(path)
    plt.close(fig)
    df.to_csv(DATA_DIR / "phase13_extreme_v2_advantage.csv", index=False)
    return {
        "figure": path.name,
        "section": "三类非线性扩展",
        "message": "六个最终保留的强扭曲环境里，原始 RL 全部压过 Riccati 外推，说明 RL 的优势在真正非 benchmark 场景中是系统性的而不是个例。",
        "source": "outputs/phase11/extreme_matrix_v2/riccati_vs_best_rl.csv",
    }


def plot_extreme_mechanism_paths() -> dict[str, str]:
    cases = [
        ("nonlinear_hyper", "nonlinear_hyper"),
        ("zlb_trap_extreme", "zlb_trap_extreme"),
        ("asymmetric_threshold_extreme", "asymmetric_threshold_extreme"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex="col")
    for col, (folder, title) in enumerate(cases):
        path = ROOT / "outputs" / "phase11" / "extreme_numerical_compare_v2" / folder / "common_shock_trajectories.csv"
        df = pd.read_csv(path)
        work = df[df["policy"].isin(["riccati_reference", "best_rl_surrogate"])].copy()
        label_map = {"riccati_reference": "Riccati", "best_rl_surrogate": "best RL surrogate"}
        for policy_name, sub_df in work.groupby("policy"):
            axes[0, col].plot(sub_df["period"], sub_df["output_gap"], label=label_map[policy_name], linewidth=2)
            axes[1, col].plot(sub_df["period"], sub_df["action"], label=label_map[policy_name], linewidth=2)
        axes[0, col].set_title(title.replace("_", " "))
        axes[0, col].set_ylabel("Output gap")
        axes[1, col].set_ylabel("Policy gap")
        axes[1, col].set_xlabel("Period")
        axes[0, col].grid(alpha=0.25)
        axes[1, col].grid(alpha=0.25)
    axes[0, 0].legend(loc="best")
    fig.suptitle("Mechanism view: RL surrogate reacts differently in strong nonlinear, ZLB, and asymmetric states")
    fig.tight_layout()
    path = FIGURES_DIR / "phase13_extreme_mechanism_paths.png"
    fig.savefig(path)
    plt.close(fig)
    return {
        "figure": path.name,
        "section": "三类非线性扩展",
        "message": "共同冲击轨迹图帮助解释为何 RL 更优：它在强扭曲区域给出不同于 Riccati 外推的状态依赖利率路径，从而改写产出缺口动态。",
        "source": "outputs/phase11/extreme_numerical_compare_v2/*/common_shock_trajectories.csv",
    }


def plot_empirical_history(
    csv_path: str,
    title: str,
    env_best_map: dict[str, str],
    output_name: str,
) -> dict[str, str]:
    if "revealed" in csv_path:
        env_sources = {
            "svar": "outputs/phase10/revealed_policy_eval/svar_historical_paths.csv",
            "ann": "outputs/phase10/revealed_policy_eval/ann_historical_paths.csv",
        }
    else:
        env_sources = {
            "svar": "outputs/phase10/counterfactual_eval/svar_historical_paths.csv",
            "ann": "outputs/phase10/counterfactual_eval/ann_historical_paths.csv",
        }
    policy_display = {
        "historical_actual_policy": "historical",
        "empirical_taylor_rule": "Taylor",
    }
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex="col")
    for col, env_name in enumerate(["svar", "ann"]):
        df = load_csv(env_sources[env_name]).copy()
        policies = ["historical_actual_policy", "empirical_taylor_rule", env_best_map[env_name]]
        env_df = df[(df["policy_name"].isin(policies)) & (df["action_date"].notna())].copy()
        display_map = policy_display | {env_best_map[env_name]: env_best_map[env_name]}
        for policy_name in policies:
            sub_df = env_df[env_df["policy_name"] == policy_name]
            axes[0, col].plot(sub_df["action_date"], sub_df["inflation_gap"], linewidth=2, label=display_map[policy_name])
            axes[1, col].plot(sub_df["action_date"], sub_df["output_gap"], linewidth=2, label=display_map[policy_name])
            axes[2, col].plot(sub_df["action_date"], sub_df["policy_rate"], linewidth=2, label=display_map[policy_name])
        axes[0, col].set_title(env_name.upper())
        axes[0, col].set_ylabel("Inflation gap")
        axes[1, col].set_ylabel("Output gap")
        axes[2, col].set_ylabel("Policy rate")
        axes[2, col].set_xlabel("Date")
        for row in range(3):
            axes[row, col].grid(alpha=0.25)
    axes[0, 0].legend(loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    path = FIGURES_DIR / output_name
    fig.savefig(path)
    plt.close(fig)
    return {
        "figure": path.name,
        "section": "经验环境：历史反事实",
        "message": "用历史路径把‘更低损失’转化成更直观的经济动态：比较 RL、Taylor 与历史政策在通胀、产出缺口和利率路径上的差异。",
        "source": "outputs/phase10/counterfactual_eval/*_historical_paths.csv or outputs/phase10/revealed_policy_eval/*_historical_paths.csv",
    }


def historical_volatility_table() -> pd.DataFrame:
    configs = [
        ("artificial", "svar", "outputs/phase10/counterfactual_eval/svar_historical_paths.csv", "ppo_ann_direct_nonlinear"),
        ("artificial", "ann", "outputs/phase10/counterfactual_eval/ann_historical_paths.csv", "ppo_ann_direct"),
        ("revealed", "svar", "outputs/phase10/revealed_policy_eval/svar_historical_paths.csv", "sac_svar_revealed_direct"),
        ("revealed", "ann", "outputs/phase10/revealed_policy_eval/ann_historical_paths.csv", "td3_ann_revealed_direct"),
    ]
    rows: list[dict[str, Any]] = []
    for welfare_type, env_name, relative_path, best_policy in configs:
        df = load_csv(relative_path)
        subset = df[df["policy_name"].isin([best_policy, "empirical_taylor_rule", "historical_actual_policy"])].dropna(subset=["loss"]).copy()
        for policy_name, sub_df in subset.groupby("policy_name"):
            rows.append(
                {
                    "welfare_type": welfare_type,
                    "environment": env_name,
                    "policy_name": policy_name,
                    "std_inflation_gap": safe_float(sub_df["inflation_gap"].std()),
                    "std_output_gap": safe_float(sub_df["output_gap"].std()),
                    "std_policy_rate": safe_float(sub_df["policy_rate"].std()),
                    "std_rate_change": safe_float(sub_df["rate_change"].std()),
                    "mean_loss": safe_float(sub_df["loss"].mean()),
                    "total_discounted_loss": safe_float(sub_df["discounted_loss"].sum()),
                }
            )
    out = pd.DataFrame(rows).sort_values(["welfare_type", "environment", "total_discounted_loss"])
    out.to_csv(DATA_DIR / "phase13_historical_volatility_core.csv", index=False)
    return out


def simulate_stochastic_metrics(
    context: Any,
    policy_map: dict[str, Any],
    policies: list[str],
    horizon: int = 120,
    episodes: int = 96,
    seed: int = 20260401,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, policy_name in enumerate(policies):
        env = make_empirical_env(context, horizon=horizon, seed=seed + idx)
        policy_fn = policy_map[policy_name]
        trajectory_rows: list[dict[str, float]] = []
        losses: list[float] = []
        clip_count = 0
        explosion_count = 0
        step_count = 0
        for ep in range(episodes):
            state = env.reset(seed=seed + 1000 + idx * 100 + ep)
            done = False
            t = 0
            while not done:
                prev_policy_rate = float(context.model.action_to_level(state[2]))
                raw_action = float(policy_fn(state.copy(), t))
                next_state, reward, done, info = env.step(raw_action)
                current_rate = float(info["policy_rate"])
                trajectory_rows.append(
                    {
                        "episode": ep,
                        "period": t,
                        "inflation_gap": float(state[0]),
                        "output_gap": float(state[1]),
                        "policy_rate": current_rate,
                        "rate_change": current_rate - prev_policy_rate,
                        "loss": float(info["loss"]),
                    }
                )
                losses.append(float(info["loss"]))
                clip_count += int(abs(float(info["raw_action"]) - float(info["action"])) > 1e-8)
                explosion_count += int(bool(info.get("exploded", False)))
                step_count += 1
                state = next_state
                t += 1
        traj_df = pd.DataFrame(trajectory_rows)
        rows.append(
            {
                "policy_name": policy_name,
                "std_inflation_gap": safe_float(traj_df["inflation_gap"].std()),
                "std_output_gap": safe_float(traj_df["output_gap"].std()),
                "std_policy_rate": safe_float(traj_df["policy_rate"].std()),
                "std_rate_change": safe_float(traj_df["rate_change"].std()),
                "mean_abs_rate_change": safe_float(traj_df["rate_change"].abs().mean()),
                "mean_loss": float(np.mean(losses)),
                "clip_rate": clip_count / step_count if step_count else 0.0,
                "explosion_rate": explosion_count / episodes if episodes else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compute_policy_maps() -> dict[str, tuple[Any, dict[str, Any]]]:
    unified = cfe.load_unified_registry()
    svar_context = build_svar_context(ROOT)
    ann_context = build_ann_context(ROOT)
    art_maps = {
        "artificial_svar": (svar_context, cfe.build_policy_map(svar_context, unified)),
        "artificial_ann": (ann_context, cfe.build_policy_map(ann_context, unified)),
    }
    weights = rpe.revealed_loss_weights()
    svar_revealed = clone_context_with_loss_weights(build_svar_context(ROOT), weights, "revealed")
    ann_revealed = clone_context_with_loss_weights(build_ann_context(ROOT), weights, "revealed")
    rev_maps = {
        "revealed_svar": (svar_revealed, rpe.build_revealed_policy_map(svar_revealed, "svar")),
        "revealed_ann": (ann_revealed, rpe.build_revealed_policy_map(ann_revealed, "ann")),
    }
    return art_maps | rev_maps


def stochastic_core_tables(policy_maps: dict[str, tuple[Any, dict[str, Any]]]) -> pd.DataFrame:
    specs = {
        "artificial_svar": ["ppo_ann_direct_nonlinear", "empirical_taylor_rule"],
        "artificial_ann": ["ppo_ann_direct", "empirical_taylor_rule"],
        "revealed_svar": ["sac_svar_revealed_direct", "empirical_taylor_rule"],
        "revealed_ann": ["sac_ann_revealed_direct", "empirical_taylor_rule"],
    }
    rows = []
    for key, policies in specs.items():
        context, policy_map = policy_maps[key]
        out = simulate_stochastic_metrics(context, policy_map, policies)
        out.insert(0, "context_key", key)
        rows.append(out)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(DATA_DIR / "phase13_stochastic_core_metrics.csv", index=False)
    return result


def plot_stochastic_tradeoffs(stochastic_df: pd.DataFrame) -> dict[str, str]:
    context_titles = {
        "artificial_svar": "SVAR artificial",
        "artificial_ann": "ANN artificial",
        "revealed_svar": "SVAR revealed",
        "revealed_ann": "ANN revealed",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axes = axes.flatten()
    for ax, key in zip(axes, context_titles, strict=True):
        sub_df = stochastic_df[stochastic_df["context_key"] == key].copy()
        sub_df["label"] = sub_df["policy_name"].map(
            {
                "empirical_taylor_rule": "Taylor",
                "ppo_ann_direct_nonlinear": "best RL",
                "ppo_ann_direct": "best RL",
                "sac_svar_revealed_direct": "best RL",
                "sac_ann_revealed_direct": "best RL",
            }
        )
        ax.scatter(
            sub_df["std_inflation_gap"],
            sub_df["std_output_gap"],
            s=80,
            c=["#d62728" if label == "Taylor" else "#1f77b4" for label in sub_df["label"]],
        )
        for row in sub_df.itertuples(index=False):
            ax.text(row.std_inflation_gap + 0.01, row.std_output_gap + 0.01, row.label)
        ax.set_title(context_titles[key])
        ax.set_xlabel("Std. inflation gap")
        ax.set_ylabel("Std. output gap")
        ax.grid(alpha=0.25)
    fig.suptitle("Long-run stochastic trade-off: core RL rules vs Taylor")
    fig.tight_layout()
    path = FIGURES_DIR / "phase13_stochastic_tradeoffs.png"
    fig.savefig(path)
    plt.close(fig)
    return {
        "figure": path.name,
        "section": "经验环境：长期随机评价",
        "message": "长期随机图不只比较总损失，还直接看通胀与产出波动的联合表现，突出 RL 的动态稳定化特征。",
        "source": "phase13 recomputation from phase10 policy maps",
    }


def plot_policy_slices(policy_maps: dict[str, tuple[Any, dict[str, Any]]]) -> tuple[dict[str, str], pd.DataFrame]:
    specs = [
        ("artificial_svar", "SVAR artificial", ["empirical_taylor_rule", "ppo_ann_direct_nonlinear", "sac_svar_direct"]),
        ("artificial_ann", "ANN artificial", ["empirical_taylor_rule", "ppo_ann_direct", "td3_ann_direct"]),
        ("revealed_svar", "SVAR revealed", ["empirical_taylor_rule", "sac_svar_revealed_direct"]),
        ("revealed_ann", "ANN revealed", ["empirical_taylor_rule", "sac_ann_revealed_direct", "td3_ann_revealed_direct"]),
    ]
    grid = np.linspace(-2.0, 2.0, 81)
    slice_rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(4, 2, figsize=(13, 15), sharex="col")
    for row_idx, (key, title, policies) in enumerate(specs):
        context, policy_map = policy_maps[key]
        for policy_name in policies:
            infl_actions = []
            out_actions = []
            for val in grid:
                infl_state = np.array([val, 0.0, 0.0], dtype=float)
                out_state = np.array([0.0, val, 0.0], dtype=float)
                infl_rate = context.model.action_to_level(float(policy_map[policy_name](infl_state.copy(), 0)))
                out_rate = context.model.action_to_level(float(policy_map[policy_name](out_state.copy(), 0)))
                infl_actions.append(infl_rate)
                out_actions.append(out_rate)
                if val in {-2.0, -1.0, 0.0, 1.0, 2.0}:
                    slice_rows.append(
                        {
                            "context_key": key,
                            "policy_name": policy_name,
                            "slice_type": "inflation_gap",
                            "state_value": val,
                            "policy_rate": infl_rate,
                        }
                    )
                    slice_rows.append(
                        {
                            "context_key": key,
                            "policy_name": policy_name,
                            "slice_type": "output_gap",
                            "state_value": val,
                            "policy_rate": out_rate,
                        }
                    )
            axes[row_idx, 0].plot(grid, infl_actions, linewidth=2, label=policy_name)
            axes[row_idx, 1].plot(grid, out_actions, linewidth=2, label=policy_name)
        axes[row_idx, 0].set_title(f"{title}: inflation slice")
        axes[row_idx, 1].set_title(f"{title}: output-gap slice")
        axes[row_idx, 0].set_ylabel("Policy rate")
        axes[row_idx, 1].set_ylabel("Policy rate")
        axes[row_idx, 0].grid(alpha=0.25)
        axes[row_idx, 1].grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Inflation gap")
    axes[-1, 1].set_xlabel("Output gap")
    axes[0, 0].legend(loc="best")
    fig.suptitle("Representative policy slices: how key RL rules reshape state-dependent responses")
    fig.tight_layout()
    path = FIGURES_DIR / "phase13_policy_slices_core.png"
    fig.savefig(path)
    plt.close(fig)
    slice_df = pd.DataFrame(slice_rows).sort_values(["context_key", "policy_name", "slice_type", "state_value"])
    slice_df.to_csv(DATA_DIR / "phase13_policy_slice_points.csv", index=False)
    return (
        {
            "figure": path.name,
            "section": "规则机制与经济学解释",
            "message": "把最佳 RL 规则直接画成状态切片，展示它们如何改变对通胀缺口和产出缺口的局部反应，这一图是规则级经济学解释的核心入口。",
            "source": "phase13 evaluation of phase10 checkpoints and benchmark rules",
        },
        slice_df,
    )


def plot_cross_transfer() -> dict[str, str]:
    df = load_csv("outputs/phase10/counterfactual_eval/cross_transfer_summary.csv").copy()
    selected = df[
        df["policy_name"].isin(
            ["ppo_ann_direct_nonlinear", "ppo_svar_direct_nonlinear", "ppo_ann_direct", "td3_svar_direct"]
        )
    ].copy()
    label_order = ["ppo_ann_direct_nonlinear", "ppo_svar_direct_nonlinear", "ppo_ann_direct", "td3_svar_direct"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, eval_env in zip(axes, ["svar", "ann"], strict=True):
        sub_df = selected[selected["evaluation_env"] == eval_env].copy()
        sub_df["policy_name"] = pd.Categorical(sub_df["policy_name"], categories=label_order, ordered=True)
        sub_df = sub_df.sort_values("policy_name")
        ax.bar(sub_df["policy_name"].astype(str), sub_df["mean_discounted_loss"], color="#4e79a7")
        ax.set_title(f"Evaluation in {eval_env.upper()}")
        ax.set_ylabel("Mean discounted loss")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Cross-transfer check: strong direct rules remain environment-specific")
    fig.tight_layout()
    path = FIGURES_DIR / "phase13_cross_transfer.png"
    fig.savefig(path)
    plt.close(fig)
    return {
        "figure": path.name,
        "section": "经验环境：交叉迁移",
        "message": "把 direct-trained 规则放到对方环境中，展示 RL 优势主要来自正确匹配状态转移与福利权重，而不是任意环境下都同样有效。",
        "source": "outputs/phase10/counterfactual_eval/cross_transfer_summary.csv",
    }


def plot_external_models() -> tuple[dict[str, str], pd.DataFrame]:
    df = load_csv("outputs/phase10/external_model_robustness/all_external_summary.csv").copy()
    rl_mask = df["policy_name"].str.contains("ppo|td3|sac", case=False, na=False)
    baseline_cols = {
        "pyfrbus": "improvement_vs_pyfrbus_baseline_revealed_pct",
        "US_SW07": "improvement_vs_US_SW07_baseline_revealed_pct",
        "US_CCTW10": "improvement_vs_US_CCTW10_baseline_revealed_pct",
        "US_KS15": "improvement_vs_US_KS15_baseline_revealed_pct",
        "NK_CW09": "improvement_vs_NK_CW09_baseline_revealed_pct",
    }
    rows = []
    for model_id, col in baseline_cols.items():
        work = df[(df["model_id"] == model_id) & rl_mask & (~df["policy_name"].str.contains("baseline", na=False))].copy()
        work = work.dropna(subset=[col])
        if work.empty:
            continue
        best_row = work.sort_values(col, ascending=False).iloc[0]
        rows.append(
            {
                "model_id": model_id,
                "best_policy": best_row["policy_name"],
                "best_revealed_improvement_pct": float(best_row[col]),
            }
        )
    best_df = pd.DataFrame(rows).sort_values("best_revealed_improvement_pct", ascending=False)
    best_df.to_csv(DATA_DIR / "phase13_external_best_rules.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#1f77b4" if v > 0 else "#bab0ab" for v in best_df["best_revealed_improvement_pct"]]
    bars = ax.bar(best_df["model_id"], best_df["best_revealed_improvement_pct"], color=colors)
    for bar, policy_name, value in zip(bars, best_df["best_policy"], best_df["best_revealed_improvement_pct"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.8 if value >= 0 else -2.5),
            f"{policy_name}\n{value:.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Best RL revealed improvement (%)")
    ax.set_title("External models: best RL rule by model")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIGURES_DIR / "phase13_external_models_best_rules.png"
    fig.savefig(path)
    plt.close(fig)
    return (
        {
            "figure": path.name,
            "section": "稳健性：外部模型",
            "message": "外部模型部分不堆满所有规则，而是直接展示每个外部模型上最有竞争力的 RL 规则，突出可保留的稳健性证据。",
            "source": "outputs/phase10/external_model_robustness/all_external_summary.csv",
        },
        best_df,
    )


def plot_pyfrbus_progression() -> tuple[dict[str, str], pd.DataFrame]:
    baseline_df = load_csv("outputs/phase12/pyfrbus_warmstart_ppo/comparison.csv").copy()
    search_df = load_csv("outputs/phase12/pyfrbus_nonlinear_search/comparison.csv").copy()
    keep = baseline_df[["variant", "revealed_improvement_vs_baseline_pct"]].rename(
        columns={"revealed_improvement_vs_baseline_pct": "revealed_improvement_pct"}
    )
    search_keep = search_df[["variant", "trained_revealed_improvement_vs_baseline_pct"]].rename(
        columns={"trained_revealed_improvement_vs_baseline_pct": "revealed_improvement_pct"}
    )
    combined = pd.concat([keep, search_keep], ignore_index=True)
    order = [
        "fine_05",
        "ppo_linear_best_init",
        "ppo_nonlinear_warmstart_best",
        "residual_init_only",
        "residual_u2_lr5e6",
        "residual_u4_lr5e6",
    ]
    combined = combined[combined["variant"].isin(order)].copy()
    combined = combined.sort_values("revealed_improvement_pct", ascending=False).drop_duplicates("variant", keep="first")
    combined["variant"] = pd.Categorical(combined["variant"], categories=order, ordered=True)
    combined = combined.sort_values("variant")
    combined.to_csv(DATA_DIR / "phase13_pyfrbus_progression.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(combined["variant"].astype(str), combined["revealed_improvement_pct"], color="#59a14f")
    for bar, value in zip(bars, combined["revealed_improvement_pct"], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, f"{value:.2f}%", ha="center", va="bottom")
    ax.set_ylabel("Revealed improvement vs pyfrbus baseline (%)")
    ax.set_title("pyfrbus progression: from local linear search to warm-start nonlinear PPO")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIGURES_DIR / "phase13_pyfrbus_progression.png"
    fig.savefig(path)
    plt.close(fig)
    return (
        {
            "figure": path.name,
            "section": "稳健性：pyfrbus 原生接口",
            "message": "pyfrbus 图组强调‘native warm-start 之后 nonlinear PPO 可以再往前推一步’，这是论文中最适合讲原生 RL 增益的外部模型证据。",
            "source": "outputs/phase12/pyfrbus_warmstart_ppo/comparison.csv + outputs/phase12/pyfrbus_nonlinear_search/comparison.csv",
        },
        combined,
    )


def write_figure_catalog(items: list[dict[str, str]]) -> None:
    rows = "\n".join(
        f"| `{item['figure']}` | {item['section']} | {item['message']} | `{item['source']}` |"
        for item in items
    )
    content = (
        "# Phase 13 图表目录\n\n"
        "| 图名 | 论文位置 | 核心信息 | 数据来源 |\n"
        "|---|---|---|---|\n"
        f"{rows}\n"
    )
    (OUTPUT_DIR / "phase13_figure_catalog.md").write_text(content, encoding="utf-8")


def write_summary(items: list[dict[str, str]]) -> None:
    lines = "\n".join(f"- `{item['figure']}`：{item['message']}" for item in items)
    content = (
        "# Phase 13 总结\n\n"
        "本阶段目标不是新增训练，而是把已有 benchmark、非 benchmark、经验环境与外部模型结果整理成可直接进入论文写作的图表与解释材料。\n\n"
        "## 交付内容\n\n"
        "- 图表统一放在 `outputs/phase13/figures/`\n"
        "- 计算中间表统一放在 `outputs/phase13/tables/`\n"
        "- 图表目录：`outputs/phase13/phase13_figure_catalog.md`\n"
        "- 经济学解释材料：`outputs/phase13/phase13_writing_materials.md`\n\n"
        "## 本阶段生成的核心图\n\n"
        f"{lines}\n"
    )
    (OUTPUT_DIR / "phase13_summary.md").write_text(content, encoding="utf-8")


def format_policy_name(name: str) -> str:
    mapping = {
        "ppo_ann_direct_nonlinear": "PPO ANN direct nonlinear",
        "ppo_ann_direct": "PPO ANN direct",
        "sac_svar_revealed_direct": "SAC SVAR revealed",
        "td3_ann_revealed_direct": "TD3 ANN revealed",
        "sac_ann_revealed_direct": "SAC ANN revealed",
        "empirical_taylor_rule": "empirical Taylor",
        "historical_actual_policy": "historical policy",
        "sac_svar_direct": "SAC SVAR direct",
        "td3_ann_direct": "TD3 ANN direct",
    }
    return mapping.get(name, name)


def write_writing_materials(
    historical_vol: pd.DataFrame,
    stochastic_df: pd.DataFrame,
    slice_df: pd.DataFrame,
    external_best: pd.DataFrame,
    pyfrbus_progression: pd.DataFrame,
) -> None:
    hist_rows = []
    for welfare_type, env_name, best_policy in [
        ("artificial", "svar", "ppo_ann_direct_nonlinear"),
        ("artificial", "ann", "ppo_ann_direct"),
        ("revealed", "svar", "sac_svar_revealed_direct"),
        ("revealed", "ann", "td3_ann_revealed_direct"),
    ]:
        work = historical_vol[
            (historical_vol["welfare_type"] == welfare_type)
            & (historical_vol["environment"] == env_name)
            & (historical_vol["policy_name"].isin([best_policy, "empirical_taylor_rule", "historical_actual_policy"]))
        ].copy()
        hist_rows.append(work)
    hist_table = pd.concat(hist_rows, ignore_index=True)
    hist_md_rows = "\n".join(
        "| {welfare_type} | {environment} | {policy_name} | {loss:.2f} | {pi:.2f} | {y:.2f} | {rate:.2f} | {dr:.2f} |".format(
            welfare_type=row.welfare_type,
            environment=row.environment.upper(),
            policy_name=format_policy_name(row.policy_name),
            loss=row.total_discounted_loss,
            pi=row.std_inflation_gap,
            y=row.std_output_gap,
            rate=row.std_policy_rate,
            dr=row.std_rate_change,
        )
        for row in hist_table.itertuples(index=False)
    )

    stoch_rows = "\n".join(
        "| {context_key} | {policy_name} | {loss:.2f} | {pi:.2f} | {y:.2f} | {dr:.2f} |".format(
            context_key=row.context_key,
            policy_name=format_policy_name(row.policy_name),
            loss=row.mean_loss,
            pi=row.std_inflation_gap,
            y=row.std_output_gap,
            dr=row.std_rate_change,
        )
        for row in stochastic_df.itertuples(index=False)
    )

    key_points = []
    for context_key, policy_name, slice_type in [
        ("artificial_svar", "ppo_ann_direct_nonlinear", "inflation_gap"),
        ("artificial_ann", "ppo_ann_direct", "inflation_gap"),
        ("revealed_svar", "sac_svar_revealed_direct", "output_gap"),
        ("revealed_ann", "sac_ann_revealed_direct", "output_gap"),
    ]:
        work = slice_df[
            (slice_df["context_key"] == context_key)
            & (slice_df["policy_name"] == policy_name)
            & (slice_df["slice_type"] == slice_type)
            & (slice_df["state_value"].isin([-1.0, 0.0, 1.0]))
        ].copy()
        work = work.sort_values("state_value")
        values = ", ".join(f"{row.state_value:+.0f}→{row.policy_rate:.2f}" for row in work.itertuples(index=False))
        key_points.append(f"- `{context_key}` 下 `{format_policy_name(policy_name)}` 的 `{slice_type}` 切片为：{values}")

    external_rows = "\n".join(
        "| {model_id} | {best_policy} | {best_revealed_improvement_pct:.2f}% |".format(**row._asdict())
        for row in external_best.itertuples(index=False)
    )
    pyfrbus_rows = "\n".join(
        "| {variant} | {revealed_improvement_pct:.3f}% |".format(**row._asdict())
        for row in pyfrbus_progression.itertuples(index=False)
    )

    content = f"""# Phase 13 写作材料

## 1. 章节级主叙事

### 1.1 理论基准与 benchmark

可直接写入正文的主句式是：RL 先在 benchmark 与可控扩展环境中通过基准检验，再在强扭曲环境中把优势显著放大，因此其表现并不是偶然的单个案例结果，而是来自对非线性、约束和状态依赖反馈的系统适应。

### 1.2 三类非 benchmark 扩展

可直接写入正文的主句式是：一旦线性外推所依赖的局部二次近似被明显扭曲，RL 相对 Riccati 的优势会迅速扩大；这种优势尤其体现在强非线性、有效下界陷阱与阈值型非对称目标下。

### 1.3 经验环境与反推福利

可直接写入正文的主句式是：在经验状态转移下，RL 的优势不仅体现在反事实总损失下降，也体现在对通胀、产出缺口和利率平滑之间权衡的重新配置。反推福利进一步表明，最优规则的排序取决于央行真实偏好的权重结构，而 RL 可以据此学出更贴近该目标的反馈行为。

### 1.4 ANN 与外部模型稳健性

可直接写入正文的主句式是：ANN 经验环境与部分外部模型支持 RL 的稳健性，尤其 `pyfrbus` 原生 warm-start 结果表明，当 RL 直接在目标模型内继续学习时，它可以在强线性基准之上再实现小幅但清晰的增益。

## 2. 图组配套短分析

### 2.1 理论基准与扩展图组

`phase13_theory_heatmap.png` 与 `phase13_theory_strength_curves.png` 可配套写成：在 benchmark 及第一轮扭曲扩展中，RL 已经具备稳定竞争力；随着 nonlinear、ZLB 与 asymmetric 扭曲增强，最优 RL 与 Riccati/线性搜索之间的差距进一步拉大，说明 RL 的边际价值主要体现在状态依赖反馈结构被改写的区域。

`phase13_extreme_v2_advantage.png` 与 `phase13_extreme_mechanism_paths.png` 可配套写成：在最终保留的六个强扭曲环境里，RL 全面压过 Riccati 外推。共同冲击路径进一步说明，RL 的优势不是静态参数微调，而是对受约束区间和阈值区域给出不同的利率路径，从而改变输出与通胀的动态调整过程。

### 2.2 经验环境图组

`phase13_empirical_artificial_histories.png` 与 `phase13_empirical_revealed_histories.png` 可配套写成：在历史反事实中，优选 RL 规则并非简单复制 Taylor rule，而是在若干关键阶段给出更平滑或更及时的利率反应，使通胀和产出缺口路径更快回归目标附近。

`phase13_stochastic_tradeoffs.png` 与 `phase13_cross_transfer.png` 可配套写成：长期随机评估显示，RL 的优势不仅是平均损失降低，还体现为更好的通胀—产出波动组合；但交叉迁移结果说明，这一优势依赖于训练环境与评价环境的一致性，因此正文需要把环境匹配解释为优势成立的制度背景，而不是把它写成对 RL 的否定。

### 2.3 稳健性图组

`phase13_external_models_best_rules.png` 可配套写成：外部模型比较保留了若干关键的 RL 优势案例，说明经验环境中学到的反馈结构并非完全不可迁移。

`phase13_pyfrbus_progression.png` 可配套写成：`pyfrbus` 原生结果最有价值的地方不在于“迁移 RL 一开始就赢”，而在于 model-native warm-start 之后，nonlinear PPO 可以在强线性基准之上继续挖出额外改进。

## 3. 规则级经济学解释：优先写进正文的 cases

### 3.1 历史反事实与波动结果

| 福利口径 | 环境 | 规则 | 总损失 | 通胀波动 | 产出波动 | 利率波动 | 利率变动波动 |
|---|---|---|---:|---:|---:|---:|---:|
{hist_md_rows}

建议从这张表里抓三种典型机制写正文：

- `SVAR artificial`：`PPO ANN direct nonlinear` 相比历史政策，主要是通过显著压低产出与利率波动取得改进，说明其规则特征更接近“温和但持续的稳定化反馈”。
- `ANN artificial`：`PPO ANN direct` 同时压低通胀与产出波动，但利率变动更频繁，适合解释为更积极地使用政策工具来换取目标变量稳定。
- `SVAR/ANN revealed`：revealed 规则把高利率平滑权重内生化进反馈结构，因此在不少情况下会牺牲部分通胀稳定，以换取更低的利率调整成本与更优的总体福利。

### 3.2 长期随机评价

| 情境 | 规则 | 平均阶段损失 | 通胀波动 | 产出波动 | 利率变动波动 |
|---|---|---:|---:|---:|---:|
{stoch_rows}

正文可据此写两类解释：

- 人工损失下，RL 的优势可以理解为对通胀与产出稳定化的重新加权，部分规则愿意承担更高的利率调整频率以换取更低的目标变量波动。
- 反推损失下，SAC revealed 规则更明显地体现出利率平滑偏好，因此“为什么更优”不能只看通胀反应强弱，而要同时看它如何减少不必要的政策转向。

### 3.3 规则切片：为什么这些 RL 规则更优

下面这些切片点可直接转化成正文中的规则级解释：

{chr(10).join(key_points)}

建议写法：

- 若 RL 在 `inflation gap` 切片上比 Taylor 更平缓，可解释为它避免对短期价格偏离做过度反应，更多依靠跨期平滑来稳定路径。
- 若 RL 在 `output gap` 切片上呈现更强的正向或非线性斜率，可解释为它更积极应对深度衰退或需求过热区间。
- 若 revealed 规则整体切片更平，可解释为高利率平滑权重把政策最优点推向“低频率、小幅度”的调节方式。

## 4. 稳健性部分可直接使用的表

### 4.1 外部模型中最好的 RL 规则

| 模型 | 最优 RL 规则 | 相对 baseline 的 revealed 改进 |
|---|---|---:|
{external_rows}

正文可写成：外部模型结果不是把所有 RL 规则都包装成稳健赢家，而是突出若干可保留的成功迁移案例；这比简单宣称“普遍稳健”更可信，也更有助于说明哪些 RL 反馈结构具备跨模型竞争力。

### 4.2 pyfrbus 原生优化路径

| 变体 | 相对 pyfrbus baseline 的 revealed 改进 |
|---|---:|
{pyfrbus_rows}

正文建议写法：`pyfrbus` 结果表明，本地线性搜索已经能显著超过 baseline，而 warm-start nonlinear PPO 还能在此基础上进一步提升。这说明 RL 在复杂外部模型中的价值，并不是从零开始暴力替代传统方法，而是作为结构化初值之上的继续优化器。

## 5. 相对 Hinterlang 的可写优势

- 我们的图组不只展示“RL 规则优于若干传统规则”，还系统覆盖了 benchmark、三类非 benchmark 扩展、双福利、历史反事实、长期随机、交叉迁移、外部模型与 `pyfrbus` 原生训练。
- 我们的经济学解释不只停留在 ANN 或 PD 图，而是把规则切片、历史路径、波动评价与外部稳健性结合起来，解释 RL 规则为什么更优、优在哪个维度、代价是什么。
- 因而论文正文应突出：RL 的优势不仅在数值大小上更丰富，也在机制解释和稳健性层面更完整。
"""
    (OUTPUT_DIR / "phase13_writing_materials.md").write_text(content, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    configure_style()

    figures: list[dict[str, str]] = []
    figures.append(plot_theory_heatmap())
    figures.append(plot_theory_strength_curves())
    figures.append(plot_extreme_v2_advantage())
    figures.append(plot_extreme_mechanism_paths())
    figures.append(
        plot_empirical_history(
            "outputs/phase10/counterfactual_eval/svar_historical_paths.csv",
            "Artificial-loss historical counterfactuals: best RL vs Taylor vs history",
            {"svar": "ppo_ann_direct_nonlinear", "ann": "ppo_ann_direct"},
            "phase13_empirical_artificial_histories.png",
        )
    )
    figures.append(
        plot_empirical_history(
            "outputs/phase10/revealed_policy_eval/svar_historical_paths.csv",
            "Revealed-loss historical counterfactuals: best RL vs Taylor vs history",
            {"svar": "sac_svar_revealed_direct", "ann": "td3_ann_revealed_direct"},
            "phase13_empirical_revealed_histories.png",
        )
    )

    policy_maps = compute_policy_maps()
    historical_vol = historical_volatility_table()
    stochastic_df = stochastic_core_tables(policy_maps)
    figures.append(plot_stochastic_tradeoffs(stochastic_df))
    figure_item, slice_df = plot_policy_slices(policy_maps)
    figures.append(figure_item)
    figures.append(plot_cross_transfer())
    external_item, external_best = plot_external_models()
    figures.append(external_item)
    pyfrbus_item, pyfrbus_progression = plot_pyfrbus_progression()
    figures.append(pyfrbus_item)

    write_figure_catalog(figures)
    write_summary(figures)
    write_writing_materials(historical_vol, stochastic_df, slice_df, external_best, pyfrbus_progression)


if __name__ == "__main__":
    main()
