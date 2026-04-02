from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PYFRBUS_ROOT = ROOT / "external_models" / "frbus_extracted" / "pyfrbus"
if str(PYFRBUS_ROOT) not in sys.path:
    sys.path.insert(0, str(PYFRBUS_ROOT))

from pyfrbus.frbus import Frbus
from pyfrbus.load_data import load_data

from monetary_rl.agents.sac import SACConfig, SACTrainer
from monetary_rl.evaluation import fit_linear_policy_response
from monetary_rl.phase10_utils import build_linear_policy
from phase10_external_model_robustness import BENCHMARK_CONFIG, REVEALED_WEIGHTS, run_pyfrbus_fixed_point, welfare_summary


PYFRBUS_MODEL = PYFRBUS_ROOT / "models" / "model.xml"
PYFRBUS_DATA = PYFRBUS_ROOT / "data" / "LONGBASE.TXT"
TARGET_RATE = 2.0


@dataclass(frozen=True)
class PyfrbusEnvConfig:
    start: str = "2040Q1"
    end: str = "2045Q4"
    action_low: float = -1.0
    action_high: float = 2.5
    target_rate: float = TARGET_RATE
    terminal_penalty: float = 50.0


class PyfrbusModelProxy:
    def __init__(self, discount_factor: float, target_rate: float) -> None:
        self.state_dim = 3
        self.config = SimpleNamespace(discount_factor=discount_factor)
        self.target_rate = target_rate

    def action_to_level(self, action_gap: float) -> float:
        return self.target_rate + float(action_gap)


class PyfrbusNativeEnv:
    def __init__(
        self,
        config: PyfrbusEnvConfig,
        loss_weights: dict[str, float],
        horizon_end: str | None = None,
    ) -> None:
        self.config = config
        self.loss_weights = {k: float(v) for k, v in loss_weights.items()}
        self.model = PyfrbusModelProxy(float(BENCHMARK_CONFIG["discount_factor"]), config.target_rate)
        self.frbus = Frbus(str(PYFRBUS_MODEL))
        self.frbus.exogenize(["rff"])
        self.data = load_data(str(PYFRBUS_DATA))
        self.start = pd.Period(config.start, freq="Q")
        self.end = pd.Period(horizon_end or config.end, freq="Q")
        self.periods = pd.period_range(self.start, self.end, freq="Q")
        self.data.loc[self.start : self.end, "dfpdbt"] = 0.0
        self.data.loc[self.start : self.end, "dfpsrp"] = 1.0
        self.base = self.frbus.init_trac(self.start, self.end, self.data)
        self.baseline = self.frbus.solve(self.start, self.end, self.base, options={"newton": "newton", "single_block": True})
        self.current = self.baseline.copy()
        self.t = 0
        self.failed = False

    def reset(self, seed: int | None = None) -> np.ndarray:
        del seed
        self.current = self.baseline.copy()
        self.t = 0
        self.failed = False
        return self._state_at(self.periods[0])

    def _state_at(self, period: pd.Period) -> np.ndarray:
        inflation_gap = float(self.current.loc[period, "picxfe"] - self.current.loc[period, "pitarg"])
        output_gap = float(self.current.loc[period, "xgap"])
        lagged_period = period - 1
        lagged_rate = float(self.current.loc[lagged_period, "rff"]) if lagged_period in self.current.index else TARGET_RATE
        lagged_rate_gap = lagged_rate - self.config.target_rate
        return np.array([inflation_gap, output_gap, lagged_rate_gap], dtype=float)

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        period = self.periods[self.t]
        clipped_action = float(np.clip(action, self.config.action_low, self.config.action_high))
        policy_rate = self.model.action_to_level(clipped_action)
        lagged_period = period - 1
        lagged_rate = float(self.current.loc[lagged_period, "rff"]) if lagged_period in self.current.index else TARGET_RATE
        trial = self.current.copy()
        trial.loc[period, "rff"] = policy_rate

        exploded = False
        try:
            sim = self.frbus.solve(period, self.end, trial, options={"newton": "newton", "single_block": True})
            self.current = sim.copy()
            inflation_gap = float(sim.loc[period, "picxfe"] - sim.loc[period, "pitarg"])
            output_gap = float(sim.loc[period, "xgap"])
            rate_change = policy_rate - lagged_rate
            loss = (
                self.loss_weights["inflation"] * inflation_gap**2
                + self.loss_weights["output_gap"] * output_gap**2
                + self.loss_weights["rate_smoothing"] * rate_change**2
            )
            reward = -float(loss)
        except Exception:
            exploded = True
            reward = -float(self.config.terminal_penalty)
            loss = float(self.config.terminal_penalty)
            inflation_gap = np.nan
            output_gap = np.nan
            rate_change = np.nan
            self.failed = True

        self.t += 1
        done = exploded or self.t >= len(self.periods)
        next_state = np.zeros(3, dtype=float) if done else self._state_at(self.periods[self.t])
        info = {
            "loss": float(loss),
            "raw_action": float(action),
            "action": clipped_action,
            "exploded": exploded,
            "policy_rate": policy_rate,
            "inflation_gap": float(inflation_gap) if np.isfinite(inflation_gap) else np.nan,
            "output_gap": float(output_gap) if np.isfinite(output_gap) else np.nan,
            "rate_change": float(rate_change) if np.isfinite(rate_change) else np.nan,
        }
        return next_state, reward, done, info


def artificial_loss_weights() -> dict[str, float]:
    weights = BENCHMARK_CONFIG["loss_weights"]
    return {
        "inflation": float(weights["inflation"]),
        "output_gap": float(weights["output_gap"]),
        "rate_smoothing": float(weights["rate_smoothing"]),
    }


def revealed_loss_weights() -> dict[str, float]:
    return {
        "inflation": float(REVEALED_WEIGHTS["inflation_weight"]),
        "output_gap": float(REVEALED_WEIGHTS["output_gap_weight"]),
        "rate_smoothing": float(REVEALED_WEIGHTS["rate_smoothing_weight"]),
    }


def load_reference_registry() -> pd.DataFrame:
    frames = [
        pd.read_csv(ROOT / "outputs" / "phase10" / "counterfactual_eval" / "unified_policy_registry.csv"),
        pd.read_csv(ROOT / "outputs" / "phase10" / "revealed_policy_training" / "policy_registry.csv"),
    ]
    return pd.concat(frames, ignore_index=True, sort=False).drop_duplicates(subset=["policy_name"], keep="last").reset_index(drop=True)


def load_reference_row(policy_name: str) -> pd.Series:
    registry = load_reference_registry()
    row = registry.loc[registry["policy_name"] == policy_name]
    if row.empty:
        raise KeyError(f"Policy {policy_name} not found in reference registry.")
    return row.iloc[0]


def evaluate_policy_fixed_point(policy_name: str, policy_fn) -> dict[str, Any]:
    window, meta = run_pyfrbus_fixed_point(policy_name, policy_fn)
    summary = welfare_summary(window.set_index("period"), policy_name)
    summary.update(meta)
    return summary


def evaluate_linear_params(policy_name: str, intercept: float, inflation_gap: float, output_gap: float, lagged_policy_rate_gap: float) -> dict[str, Any]:
    policy_fn = build_linear_policy(policy_name, intercept, inflation_gap, output_gap, lagged_policy_rate_gap)
    summary = evaluate_policy_fixed_point(policy_name, policy_fn)
    summary.update(
        {
            "intercept": float(intercept),
            "inflation_coeff": float(inflation_gap),
            "output_coeff": float(output_gap),
            "lagged_rate_coeff": float(lagged_policy_rate_gap),
            "policy_parameterization": "linear_rule",
        }
    )
    return summary


def baseline_observation_bounds() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    env = PyfrbusNativeEnv(PyfrbusEnvConfig(), revealed_loss_weights())
    rows = []
    for period in env.periods:
        state = env._state_at(period)
        rows.append(state)
    obs = np.asarray(rows, dtype=float)
    low = np.quantile(obs, 0.05, axis=0)
    high = np.quantile(obs, 0.95, axis=0)
    spread = np.maximum(high - low, 1e-3)
    low = low - 0.5 * spread
    high = high + 0.5 * spread
    return tuple(float(x) for x in low), tuple(float(x) for x in high)


def build_sac_policy_from_checkpoint(checkpoint_path: Path, config_payload: dict[str, Any], env: PyfrbusNativeEnv):
    trainer = SACTrainer(env, SACConfig(**config_payload))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    trainer.actor.load_state_dict(checkpoint)
    return lambda state, t: trainer._deterministic_action(np.asarray(state, dtype=float))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def summarize_results_md(path: Path, title: str, tables: list[tuple[str, pd.DataFrame]]) -> None:
    lines = [f"# {title}", ""]
    for heading, frame in tables:
        lines.extend([f"## {heading}", "", frame.to_markdown(index=False), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def fit_surrogate(policy_name: str, policy_fn) -> dict[str, float]:
    low, high = baseline_observation_bounds()
    return fit_linear_policy_response(policy_name, policy_fn, low, high, grid_points=7)


def as_serializable_config(config: PyfrbusEnvConfig) -> dict[str, Any]:
    return asdict(config)
