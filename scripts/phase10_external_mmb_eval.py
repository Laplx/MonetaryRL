from __future__ import annotations

import json
import re
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "phase10" / "external_model_robustness"
WORK_ROOT = OUTPUT_DIR / "mmb_work"
COUNTERFACTUAL_DIR = ROOT / "outputs" / "phase10" / "counterfactual_eval"
REVEALED_TRAIN_DIR = ROOT / "outputs" / "phase10" / "revealed_policy_training"
REVEALED_DIR = ROOT / "outputs" / "phase10" / "revealed_welfare"
MMB_ROOT = ROOT / "external_models" / "mmb_extracted" / "mmb-rep-master"
BENCHMARK_CONFIG = json.loads((ROOT / "src" / "monetary_rl" / "config" / "benchmark_lq.json").read_text(encoding="utf-8"))
REVEALED_WEIGHTS = json.loads((REVEALED_DIR / "revealed_weights.json").read_text(encoding="utf-8"))
DYNARE_MATLAB_PATH = Path(r"C:\dynare\7.0\matlab")
MATLAB_BIN = "matlab"

DISCOUNT = float(BENCHMARK_CONFIG["discount_factor"])
LOSS_WEIGHTS = BENCHMARK_CONFIG["loss_weights"]
SIM_PERIODS = 3000
BURN_IN = 500
MATLAB_TIMEOUT_SEC = 1200


def _extract_cached_series(spec: "ModelSpec", results_mat: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = loadmat(results_mat, squeeze_me=True, struct_as_record=False)
    endo_names = np.asarray(payload["M_"].endo_names).reshape(-1)
    names = [str(name).strip() for name in endo_names.tolist()]
    sim = np.asarray(payload["oo_"].endo_simul, dtype=float)
    sim = np.atleast_2d(sim)
    if sim.shape[0] != len(names) and sim.shape[1] == len(names):
        sim = sim.T
    if sim.shape[0] != len(names):
        raise ValueError(f"Unexpected simulation shape {sim.shape} for model {spec.model_id}")
    sim = sim[:, max(0, sim.shape[1] - 2500) :]
    loc = {name: idx for idx, name in enumerate(names)}

    if spec.model_id in {"US_SW07", "US_CCTW10"}:
        rate_series = 4.0 * sim[loc["r"], :]
        inflation_gap = 4.0 * sim[loc["pinf"], :]
        output_gap = sim[loc["y"], :] - sim[loc["yf"], :]
    elif spec.model_id == "US_KS15":
        rate_series = sim[loc["R"], :]
        inflation_gap = sim[loc["pit"], :]
        output_gap = sim[loc["y"], :]
    elif spec.model_id == "NK_CW09":
        rate_series = 4.0 * sim[loc["i_d_hat"], :]
        inflation_gap = 4.0 * sim[loc["Pi_hat"], :]
        output_gap = sim[loc["Y_hat"], :] - sim[loc["Y_n_hat"], :]
    else:
        raise ValueError(f"Unsupported cached extraction for model {spec.model_id}")
    return inflation_gap.astype(float), output_gap.astype(float), rate_series.astype(float)


def cached_summary_frame(spec: "ModelSpec", policy: "PolicySpec", model_work: Path) -> pd.DataFrame | None:
    results = list(model_work.glob("*/Output/*_results.mat"))
    if not results:
        return None
    try:
        inflation_gap, output_gap, rate_series = _extract_cached_series(spec, results[0])
    except Exception:
        return None
    if rate_series.size <= BURN_IN:
        return None
    inflation_gap = inflation_gap[BURN_IN:]
    output_gap = output_gap[BURN_IN:]
    rate_series = rate_series[BURN_IN:]
    rate_change = np.concatenate([[np.nan], np.diff(rate_series)])
    valid = np.isfinite(inflation_gap) & np.isfinite(output_gap) & np.isfinite(rate_series) & np.isfinite(rate_change)
    inflation_gap = inflation_gap[valid]
    output_gap = output_gap[valid]
    rate_series = rate_series[valid]
    rate_change = rate_change[valid]
    if inflation_gap.size == 0:
        return None

    per_loss = (
        LOSS_WEIGHTS["inflation"] * inflation_gap**2
        + LOSS_WEIGHTS["output_gap"] * output_gap**2
        + LOSS_WEIGHTS["rate_smoothing"] * rate_change**2
    )
    per_revealed_loss = (
        float(REVEALED_WEIGHTS["inflation_weight"]) * inflation_gap**2
        + float(REVEALED_WEIGHTS["output_gap_weight"]) * output_gap**2
        + float(REVEALED_WEIGHTS["rate_smoothing_weight"]) * rate_change**2
    )
    discounts = DISCOUNT ** np.arange(per_loss.size, dtype=float)
    return pd.DataFrame(
        [
            {
                "policy_name": policy.policy_name,
                "model_id": spec.model_id,
                "rule_family": policy.rule_family,
                "inflation_coeff": policy.inflation_gap,
                "output_coeff": policy.output_gap,
                "lagged_rate_coeff": policy.lagged_policy_rate_gap,
                "total_discounted_loss": float(np.sum(per_loss * discounts)),
                "mean_period_loss": float(np.mean(per_loss)),
                "total_discounted_revealed_loss": float(np.sum(per_revealed_loss * discounts)),
                "mean_period_revealed_loss": float(np.mean(per_revealed_loss)),
                "mean_sq_inflation_gap": float(np.mean(inflation_gap**2)),
                "mean_sq_output_gap": float(np.mean(output_gap**2)),
                "mean_sq_rate_change": float(np.mean(rate_change**2)),
                "mean_policy_rate": float(np.mean(rate_series)),
                "std_policy_rate": float(np.std(rate_series, ddof=0)),
            }
        ]
    )

MAIN_POLICY_ORDER = [
    "ppo_benchmark_transfer",
    "td3_benchmark_transfer",
    "sac_benchmark_transfer",
    "ppo_svar_direct",
    "ppo_svar_direct_nonlinear",
    "td3_svar_direct",
    "sac_svar_direct",
    "ppo_ann_direct",
    "ppo_ann_direct_nonlinear",
    "td3_ann_direct",
    "sac_ann_direct",
    "ppo_svar_revealed_direct",
    "ppo_svar_revealed_direct_nonlinear",
    "td3_svar_revealed_direct",
    "sac_svar_revealed_direct",
    "ppo_ann_revealed_direct",
    "ppo_ann_revealed_direct_nonlinear",
    "td3_ann_revealed_direct",
    "sac_ann_revealed_direct",
    "empirical_taylor_rule",
]


@dataclass(frozen=True)
class PolicySpec:
    policy_name: str
    rule_family: str
    source_env: str
    training_env: str
    algo: str
    inflation_gap: float
    output_gap: float
    lagged_policy_rate_gap: float
    intercept: float = 0.0


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    relative_dir: str
    template_file: str
    patch_style: str
    summary_code: str
    dynare_args: tuple[str, ...] = ("noclearall", "nolog")
    baseline_policy: PolicySpec | None = None


def load_policy_specs() -> list[PolicySpec]:
    unified = pd.read_csv(COUNTERFACTUAL_DIR / "unified_policy_registry.csv")
    revealed_frames = []
    for env_name in ["svar", "ann"]:
        path = REVEALED_TRAIN_DIR / env_name / "policy_registry.csv"
        if path.exists():
            revealed_frames.append(pd.read_csv(path))
    revealed = pd.concat(revealed_frames, ignore_index=True) if revealed_frames else pd.DataFrame()

    frames = [unified]
    if not revealed.empty:
        frames.append(revealed)
    all_rules = pd.concat(frames, ignore_index=True, sort=False)
    all_rules = all_rules.loc[all_rules["policy_name"].isin(MAIN_POLICY_ORDER)].copy()
    all_rules = all_rules.drop_duplicates(subset=["policy_name"], keep="last")
    order = {name: idx for idx, name in enumerate(MAIN_POLICY_ORDER)}
    all_rules["order"] = all_rules["policy_name"].map(order)
    all_rules = all_rules.sort_values("order").reset_index(drop=True)

    specs: list[PolicySpec] = []
    for row in all_rules.to_dict("records"):
        specs.append(
            PolicySpec(
                policy_name=str(row["policy_name"]),
                rule_family=str(row.get("rule_family", "")),
                source_env=str(row.get("source_env", "")),
                training_env=str(row.get("training_env", "")),
                algo=str(row.get("algo", "")),
                inflation_gap=float(row.get("inflation_gap", 0.0)),
                output_gap=float(row.get("output_gap", 0.0)),
                lagged_policy_rate_gap=float(row.get("lagged_policy_rate_gap", 0.0)),
                intercept=float(row.get("intercept", 0.0) if pd.notna(row.get("intercept", np.nan)) else 0.0),
            )
        )
    return specs


MODEL_SPECS = [
    ModelSpec(
        model_id="US_SW07",
        relative_dir="US_SW07/US_SW07_rep",
        template_file="US_SW07_rep.mod",
        patch_style="us_sw07",
        summary_code="""
names = string(strtrim(cellstr(M_.endo_names)));
sim = oo_.endo_simul;
sim = sim(:, max(1, size(sim, 2) - 2499):size(sim, 2));
loc = @(n) find(names == n, 1);
rate_series = 4 * sim(loc("r"), :);
inflation_gap = 4 * sim(loc("pinf"), :);
output_gap = sim(loc("y"), :) - sim(loc("yf"), :);
""",
        baseline_policy=PolicySpec(
            policy_name="US_SW07_baseline",
            rule_family="external_baseline",
            source_env="external",
            training_env="external",
            algo="baseline",
            inflation_gap=0.0,
            output_gap=0.0,
            lagged_policy_rate_gap=0.0,
        ),
    ),
    ModelSpec(
        model_id="US_CCTW10",
        relative_dir="US_CCTW10/Code_CCTW_2010_JEDC",
        template_file="SW_US_fiscal.mod",
        patch_style="modelbase_coeffs",
        summary_code="""
names = string(strtrim(cellstr(M_.endo_names)));
sim = oo_.endo_simul;
sim = sim(:, max(1, size(sim, 2) - 2499):size(sim, 2));
loc = @(n) find(names == n, 1);
rate_series = 4 * sim(loc("r"), :);
inflation_gap = 4 * sim(loc("pinf"), :);
output_gap = sim(loc("y"), :) - sim(loc("yf"), :);
""",
        dynare_args=("noclearall", "nolog", "nostrict"),
        baseline_policy=PolicySpec(
            policy_name="US_CCTW10_baseline",
            rule_family="external_baseline",
            source_env="external",
            training_env="external",
            algo="baseline",
            inflation_gap=0.0,
            output_gap=0.0,
            lagged_policy_rate_gap=0.0,
        ),
    ),
    ModelSpec(
        model_id="US_KS15",
        relative_dir="US_KS15/US_KS15_replication",
        template_file="US_KS15_R3.mod",
        patch_style="us_ks15",
        summary_code="""
names = string(strtrim(cellstr(M_.endo_names)));
sim = oo_.endo_simul;
sim = sim(:, max(1, size(sim, 2) - 2499):size(sim, 2));
loc = @(n) find(names == n, 1);
rate_series = sim(loc("R"), :);
inflation_gap = sim(loc("pit"), :);
output_gap = sim(loc("y"), :);
""",
        baseline_policy=PolicySpec(
            policy_name="US_KS15_baseline",
            rule_family="external_baseline",
            source_env="external",
            training_env="external",
            algo="baseline",
            inflation_gap=0.0,
            output_gap=0.0,
            lagged_policy_rate_gap=0.0,
        ),
    ),
    ModelSpec(
        model_id="NK_CW09",
        relative_dir="NK_CW09",
        template_file="NK_CW09.mod",
        patch_style="nk_cw09",
        summary_code="""
names = string(strtrim(cellstr(M_.endo_names)));
sim = oo_.endo_simul;
sim = sim(:, max(1, size(sim, 2) - 2499):size(sim, 2));
loc = @(n) find(names == n, 1);
rate_series = 4 * sim(loc("i_d_hat"), :);
inflation_gap = 4 * sim(loc("Pi_hat"), :);
output_gap = sim(loc("Y_hat"), :) - sim(loc("Y_n_hat"), :);
""",
        baseline_policy=PolicySpec(
            policy_name="NK_CW09_baseline",
            rule_family="external_baseline",
            source_env="external",
            training_env="external",
            algo="baseline",
            inflation_gap=2.0,
            output_gap=0.25,
            lagged_policy_rate_gap=0.0,
        ),
    ),
]


def replace_once(text: str, pattern: str, repl: str) -> str:
    updated, count = re.subn(pattern, repl, text, flags=re.MULTILINE)
    if count < 1:
        raise ValueError(f"Pattern not found: {pattern}")
    return updated


def read_text_any(path: Path) -> str:
    for encoding in ["utf-8", "cp1252", "latin-1"]:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("latin-1", errors="replace")


def cleanup_generated_package_dirs(workdir: Path) -> None:
    for path in workdir.iterdir():
        if path.is_dir() and path.name.startswith("+"):
            shutil.rmtree(path, ignore_errors=True)


def patch_model_text(spec: ModelSpec, template: str, policy: PolicySpec, use_original: bool) -> str:
    text = template
    if spec.patch_style == "us_sw07":
        if not use_original:
            rule = (
                f"          //Monetary Policy Rule\n"
                f"\t      r =  {policy.inflation_gap:.12f}*pinf + {policy.output_gap:.12f}*(y-yf) + "
                f"{policy.lagged_policy_rate_gap:.12f}*r(-1) +ms  ;"
            )
            text = replace_once(
                text,
                r"\s*//Monetary Policy Rule\s*[\r\n]+\s*r =  crpi\*\(1-crr\)\*pinf \+cry\*\(1-crr\)\*\(y-yf\) \+crdy\*\(y-yf-y\(-1\)\+yf\(-1\)\)\+crr\*r\(-1\) \+ms  ;",
                "\n" + rule,
            )
        text = replace_once(
            text,
            r"stoch_simul\(irf=20, noprint, nograph\) r pinf lab y;",
            f"stoch_simul(order=1, periods={SIM_PERIODS}, irf=0, noprint, nograph) r pinf y yf;",
        )
        return text

    if spec.patch_style == "modelbase_coeffs":
        if not use_original:
            text = replace_once(
                text,
                r"cofintintb1 =  .*?;",
                (
                    f"cofintintb1 =  {policy.lagged_policy_rate_gap:.12f}; "
                    "cofintintb2 = 0; cofintintb3 = 0; cofintintb4 = 0;"
                ),
            )
            text = replace_once(
                text,
                r"cofintinf0 = .*?;",
                (
                    f"cofintinf0 = {policy.inflation_gap:.12f}; "
                    "cofintinfb1 = 0; cofintinfb2 = 0; cofintinfb3 = 0; cofintinfb4 = 0; "
                    "cofintinff1 = 0; cofintinff2 = 0; cofintinff3 = 0; cofintinff4 = 0;"
                ),
            )
            text = replace_once(
                text,
                r"cofintout = .*?;",
                (
                    f"cofintout = {policy.output_gap:.12f}; "
                    "cofintoutb1 = 0; cofintoutb2 = 0; cofintoutb3 = 0; cofintoutb4 = 0; "
                    "cofintoutf1 = 0; cofintoutf2 = 0; cofintoutf3 = 0; cofintoutf4 = 0;"
                ),
            )
        if "stoch_simul(" in text:
            text = replace_once(
                text,
                r"stoch_simul\([^;]*;",
                f"stoch_simul(order=1, periods={SIM_PERIODS}, irf=0, nograph, noprint) r pinf y yf;",
            )
        else:
            text += f"\n\nstoch_simul(order=1, periods={SIM_PERIODS}, irf=0, nograph, noprint) r pinf y yf;\n"
        return text

    if spec.patch_style == "us_ks15":
        if not use_original:
            text = replace_once(text, r"rho_R\s*=\s*[-0-9.eE+]+\s*;", f"rho_R   = {policy.lagged_policy_rate_gap:.12f};")
            text = replace_once(text, r"rho_pi\s*=\s*[-0-9.eE+]+\s*;", f"rho_pi  = {policy.inflation_gap:.12f};")
            text = replace_once(text, r"rho_y\s*=\s*[-0-9.eE+]+\s*;", f"rho_y   = {policy.output_gap:.12f};")
        text = replace_once(
            text,
            r"stoch_simul\(irf=20, nograph\) c y m pit RR R;",
            f"stoch_simul(order=1, periods={SIM_PERIODS}, irf=0, nograph, noprint) R pit y;",
        )
        return text

    if spec.patch_style == "nk_cw09":
        coeff_block = textwrap.dedent(
            f"""
            // Load Modelbase Monetary Policy Parameters                             //*
            cofintintb1 = {policy.lagged_policy_rate_gap:.12f}; cofintintb2 = 0; cofintintb3 = 0; cofintintb4 = 0;
            cofintinf0 = {policy.inflation_gap:.12f}; cofintinfb1 = 0; cofintinfb2 = 0; cofintinfb3 = 0; cofintinfb4 = 0;
            cofintinff1 = 0; cofintinff2 = 0; cofintinff3 = 0; cofintinff4 = 0;
            cofintout = {policy.output_gap:.12f}; cofintoutb1 = 0; cofintoutb2 = 0; cofintoutb3 = 0; cofintoutb4 = 0;
            cofintoutf1 = 0; cofintoutf2 = 0; cofintoutf3 = 0; cofintoutf4 = 0;
            cofintoutp = 0; cofintoutpb1 = 0; cofintoutpb2 = 0; cofintoutpb3 = 0; cofintoutpb4 = 0;
            cofintoutpf1 = 0; cofintoutpf2 = 0; cofintoutpf3 = 0; cofintoutpf4 = 0;
            std_r_ = 1;
            std_r_quart = 0.25;
            """
        ).strip()
        text = replace_once(
            text,
            r"// Load Modelbase Monetary Policy Parameters[\s\S]*?cd\(thispath\);\s*",
            coeff_block + "\n",
        )
        if "stoch_simul(" in text:
            text = replace_once(
                text,
                r"//stoch_simul\([^;\n]*\)[\s\S]*",
                f"stoch_simul(order=1, periods={SIM_PERIODS}, irf=0, nograph, noprint) i_d_hat Pi_hat Y_hat Y_n_hat;\n",
            )
        else:
            text += f"\n\nstoch_simul(order=1, periods={SIM_PERIODS}, irf=0, nograph, noprint) i_d_hat Pi_hat Y_hat Y_n_hat;\n"
        return text

    raise ValueError(f"Unsupported patch style: {spec.patch_style}")


def matlab_quote(path: Path) -> str:
    return path.resolve().as_posix()


def build_runner_script(spec: ModelSpec, mod_path: Path, summary_path: Path, policy: PolicySpec) -> str:
    dynare_args = ", ".join(f"'{arg}'" for arg in spec.dynare_args)
    return textwrap.dedent(
        f"""
        addpath('{matlab_quote(DYNARE_MATLAB_PATH)}');
        cd('{matlab_quote(mod_path.parent)}');
        rng(1, 'twister');
        dynare('{mod_path.name}', {dynare_args});
        {spec.summary_code}
        inflation_gap = inflation_gap(:);
        output_gap = output_gap(:);
        rate_series = rate_series(:);
        if numel(rate_series) <= {BURN_IN}
            error('Simulation too short for burn-in.');
        end
        inflation_gap = inflation_gap(({BURN_IN}+1):end);
        output_gap = output_gap(({BURN_IN}+1):end);
        rate_series = rate_series(({BURN_IN}+1):end);
        rate_change = [NaN; diff(rate_series)];
        valid = isfinite(inflation_gap) & isfinite(output_gap) & isfinite(rate_series) & isfinite(rate_change);
        inflation_gap = inflation_gap(valid);
        output_gap = output_gap(valid);
        rate_series = rate_series(valid);
        rate_change = rate_change(valid);
        per_loss = {LOSS_WEIGHTS["inflation"]:.12f} .* (inflation_gap .^ 2) ...
            + {LOSS_WEIGHTS["output_gap"]:.12f} .* (output_gap .^ 2) ...
            + {LOSS_WEIGHTS["rate_smoothing"]:.12f} .* (rate_change .^ 2);
        per_revealed_loss = {float(REVEALED_WEIGHTS["inflation_weight"]):.12f} .* (inflation_gap .^ 2) ...
            + {float(REVEALED_WEIGHTS["output_gap_weight"]):.12f} .* (output_gap .^ 2) ...
            + {float(REVEALED_WEIGHTS["rate_smoothing_weight"]):.12f} .* (rate_change .^ 2);
        discounts = ({DISCOUNT:.12f}) .^ (0:(numel(per_loss)-1))';
        total_discounted_loss = sum(per_loss .* discounts);
        total_discounted_revealed_loss = sum(per_revealed_loss .* discounts);
        row = table( ...
            string('{policy.policy_name}'), ...
            string('{spec.model_id}'), ...
            "{policy.rule_family}", ...
            {policy.inflation_gap:.12f}, ...
            {policy.output_gap:.12f}, ...
            {policy.lagged_policy_rate_gap:.12f}, ...
            total_discounted_loss, ...
            mean(per_loss), ...
            total_discounted_revealed_loss, ...
            mean(per_revealed_loss), ...
            mean(inflation_gap .^ 2), ...
            mean(output_gap .^ 2), ...
            mean(rate_change .^ 2), ...
            mean(rate_series), ...
            std(rate_series, 1), ...
            'VariableNames', {{ ...
                'policy_name', 'model_id', 'rule_family', 'inflation_coeff', 'output_coeff', 'lagged_rate_coeff', ...
                'total_discounted_loss', 'mean_period_loss', 'total_discounted_revealed_loss', 'mean_period_revealed_loss', 'mean_sq_inflation_gap', 'mean_sq_output_gap', ...
                'mean_sq_rate_change', 'mean_policy_rate', 'std_policy_rate' ...
            }} ...
        );
        writetable(row, '{matlab_quote(summary_path)}');
        """
    ).strip()


def run_model_policy(spec: ModelSpec, policy: PolicySpec, template_text: str) -> tuple[dict[str, object], pd.DataFrame | None]:
    model_work = WORK_ROOT / spec.model_id / policy.policy_name
    model_work.mkdir(parents=True, exist_ok=True)
    mod_stem = f"{spec.model_id}_{policy.policy_name}"
    mod_path = model_work / f"{mod_stem}.mod"
    runner_path = model_work / f"run_{mod_stem}.m"
    summary_path = model_work / "summary.csv"

    use_original = policy.policy_name == f"{spec.model_id}_baseline" and spec.baseline_policy is not None and spec.patch_style != "nk_cw09"
    mod_path.write_text(patch_model_text(spec, template_text, policy, use_original), encoding="utf-8")
    runner_path.write_text(build_runner_script(spec, mod_path, summary_path, policy), encoding="utf-8")

    cmd = [MATLAB_BIN, "-batch", f"run('{matlab_quote(runner_path)}')"]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=MATLAB_TIMEOUT_SEC,
    )
    cleanup_generated_package_dirs(model_work)
    status = {
        "model_id": spec.model_id,
        "policy_name": policy.policy_name,
        "returncode": completed.returncode,
        "runner_path": str(runner_path),
        "summary_path": str(summary_path),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "status": "ok" if completed.returncode == 0 and summary_path.exists() else "failed",
    }
    if status["status"] != "ok":
        cached = cached_summary_frame(spec, policy, model_work)
        if cached is not None:
            status["status"] = "cached"
            return status, cached
        return status, None
    return status, pd.read_csv(summary_path)


def failure_placeholder(spec: ModelSpec, policy: PolicySpec) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policy_name": policy.policy_name,
                "model_id": spec.model_id,
                "rule_family": policy.rule_family,
                "inflation_coeff": policy.inflation_gap,
                "output_coeff": policy.output_gap,
                "lagged_rate_coeff": policy.lagged_policy_rate_gap,
                "total_discounted_loss": np.inf,
                "mean_period_loss": np.inf,
                "total_discounted_revealed_loss": np.inf,
                "mean_period_revealed_loss": np.inf,
                "mean_sq_inflation_gap": np.nan,
                "mean_sq_output_gap": np.nan,
                "mean_sq_rate_change": np.nan,
                "mean_policy_rate": np.nan,
                "std_policy_rate": np.nan,
                "solver_status": "failed",
            }
        ]
    )


def supported_policies_for_model(spec: ModelSpec, policies: list[PolicySpec]) -> list[PolicySpec]:
    rows = policies.copy()
    if spec.baseline_policy is not None:
        rows = [spec.baseline_policy] + rows
    return rows


def summarise_model(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    baseline_name = f"{spec.model_id}_baseline"
    baseline_loss = float(df.loc[df["policy_name"] == baseline_name, "total_discounted_loss"].iloc[0])
    baseline_revealed_loss = float(df.loc[df["policy_name"] == baseline_name, "total_discounted_revealed_loss"].iloc[0])
    df = df.sort_values("total_discounted_loss").reset_index(drop=True)
    df[f"improvement_vs_{baseline_name}_pct"] = (baseline_loss - df["total_discounted_loss"]) / baseline_loss * 100.0
    df[f"improvement_vs_{baseline_name}_revealed_pct"] = (
        (baseline_revealed_loss - df["total_discounted_revealed_loss"]) / baseline_revealed_loss * 100.0
    )
    return df


def write_markdown(combined: pd.DataFrame, runtime_df: pd.DataFrame) -> None:
    lines = [
        "# Phase 10 MMB External Robustness",
        "",
        "- 口径说明：这些 `Dynare/MMB` 外部模型只能接 simple rule，因此所有 RL 规则在此处统一用其经验环境回归出的线性反馈系数代理；`benchmark transfer`、`empirical direct-trained`、`revealed direct-trained` 仍严格分开报告。",
        "- `pyfrbus` 仍保留原先可调用版本；本文件只汇总新增的 `Dynare/MMB` 批次。",
        "- `Lucas critique` 仍成立：这些结果是固定反馈规则跨模型稳健性，不等于结构最优政策比较。",
        "",
    ]
    for model_id, frame in combined.groupby("model_id", sort=False):
        lines.extend(
            [
                f"## {model_id}",
                "",
                frame.round(6).to_markdown(index=False),
                "",
            ]
        )
    lines.extend(
        [
            "## Runtime Status",
            "",
            runtime_df.to_markdown(index=False),
            "",
        ]
    )
    (OUTPUT_DIR / "mmb_external_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    policies = load_policy_specs()

    all_status_rows: list[dict[str, object]] = []
    all_frames: list[pd.DataFrame] = []

    for spec in MODEL_SPECS:
        template_path = MMB_ROOT / spec.relative_dir / spec.template_file
        template_text = read_text_any(template_path)
        model_frames: list[pd.DataFrame] = []
        for policy in supported_policies_for_model(spec, policies):
            status, frame = run_model_policy(spec, policy, template_text)
            all_status_rows.append(status)
            if frame is not None:
                frame["solver_status"] = "ok"
                model_frames.append(frame)
            else:
                model_frames.append(failure_placeholder(spec, policy))
        if model_frames:
            model_df = pd.concat(model_frames, ignore_index=True)
            model_df = summarise_model(model_df, spec)
            model_df.to_csv(OUTPUT_DIR / f"{spec.model_id.lower()}_summary.csv", index=False)
            all_frames.append(model_df)

    runtime_df = pd.DataFrame(all_status_rows)
    runtime_df.to_csv(OUTPUT_DIR / "mmb_runtime_status.csv", index=False)

    if not all_frames:
        raise RuntimeError("No MMB external model evaluation succeeded.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "mmb_summary.csv", index=False)

    pyfrbus_path = OUTPUT_DIR / "pyfrbus_summary.csv"
    if pyfrbus_path.exists():
        pyfrbus = pd.read_csv(pyfrbus_path)
        all_external = pd.concat([pyfrbus, combined], ignore_index=True, sort=False)
        all_external.to_csv(OUTPUT_DIR / "all_external_summary.csv", index=False)

    write_markdown(combined, runtime_df)


if __name__ == "__main__":
    main()
