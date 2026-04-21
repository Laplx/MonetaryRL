from __future__ import annotations

import textwrap
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import phase10_external_mmb_eval as p10


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "phase16"
WORK_ROOT = OUTPUT_DIR / "mmb_candidate_work"
PHASE10_EXTERNAL_DIR = ROOT / "outputs" / "phase10" / "external_model_robustness"

CORE_POLICY_NAMES = [
    "empirical_taylor_rule",
    "td3_benchmark_transfer",
    "sac_benchmark_transfer",
    "td3_svar_direct",
    "sac_svar_direct",
    "sac_svar_revealed_direct",
]


@dataclass(frozen=True)
class CandidateInfo:
    priority: int
    model_id: str
    relative_dir: str
    template_file: str
    patch_style: str
    summary_code: str
    rule_form: str
    expected_fit: str
    rationale: str

    def to_model_spec(self) -> p10.ModelSpec:
        return p10.ModelSpec(
            model_id=self.model_id,
            relative_dir=self.relative_dir,
            template_file=self.template_file,
            patch_style=self.patch_style,
            summary_code=self.summary_code,
            baseline_policy=p10.PolicySpec(
                policy_name=f"{self.model_id}_baseline",
                rule_family="external_baseline",
                source_env="external",
                training_env="external",
                algo="baseline",
                inflation_gap=0.0,
                output_gap=0.0,
                lagged_policy_rate_gap=0.0,
            ),
        )


CANDIDATES = [
    CandidateInfo(
        priority=1,
        model_id="US_CPS10",
        relative_dir="US_CPS10/US_CPS10_rep",
        template_file="US_CPS10_rep1.mod",
        patch_style="us_cps10",
        summary_code="""
names = string(strtrim(cellstr(M_.endo_names)));
sim = oo_.endo_simul;
sim = sim(:, max(1, size(sim, 2) - 2499):size(sim, 2));
loc = @(n) find(names == n, 1);
rate_series = 4 * sim(loc("R"), :);
inflation_gap = sim(loc("inflgap"), :);
output_gap = sim(loc("outpgap"), :);
""",
        rule_form="explicit smooth Taylor rule on inflgap and outpgap",
        expected_fit="high",
        rationale="直接以通胀缺口和产出缺口入规则，且带利率平滑，最接近本文 simple-rule 代理设定。",
    ),
    CandidateInfo(
        priority=2,
        model_id="US_RA07",
        relative_dir="US_RA07/replication_code",
        template_file="replication_code.mod",
        patch_style="us_ra07",
        summary_code="""
names = string(strtrim(cellstr(M_.endo_names)));
sim = oo_.endo_simul;
sim = sim(:, max(1, size(sim, 2) - 2499):size(sim, 2));
loc = @(n) find(names == n, 1);
rate_series = 4 * sim(loc("r"), :);
inflation_gap = 4 * sim(loc("pi"), :);
output_gap = sim(loc("y"), :);
""",
        rule_form="explicit Taylor rule with lagged rate, inflation, and output",
        expected_fit="high",
        rationale="规则结构就是三参数泰勒式，替换成本低；若结果为正，可作为额外 U.S. 模型支撑。",
    ),
    CandidateInfo(
        priority=3,
        model_id="NK_CFP10",
        relative_dir="NK_CFP10/NK_CFP10_rep",
        template_file="NK_CFP10_rep.mod",
        patch_style="nk_cfp10",
        summary_code="""
names = string(strtrim(cellstr(M_.endo_names)));
sim = oo_.endo_simul;
sim = sim(:, max(1, size(sim, 2) - 2499):size(sim, 2));
loc = @(n) find(names == n, 1);
rate_series = 4 * sim(loc("R"), :);
inflation_gap = 4 * sim(loc("pi"), :);
output_gap = sim(loc("yg"), :);
""",
        rule_form="explicit Taylor rule on pi and output gap yg",
        expected_fit="medium-high",
        rationale="有显式 output gap 变量 `yg`，与 `NK_CW09` 同属 NK 类 simple-rule 外部检验。",
    ),
    CandidateInfo(
        priority=4,
        model_id="US_FRB03",
        relative_dir="US_FRB03/US_FRB03_rep",
        template_file="US_FRB03_rep.mod",
        patch_style="us_frb03",
        summary_code="""
names = string(strtrim(cellstr(M_.endo_names)));
sim = oo_.endo_simul;
sim = sim(:, max(1, size(sim, 2) - 2499):size(sim, 2));
loc = @(n) find(names == n, 1);
rate_series = sim(loc("interest"), :);
inflation_gap = sim(loc("inflationq"), :);
output_gap = sim(loc("outputgap"), :);
""",
        rule_form="FRB-family rule on interest, inflationq, and outputgap",
        expected_fit="medium",
        rationale="与 `pyfrbus/US_CCTW10` 同属 FRB-family 结构，若可稳定 patch，论文叙事最自然。",
    ),
]


def selected_policy_specs() -> list[p10.PolicySpec]:
    specs = p10.load_policy_specs()
    keep = [spec for spec in specs if spec.policy_name in CORE_POLICY_NAMES]
    order = {name: idx for idx, name in enumerate(CORE_POLICY_NAMES)}
    return sorted(keep, key=lambda item: order[item.policy_name])


def patch_candidate_text(spec: CandidateInfo, template: str, policy: p10.PolicySpec, use_original: bool) -> str:
    text = template
    if spec.patch_style == "us_cps10":
        if not use_original:
            rule = (
                f"% eq 5, MP rule\n"
                f"R = {policy.lagged_policy_rate_gap:.12f}*R(-1) + "
                f"{policy.inflation_gap:.12f}*inflgap + {policy.output_gap:.12f}*outpgap + Rs;"
            )
            text = p10.replace_once(
                text,
                r"% eq 5, MP rule[\s\S]*?% eq 6 - 9, exogenous shocks",
                rule + "\n\n% eq 6 - 9, exogenous shocks",
            )
        text = p10.replace_once(
            text,
            r"stoch_simul\(order=1,irf=17,solve_algo=1\) inflgap realR;",
            f"stoch_simul(order=1, periods={p10.SIM_PERIODS}, irf=0, nograph, noprint) inflgap outpgap R;",
        )
        return text

    if spec.patch_style == "us_ra07":
        if not use_original:
            text = p10.replace_once(
                text,
                r"r=rhor\*r\(-1\)\+\(1-rhor\)\*gammap\*pi\+\(1-rhor\)\*gammay\*y\+epsz;",
                f"r={policy.lagged_policy_rate_gap:.12f}*r(-1)+{policy.inflation_gap:.12f}*pi+{policy.output_gap:.12f}*y+epsz;",
            )
        text = p10.replace_once(
            text,
            r"stoch_simul\(irf = 25,nograph\);",
            f"stoch_simul(order=1, periods={p10.SIM_PERIODS}, irf=0, nograph, noprint) r pi y;",
        )
        return text

    if spec.patch_style == "nk_cfp10":
        if not use_original:
            text = p10.replace_once(
                text,
                r"R = tau\*pi \+ tau_g\*yg \+ eps_R;",
                f"R = {policy.lagged_policy_rate_gap:.12f}*R(-1) + {policy.inflation_gap:.12f}*pi + {policy.output_gap:.12f}*yg + eps_R;",
            )
        text = p10.replace_once(
            text,
            r"stoch_simul\(order=1,irf=21, noprint\) pi R y;",
            f"stoch_simul(order=1, periods={p10.SIM_PERIODS}, irf=0, nograph, noprint) pi R yg;",
        )
        return text

    if spec.patch_style == "us_frb03":
        if not use_original:
            rule = textwrap.dedent(
                f"""
                // Monetary Policy Rule

                interest = {policy.lagged_policy_rate_gap:.12f}*interest(-1)
                + {policy.inflation_gap:.12f}*inflationq
                + {policy.output_gap:.12f}*outputgap
                + interest_;
                """
            ).strip()
            text = p10.replace_once(
                text,
                r"// Monetary Policy Rule[\s\S]*?// Original Model Code:",
                rule + "\n\n\n// Original Model Code:",
            )
        text = p10.replace_once(
            text,
            r"stoch_simul \(irf =16, nograph, noprint\) interest inflationq outputgap;",
            f"stoch_simul(order=1, periods={p10.SIM_PERIODS}, irf=0, nograph, noprint) interest inflationq outputgap;",
        )
        return text

    raise ValueError(f"Unsupported patch style: {spec.patch_style}")


def matlab_blocked() -> bool:
    try:
        completed = subprocess.run(
            [p10.MATLAB_BIN, "-batch", "disp('matlab_ok'); exit"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except Exception:
        return True
    return completed.returncode != 0


def packaged_results_exists(spec: CandidateInfo) -> bool:
    target = p10.MMB_ROOT / spec.relative_dir
    return any(target.rglob("*_results.mat"))


def confirmed_positive_models() -> pd.DataFrame:
    mmb = pd.read_csv(PHASE10_EXTERNAL_DIR / "mmb_summary.csv")
    rl = mmb[mmb["policy_name"].str.contains("ppo|td3|sac", case=False, na=False)].copy()
    rows: list[dict[str, object]] = []
    for model_id, frame in rl.groupby("model_id", sort=False):
        improvement_col = f"improvement_vs_{model_id}_baseline_revealed_pct"
        work = frame.dropna(subset=[improvement_col]).copy()
        if work.empty:
            continue
        best = work.sort_values(improvement_col, ascending=False).iloc[0]
        rows.append(
            {
                "model_id": model_id,
                "best_rl_policy": best["policy_name"],
                "best_rl_rule_family": best["rule_family"],
                "best_revealed_improvement_pct": float(best[improvement_col]),
                "is_positive": float(best[improvement_col]) > 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["is_positive", "best_revealed_improvement_pct"], ascending=[False, False])


def prepare_assets() -> pd.DataFrame:
    policies = selected_policy_specs()
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        spec = candidate.to_model_spec()
        template_path = p10.MMB_ROOT / spec.relative_dir / spec.template_file
        template_text = p10.read_text_any(template_path)
        policy_grid = [spec.baseline_policy] + policies if spec.baseline_policy is not None else policies
        for policy in policy_grid:
            workdir = WORK_ROOT / spec.model_id / policy.policy_name
            workdir.mkdir(parents=True, exist_ok=True)
            patched_text = patch_candidate_text(candidate, template_text, policy, use_original=policy.policy_name == f"{spec.model_id}_baseline")
            patched_path = workdir / spec.template_file
            patched_path.write_text(patched_text, encoding="utf-8")
            summary_path = workdir / "summary.csv"
            runner_text = p10.build_runner_script(spec, patched_path, summary_path, policy)
            runner_path = workdir / f"run_{spec.model_id}_{policy.policy_name}.m"
            runner_path.write_text(runner_text, encoding="utf-8")
            rows.append(
                {
                    "priority": candidate.priority,
                    "model_id": spec.model_id,
                    "policy_name": policy.policy_name,
                    "rule_family": policy.rule_family,
                    "inflation_coeff": policy.inflation_gap,
                    "output_coeff": policy.output_gap,
                    "lagged_rate_coeff": policy.lagged_policy_rate_gap,
                    "runner_path": str(runner_path.relative_to(ROOT)),
                    "patched_model_path": str(patched_path.relative_to(ROOT)),
                    "summary_path": str(summary_path.relative_to(ROOT)),
                    "prepared_only": True,
                }
            )
    return pd.DataFrame(rows)


def run_candidate_policy(
    candidate: CandidateInfo,
    spec: p10.ModelSpec,
    policy: p10.PolicySpec,
    template_text: str,
) -> tuple[dict[str, object], pd.DataFrame | None]:
    model_work = WORK_ROOT / spec.model_id / policy.policy_name
    model_work.mkdir(parents=True, exist_ok=True)
    mod_stem = f"{spec.model_id}_{policy.policy_name}"
    mod_path = model_work / f"{mod_stem}.mod"
    runner_path = model_work / f"run_{mod_stem}.m"
    summary_path = model_work / "summary.csv"

    use_original = policy.policy_name == f"{spec.model_id}_baseline" and spec.baseline_policy is not None
    if not summary_path.exists():
        mod_path.write_text(patch_candidate_text(candidate, template_text, policy, use_original), encoding="utf-8")
        runner_path.write_text(p10.build_runner_script(spec, mod_path, summary_path, policy), encoding="utf-8")
        completed = subprocess.run(
            [p10.MATLAB_BIN, "-batch", f"run('{p10.matlab_quote(runner_path)}')"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=p10.MATLAB_TIMEOUT_SEC,
        )
        p10.cleanup_generated_package_dirs(model_work)
    else:
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="cached summary.csv", stderr="")

    status = {
        "model_id": spec.model_id,
        "policy_name": policy.policy_name,
        "returncode": completed.returncode,
        "runner_path": str(runner_path),
        "summary_path": str(summary_path),
        "stdout_tail": completed.stdout[-4000:] if completed.stdout else "",
        "stderr_tail": completed.stderr[-4000:] if completed.stderr else "",
        "status": "ok" if completed.returncode == 0 and summary_path.exists() else "failed",
    }
    if status["status"] == "ok":
        return status, pd.read_csv(summary_path)
    return status, None


def run_candidate_models() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policies = selected_policy_specs()
    all_status_rows: list[dict[str, object]] = []
    all_frames: list[pd.DataFrame] = []

    for candidate in CANDIDATES:
        spec = candidate.to_model_spec()
        template_path = p10.MMB_ROOT / spec.relative_dir / spec.template_file
        template_text = p10.read_text_any(template_path)
        model_frames: list[pd.DataFrame] = []
        policy_grid = [spec.baseline_policy] + policies if spec.baseline_policy is not None else policies
        for policy in policy_grid:
            status, frame = run_candidate_policy(candidate, spec, policy, template_text)
            all_status_rows.append(status)
            if frame is not None:
                frame["solver_status"] = "ok"
                model_frames.append(frame)
            else:
                model_frames.append(p10.failure_placeholder(spec, policy))
        if model_frames:
            model_df = pd.concat(model_frames, ignore_index=True)
            model_df = p10.summarise_model(model_df, spec)
            model_df.to_csv(OUTPUT_DIR / f"{spec.model_id.lower()}_summary.csv", index=False)
            all_frames.append(model_df)

    runtime_df = pd.DataFrame(all_status_rows)
    runtime_df.to_csv(OUTPUT_DIR / "phase16_mmb_runtime_status.csv", index=False)
    combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    if not combined.empty:
        combined.to_csv(OUTPUT_DIR / "phase16_mmb_summary.csv", index=False)
    best = best_rl_by_model(combined) if not combined.empty else pd.DataFrame()
    if not best.empty:
        best.to_csv(OUTPUT_DIR / "phase16_best_rl_by_model.csv", index=False)
    return combined, runtime_df, best


def best_rl_by_model(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()
    rl = combined[combined["policy_name"].str.contains("ppo|td3|sac", case=False, na=False)].copy()
    rows: list[dict[str, object]] = []
    for model_id, frame in rl.groupby("model_id", sort=False):
        improvement_col = f"improvement_vs_{model_id}_baseline_revealed_pct"
        work = frame.dropna(subset=[improvement_col]).copy()
        work = work[work["solver_status"] == "ok"].copy()
        if work.empty:
            continue
        best = work.sort_values(improvement_col, ascending=False).iloc[0]
        rows.append(
            {
                "model_id": model_id,
                "best_rl_policy": best["policy_name"],
                "best_rl_rule_family": best["rule_family"],
                "total_discounted_revealed_loss": float(best["total_discounted_revealed_loss"]),
                "best_revealed_improvement_pct": float(best[improvement_col]),
            }
        )
    return pd.DataFrame(rows).sort_values("best_revealed_improvement_pct", ascending=False)


def candidate_table() -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        rows.append(
            {
                "priority": candidate.priority,
                "model_id": candidate.model_id,
                "patch_style": candidate.patch_style,
                "rule_form": candidate.rule_form,
                "expected_fit": candidate.expected_fit,
                "packaged_results_mat": packaged_results_exists(candidate),
                "rationale": candidate.rationale,
            }
        )
    return pd.DataFrame(rows).sort_values("priority").reset_index(drop=True)


def write_summary(
    confirmed: pd.DataFrame,
    candidates: pd.DataFrame,
    prepared: pd.DataFrame,
    runtime_df: pd.DataFrame | None = None,
    best: pd.DataFrame | None = None,
) -> None:
    confirmed_md = confirmed.round(6).to_markdown(index=False) if not confirmed.empty else "_none_"
    candidates_md = candidates.to_markdown(index=False)
    best_md = best.round(6).to_markdown(index=False) if best is not None and not best.empty else "_none_"
    runtime_counts = (
        runtime_df["status"].value_counts().to_dict()
        if runtime_df is not None and not runtime_df.empty and "status" in runtime_df.columns
        else {}
    )
    blocked = matlab_blocked()
    policy_count_per_model = int(prepared.groupby("model_id")["policy_name"].nunique().max()) if not prepared.empty else 0
    content = "\n".join(
        [
            "# Phase 16 MMB 候选扩展",
            "",
            "## 目标",
            "",
            "- 在不利于正文的 `US_SW07` 之外，继续从 `MMB` 中寻找更适合写入论文第 6.2 节的外部模型候选。",
            "- 复用 `phase10` 的 external robustness 规则代理口径，只优先准备更可能得到正向 revealed 改进的模型。",
            "",
            "## 已确认的 phase10 MMB 结果",
            "",
            confirmed_md,
            "",
            "## Phase16 新候选",
            "",
            candidates_md,
            "",
            "## 执行状态",
            "",
            f"- MATLAB license blocked: `{blocked}`",
            f"- Prepared candidate models: `{len(candidates)}`",
            f"- Prepared policy variants per model: `{policy_count_per_model}`",
            f"- Prepared runner assets: `{len(prepared)}`",
            f"- Runtime status counts: `{runtime_counts}`",
            "",
            "## Phase16 新模型最优 RL",
            "",
            best_md,
            "",
            "## 当前结论",
            "",
            "- 现有可直接用于正文的正向 MMB 结果仍是 `US_CCTW10`、`US_KS15`、`NK_CW09`。",
            "- `phase16` 的新增模型用于替代或弱化 `US_SW07` 的负向叙事；是否进入正文取决于上表中 revealed 改进是否为正且 solver 是否稳定。",
            "- 若新模型结果为正，正文外部模型表建议优先报告 `US_CCTW10`、`US_KS15`、`NK_CW09` 与新增正向模型；`US_SW07` 可移入稳健性限制或附录。",
            "",
        ]
    )
    (OUTPUT_DIR / "phase16_summary.md").write_text(content, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    confirmed = confirmed_positive_models()
    candidates = candidate_table()
    prepared = prepare_assets()
    combined = pd.DataFrame()
    runtime_df = pd.DataFrame()
    best = pd.DataFrame()

    if not matlab_blocked():
        combined, runtime_df, best = run_candidate_models()

    confirmed.to_csv(OUTPUT_DIR / "phase16_confirmed_positive_mmb.csv", index=False)
    candidates.to_csv(OUTPUT_DIR / "phase16_candidate_models.csv", index=False)
    prepared.to_csv(OUTPUT_DIR / "phase16_prepared_runs.csv", index=False)
    write_summary(confirmed, candidates, prepared, runtime_df, best)


if __name__ == "__main__":
    main()
