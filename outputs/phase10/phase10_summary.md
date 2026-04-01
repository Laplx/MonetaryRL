# Phase 10 Summary

## Direct-Trained Empirical RL

| policy_name     | training_env   | algo   | policy_parameterization   |   mean_discounted_loss |   clip_rate |   explosion_rate |
|:----------------|:---------------|:-------|:--------------------------|-----------------------:|------------:|-----------------:|
| sac_svar_direct | svar           | sac    | standard_nonlinear        |                89.1856 |           0 |                0 |
| td3_svar_direct | svar           | td3    | standard_nonlinear        |                96.7765 |           0 |                0 |
| ppo_svar_direct | svar           | ppo    | linear_policy             |               107.215  |           0 |                0 |
| td3_ann_direct  | ann            | td3    | standard_nonlinear        |                64.1953 |           0 |                0 |
| ppo_ann_direct  | ann            | ppo    | linear_policy             |                64.3898 |           0 |                0 |
| sac_ann_direct  | ann            | sac    | standard_nonlinear        |                74.4977 |           0 |                0 |

## PPO Variants

| policy_name               | training_env   | policy_parameterization   |   mean_discounted_loss |   clip_rate |   explosion_rate |
|:--------------------------|:---------------|:--------------------------|-----------------------:|------------:|-----------------:|
| ppo_ann_direct_linear     | ann            | linear_policy             |                64.3898 |           0 |                0 |
| ppo_svar_direct_nonlinear | svar           | nonlinear_policy          |                91.9932 |           0 |                0 |
| ppo_svar_direct_linear    | svar           | linear_policy             |               107.215  |           0 |                0 |
| ppo_ann_direct_nonlinear  | ann            | nonlinear_policy          |               134.862  |           0 |                0 |

## Preferred Rule Bundle For External Interface

| evaluation_env   | rule_family        | policy_name            | source_env   | policy_parameterization   |   total_discounted_loss |
|:-----------------|:-------------------|:-----------------------|:-------------|:--------------------------|------------------------:|
| ann              | ann_direct         | ppo_ann_direct         | ann          | linear_policy             |                 69.2297 |
| ann              | benchmark_transfer | td3_benchmark_transfer | benchmark    | linear_surrogate          |                155.226  |
| ann              | svar_direct        | td3_svar_direct        | svar         | standard_nonlinear        |                111.444  |
| svar             | ann_direct         | ppo_ann_direct         | ann          | linear_policy             |                150.19   |
| svar             | benchmark_transfer | sac_benchmark_transfer | benchmark    | linear_surrogate          |                114.455  |
| svar             | svar_direct        | sac_svar_direct        | svar         | standard_nonlinear        |                 92.1016 |

## Cross-Transfer Snapshot

| policy_name                   | rule_family        | source_env   | evaluation_env   |   mean_discounted_loss |   clip_rate |   explosion_rate |
|:------------------------------|:-------------------|:-------------|:-----------------|-----------------------:|------------:|-----------------:|
| td3_svar_direct               | svar_direct        | svar         | ann              |                139.699 |    0        |                0 |
| td3_benchmark_transfer        | benchmark_transfer | benchmark    | ann              |                143.358 |    0.040278 |                0 |
| sac_benchmark_transfer        | benchmark_transfer | benchmark    | ann              |                156.729 |    0.027778 |                0 |
| linear_policy_search_transfer | benchmark_transfer | benchmark    | ann              |                181.761 |    0.013628 |                0 |
| sac_svar_direct               | svar_direct        | svar         | ann              |                191.385 |    0        |                0 |
| ppo_svar_direct               | svar_direct        | svar         | ann              |                200.37  |    0        |                0 |
| ppo_benchmark_transfer        | benchmark_transfer | benchmark    | ann              |                327.935 |    8.7e-05  |                0 |
| sac_benchmark_transfer        | benchmark_transfer | benchmark    | svar             |                131.149 |    0.044878 |                0 |
| linear_policy_search_transfer | benchmark_transfer | benchmark    | svar             |                149.463 |    0.035156 |                0 |
| td3_benchmark_transfer        | benchmark_transfer | benchmark    | svar             |                150.658 |    0.089323 |                0 |
| ppo_ann_direct                | ann_direct         | ann          | svar             |                206.105 |    0        |                0 |
| ppo_benchmark_transfer        | benchmark_transfer | benchmark    | svar             |                228.303 |    0.00191  |                0 |
| td3_ann_direct                | ann_direct         | ann          | svar             |                315.011 |    0        |                0 |
| sac_ann_direct                | ann_direct         | ann          | svar             |                490.104 |    0        |                0 |

## Revealed Welfare Weights

|   inflation_weight |   output_gap_weight |   rate_smoothing_weight |   objective | success   |   implied_phi_pi |   implied_phi_x |   implied_phi_i |   target_phi_pi |   target_phi_x |   target_phi_i |
|-------------------:|--------------------:|------------------------:|------------:|:----------|-----------------:|----------------:|----------------:|----------------:|---------------:|---------------:|
|                  1 |            0.881713 |                 20.0855 |    0.830222 | True      |        -0.013953 |       -0.040073 |        0.981491 |        0.193173 |       0.265769 |         0.8692 |

## PyFRBUS External Results

| policy_name              | model_id   |   total_discounted_loss |   mean_period_loss |   mean_sq_inflation_gap |   mean_sq_output_gap |   mean_sq_rate_change |   mean_policy_rate |   std_policy_rate |   improvement_vs_pyfrbus_baseline_pct |
|:-------------------------|:-----------|------------------------:|-------------------:|------------------------:|---------------------:|----------------------:|-------------------:|------------------:|--------------------------------------:|
| pyfrbus_baseline         | pyfrbus    |                0.014138 |           0.000647 |                0.00039  |             0.000515 |              0        |            3.00003 |          2.4e-05  |                           0           |
| sac_svar_revealed_direct | pyfrbus    |                0.027163 |           0.001276 |                0.00053  |             0.001476 |              8.1e-05  |            3.00463 |          0.043821 |                         -92.1257      |
| empirical_taylor_rule    | pyfrbus    |                0.80597  |           0.038684 |                0.001614 |             0.073919 |              0.001105 |            3.10403 |          0.113153 |                       -5600.71        |
| sac_benchmark_transfer   | pyfrbus    |                1.18525  |           0.056032 |                9.7e-05  |             0.110219 |              0.008256 |            2.87521 |          0.126559 |                       -8283.36        |
| riccati_reference        | pyfrbus    |                2.09249  |           0.09879  |                0.000274 |             0.193747 |              0.016425 |            2.83452 |          0.181575 |                      -14700.4         |
| td3_benchmark_transfer   | pyfrbus    |                2.48128  |           0.117058 |                0.000372 |             0.228904 |              0.02234  |            2.81984 |          0.204532 |                      -17450.4         |
| sac_svar_direct          | pyfrbus    |               12.5555   |           0.582159 |                0.010681 |             1.04344  |              0.497603 |            3.41147 |          0.829165 |                      -88706.1         |
| td3_svar_revealed_direct | pyfrbus    |               22.0892   |           1.06155  |                0.008129 |             2.09687  |              0.049888 |            2.4335  |          0.374777 |                     -156139           |
| td3_svar_direct          | pyfrbus    |               24.1411   |           1.14707  |                0.018991 |             2.20016  |              0.280054 |            3.65251 |          0.58423  |                     -170652           |
| td3_ann_revealed_direct  | pyfrbus    |               26.3842   |           1.25496  |                0.019294 |             2.45944  |              0.059479 |            3.40972 |          0.969153 |                     -186518           |
| ppo_ann_direct           | pyfrbus    |               38.7951   |           1.87938  |                0.028058 |             3.67652  |              0.130598 |            3.94736 |          0.361957 |                     -274302           |
| sac_ann_revealed_direct  | pyfrbus    |               91.1481   |           4.5302   |                0.054495 |             8.94435  |              0.035337 |            4.71209 |          0.572325 |                     -644600           |
| ppo_svar_revealed_direct | pyfrbus    |              673.659    |          32.9264   |                0.472159 |            64.7644   |              0.719735 |            8.61419 |          0.555495 |                          -4.76476e+06 |
| ppo_ann_revealed_direct  | pyfrbus    |              795.656    |          39.0196   |                0.56682  |            76.7772   |              0.641535 |            9.52754 |          0.794862 |                          -5.62766e+06 |

## External Runtime Status

| component              | status               | detail                                                                                       |
|:-----------------------|:---------------------|:---------------------------------------------------------------------------------------------|
| dynare_preprocessor    | available            | C:\dynare\7.0\matlab\preprocessor64\dynare_m.exe                                             |
| matlab_runtime_for_mmb | license_check_failed | Current shell still reports MATLAB license error -9/57; MMB numerical solve remains blocked. |

## External Model Inventory

| model_id   | priority_group   | runner_family   | available_locally   | interface_status   | candidate_inflation_vars                                                                                  | candidate_output_gap_vars                                           | candidate_policy_rate_vars                                                                            |
|:-----------|:-----------------|:----------------|:--------------------|:-------------------|:----------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------|
| pyfrbus    | priority         | pyfrbus         | True                | mapping_stub_ready | capital, cpi, dmpintay, dmptpi, dmptpi_aerr, empirical, episode, fpi10                                    | dpgap, dpgap_aerr, fxgap, fxgap_aerr, gap, gaps, xgap, xgap2        | cointegrating, concentrated, constraint, constraints, corporate, delrff, delrff_aerr, dmpintay        |
| US_FRB03   | priority         | dynare_mmb      | True                | mapping_stub_ready | inflation, inflationq, lagpi1, lagpi2, lagpi3, lagpi4, leadpi0, leadpi1                                   | lzgapc1, lzgapc2, lzgapc3, outputgap, xgap, xgap2, xgap2f2, xgap2f4 | drffe, generated, gfintn1, gfintnr, gsintn1, gsintn2, gsintnr, interest                               |
| US_SW07    | priority         | dynare_mmb      | True                | mapping_stub_ready | capital, constepinf, cpie, crdpi, crhopinf, crpi, epinf, epinfma                                          | gap                                                                 | constraint, ffr, interest, intertemporal, noprint, rate                                               |
| US_CCTW10  | priority         | dynare_mmb      | True                | mapping_stub_ready | capital, constepinf, cpie, crdpi, crhopinf, crpi, epinf, epinfma                                          | gap, outputgap                                                      | cofintinf0, cofintinfb1, cofintinfb2, cofintinfb3, cofintinfb4, cofintinff1, cofintinff2, cofintinff3 |
| US_CPS10   | priority         | dynare_mmb      | True                | mapping_stub_ready | housekeeping, inflation, inflgap, pit, pits, rhopit, sdpit                                                | gap, inflgap, outpgap                                               | balint, calibrated, interest, rate                                                                    |
| US_KS15    | priority         | dynare_mmb      | True                | mapping_stub_ready | inflation, pit, rho_pi                                                                                    |                                                                     | interest, rate                                                                                        |
| US_RA07    | priority         | dynare_mmb      | True                | mapping_stub_ready | capital, inflation, pi, pi_epsa, pi_epsa_baselineestimation, pi_epsg, pi_epsg_baselineestimation, pi_epsp | omegap                                                              | constraint, interest, rate                                                                            |
| NK_CW09    | fallback         | dynare_mmb      | True                | mapping_stub_ready | inflation, inflationq, phi_pi, pi, pi_b, pi_bar, pi_hat, pi_hat_a                                         | outputgap                                                           | cofintinf0, cofintinfb1, cofintinfb2, cofintinfb3, cofintinfb4, cofintinff1, cofintinff2, cofintinff3 |
| NK_CFP10   | fallback         | dynare_mmb      | True                | mapping_stub_ready | capital, eps_pi, eta_pi, inflation, pi, rho_pi                                                            | gap                                                                 | constraint, interest, noprint, rate                                                                   |
| NK_GLSV07  | fallback         | dynare_mmb      | True                | mapping_stub_ready | capital, inflation, phi_pi, pi                                                                            |                                                                     | generated, noprint, rate                                                                              |
| NK_GK13    | fallback         | dynare_mmb      | True                | mapping_stub_ready | capital, infl, infl_ss, inflstar, inflstar_ss, inflstarf, kappa_pi, kappa_pi_ex                           |                                                                     | constraint, corporate, interest, intermediaries, intermediary, intermediate, international, rate      |

## Notes

- `pyfrbus` 已实际跑通；当前外部数值结果先来自 `pyfrbus` 闭环固定点评估。
- MMB/Dynare 模型文件已确认在本地，但当前 shell 下 MATLAB 许可证仍报错，因此未能完成数值求解阶段。
- `Phase 8/9` 仍是 benchmark-transfer baseline，本轮新增的核心是 `SVAR direct` 与 `ANN direct` 的外部接口延伸。
- Lucas critique 仍是经验环境与外部模型迁移时必须明确的边界。

## MMB External Addendum

| model_id | 最佳可解规则 | 结果摘要 |
|---|---|---|
| `US_SW07` | `US_SW07_baseline` | 外部 baseline 最稳；可解规则里 `sac_svar_direct` 最好，`PPO` 与多数 `ANN/revealed direct` 在该模型下直接失稳或不可解 |
| `US_CCTW10` | `td3_svar_direct` | `SVAR direct` 明显最强，优于 `benchmark transfer`、经验 Taylor 与模型 baseline；多数组 `PPO/ANN/revealed direct` 不可解 |
| `US_KS15` | `td3_svar_direct` | `SVAR direct` 与 `benchmark transfer` 都能击败模型 baseline；`ANN direct` 大多不可解，`revealed direct` 仅少数可解且排序靠后 |
| `NK_CW09` | `sac_benchmark_transfer` | `benchmark transfer` 与 `SVAR direct` 都优于该 NK baseline；`revealed direct` 只有 `sac_svar_revealed_direct` 可解且表现较弱 |

- 新增外部模型批次输出位于 `outputs/phase10/external_model_robustness/mmb_summary.csv` 与 `outputs/phase10/external_model_robustness/mmb_external_summary.md`。
- `Dynare/MMB` 接口统一使用规则在线性 simple-rule 空间中的 surrogate 系数；这不改变 `benchmark transfer`、`empirical direct-trained`、`revealed direct-trained` 的分组口径。
- `US_FRB03`、`US_CPS10`、`US_RA07` 仍受 legacy `Dynare/MMB` 兼容问题阻塞，不再是 MATLAB 许可证本身。
