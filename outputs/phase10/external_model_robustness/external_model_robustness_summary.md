# Phase 10 External Model Robustness Interface

## Preferred Rules

| evaluation_env   | rule_family        | policy_name            | source_env   | policy_parameterization   |   total_discounted_loss |
|:-----------------|:-------------------|:-----------------------|:-------------|:--------------------------|------------------------:|
| ann              | ann_direct         | ppo_ann_direct         | ann          | linear_policy             |                 69.2297 |
| ann              | benchmark_transfer | td3_benchmark_transfer | benchmark    | linear_surrogate          |                155.226  |
| ann              | svar_direct        | td3_svar_direct        | svar         | standard_nonlinear        |                111.444  |
| svar             | ann_direct         | ppo_ann_direct         | ann          | linear_policy             |                150.19   |
| svar             | benchmark_transfer | sac_benchmark_transfer | benchmark    | linear_surrogate          |                114.455  |
| svar             | svar_direct        | sac_svar_direct        | svar         | standard_nonlinear        |                 92.1016 |

## PyFRBUS Results

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

## PyFRBUS Fixed-Point Status

| policy_name              | model_id   | converged   |   iterations |   max_fixed_point_gap |   clip_rate | start   | end    |
|:-------------------------|:-----------|:------------|-------------:|----------------------:|------------:|:--------|:-------|
| empirical_taylor_rule    | pyfrbus    | True        |           13 |                     0 |           0 | 2040Q1  | 2045Q4 |
| ppo_ann_direct           | pyfrbus    | True        |           10 |                     0 |           0 | 2040Q1  | 2045Q4 |
| ppo_ann_revealed_direct  | pyfrbus    | True        |           11 |                     0 |           0 | 2040Q1  | 2045Q4 |
| ppo_svar_revealed_direct | pyfrbus    | True        |           10 |                     0 |           0 | 2040Q1  | 2045Q4 |
| pyfrbus_baseline         | pyfrbus    | True        |            0 |                     0 |           0 | 2040Q1  | 2045Q4 |
| riccati_reference        | pyfrbus    | True        |           18 |                     0 |           0 | 2040Q1  | 2045Q4 |
| sac_ann_revealed_direct  | pyfrbus    | True        |           12 |                     0 |           0 | 2040Q1  | 2045Q4 |
| sac_benchmark_transfer   | pyfrbus    | True        |           16 |                     0 |           0 | 2040Q1  | 2045Q4 |
| sac_svar_direct          | pyfrbus    | False       |           20 |                     0 |           0 | 2040Q1  | 2045Q4 |
| sac_svar_revealed_direct | pyfrbus    | True        |           12 |                     0 |           0 | 2040Q1  | 2045Q4 |
| td3_ann_revealed_direct  | pyfrbus    | True        |           15 |                     0 |           0 | 2040Q1  | 2045Q4 |
| td3_benchmark_transfer   | pyfrbus    | True        |           18 |                     0 |           0 | 2040Q1  | 2045Q4 |
| td3_svar_direct          | pyfrbus    | False       |           20 |                     0 |           0 | 2040Q1  | 2045Q4 |
| td3_svar_revealed_direct | pyfrbus    | True        |           14 |                     0 |           0 | 2040Q1  | 2045Q4 |

## Runtime Status

| component              | status               | detail                                                                                       |
|:-----------------------|:---------------------|:---------------------------------------------------------------------------------------------|
| dynare_preprocessor    | available            | C:\dynare\7.0\matlab\preprocessor64\dynare_m.exe                                             |
| matlab_runtime_for_mmb | available_via_escalated | Current shell still reports license error `-9/57` but external user context can run MATLAB batch and complete selected MMB runs. |
| mmb_selected_models    | completed            | Completed external evaluation for `US_SW07`、`US_CCTW10`、`US_KS15`、`NK_CW09`.             |
| mmb_legacy_models      | legacy_incompatible  | `US_FRB03`、`US_CPS10`、`US_RA07` remain blocked by legacy Dynare/MMB compatibility issues. |

## Model Inventory

| model_id   | priority_group   | runner_family   | available_locally   | source_path                                                                                                                            | interface_status   | candidate_inflation_vars                                                                                  | candidate_output_gap_vars                                           | candidate_policy_rate_vars                                                                            |
|:-----------|:-----------------|:----------------|:--------------------|:---------------------------------------------------------------------------------------------------------------------------------------|:-------------------|:----------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------|
| pyfrbus    | priority         | pyfrbus         | True                | C:\Users\Laplace\Documents\Code\MonetaryRL\external_models\frbus_extracted\pyfrbus\models\model.xml                                    | mapping_stub_ready | capital, cpi, dmpintay, dmptpi, dmptpi_aerr, empirical, episode, fpi10                                    | dpgap, dpgap_aerr, fxgap, fxgap_aerr, gap, gaps, xgap, xgap2        | cointegrating, concentrated, constraint, constraints, corporate, delrff, delrff_aerr, dmpintay        |
| US_FRB03   | priority         | dynare_mmb      | True                | C:\Users\Laplace\Documents\Code\MonetaryRL\external_models\mmb_extracted\mmb-rep-master\US_FRB03\US_FRB03_rep\US_FRB03_rep.mod         | mapping_stub_ready | inflation, inflationq, lagpi1, lagpi2, lagpi3, lagpi4, leadpi0, leadpi1                                   | lzgapc1, lzgapc2, lzgapc3, outputgap, xgap, xgap2, xgap2f2, xgap2f4 | drffe, generated, gfintn1, gfintnr, gsintn1, gsintn2, gsintnr, interest                               |
| US_SW07    | priority         | dynare_mmb      | True                | C:\Users\Laplace\Documents\Code\MonetaryRL\external_models\mmb_extracted\mmb-rep-master\US_SW07\US_SW07_rep\US_SW07_rep.mod            | mapping_stub_ready | capital, constepinf, cpie, crdpi, crhopinf, crpi, epinf, epinfma                                          | gap                                                                 | constraint, ffr, interest, intertemporal, noprint, rate                                               |
| US_CCTW10  | priority         | dynare_mmb      | True                | C:\Users\Laplace\Documents\Code\MonetaryRL\external_models\mmb_extracted\mmb-rep-master\US_CCTW10\Code_CCTW_2010_JEDC\SW_US_fiscal.mod | mapping_stub_ready | capital, constepinf, cpie, crdpi, crhopinf, crpi, epinf, epinfma                                          | gap, outputgap                                                      | cofintinf0, cofintinfb1, cofintinfb2, cofintinfb3, cofintinfb4, cofintinff1, cofintinff2, cofintinff3 |
| US_CPS10   | priority         | dynare_mmb      | True                | C:\Users\Laplace\Documents\Code\MonetaryRL\external_models\mmb_extracted\mmb-rep-master\US_CPS10\US_CPS10_rep\US_CPS10_rep1.mod        | mapping_stub_ready | housekeeping, inflation, inflgap, pit, pits, rhopit, sdpit                                                | gap, inflgap, outpgap                                               | balint, calibrated, interest, rate                                                                    |
| US_KS15    | priority         | dynare_mmb      | True                | C:\Users\Laplace\Documents\Code\MonetaryRL\external_models\mmb_extracted\mmb-rep-master\US_KS15\US_KS15_replication\US_KS15_R3.mod     | mapping_stub_ready | inflation, pit, rho_pi                                                                                    |                                                                     | interest, rate                                                                                        |
| US_RA07    | priority         | dynare_mmb      | True                | C:\Users\Laplace\Documents\Code\MonetaryRL\external_models\mmb_extracted\mmb-rep-master\US_RA07\replication_code\replication_code.mod  | mapping_stub_ready | capital, inflation, pi, pi_epsa, pi_epsa_baselineestimation, pi_epsg, pi_epsg_baselineestimation, pi_epsp | omegap                                                              | constraint, interest, rate                                                                            |
| NK_CW09    | fallback         | dynare_mmb      | True                | C:\Users\Laplace\Documents\Code\MonetaryRL\external_models\mmb_extracted\mmb-rep-master\NK_CW09\NK_CW09.mod                            | mapping_stub_ready | inflation, inflationq, phi_pi, pi, pi_b, pi_bar, pi_hat, pi_hat_a                                         | outputgap                                                           | cofintinf0, cofintinfb1, cofintinfb2, cofintinfb3, cofintinfb4, cofintinff1, cofintinff2, cofintinff3 |
| NK_CFP10   | fallback         | dynare_mmb      | True                | C:\Users\Laplace\Documents\Code\MonetaryRL\external_models\mmb_extracted\mmb-rep-master\NK_CFP10\NK_CFP10_rep\NK_CFP10_rep.mod         | mapping_stub_ready | capital, eps_pi, eta_pi, inflation, pi, rho_pi                                                            | gap                                                                 | constraint, interest, noprint, rate                                                                   |
| NK_GLSV07  | fallback         | dynare_mmb      | True                | C:\Users\Laplace\Documents\Code\MonetaryRL\external_models\mmb_extracted\mmb-rep-master\NK_GLSV07\NK_GLSV07_rep\NK_GLSV07_iclm_rep.mod | mapping_stub_ready | capital, inflation, phi_pi, pi                                                                            |                                                                     | generated, noprint, rate                                                                              |
| NK_GK13    | fallback         | dynare_mmb      | True                | C:\Users\Laplace\Documents\Code\MonetaryRL\external_models\mmb_extracted\mmb-rep-master\NK_GK13\NK_GK13_rep\NK_GK13_rep.mod            | mapping_stub_ready | capital, infl, infl_ss, inflstar, inflstar_ss, inflstarf, kappa_pi, kappa_pi_ex                           |                                                                     | constraint, corporate, interest, intermediaries, intermediary, intermediate, international, rate      |

## Notes

- `pyfrbus` 已完成闭环固定点评估；新增的 `Dynare/MMB` 外部批次见 `outputs/phase10/external_model_robustness/mmb_external_summary.md` 与 `outputs/phase10/external_model_robustness/mmb_summary.csv`。
- `Dynare/MMB` 口径只能接 simple rule，因此所有 RL 规则在这些模型里统一用线性 surrogate 系数接入；`benchmark transfer`、`empirical direct-trained`、`revealed direct-trained` 仍严格分开汇报。
- `pyfrbus` 中使用的状态映射是 `picxfe - pitarg`、`xgap`、`rff(-1) - 2`，并对规则路径做固定点迭代。
