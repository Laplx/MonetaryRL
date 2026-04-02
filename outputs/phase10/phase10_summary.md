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

| evaluation_env   | rule_family          | policy_name               | source_env   | policy_parameterization   |   total_discounted_loss |
|:-----------------|:---------------------|:--------------------------|:-------------|:--------------------------|------------------------:|
| ann              | ann_direct           | ppo_ann_direct            | ann          | linear_policy             |                 69.2297 |
| ann              | ann_revealed_direct  | td3_ann_revealed_direct   | ann          | standard_nonlinear        |                114.154  |
| ann              | benchmark_transfer   | td3_benchmark_transfer    | benchmark    | linear_surrogate          |                155.226  |
| ann              | svar_direct          | td3_svar_direct           | svar         | standard_nonlinear        |                111.444  |
| ann              | svar_revealed_direct | sac_svar_revealed_direct  | svar         | standard_nonlinear        |                172.599  |
| svar             | ann_direct           | ppo_ann_direct_nonlinear  | ann          | nonlinear_policy          |                 87.3361 |
| svar             | ann_revealed_direct  | sac_ann_revealed_direct   | ann          | standard_nonlinear        |                110.788  |
| svar             | benchmark_transfer   | sac_benchmark_transfer    | benchmark    | linear_surrogate          |                114.455  |
| svar             | svar_direct          | ppo_svar_direct_nonlinear | svar         | nonlinear_policy          |                 90.7985 |
| svar             | svar_revealed_direct | sac_svar_revealed_direct  | svar         | standard_nonlinear        |                105.451  |

## Cross-Transfer Snapshot

| policy_name                   | rule_family        | source_env   | evaluation_env   |   mean_discounted_loss |   clip_rate |   explosion_rate |
|:------------------------------|:-------------------|:-------------|:-----------------|-----------------------:|------------:|-----------------:|
| td3_svar_direct               | svar_direct        | svar         | ann              |                137.659 |    0        |                0 |
| td3_benchmark_transfer        | benchmark_transfer | benchmark    | ann              |                148.608 |    0.042622 |                0 |
| sac_benchmark_transfer        | benchmark_transfer | benchmark    | ann              |                159.817 |    0.028299 |                0 |
| linear_policy_search_transfer | benchmark_transfer | benchmark    | ann              |                179.944 |    0.013455 |                0 |
| sac_svar_direct               | svar_direct        | svar         | ann              |                184.183 |    0        |                0 |
| ppo_svar_direct_nonlinear     | svar_direct        | svar         | ann              |                194.144 |    0        |                0 |
| ppo_svar_direct               | svar_direct        | svar         | ann              |                202.591 |    0        |                0 |
| ppo_benchmark_transfer        | benchmark_transfer | benchmark    | ann              |                327.15  |    0.000781 |                0 |
| ppo_ann_direct_nonlinear      | ann_direct         | ann          | svar             |                 98.318 |    0        |                0 |
| linear_policy_search_transfer | benchmark_transfer | benchmark    | svar             |                134.453 |    0.031424 |                0 |
| sac_benchmark_transfer        | benchmark_transfer | benchmark    | svar             |                135.267 |    0.040538 |                0 |
| td3_benchmark_transfer        | benchmark_transfer | benchmark    | svar             |                150.971 |    0.072656 |                0 |
| ppo_ann_direct                | ann_direct         | ann          | svar             |                206.105 |    0        |                0 |
| ppo_benchmark_transfer        | benchmark_transfer | benchmark    | svar             |                238.086 |    0.001997 |                0 |
| td3_ann_direct                | ann_direct         | ann          | svar             |                371.643 |    0        |                0 |
| sac_ann_direct                | ann_direct         | ann          | svar             |                439.546 |    0        |                0 |

## Revealed Welfare Weights

|   inflation_weight |   output_gap_weight |   rate_smoothing_weight |   objective | success   |   implied_phi_pi |   implied_phi_x |   implied_phi_i |   target_phi_pi |   target_phi_x |   target_phi_i |
|-------------------:|--------------------:|------------------------:|------------:|:----------|-----------------:|----------------:|----------------:|----------------:|---------------:|---------------:|
|                  1 |            0.881713 |                 20.0855 |    0.830222 | True      |        -0.013953 |       -0.040073 |        0.981491 |        0.193173 |       0.265769 |         0.8692 |

## PyFRBUS External Results

| policy_name                        | model_id   |   total_discounted_loss |   mean_period_loss |   total_discounted_revealed_loss |   mean_period_revealed_loss |   mean_sq_inflation_gap |   mean_sq_output_gap |   mean_sq_rate_change |   mean_policy_rate |   std_policy_rate |   improvement_vs_pyfrbus_baseline_pct |   improvement_vs_pyfrbus_baseline_revealed_pct |
|:-----------------------------------|:-----------|------------------------:|-------------------:|---------------------------------:|----------------------------:|------------------------:|---------------------:|----------------------:|-------------------:|------------------:|--------------------------------------:|-----------------------------------------------:|
| pyfrbus_baseline                   | pyfrbus    |                0.014138 |           0.000647 |                         0.018558 |                    0.000844 |                0.00039  |             0.000515 |              0        |            3.00003 |          2.4e-05  |                           0           |                                    0           |
| sac_svar_revealed_direct           | pyfrbus    |                0.027163 |           0.001276 |                         0.074471 |                    0.003462 |                0.00053  |             0.001476 |              8.1e-05  |            3.00463 |          0.043821 |                         -92.1257      |                                 -301.296       |
| empirical_taylor_rule              | pyfrbus    |                0.80597  |           0.038684 |                         1.90248  |                    0.088981 |                0.001614 |             0.073919 |              0.001105 |            3.10403 |          0.113153 |                       -5600.71        |                               -10151.8         |
| sac_benchmark_transfer             | pyfrbus    |                1.18525  |           0.056032 |                         6.00916  |                    0.263102 |                9.7e-05  |             0.110219 |              0.008256 |            2.87521 |          0.126559 |                       -8283.36        |                               -32281.3         |
| riccati_reference                  | pyfrbus    |                2.09249  |           0.09879  |                        11.4796   |                    0.501016 |                0.000274 |             0.193747 |              0.016425 |            2.83452 |          0.181575 |                      -14700.4         |                               -61759.7         |
| td3_benchmark_transfer             | pyfrbus    |                2.48128  |           0.117058 |                        14.9726   |                    0.650903 |                0.000372 |             0.228904 |              0.02234  |            2.81984 |          0.204532 |                      -17450.4         |                               -80581.9         |
| td3_svar_revealed_direct           | pyfrbus    |               22.0892   |           1.06155  |                        62.5016   |                    2.85899  |                0.008129 |             2.09687  |              0.049888 |            2.4335  |          0.374777 |                     -156139           |                              -336699           |
| td3_svar_direct                    | pyfrbus    |               24.1411   |           1.14707  |                       175.674    |                    7.58394  |                0.018991 |             2.20016  |              0.280054 |            3.65251 |          0.58423  |                     -170652           |                              -946546           |
| td3_ann_revealed_direct            | pyfrbus    |               26.3842   |           1.25496  |                        72.8468   |                    3.38247  |                0.019294 |             2.45944  |              0.059479 |            3.40972 |          0.969153 |                     -186518           |                              -392446           |
| ppo_ann_direct                     | pyfrbus    |               38.7951   |           1.87938  |                       130.278    |                    5.89283  |                0.028058 |             3.67652  |              0.130598 |            3.94736 |          0.361957 |                     -274302           |                              -701923           |
| ppo_ann_direct_nonlinear           | pyfrbus    |               51.9825   |           2.5482   |                       142.61     |                    6.61253  |                0.035371 |             5.0041   |              0.107788 |            4.21827 |          0.121217 |                     -367578           |                              -768377           |
| sac_ann_revealed_direct            | pyfrbus    |               91.1481   |           4.5302   |                       175.218    |                    8.65061  |                0.054495 |             8.94435  |              0.035337 |            4.71209 |          0.572325 |                     -644600           |                              -944091           |
| ppo_svar_direct_nonlinear          | pyfrbus    |              157.646    |           7.66523  |                       488.438    |                   22.2802   |                0.105025 |            15.0316   |              0.444184 |            5.14149 |          0.502437 |                          -1.11495e+06 |                                   -2.63192e+06 |
| ppo_svar_revealed_direct           | pyfrbus    |              673.659    |          32.9264   |                      1523.47     |                   72.0321   |                0.472159 |            64.7644   |              0.719735 |            8.61419 |          0.555495 |                          -4.76476e+06 |                                   -8.20932e+06 |
| ppo_ann_revealed_direct            | pyfrbus    |              795.656    |          39.0196   |                      1699.6      |                   81.1478   |                0.56682  |            76.7772   |              0.641535 |            9.52754 |          0.794862 |                          -5.62766e+06 |                                   -9.15845e+06 |
| ppo_svar_revealed_direct_nonlinear | pyfrbus    |              834.218    |          40.7484   |                      2329.13     |                  107.536    |                0.603451 |            79.9268   |              1.81522  |            9.60052 |          0.000127 |                          -5.90041e+06 |                                   -1.25508e+07 |
| ppo_ann_revealed_direct_nonlinear  | pyfrbus    |              866.979    |          42.3409   |                      2434.39     |                  112.319    |                0.630409 |            83.0379   |              1.91547  |            9.7803  |          0        |                          -6.13213e+06 |                                   -1.3118e+07  |

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

- `pyfrbus` 已实际跑通；`Dynare/MMB` 批次由 `phase10_external_mmb_eval.py` 单独维护并汇总。
- `Phase 8/9` 仍是 benchmark-transfer baseline，本轮新增的核心是 `SVAR direct` 与 `ANN direct` 的外部接口延伸。
- Lucas critique 仍是经验环境与外部模型迁移时必须明确的边界。