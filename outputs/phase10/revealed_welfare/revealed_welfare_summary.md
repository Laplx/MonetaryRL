# Phase 10 Revealed Welfare Summary

## Revealed Weights

|   inflation_weight |   output_gap_weight |   rate_smoothing_weight |   objective | success   |   implied_phi_pi |   implied_phi_x |   implied_phi_i |   target_phi_pi |   target_phi_x |   target_phi_i |
|-------------------:|--------------------:|------------------------:|------------:|:----------|-----------------:|----------------:|----------------:|----------------:|---------------:|---------------:|
|                  1 |            0.881713 |                 20.0855 |    0.830222 | True      |        -0.013953 |       -0.040073 |        0.981491 |        0.193173 |       0.265769 |         0.8692 |

## Top Objective Grid Points

|   output_weight |   rate_smoothing_weight |   objective |
|----------------:|------------------------:|------------:|
|        0.594604 |                       5 |     1.2603  |
|        0.3884   |                       5 |     1.27049 |
|        0.910282 |                       5 |     1.28899 |
|        0.253706 |                       5 |     1.29717 |
|        0.165723 |                       5 |     1.32775 |
|        0.108251 |                       5 |     1.35591 |
|        0.070711 |                       5 |     1.37911 |
|        1.39356  |                       5 |     1.39345 |
|        0.046189 |                       5 |     1.39693 |
|        0.030171 |                       5 |     1.40996 |

## Historical Re-Scoring

| evaluation_env   | policy_name                        |   total_discounted_revealed_loss |   mean_period_revealed_loss |
|:-----------------|:-----------------------------------|---------------------------------:|----------------------------:|
| ann              | td3_ann_revealed_direct            |                          186.231 |                     3.33914 |
| ann              | ppo_svar_direct                    |                          206.491 |                     3.76942 |
| ann              | ppo_svar_direct_linear             |                          206.491 |                     3.76942 |
| ann              | sac_ann_revealed_direct            |                          207.699 |                     3.80104 |
| ann              | sac_svar_revealed_direct           |                          211.276 |                     3.91791 |
| ann              | ppo_ann_direct_nonlinear           |                          284.81  |                     5.20634 |
| ann              | empirical_taylor_rule              |                          285.352 |                     5.25091 |
| ann              | ppo_ann_direct                     |                          376.382 |                     6.47642 |
| ann              | ppo_ann_direct_linear              |                          376.382 |                     6.47642 |
| ann              | ppo_svar_revealed_direct           |                          398.194 |                     7.29758 |
| ann              | ppo_svar_direct_nonlinear          |                          462.658 |                     8.21768 |
| ann              | historical_actual_policy           |                          483.772 |                     9.27975 |
| ann              | ppo_ann_revealed_direct            |                          484.76  |                     9.03991 |
| ann              | ppo_svar_revealed_direct_nonlinear |                          549.025 |                     9.78431 |
| ann              | ppo_benchmark_transfer             |                          574.596 |                    10.353   |
| ann              | ppo_ann_revealed_direct_nonlinear  |                          598.721 |                    10.6657  |
| ann              | td3_svar_revealed_direct           |                          700.003 |                    13.6861  |
| ann              | linear_policy_search_transfer      |                          835.378 |                    15.4146  |
| ann              | riccati_reference                  |                         1012.99  |                    18.4673  |
| ann              | sac_benchmark_transfer             |                         1014.5   |                    18.6839  |
| ann              | td3_benchmark_transfer             |                         1053.46  |                    19.3217  |
| ann              | sac_svar_direct                    |                         1365.22  |                    25.4324  |
| ann              | td3_ann_direct                     |                         2216.44  |                    40.4764  |
| ann              | td3_svar_direct                    |                         2471.02  |                    44.9374  |
| ann              | sac_ann_direct                     |                         2907.21  |                    52.1695  |
| svar             | ppo_svar_direct                    |                          134.662 |                     2.40549 |
| svar             | ppo_svar_direct_linear             |                          134.662 |                     2.40549 |
| svar             | sac_svar_revealed_direct           |                          148.76  |                     2.68005 |
| svar             | sac_ann_revealed_direct            |                          153.027 |                     2.79718 |
| svar             | ppo_ann_direct_nonlinear           |                          243.23  |                     4.33832 |
| svar             | td3_ann_revealed_direct            |                          246.754 |                     4.34979 |
| svar             | empirical_taylor_rule              |                          254.738 |                     4.82238 |
| svar             | ppo_svar_direct_nonlinear          |                          300.949 |                     5.29313 |
| svar             | td3_svar_revealed_direct           |                          320.306 |                     5.67029 |
| svar             | ppo_ann_revealed_direct            |                          338.837 |                     6.06157 |
| svar             | historical_actual_policy           |                          375.37  |                     6.71249 |
| svar             | ppo_svar_revealed_direct           |                          388.475 |                     7.44599 |
| svar             | ppo_ann_direct                     |                          440.062 |                     7.09901 |
| svar             | ppo_ann_direct_linear              |                          440.062 |                     7.09901 |
| svar             | ppo_svar_revealed_direct_nonlinear |                          465.048 |                     8.4016  |
| svar             | ppo_benchmark_transfer             |                          488.714 |                     7.71732 |
| svar             | ppo_ann_revealed_direct_nonlinear  |                          505.235 |                     9.09919 |
| svar             | linear_policy_search_transfer      |                          691.114 |                    12.2928  |
| svar             | sac_ann_direct                     |                          786.703 |                    12.9036  |
| svar             | td3_ann_direct                     |                          793.308 |                    13.6199  |
| svar             | sac_benchmark_transfer             |                          799.379 |                    14.3163  |
| svar             | sac_svar_direct                    |                          858.19  |                    14.5034  |
| svar             | riccati_reference                  |                          903.782 |                    15.9533  |
| svar             | td3_benchmark_transfer             |                          989.764 |                    17.2322  |
| svar             | td3_svar_direct                    |                         1160.16  |                    22.1874  |

## Stochastic Re-Scoring

| evaluation_env   | policy_name                        |   mean_discounted_revealed_loss |   std_discounted_revealed_loss |
|:-----------------|:-----------------------------------|--------------------------------:|-------------------------------:|
| svar             | sac_svar_revealed_direct           |                         203.521 |                        70.4612 |
| svar             | sac_ann_revealed_direct            |                         251.209 |                       129.085  |
| svar             | ppo_svar_direct                    |                         272.777 |                       123.101  |
| svar             | empirical_taylor_rule              |                         275.661 |                        86.3683 |
| svar             | td3_ann_revealed_direct            |                         306.52  |                       144.792  |
| svar             | ppo_ann_direct_nonlinear           |                         359.523 |                       106.263  |
| svar             | td3_svar_revealed_direct           |                         415.92  |                       106.092  |
| svar             | ppo_svar_direct_nonlinear          |                         488.888 |                       316.944  |
| svar             | ppo_benchmark_transfer             |                         501.38  |                       134.5    |
| svar             | ppo_ann_direct                     |                         512.397 |                       217.142  |
| svar             | ppo_ann_revealed_direct            |                         519.396 |                       191.618  |
| svar             | ppo_svar_revealed_direct           |                         683.67  |                       214.346  |
| svar             | linear_policy_search_transfer      |                         744.593 |                       150.781  |
| svar             | sac_svar_direct                    |                         772.091 |                       248.906  |
| svar             | sac_benchmark_transfer             |                         872.029 |                       205.524  |
| svar             | ppo_svar_revealed_direct_nonlinear |                         959.669 |                       454.814  |
| svar             | riccati_reference                  |                         978.205 |                       196.498  |
| svar             | ppo_ann_revealed_direct_nonlinear  |                        1032.09  |                       478.988  |
| svar             | td3_benchmark_transfer             |                        1063.25  |                       218.559  |
| svar             | td3_ann_direct                     |                        1175.09  |                       361.782  |
| svar             | sac_ann_direct                     |                        1237.2   |                       422.874  |
| svar             | td3_svar_direct                    |                        1390.77  |                       461.161  |
| ann              | sac_ann_revealed_direct            |                         253.47  |                        48.4531 |
| ann              | empirical_taylor_rule              |                         286.009 |                        43.607  |
| ann              | td3_ann_revealed_direct            |                         309.792 |                        55.0108 |
| ann              | ppo_svar_direct                    |                         332.075 |                       113.838  |
| ann              | sac_svar_revealed_direct           |                         370.906 |                       101.317  |
| ann              | ppo_ann_direct_nonlinear           |                         414.587 |                       105.64   |
| ann              | ppo_ann_direct                     |                         500.301 |                       157.768  |
| ann              | ppo_svar_revealed_direct           |                         607.188 |                       153.407  |
| ann              | ppo_ann_revealed_direct            |                         664.239 |                       190.599  |
| ann              | ppo_svar_direct_nonlinear          |                         791.499 |                       356.102  |
| ann              | ppo_benchmark_transfer             |                         797.518 |                       144.624  |
| ann              | ppo_svar_revealed_direct_nonlinear |                         975.011 |                       428.343  |
| ann              | ppo_ann_revealed_direct_nonlinear  |                        1057.52  |                       457.436  |
| ann              | linear_policy_search_transfer      |                        1061.55  |                       186.486  |
| ann              | sac_benchmark_transfer             |                        1135.59  |                       218.004  |
| ann              | sac_svar_direct                    |                        1217.35  |                       321.562  |
| ann              | riccati_reference                  |                        1229.08  |                       215.264  |
| ann              | td3_svar_revealed_direct           |                        1266.08  |                       352.95   |
| ann              | td3_benchmark_transfer             |                        1324.35  |                       270.957  |
| ann              | td3_svar_direct                    |                        1840.51  |                       501.878  |
| ann              | td3_ann_direct                     |                        2408.7   |                       507.659  |
| ann              | sac_ann_direct                     |                        3250.93  |                       644.125  |

## Historical Rank Change

| evaluation_env   | policy_name                        |   baseline_rank |   revealed_rank |   rank_change |
|:-----------------|:-----------------------------------|----------------:|----------------:|--------------:|
| ann              | td3_ann_revealed_direct            |               5 |               1 |             4 |
| ann              | ppo_svar_direct                    |              16 |               2 |            14 |
| ann              | ppo_svar_direct_linear             |              16 |               2 |            14 |
| ann              | sac_ann_revealed_direct            |              17 |               3 |            14 |
| ann              | sac_svar_revealed_direct           |              15 |               4 |            11 |
| ann              | ppo_ann_direct_nonlinear           |              12 |               5 |             7 |
| ann              | empirical_taylor_rule              |               6 |               6 |             0 |
| ann              | ppo_ann_direct                     |               1 |               7 |            -6 |
| ann              | ppo_ann_direct_linear              |               1 |               7 |            -6 |
| ann              | ppo_svar_revealed_direct           |              18 |               8 |            10 |
| ann              | ppo_svar_direct_nonlinear          |              14 |               9 |             5 |
| ann              | historical_actual_policy           |              13 |              10 |             3 |
| ann              | ppo_ann_revealed_direct            |              21 |              11 |            10 |
| ann              | ppo_svar_revealed_direct_nonlinear |              20 |              12 |             8 |
| ann              | ppo_benchmark_transfer             |              19 |              13 |             6 |
| ann              | ppo_ann_revealed_direct_nonlinear  |              22 |              14 |             8 |
| ann              | td3_svar_revealed_direct           |              23 |              15 |             8 |
| ann              | linear_policy_search_transfer      |              10 |              16 |            -6 |
| ann              | riccati_reference                  |               8 |              17 |            -9 |
| ann              | sac_benchmark_transfer             |              11 |              18 |            -7 |
| ann              | td3_benchmark_transfer             |               7 |              19 |           -12 |
| ann              | sac_svar_direct                    |               9 |              20 |           -11 |
| ann              | td3_ann_direct                     |               2 |              21 |           -19 |
| ann              | td3_svar_direct                    |               4 |              22 |           -18 |
| ann              | sac_ann_direct                     |               3 |              23 |           -20 |
| svar             | ppo_svar_direct                    |               6 |               1 |             5 |
| svar             | ppo_svar_direct_linear             |               6 |               1 |             5 |
| svar             | sac_svar_revealed_direct           |               8 |               2 |             6 |
| svar             | sac_ann_revealed_direct            |               9 |               3 |             6 |
| svar             | ppo_ann_direct_nonlinear           |               1 |               4 |            -3 |
| svar             | td3_ann_revealed_direct            |              14 |               5 |             9 |
| svar             | empirical_taylor_rule              |               5 |               6 |            -1 |
| svar             | ppo_svar_direct_nonlinear          |               3 |               7 |            -4 |
| svar             | td3_svar_revealed_direct           |              18 |               8 |            10 |
| svar             | ppo_ann_revealed_direct            |              17 |               9 |             8 |
| svar             | historical_actual_policy           |               2 |              10 |            -8 |
| svar             | ppo_svar_revealed_direct           |              21 |              11 |            10 |
| svar             | ppo_ann_direct                     |              15 |              12 |             3 |
| svar             | ppo_ann_direct_linear              |              15 |              12 |             3 |
| svar             | ppo_svar_revealed_direct_nonlinear |              20 |              13 |             7 |
| svar             | ppo_benchmark_transfer             |              16 |              14 |             2 |
| svar             | ppo_ann_revealed_direct_nonlinear  |              22 |              15 |             7 |
| svar             | linear_policy_search_transfer      |              11 |              16 |            -5 |
| svar             | sac_ann_direct                     |              23 |              17 |             6 |
| svar             | td3_ann_direct                     |              19 |              18 |             1 |
| svar             | sac_benchmark_transfer             |              10 |              19 |            -9 |
| svar             | sac_svar_direct                    |               4 |              20 |           -16 |
| svar             | riccati_reference                  |              12 |              21 |            -9 |
| svar             | td3_benchmark_transfer             |              13 |              22 |            -9 |
| svar             | td3_svar_direct                    |               7 |              23 |           -16 |

## Stochastic Rank Change

| evaluation_env   | policy_name                        |   baseline_rank |   revealed_rank |   rank_change |
|:-----------------|:-----------------------------------|----------------:|----------------:|--------------:|
| ann              | sac_ann_revealed_direct            |              14 |               1 |            13 |
| ann              | empirical_taylor_rule              |              10 |               2 |             8 |
| ann              | td3_ann_revealed_direct            |               6 |               3 |             3 |
| ann              | ppo_svar_direct                    |              17 |               4 |            13 |
| ann              | sac_svar_revealed_direct           |              18 |               5 |            13 |
| ann              | ppo_ann_direct_nonlinear           |              11 |               6 |             5 |
| ann              | ppo_ann_direct                     |               3 |               7 |            -4 |
| ann              | ppo_svar_revealed_direct           |              19 |               8 |            11 |
| ann              | ppo_ann_revealed_direct            |              20 |               9 |            11 |
| ann              | ppo_svar_direct_nonlinear          |              15 |              10 |             5 |
| ann              | ppo_benchmark_transfer             |              22 |              11 |            11 |
| ann              | ppo_svar_revealed_direct_nonlinear |              21 |              12 |             9 |
| ann              | ppo_ann_revealed_direct_nonlinear  |              23 |              13 |            10 |
| ann              | linear_policy_search_transfer      |              12 |              14 |            -2 |
| ann              | sac_benchmark_transfer             |               9 |              15 |            -6 |
| ann              | sac_svar_direct                    |              13 |              16 |            -3 |
| ann              | riccati_reference                  |               8 |              17 |            -9 |
| ann              | td3_svar_revealed_direct           |              24 |              18 |             6 |
| ann              | td3_benchmark_transfer             |               7 |              19 |           -12 |
| ann              | td3_svar_direct                    |               5 |              20 |           -15 |
| ann              | td3_ann_direct                     |               2 |              21 |           -19 |
| ann              | sac_ann_direct                     |               4 |              22 |           -18 |
| svar             | sac_svar_revealed_direct           |               9 |               1 |             8 |
| svar             | sac_ann_revealed_direct            |              14 |               2 |            12 |
| svar             | ppo_svar_direct                    |               7 |               3 |             4 |
| svar             | empirical_taylor_rule              |               5 |               4 |             1 |
| svar             | td3_ann_revealed_direct            |              15 |               5 |            10 |
| svar             | ppo_ann_direct_nonlinear           |               3 |               6 |            -3 |
| svar             | td3_svar_revealed_direct           |              19 |               7 |            12 |
| svar             | ppo_svar_direct_nonlinear          |               2 |               8 |            -6 |
| svar             | ppo_benchmark_transfer             |              17 |               9 |             8 |
| svar             | ppo_ann_direct                     |              16 |              10 |             6 |
| svar             | ppo_ann_revealed_direct            |              18 |              11 |             7 |
| svar             | ppo_svar_revealed_direct           |              21 |              12 |             9 |
| svar             | linear_policy_search_transfer      |              10 |              13 |            -3 |
| svar             | sac_svar_direct                    |               1 |              14 |           -13 |
| svar             | sac_benchmark_transfer             |               8 |              15 |            -7 |
| svar             | ppo_svar_revealed_direct_nonlinear |              22 |              16 |             6 |
| svar             | riccati_reference                  |              11 |              17 |            -6 |
| svar             | ppo_ann_revealed_direct_nonlinear  |              23 |              18 |             5 |
| svar             | td3_benchmark_transfer             |              12 |              19 |            -7 |
| svar             | td3_ann_direct                     |              20 |              20 |             0 |
| svar             | sac_ann_direct                     |              24 |              21 |             3 |
| svar             | td3_svar_direct                    |               4 |              22 |           -18 |

## Notes

- Inflation weight is normalized to `1`; only output-gap and rate-smoothing weights are revealed.
- This is a report-only welfare metric based on `SVAR + empirical Taylor` and does not replace the main training objective.
- Lucas critique still applies because the revealed metric is inferred under a fixed reduced-form transition.