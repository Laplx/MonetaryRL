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

| evaluation_env   | policy_name                   |   total_discounted_revealed_loss |   mean_period_revealed_loss |
|:-----------------|:------------------------------|---------------------------------:|----------------------------:|
| ann              | ppo_svar_direct               |                          206.491 |                     3.76942 |
| ann              | empirical_taylor_rule         |                          285.352 |                     5.25091 |
| ann              | ppo_ann_direct                |                          376.382 |                     6.47642 |
| ann              | historical_actual_policy      |                          483.772 |                     9.27975 |
| ann              | ppo_benchmark_transfer        |                          574.596 |                    10.353   |
| ann              | linear_policy_search_transfer |                          835.378 |                    15.4146  |
| ann              | riccati_reference             |                         1012.99  |                    18.4673  |
| ann              | sac_benchmark_transfer        |                         1014.5   |                    18.6839  |
| ann              | td3_benchmark_transfer        |                         1053.46  |                    19.3217  |
| ann              | sac_svar_direct               |                         1365.22  |                    25.4324  |
| ann              | td3_ann_direct                |                         2216.44  |                    40.4764  |
| ann              | td3_svar_direct               |                         2471.02  |                    44.9374  |
| ann              | sac_ann_direct                |                         2907.21  |                    52.1695  |
| svar             | ppo_svar_direct               |                          134.662 |                     2.40549 |
| svar             | empirical_taylor_rule         |                          254.738 |                     4.82238 |
| svar             | historical_actual_policy      |                          375.37  |                     6.71249 |
| svar             | ppo_ann_direct                |                          440.062 |                     7.09901 |
| svar             | ppo_benchmark_transfer        |                          488.714 |                     7.71732 |
| svar             | linear_policy_search_transfer |                          691.114 |                    12.2928  |
| svar             | sac_ann_direct                |                          786.703 |                    12.9036  |
| svar             | td3_ann_direct                |                          793.308 |                    13.6199  |
| svar             | sac_benchmark_transfer        |                          799.379 |                    14.3163  |
| svar             | sac_svar_direct               |                          858.19  |                    14.5034  |
| svar             | riccati_reference             |                          903.782 |                    15.9533  |
| svar             | td3_benchmark_transfer        |                          989.764 |                    17.2322  |
| svar             | td3_svar_direct               |                         1160.16  |                    22.1874  |

## Stochastic Re-Scoring

| evaluation_env   | policy_name                   |   mean_discounted_revealed_loss |   std_discounted_revealed_loss |
|:-----------------|:------------------------------|--------------------------------:|-------------------------------:|
| svar             | ppo_svar_direct               |                         260.92  |                       127.344  |
| svar             | empirical_taylor_rule         |                         276.378 |                        74.3971 |
| svar             | ppo_benchmark_transfer        |                         485.798 |                       149.178  |
| svar             | ppo_ann_direct                |                         512.397 |                       217.142  |
| svar             | sac_svar_direct               |                         727.952 |                       261.896  |
| svar             | linear_policy_search_transfer |                         778.117 |                       144.937  |
| svar             | sac_benchmark_transfer        |                         874.518 |                       190.721  |
| svar             | riccati_reference             |                         970.622 |                       204.478  |
| svar             | td3_benchmark_transfer        |                        1079.32  |                       240.417  |
| svar             | sac_ann_direct                |                        1194.23  |                       349.834  |
| svar             | td3_ann_direct                |                        1233.66  |                       388.417  |
| svar             | td3_svar_direct               |                        1428.96  |                       464.809  |
| ann              | empirical_taylor_rule         |                         283.061 |                        41.5765 |
| ann              | ppo_svar_direct               |                         327.939 |                       112.102  |
| ann              | ppo_ann_direct                |                         500.301 |                       157.768  |
| ann              | ppo_benchmark_transfer        |                         804.758 |                       147.051  |
| ann              | linear_policy_search_transfer |                        1074.91  |                       194.901  |
| ann              | sac_benchmark_transfer        |                        1181.38  |                       224.314  |
| ann              | sac_svar_direct               |                        1238.31  |                       328.591  |
| ann              | riccati_reference             |                        1254.26  |                       221.734  |
| ann              | td3_benchmark_transfer        |                        1346.99  |                       263.788  |
| ann              | td3_svar_direct               |                        1898.88  |                       534.393  |
| ann              | td3_ann_direct                |                        2300.96  |                       470.144  |
| ann              | sac_ann_direct                |                        3193.36  |                       558.171  |

## Historical Rank Change

| evaluation_env   | policy_name                   |   baseline_rank |   revealed_rank |   rank_change |
|:-----------------|:------------------------------|----------------:|----------------:|--------------:|
| ann              | ppo_svar_direct               |              12 |               1 |            11 |
| ann              | empirical_taylor_rule         |               5 |               2 |             3 |
| ann              | ppo_ann_direct                |               1 |               3 |            -2 |
| ann              | historical_actual_policy      |              11 |               4 |             7 |
| ann              | ppo_benchmark_transfer        |              13 |               5 |             8 |
| ann              | linear_policy_search_transfer |               9 |               6 |             3 |
| ann              | riccati_reference             |               7 |               7 |             0 |
| ann              | sac_benchmark_transfer        |              10 |               8 |             2 |
| ann              | td3_benchmark_transfer        |               6 |               9 |            -3 |
| ann              | sac_svar_direct               |               8 |              10 |            -2 |
| ann              | td3_ann_direct                |               2 |              11 |            -9 |
| ann              | td3_svar_direct               |               4 |              12 |            -8 |
| ann              | sac_ann_direct                |               3 |              13 |           -10 |
| svar             | ppo_svar_direct               |               4 |               1 |             3 |
| svar             | empirical_taylor_rule         |               3 |               2 |             1 |
| svar             | historical_actual_policy      |               1 |               3 |            -2 |
| svar             | ppo_ann_direct                |              10 |               4 |             6 |
| svar             | ppo_benchmark_transfer        |              11 |               5 |             6 |
| svar             | linear_policy_search_transfer |               7 |               6 |             1 |
| svar             | sac_ann_direct                |              13 |               7 |             6 |
| svar             | td3_ann_direct                |              12 |               8 |             4 |
| svar             | sac_benchmark_transfer        |               6 |               9 |            -3 |
| svar             | sac_svar_direct               |               2 |              10 |            -8 |
| svar             | riccati_reference             |               8 |              11 |            -3 |
| svar             | td3_benchmark_transfer        |               9 |              12 |            -3 |
| svar             | td3_svar_direct               |               5 |              13 |            -8 |

## Stochastic Rank Change

| evaluation_env   | policy_name                   |   baseline_rank |   revealed_rank |   rank_change |
|:-----------------|:------------------------------|----------------:|----------------:|--------------:|
| ann              | empirical_taylor_rule         |               8 |               1 |             7 |
| ann              | ppo_svar_direct               |              11 |               2 |             9 |
| ann              | ppo_ann_direct                |               2 |               3 |            -1 |
| ann              | ppo_benchmark_transfer        |              12 |               4 |             8 |
| ann              | linear_policy_search_transfer |               9 |               5 |             4 |
| ann              | sac_benchmark_transfer        |               7 |               6 |             1 |
| ann              | sac_svar_direct               |              10 |               7 |             3 |
| ann              | riccati_reference             |               6 |               8 |            -2 |
| ann              | td3_benchmark_transfer        |               5 |               9 |            -4 |
| ann              | td3_svar_direct               |               4 |              10 |            -6 |
| ann              | td3_ann_direct                |               1 |              11 |           -10 |
| ann              | sac_ann_direct                |               3 |              12 |            -9 |
| svar             | ppo_svar_direct               |               4 |               1 |             3 |
| svar             | empirical_taylor_rule         |               3 |               2 |             1 |
| svar             | ppo_benchmark_transfer        |              10 |               3 |             7 |
| svar             | ppo_ann_direct                |               9 |               4 |             5 |
| svar             | sac_svar_direct               |               1 |               5 |            -4 |
| svar             | linear_policy_search_transfer |               7 |               6 |             1 |
| svar             | sac_benchmark_transfer        |               5 |               7 |            -2 |
| svar             | riccati_reference             |               6 |               8 |            -2 |
| svar             | td3_benchmark_transfer        |               8 |               9 |            -1 |
| svar             | sac_ann_direct                |              12 |              10 |             2 |
| svar             | td3_ann_direct                |              11 |              11 |             0 |
| svar             | td3_svar_direct               |               2 |              12 |           -10 |

## Notes

- Inflation weight is normalized to `1`; only output-gap and rate-smoothing weights are revealed.
- This is a report-only welfare metric based on `SVAR + empirical Taylor` and does not replace the main training objective.
- Lucas critique still applies because the revealed metric is inferred under a fixed reduced-form transition.