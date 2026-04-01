# Phase 10 Unified Counterfactual Evaluation

## Unified Policy Registry

| policy_name                   | rule_family        | source_env   | policy_parameterization   | callable_type   |
|:------------------------------|:-------------------|:-------------|:--------------------------|:----------------|
| ppo_ann_direct                | ann_direct         | ann          | linear_policy             | checkpoint      |
| sac_ann_direct                | ann_direct         | ann          | standard_nonlinear        | checkpoint      |
| td3_ann_direct                | ann_direct         | ann          | standard_nonlinear        | checkpoint      |
| linear_policy_search_transfer | benchmark_transfer | benchmark    | fixed_rule                | linear          |
| ppo_benchmark_transfer        | benchmark_transfer | benchmark    | linear_surrogate          | linear          |
| sac_benchmark_transfer        | benchmark_transfer | benchmark    | linear_surrogate          | linear          |
| td3_benchmark_transfer        | benchmark_transfer | benchmark    | linear_surrogate          | linear          |
| empirical_taylor_rule         | empirical_rule     | svar         | fixed_rule                | linear          |
| historical_actual_policy      | historical_actual  | historical   | fixed_rule                | historical      |
| ppo_svar_direct               | svar_direct        | svar         | linear_policy             | checkpoint      |
| sac_svar_direct               | svar_direct        | svar         | standard_nonlinear        | checkpoint      |
| td3_svar_direct               | svar_direct        | svar         | standard_nonlinear        | checkpoint      |
| riccati_reference             | theory_reference   | benchmark    | fixed_rule                | linear          |

## SVAR Historical Counterfactual

| evaluation_env   | policy_name                   |   total_discounted_loss |   mean_period_loss |   mean_sq_inflation_gap |   mean_sq_output_gap |   mean_sq_rate_change |   mean_policy_rate |   std_policy_rate |   improvement_vs_actual_pct |   improvement_vs_taylor_pct |
|:-----------------|:------------------------------|------------------------:|-------------------:|------------------------:|---------------------:|----------------------:|-------------------:|------------------:|----------------------------:|----------------------------:|
| svar             | historical_actual_policy      |                 87.916  |            1.54727 |                0.723281 |              1.60241 |              0.227843 |           4.77892  |          2.18807  |                     0       |                     6.96902 |
| svar             | sac_svar_direct               |                 92.1016 |            1.55643 |                0.708636 |              1.57203 |              0.617792 |           5.34625  |          1.5752   |                    -4.76086 |                     2.53994 |
| svar             | empirical_taylor_rule         |                 94.5018 |            1.64709 |                0.693088 |              1.88343 |              0.122907 |           4.73712  |          1.99524  |                    -7.49107 |                     0       |
| svar             | ppo_svar_direct               |                100.998  |            1.77717 |                1.15267  |              1.24748 |              0.007613 |           6.47433  |          0.399381 |                   -14.88    |                    -6.87398 |
| svar             | td3_svar_direct               |                103.644  |            1.8112  |                0.86723  |              1.6905  |              0.987259 |           5.21452  |          2.09872  |                   -17.8897  |                    -9.67398 |
| svar             | sac_benchmark_transfer        |                114.455  |            1.99224 |                1.00172  |              1.86482 |              0.581034 |           4.27898  |          2.34381  |                   -30.1866  |                   -21.1139  |
| svar             | linear_policy_search_transfer |                120.575  |            2.10817 |                1.17866  |              1.76384 |              0.475911 |           4.02609  |          2.08302  |                   -37.1484  |                   -27.5905  |
| svar             | riccati_reference             |                121.072  |            2.10878 |                1.09218  |              1.90192 |              0.656402 |           4.15918  |          2.46192  |                   -37.7134  |                   -28.1161  |
| svar             | td3_benchmark_transfer        |                129.555  |            2.25366 |                1.24125  |              1.88213 |              0.713523 |           3.98097  |          2.44256  |                   -47.3624  |                   -37.0927  |
| svar             | ppo_ann_direct                |                150.19   |            2.47015 |                1.87964  |              1.13904 |              0.209855 |           3.73454  |          0.968385 |                   -70.8331  |                   -58.9277  |
| svar             | ppo_benchmark_transfer        |                171.875  |            3.02802 |                2.37379  |              1.26636 |              0.210448 |           2.89154  |          0.772566 |                   -95.4995  |                   -81.8751  |
| svar             | td3_ann_direct                |                270.191  |            4.90445 |                4.33277  |              1.06019 |              0.415838 |           1.65881  |          0.770966 |                  -207.329   |                  -185.911   |
| svar             | sac_ann_direct                |                407.733  |            7.76666 |                7.24039  |              1.00497 |              0.237839 |           0.345926 |          0.496773 |                  -363.775   |                  -331.455   |

## ANN Historical Counterfactual

| evaluation_env   | policy_name                   |   total_discounted_loss |   mean_period_loss |   mean_sq_inflation_gap |   mean_sq_output_gap |   mean_sq_rate_change |   mean_policy_rate |   std_policy_rate |   improvement_vs_actual_pct |   improvement_vs_taylor_pct |
|:-----------------|:------------------------------|------------------------:|-------------------:|------------------------:|---------------------:|----------------------:|-------------------:|------------------:|----------------------------:|----------------------------:|
| ann              | ppo_ann_direct                |                 69.2297 |            1.23336 |                0.571374 |              1.27638 |              0.237965 |            4.85375 |          0.986362 |                    58.8507  |                     46.9126 |
| ann              | td3_ann_direct                |                 71.239  |            1.25295 |                0.305417 |              1.5083  |              1.93379  |            3.91172 |          2.27815  |                    57.6564  |                     45.3718 |
| ann              | sac_ann_direct                |                 86.01   |            1.54357 |                0.38708  |              1.81328 |              2.49849  |            3.72476 |          2.46862  |                    48.8766  |                     34.0449 |
| ann              | td3_svar_direct               |                111.444  |            2.09229 |                1.1918   |              1.37749 |              2.11749  |            5.3367  |          2.16721  |                    33.759   |                     14.5415 |
| ann              | empirical_taylor_rule         |                130.407  |            2.33591 |                1.55388  |              1.54077 |              0.116428 |            5.21069 |          2.06386  |                    22.4876  |                      0      |
| ann              | td3_benchmark_transfer        |                155.226  |            2.84425 |                2.09077  |              1.3472  |              0.79874  |            4.94194 |          2.55687  |                     7.73555 |                    -19.0318 |
| ann              | riccati_reference             |                157.011  |            2.8811  |                2.106    |              1.39958 |              0.753141 |            5.14884 |          2.70275  |                     6.67434 |                    -20.4009 |
| ann              | sac_svar_direct               |                157.371  |            2.94634 |                1.79042  |              2.09482 |              1.08511  |            6.45629 |          2.05647  |                     6.46045 |                    -20.6769 |
| ann              | linear_policy_search_transfer |                159.9    |            2.9363  |                2.20673  |              1.33939 |              0.598787 |            5.02625 |          2.47133  |                     4.95706 |                    -22.6164 |
| ann              | sac_benchmark_transfer        |                161.911  |            2.97455 |                2.16364  |              1.47022 |              0.757954 |            5.34278 |          2.71145  |                     3.76219 |                    -24.1579 |
| ann              | historical_actual_policy      |                168.24   |            3.34058 |                1.55898  |              3.51721 |              0.229996 |            4.75412 |          2.19169  |                     0       |                    -29.0116 |
| ann              | ppo_svar_direct               |                172.814  |            3.16724 |                2.58718  |              1.15851 |              0.008004 |            7.00066 |          0.234108 |                    -2.7189  |                    -32.5193 |
| ann              | ppo_benchmark_transfer        |                225.277  |            4.38216 |                0.335505 |              8.06437 |              0.144733 |            3.64916 |          0.525939 |                   -33.9023  |                    -72.7495 |

## SVAR Long-Run Stochastic Evaluation

| policy_name                   | evaluation_env   |   mean_discounted_loss |   std_discounted_loss |   mean_reward |   mean_abs_action |   clip_rate |   explosion_rate |
|:------------------------------|:-----------------|-----------------------:|----------------------:|--------------:|------------------:|------------:|-----------------:|
| sac_svar_direct               | svar             |                105.523 |               28.4863 |      -179.097 |           3.37868 |    0        |                0 |
| td3_svar_direct               | svar             |                111.906 |               35.529  |      -190.568 |           3.42763 |    0        |                0 |
| empirical_taylor_rule         | svar             |                114     |               30.1481 |      -195.278 |           3.06149 |    0.007031 |                0 |
| ppo_svar_direct               | svar             |                124.516 |               42.9543 |      -216.235 |           4.43068 |    0        |                0 |
| sac_benchmark_transfer        | svar             |                131.149 |               33.5387 |      -227.187 |           2.66632 |    0.044878 |                0 |
| riccati_reference             | svar             |                139.479 |               35.6771 |      -239.046 |           2.64253 |    0.057118 |                0 |
| linear_policy_search_transfer | svar             |                149.463 |               38.7141 |      -258.117 |           2.48097 |    0.035156 |                0 |
| td3_benchmark_transfer        | svar             |                150.658 |               41.5035 |      -257.882 |           2.52317 |    0.089323 |                0 |
| ppo_ann_direct                | svar             |                206.105 |              145.304  |      -375.859 |           2.50719 |    0        |                0 |
| ppo_benchmark_transfer        | svar             |                228.303 |               78.7042 |      -412.084 |           1.12259 |    0.00191  |                0 |
| td3_ann_direct                | svar             |                315.011 |              165.049  |      -585.866 |           2.11178 |    0        |                0 |
| sac_ann_direct                | svar             |                490.104 |              237.126  |      -940.695 |           2.81523 |    0        |                0 |

## ANN Long-Run Stochastic Evaluation

| policy_name                   | evaluation_env   |   mean_discounted_loss |   std_discounted_loss |   mean_reward |   mean_abs_action |   clip_rate |   explosion_rate |
|:------------------------------|:-----------------|-----------------------:|----------------------:|--------------:|------------------:|------------:|-----------------:|
| td3_ann_direct                | ann              |                77.7712 |               11.2404 |      -132.051 |           2.66443 |    0        |                0 |
| ppo_ann_direct                | ann              |                78.4765 |               18.1763 |      -131.711 |           3.1319  |    0        |                0 |
| sac_ann_direct                | ann              |                91.0708 |               10.991  |      -156.291 |           2.76287 |    0        |                0 |
| td3_svar_direct               | ann              |               139.699  |               27.1242 |      -241.913 |           3.81753 |    0        |                0 |
| td3_benchmark_transfer        | ann              |               143.358  |               19.0159 |      -245.526 |           2.78548 |    0.040278 |                0 |
| riccati_reference             | ann              |               154.293  |               19.5498 |      -264.87  |           2.9916  |    0.040538 |                0 |
| sac_benchmark_transfer        | ann              |               156.729  |               17.275  |      -271.008 |           3.10593 |    0.027778 |                0 |
| empirical_taylor_rule         | ann              |               162.707  |               32.9417 |      -287.265 |           3.88596 |    8.7e-05  |                0 |
| linear_policy_search_transfer | ann              |               181.761  |               21.8113 |      -315.7   |           2.79709 |    0.013628 |                0 |
| sac_svar_direct               | ann              |               191.385  |               20.5371 |      -333.4   |           4.712   |    0        |                0 |
| ppo_svar_direct               | ann              |               200.37   |               21.3704 |      -350.24  |           4.96519 |    0        |                0 |
| ppo_benchmark_transfer        | ann              |               327.935  |               36.9645 |      -583.891 |           1.75707 |    8.7e-05  |                0 |

## Cross-Transfer Summary

| policy_name                   | rule_family        | source_env   | evaluation_env   | policy_parameterization   |   mean_discounted_loss |   std_discounted_loss |   clip_rate |   explosion_rate |
|:------------------------------|:-------------------|:-------------|:-----------------|:--------------------------|-----------------------:|----------------------:|------------:|-----------------:|
| td3_svar_direct               | svar_direct        | svar         | ann              | standard_nonlinear        |                139.699 |               27.1242 |    0        |                0 |
| td3_benchmark_transfer        | benchmark_transfer | benchmark    | ann              | linear_surrogate          |                143.358 |               19.0159 |    0.040278 |                0 |
| sac_benchmark_transfer        | benchmark_transfer | benchmark    | ann              | linear_surrogate          |                156.729 |               17.275  |    0.027778 |                0 |
| linear_policy_search_transfer | benchmark_transfer | benchmark    | ann              | fixed_rule                |                181.761 |               21.8113 |    0.013628 |                0 |
| sac_svar_direct               | svar_direct        | svar         | ann              | standard_nonlinear        |                191.385 |               20.5371 |    0        |                0 |
| ppo_svar_direct               | svar_direct        | svar         | ann              | linear_policy             |                200.37  |               21.3704 |    0        |                0 |
| ppo_benchmark_transfer        | benchmark_transfer | benchmark    | ann              | linear_surrogate          |                327.935 |               36.9645 |    8.7e-05  |                0 |
| sac_benchmark_transfer        | benchmark_transfer | benchmark    | svar             | linear_surrogate          |                131.149 |               33.5387 |    0.044878 |                0 |
| linear_policy_search_transfer | benchmark_transfer | benchmark    | svar             | fixed_rule                |                149.463 |               38.7141 |    0.035156 |                0 |
| td3_benchmark_transfer        | benchmark_transfer | benchmark    | svar             | linear_surrogate          |                150.658 |               41.5035 |    0.089323 |                0 |
| ppo_ann_direct                | ann_direct         | ann          | svar             | linear_policy             |                206.105 |              145.304  |    0        |                0 |
| ppo_benchmark_transfer        | benchmark_transfer | benchmark    | svar             | linear_surrogate          |                228.303 |               78.7042 |    0.00191  |                0 |
| td3_ann_direct                | ann_direct         | ann          | svar             | standard_nonlinear        |                315.011 |              165.049  |    0        |                0 |
| sac_ann_direct                | ann_direct         | ann          | svar             | standard_nonlinear        |                490.104 |              237.126  |    0        |                0 |

## Notes

- `Phase 8/9` remains the benchmark-transfer baseline; this file adds the direct-trained empirical rules under the same evaluator.
- `benchmark transfer` and empirical direct-trained rules are kept distinct in all tables.
- Lucas critique still applies because both empirical environments hold reduced-form transitions fixed under policy changes.