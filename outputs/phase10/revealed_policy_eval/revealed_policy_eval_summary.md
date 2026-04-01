# Phase 10 Revealed-Welfare Policy Evaluation

## SVAR Historical Counterfactual

| evaluation_env   | policy_name              |   total_discounted_loss |   mean_period_loss |   mean_sq_inflation_gap |   mean_sq_output_gap |   mean_sq_rate_change |   mean_policy_rate |   std_policy_rate |   improvement_vs_actual_pct |   improvement_vs_taylor_pct |
|:-----------------|:-------------------------|------------------------:|-------------------:|------------------------:|---------------------:|----------------------:|-------------------:|------------------:|----------------------------:|----------------------------:|
| svar             | sac_svar_revealed_direct |                 148.76  |            2.68005 |                1.31262  |              1.13371 |              0.018313 |            6.69631 |          0.526392 |                    60.3697  |                     41.6026 |
| svar             | empirical_taylor_rule    |                 254.738 |            4.82238 |                0.693088 |              1.88343 |              0.122907 |            4.73712 |          1.99524  |                    32.1368  |                      0      |
| svar             | td3_svar_revealed_direct |                 320.306 |            5.67029 |                2.98347  |              1.18981 |              0.081539 |            2.33346 |          0.998335 |                    14.6691  |                    -25.7396 |
| svar             | historical_actual_policy |                 375.37  |            6.71249 |                0.723281 |              1.60241 |              0.227843 |            4.77892 |          2.18807  |                     0       |                    -47.3553 |
| svar             | ppo_svar_revealed_direct |                 388.475 |            7.44599 |                5.43124  |              1.27748 |              0.04423  |            9.58647 |          0.208574 |                    -3.49135 |                    -52.5    |

## ANN Historical Counterfactual

| evaluation_env   | policy_name              |   total_discounted_loss |   mean_period_loss |   mean_sq_inflation_gap |   mean_sq_output_gap |   mean_sq_rate_change |   mean_policy_rate |   std_policy_rate |   improvement_vs_actual_pct |   improvement_vs_taylor_pct |
|:-----------------|:-------------------------|------------------------:|-------------------:|------------------------:|---------------------:|----------------------:|-------------------:|------------------:|----------------------------:|----------------------------:|
| ann              | td3_ann_revealed_direct  |                 186.231 |            3.33914 |                 1.46768 |             1.11815  |              0.04409  |            5.0828  |          0.912082 |                   61.5044   |                     34.7365 |
| ann              | sac_ann_revealed_direct  |                 207.699 |            3.80104 |                 2.8279  |             0.923449 |              0.007912 |            5.02788 |          0.350264 |                   57.0667   |                     27.2131 |
| ann              | empirical_taylor_rule    |                 285.352 |            5.25091 |                 1.55388 |             1.54077  |              0.116428 |            5.21069 |          2.06386  |                   41.0151   |                      0      |
| ann              | historical_actual_policy |                 483.772 |            9.27975 |                 1.55898 |             3.51721  |              0.229996 |            4.75412 |          2.19169  |                    0        |                    -69.5349 |
| ann              | ppo_ann_revealed_direct  |                 484.76  |            9.03991 |                 1.60659 |             6.93982  |              0.065439 |            9.65003 |          0.126661 |                   -0.204328 |                    -69.8813 |

## SVAR Long-Run Stochastic

| policy_name              | evaluation_env   |   mean_discounted_loss |   std_discounted_loss |   mean_reward |   mean_abs_action |   clip_rate |   explosion_rate |
|:-------------------------|:-----------------|-----------------------:|----------------------:|--------------:|------------------:|------------:|-----------------:|
| sac_svar_revealed_direct | svar_revealed    |                199.486 |               64.8218 |      -349.991 |          2.97017  |     0       |                0 |
| empirical_taylor_rule    | svar_revealed    |                259.714 |               84.0344 |      -441.394 |          3.03625  |     0.00599 |                0 |
| td3_svar_revealed_direct | svar_revealed    |                441.624 |              115.533  |      -778.306 |          0.872217 |     0       |                0 |
| historical_actual_policy | svar_revealed    |                581.382 |              192.115  |      -831.382 |          3.0715   |     0       |                0 |
| ppo_svar_revealed_direct | svar_revealed    |                653.976 |              212.672  |     -1044.87  |          7.64274  |     0       |                0 |

## ANN Long-Run Stochastic

| policy_name              | evaluation_env   |   mean_discounted_loss |   std_discounted_loss |   mean_reward |   mean_abs_action |   clip_rate |   explosion_rate |
|:-------------------------|:-----------------|-----------------------:|----------------------:|--------------:|------------------:|------------:|-----------------:|
| sac_ann_revealed_direct  | ann_revealed     |                255.241 |               56.8712 |      -429.674 |           3.26219 |     0       |                0 |
| empirical_taylor_rule    | ann_revealed     |                284.628 |               41.0572 |      -486.064 |           3.82767 |     8.7e-05 |                0 |
| td3_ann_revealed_direct  | ann_revealed     |                315.635 |               58.1501 |      -535.98  |           3.2143  |     0       |                0 |
| ppo_ann_revealed_direct  | ann_revealed     |                664.239 |              190.599  |     -1047.56  |           7.64104 |     0       |                0 |
| historical_actual_policy | ann_revealed     |                786.409 |              225.252  |     -1190.3   |           3.05972 |     0       |                0 |

## Cross Transfer

| policy_name              | evaluation_env   |   mean_discounted_loss |   std_discounted_loss |   mean_reward |   mean_abs_action |   clip_rate |   explosion_rate | source_env   | rule_family          |
|:-------------------------|:-----------------|-----------------------:|----------------------:|--------------:|------------------:|------------:|-----------------:|:-------------|:---------------------|
| sac_ann_revealed_direct  | ann_revealed     |                255.241 |               56.8712 |      -429.674 |          3.26219  |           0 |                0 | ann          | ann_revealed_direct  |
| td3_ann_revealed_direct  | ann_revealed     |                315.635 |               58.1501 |      -535.98  |          3.2143   |           0 |                0 | ann          | ann_revealed_direct  |
| ppo_ann_revealed_direct  | ann_revealed     |                664.239 |              190.599  |     -1047.56  |          7.64104  |           0 |                0 | ann          | ann_revealed_direct  |
| sac_svar_revealed_direct | svar_revealed    |                199.486 |               64.8218 |      -349.991 |          2.97017  |           0 |                0 | svar         | svar_revealed_direct |
| td3_svar_revealed_direct | svar_revealed    |                441.624 |              115.533  |      -778.306 |          0.872217 |           0 |                0 | svar         | svar_revealed_direct |
| ppo_svar_revealed_direct | svar_revealed    |                653.976 |              212.672  |     -1044.87  |          7.64274  |           0 |                0 | svar         | svar_revealed_direct |