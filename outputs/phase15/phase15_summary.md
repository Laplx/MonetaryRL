# Phase 15 beta / lambda 稳健性

## 经验环境：最佳 RL 相对最佳非 RL 基线

| evaluation_env   | loss_function   | evaluation_type           | scenario_group   | scenario_id            | best_rl_policy            |   best_rl_loss | best_baseline_policy         |   best_baseline_loss |   rl_advantage_pct |
|:-----------------|:----------------|:--------------------------|:-----------------|:-----------------------|:--------------------------|---------------:|:-----------------------------|---------------------:|-------------------:|
| ann              | artificial      | historical_counterfactual | beta             | beta_097               | ppo_ann_direct            |        44.4452 | ann_affine_search_artificial |              42.6332 |          -4.25023  |
| ann              | artificial      | historical_counterfactual | beta             | beta_099               | ppo_ann_direct            |        69.2297 | ann_affine_search_artificial |              65.4536 |          -5.76909  |
| ann              | artificial      | historical_counterfactual | beta             | beta_0995              | ppo_ann_direct            |        79.9652 | ann_affine_search_artificial |              74.737  |          -6.99547  |
| ann              | artificial      | historical_counterfactual | lambda           | lambda_art_output_high | ppo_ann_direct            |       101.133  | ann_affine_search_artificial |             102.868  |           1.68645  |
| ann              | artificial      | historical_counterfactual | lambda           | lambda_art_output_low  | td3_ann_direct            |        49.3548 | ann_affine_search_artificial |              46.7464 |          -5.5799   |
| ann              | artificial      | historical_counterfactual | lambda           | lambda_art_smooth_high | ppo_ann_direct            |        70.6447 | ann_affine_search_artificial |              74.8115 |           5.5698   |
| ann              | artificial      | historical_counterfactual | lambda           | lambda_art_smooth_low  | td3_ann_direct            |        65.9557 | ann_affine_search_artificial |              60.7746 |          -8.52505  |
| svar             | artificial      | historical_counterfactual | beta             | beta_097               | ppo_svar_direct_nonlinear |        57.8732 | historical_actual_policy     |              55.8129 |          -3.6913   |
| svar             | artificial      | historical_counterfactual | beta             | beta_099               | ppo_ann_direct_nonlinear  |        87.3361 | historical_actual_policy     |              87.916  |           0.659613 |
| svar             | artificial      | historical_counterfactual | beta             | beta_0995              | ppo_ann_direct_nonlinear  |        98.194  | historical_actual_policy     |             101.615  |           3.36622  |
| svar             | artificial      | historical_counterfactual | lambda           | lambda_art_output_high | ppo_ann_direct_nonlinear  |       123.686  | historical_actual_policy     |             130.55   |           5.25769  |
| svar             | artificial      | historical_counterfactual | lambda           | lambda_art_output_low  | ppo_ann_direct_nonlinear  |        69.1609 | historical_actual_policy     |              66.5988 |          -3.84707  |
| svar             | artificial      | historical_counterfactual | lambda           | lambda_art_smooth_high | ppo_ann_direct_nonlinear  |        87.9773 | historical_actual_policy     |              89.1914 |           1.36131  |
| svar             | artificial      | historical_counterfactual | lambda           | lambda_art_smooth_low  | ppo_ann_direct_nonlinear  |        87.0155 | historical_actual_policy     |              87.2783 |           0.301071 |
| ann              | artificial      | stochastic_long_run       | beta             | beta_097               | ppo_ann_direct            |        36.0753 | ann_affine_search_artificial |              31.5548 |         -14.3259   |
| ann              | artificial      | stochastic_long_run       | beta             | beta_099               | ppo_ann_direct            |        75.1152 | ann_affine_search_artificial |              65.9113 |         -13.9641   |
| ann              | artificial      | stochastic_long_run       | beta             | beta_0995              | ppo_ann_direct            |        96.1164 | ann_affine_search_artificial |              84.6059 |         -13.6048   |
| ann              | artificial      | stochastic_long_run       | lambda           | lambda_art_output_high | ppo_ann_direct            |       113.374  | ann_affine_search_artificial |             104.494  |          -8.49763  |
| ann              | artificial      | stochastic_long_run       | lambda           | lambda_art_output_low  | td3_ann_direct            |        54.3811 | ann_affine_search_artificial |              46.6198 |         -16.648    |
| ann              | artificial      | stochastic_long_run       | lambda           | lambda_art_smooth_high | ppo_ann_direct            |        77.0668 | ann_affine_search_artificial |              75.9097 |          -1.52427  |
| ann              | artificial      | stochastic_long_run       | lambda           | lambda_art_smooth_low  | td3_ann_direct            |        72.6026 | ann_affine_search_artificial |              60.912  |         -19.1925   |
| svar             | artificial      | stochastic_long_run       | beta             | beta_097               | sac_svar_direct           |        46.1175 | empirical_taylor_rule        |              52.3523 |          11.9093   |
| svar             | artificial      | stochastic_long_run       | beta             | beta_099               | sac_svar_direct           |        96.5894 | empirical_taylor_rule        |             112.061  |          13.8062   |
| svar             | artificial      | stochastic_long_run       | beta             | beta_0995              | sac_svar_direct           |       123.94   | empirical_taylor_rule        |             144.2    |          14.05     |
| svar             | artificial      | stochastic_long_run       | lambda           | lambda_art_output_high | sac_svar_direct           |       152.292  | empirical_taylor_rule        |             173.745  |          12.3475   |
| svar             | artificial      | stochastic_long_run       | lambda           | lambda_art_output_low  | sac_svar_direct           |        68.738  | empirical_taylor_rule        |              81.2184 |          15.3664   |
| svar             | artificial      | stochastic_long_run       | lambda           | lambda_art_smooth_high | sac_svar_direct           |        99.4785 | empirical_taylor_rule        |             112.695  |          11.7275   |
| svar             | artificial      | stochastic_long_run       | lambda           | lambda_art_smooth_low  | sac_svar_direct           |        95.1448 | empirical_taylor_rule        |             111.744  |          14.8544   |
| ann              | revealed        | historical_counterfactual | beta             | beta_097               | td3_ann_revealed_direct   |       116.933  | ann_affine_search_revealed   |             126.469  |           7.54025  |
| ann              | revealed        | historical_counterfactual | beta             | beta_099               | td3_ann_revealed_direct   |       186.231  | ann_affine_search_revealed   |             207.583  |          10.286    |
| ann              | revealed        | historical_counterfactual | beta             | beta_0995              | td3_ann_revealed_direct   |       215.866  | ann_affine_search_revealed   |             241.665  |          10.6756   |
| ann              | revealed        | historical_counterfactual | lambda           | lambda_rev_output_down | td3_ann_revealed_direct   |       160.585  | ann_affine_search_revealed   |             184.228  |          12.8333   |
| ann              | revealed        | historical_counterfactual | lambda           | lambda_rev_output_up   | td3_ann_revealed_direct   |       211.876  | ann_affine_search_revealed   |             230.938  |           8.25399  |
| ann              | revealed        | historical_counterfactual | lambda           | lambda_rev_smooth_down | td3_ann_revealed_direct   |       161.17   | ann_affine_search_revealed   |             193.818  |          16.8443   |
| ann              | revealed        | historical_counterfactual | lambda           | lambda_rev_smooth_up   | td3_ann_revealed_direct   |       211.291  | ann_affine_search_revealed   |             221.348  |           4.54348  |
| svar             | revealed        | historical_counterfactual | beta             | beta_097               | ppo_svar_direct           |        80.7275 | empirical_taylor_rule        |             138.846  |          41.8583   |
| svar             | revealed        | historical_counterfactual | beta             | beta_099               | ppo_svar_direct           |       134.662  | empirical_taylor_rule        |             254.738  |          47.1371   |
| svar             | revealed        | historical_counterfactual | beta             | beta_0995              | ppo_svar_direct           |       157.076  | empirical_taylor_rule        |             305.68   |          48.614    |
| svar             | revealed        | historical_counterfactual | lambda           | lambda_rev_output_down | ppo_svar_direct           |       105.148  | empirical_taylor_rule        |             212.046  |          50.4128   |
| svar             | revealed        | historical_counterfactual | lambda           | lambda_rev_output_up   | ppo_svar_direct           |       164.176  | empirical_taylor_rule        |             297.43   |          44.8018   |
| svar             | revealed        | historical_counterfactual | lambda           | lambda_rev_smooth_down | ppo_svar_direct           |       130.587  | empirical_taylor_rule        |             192.794  |          32.266    |
| svar             | revealed        | historical_counterfactual | lambda           | lambda_rev_smooth_up   | ppo_svar_direct           |       138.737  | empirical_taylor_rule        |             316.682  |          56.1905   |
| ann              | revealed        | stochastic_long_run       | beta             | beta_097               | sac_ann_revealed_direct   |       121.496  | ann_affine_search_revealed   |             106.524  |         -14.0548   |
| ann              | revealed        | stochastic_long_run       | beta             | beta_099               | sac_ann_revealed_direct   |       253.634  | ann_affine_search_revealed   |             218.369  |         -16.1493   |
| ann              | revealed        | stochastic_long_run       | beta             | beta_0995              | sac_ann_revealed_direct   |       324.547  | ann_affine_search_revealed   |             278.962  |         -16.3409   |
| ann              | revealed        | stochastic_long_run       | lambda           | lambda_rev_output_down | sac_ann_revealed_direct   |       208.914  | ann_affine_search_revealed   |             181.396  |         -15.1701   |
| ann              | revealed        | stochastic_long_run       | lambda           | lambda_rev_output_up   | sac_ann_revealed_direct   |       298.355  | ann_affine_search_revealed   |             255.343  |         -16.8449   |
| ann              | revealed        | stochastic_long_run       | lambda           | lambda_rev_smooth_down | sac_ann_revealed_direct   |       241.747  | ann_affine_search_revealed   |             193.703  |         -24.8028   |
| ann              | revealed        | stochastic_long_run       | lambda           | lambda_rev_smooth_up   | sac_ann_revealed_direct   |       265.521  | ann_affine_search_revealed   |             243.035  |          -9.25223  |
| svar             | revealed        | stochastic_long_run       | beta             | beta_097               | sac_svar_revealed_direct  |        92.4573 | empirical_taylor_rule        |             133.138  |          30.5555   |
| svar             | revealed        | stochastic_long_run       | beta             | beta_099               | sac_svar_revealed_direct  |       202.92   | empirical_taylor_rule        |             285.868  |          29.0164   |
| svar             | revealed        | stochastic_long_run       | beta             | beta_0995              | sac_svar_revealed_direct  |       263.572  | empirical_taylor_rule        |             368.704  |          28.5139   |
| svar             | revealed        | stochastic_long_run       | lambda           | lambda_rev_output_down | sac_svar_revealed_direct  |       159.299  | empirical_taylor_rule        |             231.48   |          31.1823   |
| svar             | revealed        | stochastic_long_run       | lambda           | lambda_rev_output_up   | sac_svar_revealed_direct  |       246.54   | empirical_taylor_rule        |             340.257  |          27.543    |
| svar             | revealed        | stochastic_long_run       | lambda           | lambda_rev_smooth_down | sac_svar_revealed_direct  |       192.904  | empirical_taylor_rule        |             222.193  |          13.182    |
| svar             | revealed        | stochastic_long_run       | lambda           | lambda_rev_smooth_up   | sac_svar_revealed_direct  |       212.935  | empirical_taylor_rule        |             349.543  |          39.0819   |

## 经验环境：RL 胜出次数

|   index | loss_function   | evaluation_type           | evaluation_env   |   sum |   count |
|--------:|:----------------|:--------------------------|:-----------------|------:|--------:|
|       0 | artificial      | historical_counterfactual | ann              |     2 |       7 |
|       1 | artificial      | historical_counterfactual | svar             |     5 |       7 |
|       2 | artificial      | stochastic_long_run       | ann              |     0 |       7 |
|       3 | artificial      | stochastic_long_run       | svar             |     7 |       7 |
|       4 | revealed        | historical_counterfactual | ann              |     7 |       7 |
|       5 | revealed        | historical_counterfactual | svar             |     7 |       7 |
|       6 | revealed        | stochastic_long_run       | ann              |     0 |       7 |
|       7 | revealed        | stochastic_long_run       | svar             |     7 |       7 |

## 经验环境：少量重训检查

| evaluation_env   | loss_function   | scenario_id            | policy_name                                        | algo   |   historical_loss |   stochastic_loss |
|:-----------------|:----------------|:-----------------------|:---------------------------------------------------|:-------|------------------:|------------------:|
| svar             | artificial      | beta_097               | ppo_svar_artificial_beta_097_phase15               | ppo    |           62.818  |           51.1539 |
| svar             | artificial      | beta_097               | td3_svar_artificial_beta_097_phase15               | td3    |           60.667  |           53.3478 |
| svar             | artificial      | beta_097               | sac_svar_artificial_beta_097_phase15               | sac    |           60.1467 |           43.7954 |
| svar             | artificial      | lambda_art_output_high | ppo_svar_artificial_lambda_art_output_high_phase15 | ppo    |          140.348  |          176.803  |
| svar             | artificial      | lambda_art_output_high | td3_svar_artificial_lambda_art_output_high_phase15 | td3    |          135.792  |          178.832  |
| svar             | artificial      | lambda_art_output_high | sac_svar_artificial_lambda_art_output_high_phase15 | sac    |          128.016  |          149.404  |
| svar             | revealed        | beta_097               | ppo_svar_revealed_beta_097_phase15                 | ppo    |          223.986  |          393.392  |
| svar             | revealed        | beta_097               | td3_svar_revealed_beta_097_phase15                 | td3    |          188.845  |          179.281  |
| svar             | revealed        | beta_097               | sac_svar_revealed_beta_097_phase15                 | sac    |          115.416  |          102.57   |
| svar             | revealed        | lambda_rev_smooth_down | ppo_svar_revealed_lambda_rev_smooth_down_phase15   | ppo    |          354.758  |          558.956  |
| svar             | revealed        | lambda_rev_smooth_down | td3_svar_revealed_lambda_rev_smooth_down_phase15   | td3    |          146.289  |          216.008  |
| svar             | revealed        | lambda_rev_smooth_down | sac_svar_revealed_lambda_rev_smooth_down_phase15   | sac    |          120.505  |          174.65   |
| ann              | artificial      | beta_097               | ppo_ann_artificial_beta_097_phase15                | ppo    |           42.0867 |           31.3561 |
| ann              | artificial      | beta_097               | td3_ann_artificial_beta_097_phase15                | td3    |           46.4343 |           38.3641 |
| ann              | artificial      | beta_097               | sac_ann_artificial_beta_097_phase15                | sac    |           58.6932 |           42.7009 |
| ann              | artificial      | lambda_art_output_high | ppo_ann_artificial_lambda_art_output_high_phase15  | ppo    |          171.623  |          145.311  |
| ann              | artificial      | lambda_art_output_high | td3_ann_artificial_lambda_art_output_high_phase15  | td3    |          123.659  |          112.419  |
| ann              | artificial      | lambda_art_output_high | sac_ann_artificial_lambda_art_output_high_phase15  | sac    |          168.05   |          165.393  |
| ann              | revealed        | beta_097               | ppo_ann_revealed_beta_097_phase15                  | ppo    |          302.655  |          421.08   |
| ann              | revealed        | beta_097               | td3_ann_revealed_beta_097_phase15                  | td3    |          129.318  |          119.98   |
| ann              | revealed        | beta_097               | sac_ann_revealed_beta_097_phase15                  | sac    |          117.607  |          112.232  |
| ann              | revealed        | lambda_rev_smooth_down | ppo_ann_revealed_lambda_rev_smooth_down_phase15    | ppo    |          417.415  |          547.743  |
| ann              | revealed        | lambda_rev_smooth_down | td3_ann_revealed_lambda_rev_smooth_down_phase15    | td3    |          208.272  |          233.949  |
| ann              | revealed        | lambda_rev_smooth_down | sac_ann_revealed_lambda_rev_smooth_down_phase15    | sac    |          204.2    |          231.793  |

## phase11 六环境：最佳 RL 相对 Riccati 外推

| env_id                           | scenario_id            |   best_rl_loss |   riccati_loss |   rl_advantage_pct |
|:---------------------------------|:-----------------------|---------------:|---------------:|-------------------:|
| asymmetric_threshold_extreme     | beta_097               |       305.652  |       345.349  |           11.4948  |
| asymmetric_threshold_extreme     | beta_099               |       320.972  |       366.134  |           12.3349  |
| asymmetric_threshold_extreme     | beta_0995              |       326.567  |       373.718  |           12.6168  |
| asymmetric_threshold_extreme     | lambda_art_output_high |       330.553  |       363.788  |            9.13584 |
| asymmetric_threshold_extreme     | lambda_art_output_low  |       316.182  |       370.853  |           14.742   |
| asymmetric_threshold_extreme     | lambda_art_smooth_high |       325.996  |       385.995  |           15.544   |
| asymmetric_threshold_extreme     | lambda_art_smooth_low  |       318.46   |       350.905  |            9.24603 |
| asymmetric_threshold_very_strong | beta_097               |       101.602  |       110.561  |            8.10296 |
| asymmetric_threshold_very_strong | beta_099               |       110.35   |       119.895  |            7.96103 |
| asymmetric_threshold_very_strong | beta_0995              |       113.582  |       123.34   |            7.91195 |
| asymmetric_threshold_very_strong | lambda_art_output_high |       120.116  |       125.455  |            4.25566 |
| asymmetric_threshold_very_strong | lambda_art_output_low  |       105.467  |       118.074  |           10.6774  |
| asymmetric_threshold_very_strong | lambda_art_smooth_high |       112.835  |       127.162  |           11.2669  |
| asymmetric_threshold_very_strong | lambda_art_smooth_low  |       109.107  |       113.927  |            4.23031 |
| nonlinear_extreme_v2             | beta_097               |       866.265  |       568.382  |          -52.4089  |
| nonlinear_extreme_v2             | beta_099               |      1054.96   |       640.225  |          -64.7789  |
| nonlinear_extreme_v2             | beta_0995              |      1112.83   |       662.266  |          -68.0341  |
| nonlinear_extreme_v2             | lambda_art_output_high |      1060.69   |       770.015  |          -37.7487  |
| nonlinear_extreme_v2             | lambda_art_output_low  |      1052.09   |       677.084  |          -55.3853  |
| nonlinear_extreme_v2             | lambda_art_smooth_high |      1055.77   |       691.675  |          -52.64    |
| nonlinear_extreme_v2             | lambda_art_smooth_low  |      1054.55   |       660.361  |          -59.6923  |
| nonlinear_hyper                  | beta_097               |       848.25   |       754.415  |          -12.4381  |
| nonlinear_hyper                  | beta_099               |       947.737  |       830.154  |          -14.1641  |
| nonlinear_hyper                  | beta_0995              |       976.034  |       892.894  |           -9.3113  |
| nonlinear_hyper                  | lambda_art_output_high |       951.506  |       847.658  |          -12.2511  |
| nonlinear_hyper                  | lambda_art_output_low  |       945.853  |       891.716  |           -6.07111 |
| nonlinear_hyper                  | lambda_art_smooth_high |       948.484  |       914.819  |           -3.68    |
| nonlinear_hyper                  | lambda_art_smooth_low  |       947.364  |       958.058  |            1.11622 |
| zlb_trap_extreme                 | beta_097               |        92.4301 |        90.2243 |           -2.44485 |
| zlb_trap_extreme                 | beta_099               |       114.532  |       112.133  |           -2.13967 |
| zlb_trap_extreme                 | beta_0995              |       122.248  |       119.857  |           -1.99468 |
| zlb_trap_extreme                 | lambda_art_output_high |       144.403  |       141.375  |           -2.14187 |
| zlb_trap_extreme                 | lambda_art_output_low  |        99.5966 |        97.5041 |           -2.14613 |
| zlb_trap_extreme                 | lambda_art_smooth_high |       114.545  |       112.101  |           -2.18009 |
| zlb_trap_extreme                 | lambda_art_smooth_low  |       114.525  |       112.176  |           -2.09479 |
| zlb_trap_very_strong             | beta_097               |        58.8897 |        55.3703 |           -6.35619 |
| zlb_trap_very_strong             | beta_099               |        75.1711 |        69.6155 |           -7.98049 |
| zlb_trap_very_strong             | beta_0995              |        81.0202 |        74.7111 |           -8.44466 |
| zlb_trap_very_strong             | lambda_art_output_high |        97.4125 |        90.3213 |           -7.85101 |
| zlb_trap_very_strong             | lambda_art_output_low  |        64.0504 |        59.2469 |           -8.10766 |
| zlb_trap_very_strong             | lambda_art_smooth_high |        75.1886 |        69.7292 |           -7.82942 |
| zlb_trap_very_strong             | lambda_art_smooth_low  |        75.1624 |        69.5636 |           -8.0484  |

## phase11 代表环境：少量重训检查

| env_id                       | scenario_id            | algo   |   mean_discounted_loss |   std_discounted_loss |   mean_abs_action |   clip_rate |   explosion_rate |
|:-----------------------------|:-----------------------|:-------|-----------------------:|----------------------:|------------------:|------------:|-----------------:|
| nonlinear_hyper              | beta_097               | td3    |               813.036  |              512.484  |           1.98772 |           0 |         0.979167 |
| nonlinear_hyper              | lambda_art_smooth_high | td3    |               883.41   |              533.441  |           1.7588  |           0 |         1        |
| zlb_trap_extreme             | beta_097               | td3    |                99.7527 |               64.8701 |           0.25    |           0 |         0        |
| zlb_trap_extreme             | lambda_art_smooth_high | td3    |               121.26   |               72.1546 |           0.25    |           0 |         0        |
| asymmetric_threshold_extreme | beta_097               | sac    |               380.682  |              503.694  |           1.40417 |           0 |         0        |
| asymmetric_threshold_extreme | lambda_art_smooth_high | sac    |               413.529  |              506.76   |           1.23086 |           0 |         0        |

## 文件

- `empirical_beta_lambda_full_matrix.csv`：SVAR / ANN 全部重评估矩阵
- `empirical_rank_compare.csv`：最佳 RL 相对最佳非 RL 基线比较
- `empirical_retrain_summary.csv`：经验环境少量重训比较
- `phase11_beta_lambda_matrix.csv`：phase11 六环境稳健性表
- `phase11_selected_retrain_summary.csv`：phase11 代表环境少量重训
- `figures/phase15_empirical_rank_heatmap.png`：经验环境稳健性热图
- `figures/phase15_phase11_riccati_heatmap.png`：phase11 相对 Riccati 热图