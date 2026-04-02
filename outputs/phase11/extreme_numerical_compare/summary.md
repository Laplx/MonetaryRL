# Phase 11 新增两档数值求解对照

## 总表

| env_id                 | group      | tier        |   dp_mean_discounted_loss |   riccati_re_eval_mean_discounted_loss |   linear_search_re_eval_mean_discounted_loss | best_rl_algo   |   best_rl_seed |   best_rl_surrogate_re_eval_mean_discounted_loss |   dp_improvement_vs_riccati_pct |   dp_improvement_vs_best_rl_surrogate_pct |   solve_seconds |
|:-----------------------|:-----------|:------------|--------------------------:|---------------------------------------:|---------------------------------------------:|:---------------|---------------:|-------------------------------------------------:|--------------------------------:|------------------------------------------:|----------------:|
| nonlinear_very_strong  | nonlinear  | very_strong |                   21.3718 |                                21.9432 |                                      21.6602 | sac            |              7 |                                          24.8765 |                        2.67351  |                                 16.3988   |         41.0721 |
| nonlinear_extreme      | nonlinear  | extreme     |                  294.969  |                               432.84   |                                     398.715  | td3            |              7 |                                         850.047  |                       46.7407   |                                188.181    |         38.9094 |
| zlb_very_strong        | zlb        | very_strong |                   60.8644 |                                61.1114 |                                      61.7213 | td3            |             43 |                                          65.7672 |                        0.4058   |                                  8.05517  |         45.189  |
| zlb_extreme            | zlb        | extreme     |                  102.487  |                               102.608  |                                     102.825  | td3            |             43 |                                         103.966  |                        0.118362 |                                  1.44339  |         39.6496 |
| asymmetric_very_strong | asymmetric | very_strong |                   41.5262 |                                42.333  |                                      41.519  | sac            |             43 |                                          41.9286 |                        1.94274  |                                  0.968798 |         52.4763 |
| asymmetric_extreme     | asymmetric | extreme     |                   68.144  |                                71.1152 |                                      69.1718 | sac            |             43 |                                          72.8667 |                        4.36014  |                                  6.93041  |         52.6968 |

## 结论

- 本表只覆盖 `phase11` 新增的两档 × 三类环境。
- 若 `dp_improvement_vs_riccati_pct` 为正，则说明新增扭曲下传统 benchmark Riccati 外推已被环境内数值解压过。