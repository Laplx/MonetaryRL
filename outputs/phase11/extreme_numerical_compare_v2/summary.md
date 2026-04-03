# Phase 11 v2 数值求解对照

## 总表

| env_id                           | group      | tier           |   dp_mean_discounted_loss |   riccati_re_eval_mean_discounted_loss |   linear_search_re_eval_mean_discounted_loss | best_rl_algo   |   best_rl_seed |   best_rl_surrogate_re_eval_mean_discounted_loss |   dp_improvement_vs_riccati_pct |   dp_improvement_vs_best_rl_surrogate_pct |   solve_seconds |
|:---------------------------------|:-----------|:---------------|--------------------------:|---------------------------------------:|---------------------------------------------:|:---------------|---------------:|-------------------------------------------------:|--------------------------------:|------------------------------------------:|----------------:|
| nonlinear_extreme_v2             | nonlinear  | extreme_v2     |                  604.462  |                               852.347  |                                     940.272  | td3            |             43 |                                        1039.26   |                       41.009    |                                  71.9321  |         56.0848 |
| nonlinear_hyper                  | nonlinear  | hyper          |                  916.667  |                               851.988  |                                     815.141  | td3            |             43 |                                         890.623  |                       -7.05594  |                                  -2.84121 |         59.1244 |
| zlb_trap_very_strong             | zlb        | very_strong_v2 |                   65.6737 |                                65.9322 |                                      66.4399 | td3            |             43 |                                          70.4201 |                        0.393543 |                                   7.2273  |         61.4542 |
| zlb_trap_extreme                 | zlb        | extreme_v2     |                  105.756  |                               106.021  |                                     105.888  | td3            |             43 |                                         107.034  |                        0.250162 |                                   1.20809 |         58.7446 |
| asymmetric_threshold_very_strong | asymmetric | very_strong_v2 |                  100.95   |                               101.227  |                                     100.46   | ppo            |             43 |                                          99.8372 |                        0.274399 |                                  -1.10206 |         73.3513 |
| asymmetric_threshold_extreme     | asymmetric | extreme_v2     |                  310.113  |                               311.131  |                                     305.174  | sac            |             43 |                                         297.789  |                        0.328149 |                                  -3.97395 |         72.9991 |

## 结论

- 本表只覆盖 `phase11 v2` 六个新增环境。
- 若 `dp_improvement_vs_riccati_pct` 为正，则说明传统 benchmark Riccati 外推已被环境内数值解压过。
- 若 `dp_improvement_vs_best_rl_surrogate_pct` 为正，则说明数值最优仍优于 RL 线性 surrogate。