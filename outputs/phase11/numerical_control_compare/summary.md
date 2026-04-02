# Phase 11 传统数值求解对照

## 总表

| case_id          |   dp_mean_discounted_loss |   best_rl_actual_mean_discounted_loss | best_rl_algo   |   best_rl_seed |   riccati_re_eval_mean_discounted_loss |   linear_search_re_eval_mean_discounted_loss |   best_rl_surrogate_re_eval_mean_discounted_loss |   dp_improvement_vs_best_rl_actual_pct |   dp_improvement_vs_riccati_re_eval_pct |   solve_seconds |
|:-----------------|--------------------------:|--------------------------------------:|:---------------|---------------:|---------------------------------------:|---------------------------------------------:|-------------------------------------------------:|---------------------------------------:|----------------------------------------:|----------------:|
| nonlinear_strong |                   14.4647 |                               16.4799 | sac            |              7 |                                14.4338 |                                      14.6238 |                                          16.8512 |                                13.9324 |                               -0.213636 |         42.5218 |
| zlb_strong       |                   22.2258 |                               26.1654 | td3            |             43 |                                22.2553 |                                      22.7621 |                                          32.0988 |                                17.7256 |                                0.132801 |         59.6756 |

## 观察

- `nonlinear_strong`：数值 DP `14.465`，略差于 `Riccati reference` 的 `14.434`，但优于最优 RL 缓存结果 `16.480`。
- `zlb_strong`：数值 DP `22.226`，略优于 `Riccati reference` 的 `22.255`，也优于最优 RL 缓存结果 `26.165`。
- 这两组代表环境下，当前证据不支持把“传统数值法失效而 RL 明显更优”作为新增主结论。
- 更稳妥的表述是：在当前低维扩展环境里，传统数值控制仍然可做且表现很强；RL 的价值更多体现在统一实现框架与可扩展性，而非在这两组环境中数值上显著压过传统方法。

## 说明

- `finite_horizon_dp` 是在扩展环境上直接做状态-动作离散化与有限期 Bellman backward induction 的数值解。
- `best_rl_actual_mean_discounted_loss` 直接复用 `Phase 7` 缓存评估结果。
- `best_rl_surrogate_re_eval_mean_discounted_loss` 使用 `Phase 7` 最优 RL 的线性拟合 surrogate 在同一评价器下重评，仅作近似结构对照。
- 本轮先做 `nonlinear_strong` 与 `zlb_strong` 两个代表环境，不改动既有 `phase10` 材料。