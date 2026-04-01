# Phase 8 经验 SVAR 反事实总结

## 1. 任务完成情况

| 项目 | 结果 |
|---|---|
| 主环境 | `SVAR` |
| 已比较对象 | `historical_actual_policy`、`empirical_taylor_rule`、`riccati_reference`、`linear_policy_search_transfer`、`ppo/td3/sac_benchmark_transfer` |
| 历史反事实 | 已完成 |
| 长期随机评估 | 已完成（仅反馈规则） |
| ANN | 未进入主结果，保留到 `Phase 9` 门槛判断 |

## 2. 历史反事实主表

| policy_name                   |   total_discounted_loss |   mean_period_loss |   mean_sq_inflation_gap |   mean_sq_output_gap |   mean_sq_rate_change |   mean_policy_rate |   std_policy_rate |   improvement_vs_actual_pct |   improvement_vs_taylor_pct |
|:------------------------------|------------------------:|-------------------:|------------------------:|---------------------:|----------------------:|-------------------:|------------------:|----------------------------:|----------------------------:|
| historical_actual_policy      |                 95.8255 |            1.64986 |                0.797429 |              1.65929 |              0.227843 |            4.77892 |          2.18807  |                     0       |                      5.6056 |
| empirical_taylor_rule         |                101.516  |            1.74061 |                0.736596 |              1.98219 |              0.129198 |            4.76951 |          2.08396  |                    -5.93849 |                      0      |
| sac_benchmark_transfer        |                117.89   |            2.02736 |                1.02023  |              1.90173 |              0.562667 |            4.30086 |          2.39834  |                   -23.0257  |                    -16.1294 |
| linear_policy_search_transfer |                123.611  |            2.13808 |                1.19787  |              1.79016 |              0.451286 |            4.04752 |          2.12412  |                   -28.9958  |                    -21.7648 |
| riccati_reference             |                123.977  |            2.13599 |                1.10541  |              1.935   |              0.630797 |            4.1843  |          2.51108  |                   -29.3782  |                    -22.1258 |
| td3_benchmark_transfer        |                131.665  |            2.26934 |                1.25112  |              1.9016  |              0.674237 |            4.00699 |          2.47384  |                   -37.4005  |                    -29.6983 |
| ppo_benchmark_transfer        |                174.957  |            3.05884 |                2.40668  |              1.26538 |              0.194699 |            2.90436 |          0.775682 |                   -82.5789  |                    -72.3442 |

## 3. 长期随机评估

| policy_name                   |   mean_discounted_loss |   std_discounted_loss |   mean_reward |   mean_abs_action |   clip_rate |   explosion_rate |
|:------------------------------|-----------------------:|----------------------:|--------------:|------------------:|------------:|-----------------:|
| empirical_taylor_rule         |                90.0604 |               33.3361 |      -129.615 |           3.03917 |    0.006543 |                0 |
| sac_benchmark_transfer        |               102.726  |               31.6078 |      -148.948 |           2.63463 |    0.042432 |                0 |
| riccati_reference             |               110.196  |               33.6946 |      -159.851 |           2.65236 |    0.057666 |                0 |
| linear_policy_search_transfer |               113.073  |               35.3516 |      -164.772 |           2.43734 |    0.032471 |                0 |
| td3_benchmark_transfer        |               118.422  |               35.9631 |      -174.368 |           2.58067 |    0.083008 |                0 |
| ppo_benchmark_transfer        |               174.477  |               61.6129 |      -264.914 |           1.1213  |    0.000537 |                0 |

## 4. 规则登记表

| policy_name                   | rule_type        | source                     |   intercept |   inflation_gap |   output_gap |   lagged_policy_rate_gap |   fit_rmse | note                                                                        | algo   |   seed |   benchmark_mean_discounted_loss |
|:------------------------------|:-----------------|:---------------------------|------------:|----------------:|-------------:|-------------------------:|-----------:|:----------------------------------------------------------------------------|:-------|-------:|---------------------------------:|
| historical_actual_policy      | historical_path  | data                       |  nan        |      nan        |   nan        |               nan        | nan        | Observed policy-rate path in the sample                                     | nan    |    nan |                         nan      |
| empirical_taylor_rule         | estimated_rule   | phase2                     |    0.216573 |        0.193173 |     0.265769 |                 0.8692   |   0        | Estimated Taylor rule converted to gap form                                 | nan    |    nan |                         nan      |
| riccati_reference             | theory_reference | phase4                     |    0        |        1.0891   |     1.07844  |                 0.434052 |   0        | Theoretical Riccati benchmark rule                                          | nan    |    nan |                         nan      |
| linear_policy_search_transfer | linear           | phase6                     |    0        |        1.15759  |     0.923679 |                 0.379484 |   0        | Phase 6 benchmark linear policy search coefficients                         | nan    |    nan |                         nan      |
| ppo_benchmark_transfer        | linear_surrogate | phase7_benchmark_best_seed |   -0.101581 |        0.480967 |     0.477127 |                 0.262907 |   0.016044 | Transferred benchmark-trained RL rule represented by saved linear surrogate | ppo    |      7 |                          20.669  |
| td3_benchmark_transfer        | linear_surrogate | phase7_benchmark_best_seed |   -0.082606 |        0.968165 |     1.17828  |                 0.417105 |   0.436459 | Transferred benchmark-trained RL rule represented by saved linear surrogate | td3    |     29 |                          17.7432 |
| sac_benchmark_transfer        | linear_surrogate | phase7_benchmark_best_seed |    0.19549  |        1.17356  |     1.02139  |                 0.396672 |   0.357647 | Transferred benchmark-trained RL rule represented by saved linear surrogate | sac    |     43 |                          16.4973 |

## 5. Phase 9 门槛判断

| module           | status         | reason                                                         | next_step                                                        |
|:-----------------|:---------------|:---------------------------------------------------------------|:-----------------------------------------------------------------|
| ann_phase9_gate  | not_passed_yet | inflation equation still underperforms SVAR in Phase 2 summary | keep ANN as Phase 9 supplementary module after SVAR main results |
| dsge_phase9_gate | deferred       | model uncertainty extension is intentionally placed in Phase 9 | prepare transferable linear rule registry and coefficient table  |

## 6. 说明

- 历史实际政策在长期随机评估中未纳入，因为它是样本路径而不是固定反馈规则。
- benchmark 训练得到的 RL 规则在 `Phase 8` 中以已保存的线性 surrogate 形式迁入经验环境，而不是在 `SVAR` 环境中重新训练。
- 历史实际政策复现实验的最大绝对误差为 `0.0000000684`，说明 recovered shocks 与递归 SVAR 转移实现是对齐的。
- 写作中必须明确 `Lucas critique`：经验转移固定不变只是方法近似。
