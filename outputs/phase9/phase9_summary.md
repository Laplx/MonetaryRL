# Phase 9 ANN 补充与 Model Uncertainty 总结

## 1. 任务完成情况

| 项目 | 结果 |
|---|---|
| ANN 调优 | 已完成 |
| ANN 补充反事实 | 已完成 |
| local model uncertainty | 已完成 |
| 外部 DSGE/MMB 模型 | 仓库中暂无现成文件，已保留规则接口 |

## 2. ANN 调优选型

| equation | feature_set | stage_name | hidden_layer_sizes | activation | solver | alpha | learning_rate_init | full_sample_mse |
|---|---|---|---|---|---|---|---|---|
| output_gap | extra_lag | stage3_extra_lag | (3,) | relu | adam | 0.000100 | 0.001000 | 0.205186 |
| inflation | extra_lag | stage3_extra_lag | (3,) | tanh | lbfgs | 0.001000 | 0.001000 | 0.027753 |

## 3. ANN 拟合门槛表

| equation   |   svar_mse |   phase2_ann_mse |   phase9_tuned_ann_mse |   improvement_vs_phase2_pct |   improvement_vs_svar_pct | inflation_gate   | output_gate   |   dynamic_gate_reference |   ann_best_feedback_loss |
|:-----------|-----------:|-----------------:|-----------------------:|----------------------------:|--------------------------:|:-----------------|:--------------|-------------------------:|-------------------------:|
| output_gap |  0.210391  |        0.18223   |              0.205186  |                    -12.5972 |                   2.47381 | False            | False         |              6.83781e-08 |                  113.005 |
| inflation  |  0.0295402 |        0.0434236 |              0.0277533 |                     36.0872 |                   6.0493  | True             | False         |              6.83781e-08 |                  113.005 |

## 4. ANN 历史反事实主表

| policy_name                   |   total_discounted_loss |   mean_period_loss |   mean_sq_inflation_gap |   mean_sq_output_gap |   mean_sq_rate_change |   mean_policy_rate |   std_policy_rate |   improvement_vs_actual_pct |   improvement_vs_taylor_pct |
|:------------------------------|------------------------:|-------------------:|------------------------:|---------------------:|----------------------:|-------------------:|------------------:|----------------------------:|----------------------------:|
| historical_actual_policy      |                 95.6675 |            1.6569  |                0.79424  |              1.67933 |              0.229996 |            4.75412 |          2.19169  |                      0      |                     23.9051 |
| empirical_taylor_rule         |                125.721  |            2.26223 |                1.48167  |              1.53799 |              0.11567  |            5.15631 |          1.99894  |                    -31.4149 |                      0      |
| td3_benchmark_transfer        |                153.472  |            2.81004 |                1.99906  |              1.4527  |              0.84621  |            4.88023 |          2.57787  |                    -60.4227 |                    -22.0735 |
| riccati_reference             |                156.568  |            2.86663 |                2.03202  |              1.50695 |              0.811301 |            5.08515 |          2.72557  |                    -63.6583 |                    -24.5355 |
| linear_policy_search_transfer |                158.432  |            2.90844 |                2.14564  |              1.40151 |              0.620452 |            4.96731 |          2.49419  |                    -65.607  |                    -26.0184 |
| sac_benchmark_transfer        |                161.347  |            2.95864 |                2.10227  |              1.55521 |              0.787661 |            5.28848 |          2.73007  |                    -68.6543 |                    -28.3373 |
| ppo_benchmark_transfer        |                227.037  |            4.43848 |                0.369915 |              8.1054  |              0.158602 |            3.63152 |          0.588843 |                   -137.319  |                    -80.5874 |

## 5. ANN 长期随机评估

| policy_name                   |   mean_discounted_loss |   std_discounted_loss |   mean_reward |   mean_abs_action |   clip_rate |   explosion_rate |
|:------------------------------|-----------------------:|----------------------:|--------------:|------------------:|------------:|-----------------:|
| td3_benchmark_transfer        |                113.005 |               18.3858 |      -165.026 |           2.79609 | 0.0402995   |                0 |
| riccati_reference             |                119.109 |               16.7386 |      -173.826 |           2.94637 | 0.0421224   |                0 |
| empirical_taylor_rule         |                120.257 |               28.6803 |      -178.023 |           3.59974 | 0.000195313 |                0 |
| sac_benchmark_transfer        |                120.532 |               17.1501 |      -176.013 |           3.00337 | 0.028776    |                0 |
| linear_policy_search_transfer |                137.943 |               17.2359 |      -203.316 |           2.71201 | 0.0170573   |                0 |
| ppo_benchmark_transfer        |                242.358 |               29.7984 |      -366.348 |           1.6601  | 0.00078125  |                0 |

## 6. Local Model Uncertainty 汇总

| policy_name                   |   mean_rank |   median_rank |   win_count |   top2_count |   mean_gap_vs_best_pct |   median_gap_vs_best_pct |   max_gap_vs_best_pct |   mean_clip_rate |   mean_explosion_rate |
|:------------------------------|------------:|--------------:|------------:|-------------:|-----------------------:|-------------------------:|----------------------:|-----------------:|----------------------:|
| td3_benchmark_transfer        |     2.08333 |           1.5 |           6 |            9 |                4.99879 |                 0.832821 |               36.0548 |        0.0737901 |                     0 |
| riccati_reference             |     2.25    |           2   |           3 |            9 |                6.59948 |                 3.72502  |               30.1034 |        0.0723416 |                     0 |
| linear_policy_search_transfer |     3.08333 |           3   |           2 |            4 |                9.01504 |                 5.59706  |               32.6205 |        0.0599284 |                     0 |
| sac_benchmark_transfer        |     3.66667 |           3.5 |           0 |            0 |                9.71203 |                 8.98103  |               20.6385 |        0.0583008 |                     0 |
| ppo_benchmark_transfer        |     5.25    |           5   |           0 |            0 |               43.2748  |                33.5794   |              114.231  |        0.0317546 |                     0 |
| empirical_taylor_rule         |     5.66667 |           6   |           0 |            1 |               87.1148  |               101.765    |              131.513  |        0.0184408 |                     0 |
| historical_actual_proxy       |     6       |           7   |           1 |            1 |               87.2776  |               106.777    |              122.039  |        0.0183268 |                     0 |

## 7. Phase 9 模块状态

| module                         | status                | reason                                                                                              | next_step                                                                              |
|:-------------------------------|:----------------------|:----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------|
| ann_phase9_module              | supplementary_only    | inflation gate, output gate, and dynamic stability are jointly evaluated under Phase 9 rules        | report ANN fit comparison and limitations, but keep SVAR as empirical main result      |
| local_model_uncertainty_module | completed             | frozen policy registry has been evaluated on benchmark, extension, and empirical local environments | use aggregate robustness table and gap distribution figure in writing                  |
| external_dsge_source           | not_available_locally | no external MMB/DSGE model files are present in the repository                                      | if needed later, plug user-supplied DSGE/MMB models into the frozen registry evaluator |

## 8. 说明

- ANN 历史实际政策复现实验最大绝对误差为 `0.0000000684`。
- `Phase 9` 的 model uncertainty 模块使用仓库内可执行的本地结构模型族：benchmark、nonlinear、ZLB/ELB-tightness、asymmetric、empirical SVAR、empirical ANN。
- 由于仓库中没有现成 `MMB` / 外部 DSGE 模型文件，本轮未机械复刻 11 个外部 DSGE；规则 registry 与评估接口已经为用户后续提供模型后继续扩展预留。
- 写作中仍须明确 `Lucas critique`：经验环境中的固定转移仅为近似，Phase 9 的跨模型比较只能部分缓解该边界。