# Phase 14 ANN-native 数值搜索比较

## 新增数值搜索政策

- `ann_affine_search_artificial`：best seed = `99`，wall time = `567.97s`，train env steps = `376320`
- `ann_affine_search_revealed`：best seed = `43`，wall time = `569.62s`，train env steps = `376320`

## 关键结果

### Artificial loss / Historical

| policy_name                  |   total_discounted_loss |   std_inflation_gap |   std_output_gap |   std_rate_change |
|:-----------------------------|------------------------:|--------------------:|-----------------:|------------------:|
| ann_affine_search_artificial |                 65.4536 |            0.582899 |          1.08293 |          1.30026  |
| ppo_ann_direct               |                 69.2297 |            0.755132 |          1.13618 |          0.490789 |
| td3_ann_direct               |                 71.239  |            0.515603 |          1.18717 |          1.39977  |
| sac_ann_direct               |                 86.01   |            0.563297 |          1.34394 |          1.59101  |
| empirical_taylor_rule        |                130.407  |            0.699938 |          1.24825 |          0.342821 |
| riccati_reference            |                157.011  |            0.685622 |          1.16152 |          0.873603 |

### Artificial loss / Stochastic

| policy_name                  |   mean_discounted_loss |   std_inflation_gap |   std_output_gap |   std_rate_change |
|:-----------------------------|-----------------------:|--------------------:|-----------------:|------------------:|
| ann_affine_search_artificial |                68.1322 |            0.476808 |         0.953269 |          1.18772  |
| ppo_ann_direct               |                77.6134 |            0.716701 |         0.985525 |          0.496324 |
| td3_ann_direct               |                79.7799 |            0.522492 |         1.1577   |          1.25642  |
| sac_ann_direct               |                92.8784 |            0.574433 |         1.20664  |          1.49067  |
| riccati_reference            |               151.911  |            0.881208 |         1.18457  |          0.849514 |
| empirical_taylor_rule        |               158.791  |            0.938202 |         1.08346  |          0.255356 |

### Revealed loss / Historical

| policy_name                |   total_discounted_loss |   std_inflation_gap |   std_output_gap |   std_rate_change |
|:---------------------------|------------------------:|--------------------:|-----------------:|------------------:|
| td3_ann_revealed_direct    |                 186.231 |            0.816984 |         1.04303  |         0.209863  |
| ann_affine_search_revealed |                 207.583 |            0.286533 |         0.788305 |         0.14666   |
| sac_ann_revealed_direct    |                 207.699 |            0.215775 |         0.785473 |         0.0871388 |
| empirical_taylor_rule      |                 285.352 |            0.699938 |         1.24825  |         0.342821  |

### Revealed loss / Stochastic

| policy_name                |   mean_discounted_loss |   std_inflation_gap |   std_output_gap |   std_rate_change |
|:---------------------------|-----------------------:|--------------------:|-----------------:|------------------:|
| ann_affine_search_revealed |                217.33  |            0.989659 |         0.981777 |          0.176675 |
| sac_ann_revealed_direct    |                249.177 |            0.844297 |         1.03279  |          0.119236 |
| empirical_taylor_rule      |                289.854 |            0.938202 |         1.08346  |          0.255356 |
| td3_ann_revealed_direct    |                305.441 |            0.904202 |         0.991115 |          0.305429 |

## 计算开销对比

| policy_name                  | training_context   | method_family    |   training_env_steps |   wall_seconds |
|:-----------------------------|:-------------------|:-----------------|---------------------:|---------------:|
| ann_affine_search_artificial | ann_artificial     | numerical_search |               376320 |        567.969 |
| ann_affine_search_revealed   | ann_revealed       | numerical_search |               376320 |        569.619 |
| ppo_ann_direct               | ann                | PPO              |               225280 |        nan     |
| td3_ann_direct               | ann                | TD3              |                16000 |        nan     |
| sac_ann_direct               | ann                | SAC              |                16000 |        nan     |
| ppo_ann_direct_nonlinear     | ann                | PPO              |               225280 |        nan     |
| ppo_ann_revealed_direct      | ann                | PPO              |               225280 |        nan     |
| td3_ann_revealed_direct      | ann                | TD3              |                16000 |        nan     |
| sac_ann_revealed_direct      | ann                | SAC              |                16000 |        nan     |

## 文件

- `ann_full_matrix.csv`：phase14 最终 ANN 全矩阵
- `ann_master_matrix.csv`：整合 phase9 / phase10 / phase14 的 ANN 写作用主表
- `ann_search_seed_summary.csv`：数值搜索三 seed 汇总
- `figures/ann_numerical_search_vs_rl_matrix.png`：ANN 数值搜索与 RL 主比较图
- `figures/ann_numerical_search_compute_cost.png`：计算开销对比图