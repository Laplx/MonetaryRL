# Phase 11 v2 极端扩展矩阵

## 范围

| 维度 | 内容 |
|---|---|
| v2 环境 | `nonlinear` 保留一档并新增更高曲率一档；`zlb/asymmetric` 两档均替换为更强结构扭曲 |
| RL 算法 | `PPO`、`TD3`、`SAC` |
| Seeds | `[43]` |

## v2 RL 汇总

| env_id                           | algo   |   mean_discounted_loss |   std_discounted_loss |   loss_gap_vs_best_rl_pct |
|:---------------------------------|:-------|-----------------------:|----------------------:|--------------------------:|
| asymmetric_threshold_very_strong | ppo    |                92.9353 |                     0 |                   0       |
| asymmetric_threshold_very_strong | sac    |                95.1174 |                     0 |                   2.3479  |
| asymmetric_threshold_very_strong | td3    |                94.3867 |                     0 |                   1.56163 |
| asymmetric_threshold_extreme     | ppo    |               279.916  |                     0 |                   4.1262  |
| asymmetric_threshold_extreme     | sac    |               268.823  |                     0 |                   0       |
| asymmetric_threshold_extreme     | td3    |               282.209  |                     0 |                   4.97939 |
| nonlinear_extreme_v2             | ppo    |               881.316  |                     0 |                  87.0404  |
| nonlinear_extreme_v2             | sac    |               543.293  |                     0 |                  15.3023  |
| nonlinear_extreme_v2             | td3    |               471.19   |                     0 |                   0       |
| nonlinear_hyper                  | ppo    |               781.569  |                     0 |                   6.99808 |
| nonlinear_hyper                  | sac    |               857.833  |                     0 |                  17.4387  |
| nonlinear_hyper                  | td3    |               730.451  |                     0 |                   0       |
| zlb_trap_very_strong             | ppo    |               370.487  |                     0 |                 410.874   |
| zlb_trap_very_strong             | sac    |                73.5022 |                     0 |                   1.35403 |
| zlb_trap_very_strong             | td3    |                72.5203 |                     0 |                   0       |
| zlb_trap_extreme                 | ppo    |               596.109  |                     0 |                 453.816   |
| zlb_trap_extreme                 | sac    |               166.632  |                     0 |                  54.809   |
| zlb_trap_extreme                 | td3    |               107.637  |                     0 |                   0       |

## Riccati 相对最优策略差距

| env_id                           | group      | tier           |   riccati_loss |   best_policy_loss |   riccati_gap_pct |
|:---------------------------------|:-----------|:---------------|---------------:|-------------------:|------------------:|
| asymmetric_threshold_very_strong | asymmetric | very_strong_v2 |       104.265  |            92.9353 |         12.1911   |
| asymmetric_threshold_extreme     | asymmetric | extreme_v2     |       315.538  |           268.823  |         17.3775   |
| nonlinear_extreme_v2             | nonlinear  | extreme_v2     |       820.784  |           471.19   |         74.1937   |
| nonlinear_hyper                  | nonlinear  | hyper          |       756.034  |           730.451  |          3.5023   |
| zlb_trap_very_strong             | zlb        | very_strong_v2 |        72.6386 |            72.5203 |          0.163211 |
| zlb_trap_extreme                 | zlb        | extreme_v2     |       117.819  |           107.637  |          9.45933  |

## Riccati 对比最优 RL

| env_id                           | group      | tier           |   riccati_loss | best_rl_algo   |   best_rl_loss | rl_beats_riccati   |   riccati_gap_vs_best_rl_pct |
|:---------------------------------|:-----------|:---------------|---------------:|:---------------|---------------:|:-------------------|-----------------------------:|
| asymmetric_threshold_very_strong | asymmetric | very_strong_v2 |       104.265  | ppo            |        92.9353 | True               |                    12.1911   |
| asymmetric_threshold_extreme     | asymmetric | extreme_v2     |       315.538  | sac            |       268.823  | True               |                    17.3775   |
| nonlinear_extreme_v2             | nonlinear  | extreme_v2     |       820.784  | td3            |       471.19   | True               |                    74.1937   |
| nonlinear_hyper                  | nonlinear  | hyper          |       756.034  | td3            |       730.451  | True               |                     3.5023   |
| zlb_trap_very_strong             | zlb        | very_strong_v2 |        72.6386 | td3            |        72.5203 | True               |                     0.163211 |
| zlb_trap_extreme                 | zlb        | extreme_v2     |       117.819  | td3            |       107.637  | True               |                     9.45933  |

## 说明

- 本批次不改 `phase10`，全部输出独立放在 `phase11/extreme_matrix_v2`。
- `zlb` 通过 state-contingent trap 机制放大衰退与通缩。
- `asymmetric` 通过阈值型高阶尾部惩罚扭曲传统二次损失。