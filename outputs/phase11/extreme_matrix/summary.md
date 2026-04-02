# Phase 11 极端扩展矩阵

## 范围

| 维度 | 内容 |
|---|---|
| 新增环境 | `nonlinear/asymmetric/zlb` 各新增 `very_strong` 与 `extreme` 两档 |
| RL 算法 | `PPO`、`TD3`、`SAC` |
| Seeds | `[7, 29, 43]` |

## 新增两档 RL 汇总

| env_id                 | algo   |   mean_discounted_loss |   std_discounted_loss |   loss_gap_vs_best_rl_pct |
|:-----------------------|:-------|-----------------------:|----------------------:|--------------------------:|
| asymmetric_extreme     | ppo    |                97.8439 |              11.8365  |                 30.1917   |
| asymmetric_extreme     | sac    |                75.1537 |               6.42646 |                  0        |
| asymmetric_extreme     | td3    |                75.8947 |               4.37323 |                  0.985934 |
| asymmetric_very_strong | ppo    |                56.1574 |               4.46678 |                 25.7672   |
| asymmetric_very_strong | sac    |                44.6519 |               2.66447 |                  0        |
| asymmetric_very_strong | td3    |                47.2943 |               2.74939 |                  5.91772  |
| nonlinear_extreme      | ppo    |               915.898  |              12.145   |                113.282    |
| nonlinear_extreme      | sac    |               483.249  |              91.0585  |                 12.5326   |
| nonlinear_extreme      | td3    |               429.43   |              94.954   |                  0        |
| nonlinear_very_strong  | ppo    |               379.602  |              26.9852  |               1123.23     |
| nonlinear_very_strong  | sac    |                31.0329 |               4.35609 |                  0        |
| nonlinear_very_strong  | td3    |                37.1075 |               7.60823 |                 19.5748   |
| zlb_extreme            | ppo    |               190.017  |              22.1673  |                103.169    |
| zlb_extreme            | sac    |                93.746  |               3.9103  |                  0.234754 |
| zlb_extreme            | td3    |                93.5265 |               5.5216  |                  0        |
| zlb_very_strong        | ppo    |               143.166  |              16.341   |                144.047    |
| zlb_very_strong        | sac    |                59.1847 |               3.67396 |                  0.889055 |
| zlb_very_strong        | td3    |                58.6631 |               3.91899 |                  0        |

## 新增两档中 Riccati 相对最优策略差距

| env_id                 | group      | tier        |   riccati_loss |   best_policy_loss |   riccati_gap_pct |
|:-----------------------|:-----------|:------------|---------------:|-------------------:|------------------:|
| asymmetric_very_strong | asymmetric | very_strong |        38.9633 |            37.8515 |           2.9373  |
| asymmetric_extreme     | asymmetric | extreme     |        64.6313 |            61.8547 |           4.48879 |
| nonlinear_very_strong  | nonlinear  | very_strong |        19.189  |            18.8987 |           1.53591 |
| nonlinear_extreme      | nonlinear  | extreme     |       625.891  |           429.43   |          45.7492  |
| zlb_very_strong        | zlb        | very_strong |        60.6204 |            58.6631 |           3.33648 |
| zlb_extreme            | zlb        | extreme     |       102      |            93.5265 |           9.05956 |

## 五档合并后 RL 汇总

| env_id                 | group      | tier        | algo   |   mean_discounted_loss |
|:-----------------------|:-----------|:------------|:-------|-----------------------:|
| asymmetric_mild        | asymmetric | mild        | ppo    |                24.667  |
| asymmetric_mild        | asymmetric | mild        | sac    |                19.3156 |
| asymmetric_mild        | asymmetric | mild        | td3    |                20.6241 |
| asymmetric_medium      | asymmetric | medium      | ppo    |                28.0312 |
| asymmetric_medium      | asymmetric | medium      | sac    |                22.558  |
| asymmetric_medium      | asymmetric | medium      | td3    |                23.9619 |
| asymmetric_strong      | asymmetric | strong      | ppo    |                36.7037 |
| asymmetric_strong      | asymmetric | strong      | sac    |                28.8468 |
| asymmetric_strong      | asymmetric | strong      | td3    |                31.2276 |
| asymmetric_very_strong | asymmetric | very_strong | ppo    |                56.1574 |
| asymmetric_very_strong | asymmetric | very_strong | sac    |                44.6519 |
| asymmetric_very_strong | asymmetric | very_strong | td3    |                47.2943 |
| asymmetric_extreme     | asymmetric | extreme     | ppo    |                97.8439 |
| asymmetric_extreme     | asymmetric | extreme     | sac    |                75.1537 |
| asymmetric_extreme     | asymmetric | extreme     | td3    |                75.8947 |
| benchmark              | benchmark  | nan         | ppo    |                21.0743 |
| benchmark              | benchmark  | nan         | sac    |                16.6638 |
| benchmark              | benchmark  | nan         | td3    |                18.0437 |
| nonlinear_mild         | nonlinear  | mild        | ppo    |                20.4903 |
| nonlinear_mild         | nonlinear  | mild        | sac    |                14.3784 |
| nonlinear_mild         | nonlinear  | mild        | td3    |                15.1742 |
| nonlinear_medium       | nonlinear  | medium      | ppo    |                21.366  |
| nonlinear_medium       | nonlinear  | medium      | sac    |                15.2663 |
| nonlinear_medium       | nonlinear  | medium      | td3    |                16.1735 |
| nonlinear_strong       | nonlinear  | strong      | ppo    |                26.6718 |
| nonlinear_strong       | nonlinear  | strong      | sac    |                16.5778 |
| nonlinear_strong       | nonlinear  | strong      | td3    |                17.6738 |
| nonlinear_very_strong  | nonlinear  | very_strong | ppo    |               379.602  |
| nonlinear_very_strong  | nonlinear  | very_strong | sac    |                31.0329 |
| nonlinear_very_strong  | nonlinear  | very_strong | td3    |                37.1075 |
| nonlinear_extreme      | nonlinear  | extreme     | ppo    |               915.898  |
| nonlinear_extreme      | nonlinear  | extreme     | sac    |               483.249  |
| nonlinear_extreme      | nonlinear  | extreme     | td3    |               429.43   |
| zlb_mild               | zlb        | mild        | ppo    |                36.1241 |
| zlb_mild               | zlb        | mild        | sac    |                18.6772 |
| zlb_mild               | zlb        | mild        | td3    |                19.589  |
| zlb_medium             | zlb        | medium      | ppo    |                76.5004 |
| zlb_medium             | zlb        | medium      | sac    |                25.6919 |
| zlb_medium             | zlb        | medium      | td3    |                22.2419 |
| zlb_strong             | zlb        | strong      | ppo    |               122.193  |
| zlb_strong             | zlb        | strong      | sac    |                31.5759 |
| zlb_strong             | zlb        | strong      | td3    |                30.0394 |
| zlb_very_strong        | zlb        | very_strong | ppo    |               143.166  |
| zlb_very_strong        | zlb        | very_strong | sac    |                59.1847 |
| zlb_very_strong        | zlb        | very_strong | td3    |                58.6631 |
| zlb_extreme            | zlb        | extreme     | ppo    |               190.017  |
| zlb_extreme            | zlb        | extreme     | sac    |                93.746  |
| zlb_extreme            | zlb        | extreme     | td3    |                93.5265 |

## 说明

- 本批次不改 `phase10`，全部输出独立放在 `phase11`。
- `zlb_*` 继续按更紧的 `ELB-tightness` 解释。
- 若新增两档中 `Riccati` 仍然是最优，则说明扭曲还不够，需要继续加大。