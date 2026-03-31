# Phase 7 Matrix Summary

## Matrix Scope

| Dimension | Value |
|---|---|
| Environments | 1 benchmark + 3 nonlinear + 3 ZLB/ELB-tightness + 3 asymmetric-loss = 10 |
| RL algorithms | PPO, TD3, SAC |
| Seeds per algorithm-environment pair | [7, 29, 43] |

## RL Summary Table

| env_id            | algo   |   mean_discounted_loss |   std_discounted_loss |   mean_reward |   mean_clip_rate |   mean_explosion_rate |   loss_gap_vs_best_rl_pct |
|:------------------|:-------|-----------------------:|----------------------:|--------------:|-----------------:|----------------------:|--------------------------:|
| asymmetric_medium | ppo    |                28.0312 |              0.964418 |      -35.6295 |                0 |                     0 |                  24.2625  |
| asymmetric_medium | sac    |                22.558  |              0.959076 |      -28.532  |                0 |                     0 |                   0       |
| asymmetric_medium | td3    |                23.9619 |              0.829539 |      -30.3986 |                0 |                     0 |                   6.22352 |
| asymmetric_mild   | ppo    |                24.667  |              0.871278 |      -31.4532 |                0 |                     0 |                  27.7053  |
| asymmetric_mild   | sac    |                19.3156 |              0.48035  |      -24.4374 |                0 |                     0 |                   0       |
| asymmetric_mild   | td3    |                20.6241 |              0.41459  |      -26.1899 |                0 |                     0 |                   6.77461 |
| asymmetric_strong | ppo    |                36.7037 |              2.06996  |      -46.6144 |                0 |                     0 |                  27.2367  |
| asymmetric_strong | sac    |                28.8468 |              0.961378 |      -36.413  |                0 |                     0 |                   0       |
| asymmetric_strong | td3    |                31.2276 |              1.42325  |      -39.6041 |                0 |                     0 |                   8.2534  |
| benchmark         | ppo    |                21.0743 |              0.41762  |      -26.691  |                0 |                     0 |                  26.4678  |
| benchmark         | sac    |                16.6638 |              0.144475 |      -21.0081 |                0 |                     0 |                   0       |
| benchmark         | td3    |                18.0437 |              0.260459 |      -22.8293 |                0 |                     0 |                   8.28084 |
| nonlinear_medium  | ppo    |                21.366  |              1.10673  |      -27.8045 |                0 |                     0 |                  39.9558  |
| nonlinear_medium  | sac    |                15.2663 |              0.181287 |      -19.699  |                0 |                     0 |                   0       |
| nonlinear_medium  | td3    |                16.1735 |              0.490073 |      -20.868  |                0 |                     0 |                   5.94287 |
| nonlinear_mild    | ppo    |                20.4903 |              1.11277  |      -26.9388 |                0 |                     0 |                  42.5078  |
| nonlinear_mild    | sac    |                14.3784 |              0.222651 |      -18.7322 |                0 |                     0 |                   0       |
| nonlinear_mild    | td3    |                15.1742 |              0.28998  |      -19.8017 |                0 |                     0 |                   5.53508 |
| nonlinear_strong  | ppo    |                26.6718 |              0.870429 |      -34.4357 |                0 |                     0 |                  60.888   |
| nonlinear_strong  | sac    |                16.5778 |              0.168837 |      -21.1436 |                0 |                     0 |                   0       |
| nonlinear_strong  | td3    |                17.6738 |              0.966218 |      -22.5994 |                0 |                     0 |                   6.61083 |
| zlb_medium        | ppo    |                76.5004 |              1.93584  |     -101.883  |                0 |                     0 |                 243.947   |
| zlb_medium        | sac    |                25.6919 |              1.42161  |      -32.889  |                0 |                     0 |                  15.5112  |
| zlb_medium        | td3    |                22.2419 |              2.14117  |      -28.0294 |                0 |                     0 |                   0       |
| zlb_mild          | ppo    |                36.1241 |              3.31427  |      -47.4435 |                0 |                     0 |                  93.413   |
| zlb_mild          | sac    |                18.6772 |              1.09859  |      -23.9421 |                0 |                     0 |                   0       |
| zlb_mild          | td3    |                19.589  |              0.482622 |      -25.1178 |                0 |                     0 |                   4.88208 |
| zlb_strong        | ppo    |               122.193  |              6.29305  |     -163.248  |                0 |                     0 |                 306.778   |
| zlb_strong        | sac    |                31.5759 |              1.77822  |      -39.7065 |                0 |                     0 |                   5.11514 |
| zlb_strong        | td3    |                30.0394 |              3.35974  |      -37.6731 |                0 |                     0 |                   0       |

## Benchmark Including Reference Rules

| env_id    | group     | tier     | policy_name          |   mean_discounted_loss |   std_discounted_loss |   mean_reward |   clip_rate |   explosion_rate |
|:----------|:----------|:---------|:---------------------|-----------------------:|----------------------:|--------------:|------------:|-----------------:|
| benchmark | benchmark | baseline | empirical_taylor     |                32.2708 |              8.92757  |      -41.5123 |           0 |                0 |
| benchmark | benchmark | baseline | linear_policy_search |                15.0468 |              4.29622  |      -18.7334 |           0 |                0 |
| benchmark | benchmark | baseline | riccati_reference    |                14.8791 |              4.14729  |      -18.5216 |           0 |                0 |
| benchmark | benchmark | baseline | zero_policy          |                34.4815 |             18.0363   |      -44.1637 |           0 |                0 |
| benchmark | benchmark | baseline | ppo                  |                21.0743 |              0.41762  |      -26.691  |           0 |                0 |
| benchmark | benchmark | baseline | sac                  |                16.6638 |              0.144475 |      -21.0081 |           0 |                0 |
| benchmark | benchmark | baseline | td3                  |                18.0437 |              0.260459 |      -22.8293 |           0 |                0 |

## Notes

- `zlb_*` tiers should be read as progressively tighter effective lower-bound environments, implemented through reduced policy-rate room and more recessionary initial-state support.
- `nonlinear_*` tiers increase the strength of the nonlinear Phillips distortion.
- `asymmetric_*` tiers increase the extra penalty on upside inflation and downside output gaps.
- RL summary statistics are averaged across seeds; benchmark reference rules are single deterministic baselines.
