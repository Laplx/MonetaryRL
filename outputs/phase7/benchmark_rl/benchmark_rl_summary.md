# Phase 7 Benchmark RL Strengthening Summary

## Policy Performance

| policy               |   mean_discounted_loss |   std_discounted_loss |   mean_reward |   mean_abs_action |   clip_rate |   loss_gap_vs_riccati_pct |
|:---------------------|-----------------------:|----------------------:|--------------:|------------------:|------------:|--------------------------:|
| zero_policy          |                42.5528 |              24.6998  |      -55.5669 |          0        |           0 |                 166.945   |
| riccati_optimal      |                15.9407 |               4.57099 |      -19.9961 |          0.893118 |           0 |                   0       |
| empirical_taylor     |                32.3667 |               9.82    |      -41.9459 |          0.808122 |           0 |                 103.045   |
| linear_policy_search |                16.152  |               4.84391 |      -20.2784 |          0.811888 |           0 |                   1.32568 |
| ppo_tuned            |                16.1662 |               4.79646 |      -20.2832 |          0.90202  |           0 |                   1.41469 |
| sac                  |                18.0677 |               4.7188  |      -22.8505 |          1.01443  |           0 |                  13.3432  |
| td3                  |                17.1134 |               4.94399 |      -21.5407 |          1.01645  |           0 |                   7.35706 |

## Approximate Policy Coefficients

| policy               |   intercept |   inflation_gap |   output_gap |   lagged_policy_rate_gap |   fit_rmse |
|:---------------------|------------:|----------------:|-------------:|-------------------------:|-----------:|
| riccati_optimal      |    0        |        1.0891   |     1.07844  |                 0.434052 |   0        |
| empirical_taylor     |    0.216573 |        0.193173 |     0.265769 |                 0.8692   |   0        |
| linear_policy_search |    0        |        1.15759  |     0.923679 |                 0.379484 |   0        |
| ppo_tuned            |   -0.05276  |        0.828851 |     0.915925 |                 0.492693 |   0.09784  |
| sac                  |   -0.406556 |        1.25187  |     1.08182  |                 0.504863 |   0.449218 |
| td3                  |   -0.133531 |        1.30554  |     0.993559 |                 0.363562 |   0.729433 |

## Training Snapshot

| algo      |   seed |   update |   mean_episode_reward_in_rollout |   rollout_clip_rate |   rollout_mean_abs_action |   eval_mean_reward |   eval_mean_discounted_loss |   eval_clip_rate |   step |
|:----------|-------:|---------:|---------------------------------:|--------------------:|--------------------------:|-------------------:|----------------------------:|-----------------:|-------:|
| ppo_tuned |     43 |      160 |                         -50.3737 |                   0 |                   1.60018 |           -20.7069 |                     16.6453 |                0 |    nan |
| ppo_tuned |     43 |      170 |                         -52.3426 |                   0 |                   1.64881 |           -20.712  |                     16.7747 |                0 |    nan |
| ppo_tuned |     43 |      179 |                         -48.3969 |                   0 |                   1.64285 |           -19.8314 |                     16.0066 |                0 |    nan |
| sac       |     43 |      nan |                         nan      |                 nan |                 nan       |           -20.1818 |                     16.1216 |                0 |  10000 |
| sac       |     43 |      nan |                         nan      |                 nan |                 nan       |           -20.3605 |                     16.4239 |                0 |  12000 |
| sac       |     43 |      nan |                         nan      |                 nan |                 nan       |           -22.5485 |                     17.7133 |                0 |  14000 |
| td3       |     43 |      nan |                         nan      |                 nan |                 nan       |           -22.0901 |                     17.7492 |                0 |  10000 |
| td3       |     43 |      nan |                         nan      |                 nan |                 nan       |           -20.3332 |                     16.188  |                0 |  12000 |
| td3       |     43 |      nan |                         nan      |                 nan |                 nan       |           -21.7512 |                     17.4543 |                0 |  14000 |

## Notes

- This file strengthens the benchmark RL line after Phase 6 by adding tuned PPO, SAC and TD3 under a unified evaluation protocol.
- Linear policy search remains a strong benchmark-specific baseline because the underlying benchmark is still an LQ problem.
- Empirical Taylor rule is kept as an external benchmark policy, not an endogenous benchmark component.
