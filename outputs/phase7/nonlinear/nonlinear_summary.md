# Phase 7 Nonlinear Phillips Summary

## Policy Performance

| policy               |   mean_discounted_loss |   std_discounted_loss |   mean_reward |   mean_abs_action |   clip_rate |   loss_gap_vs_best_pct |
|:---------------------|-----------------------:|----------------------:|--------------:|------------------:|------------:|-----------------------:|
| zero_policy          |                33.9671 |              23.3153  |      -44.57   |          0        |           0 |              149.305   |
| linear_riccati_rule  |                13.6247 |               3.35645 |      -17.3855 |          0.839083 |           0 |                0       |
| empirical_taylor     |                28.8125 |               8.32304 |      -37.745  |          0.744068 |           0 |              111.472   |
| linear_policy_search |                13.8152 |               3.55168 |      -17.6311 |          0.759546 |           0 |                1.39799 |
| ppo_nonlinear        |                13.9058 |               3.56802 |      -17.7423 |          0.767397 |           0 |                2.06329 |
| sac_nonlinear        |                15.3412 |               3.24553 |      -19.653  |          1.00784  |           0 |               12.5984  |
| td3_nonlinear        |                15.4485 |               3.743   |      -19.8302 |          0.74873  |           0 |               13.3857  |

## Approximate Policy Coefficients

| policy               |   intercept |   inflation_gap |   output_gap |   lagged_policy_rate_gap |   fit_rmse |
|:---------------------|------------:|----------------:|-------------:|-------------------------:|-----------:|
| linear_riccati_rule  |    0        |        1.0891   |     1.07844  |                 0.434052 |   0        |
| empirical_taylor     |    0.216573 |        0.193173 |     0.265769 |                 0.8692   |   0        |
| linear_policy_search |    0        |        1.15759  |     0.923679 |                 0.379484 |   0        |
| ppo_nonlinear        |   -0.034715 |        0.758302 |     0.854248 |                 0.472106 |   0.047581 |
| sac_nonlinear        |   -0.233514 |        1.08605  |     1.09925  |                 0.556342 |   0.272133 |
| td3_nonlinear        |    0.087101 |        1.09189  |     1.0223   |                 0.416229 |   0.518088 |

## Training Snapshot

| algo          |   seed |   update |   mean_episode_reward_in_rollout |   rollout_clip_rate |   rollout_mean_abs_action |   eval_mean_reward |   eval_mean_discounted_loss |   eval_clip_rate |   step |
|:--------------|-------:|---------:|---------------------------------:|--------------------:|--------------------------:|-------------------:|----------------------------:|-----------------:|-------:|
| ppo_nonlinear |     43 |      160 |                         -40.8078 |                   0 |                   1.3582  |           -19.4265 |                     15.1956 |                0 |    nan |
| ppo_nonlinear |     43 |      170 |                         -42.2422 |                   0 |                   1.4016  |           -18.9471 |                     14.9138 |                0 |    nan |
| ppo_nonlinear |     43 |      179 |                         -39.8986 |                   0 |                   1.40842 |           -18.0307 |                     14.1676 |                0 |    nan |
| sac_nonlinear |     43 |      nan |                         nan      |                 nan |                 nan       |           -18.2783 |                     14.278  |                0 |  10000 |
| sac_nonlinear |     43 |      nan |                         nan      |                 nan |                 nan       |           -18.3771 |                     14.4735 |                0 |  12000 |
| sac_nonlinear |     43 |      nan |                         nan      |                 nan |                 nan       |           -20.7451 |                     15.9429 |                0 |  14000 |
| td3_nonlinear |     43 |      nan |                         nan      |                 nan |                 nan       |           -21.8078 |                     17.0552 |                0 |  10000 |
| td3_nonlinear |     43 |      nan |                         nan      |                 nan |                 nan       |           -21.0946 |                     16.2932 |                0 |  12000 |
| td3_nonlinear |     43 |      nan |                         nan      |                 nan |                 nan       |           -20.2767 |                     15.8327 |                0 |  14000 |

## Notes

- This environment introduces a nonlinear Phillips curve while keeping the same state vector and loss function as the benchmark.
- The Riccati policy shown here is the linear benchmark rule extrapolated into the nonlinear environment, not a nonlinear optimum.
- Empirical Taylor rule remains an external rule estimated from Phase 2 and translated into gap form before simulation.
