# Phase 7 ZLB Summary

## Policy Performance

| policy                |   mean_discounted_loss |   std_discounted_loss |   mean_reward |   mean_abs_action |   clip_rate |   loss_gap_vs_best_pct |
|:----------------------|-----------------------:|----------------------:|--------------:|------------------:|------------:|-----------------------:|
| zero_policy           |                35.4426 |              23.2172  |      -45.8922 |          0        |    0        |             122.666    |
| riccati_rule_with_zlb |                15.9173 |               4.55395 |      -20.0401 |          0.86504  |    0.040625 |               0        |
| empirical_taylor      |                32.46   |               7.56394 |      -42.2851 |          0.734184 |    0.002431 |             103.929    |
| linear_policy_search  |                16.0345 |               4.61667 |      -20.1879 |          0.80354  |    0.030556 |               0.736065 |
| ppo_zlb               |                19.4613 |               7.62964 |      -24.7127 |          0.478663 |    0        |              22.2648   |
| sac_zlb               |                17.5452 |               7.31797 |      -22.0729 |          0.690822 |    0        |              10.2267   |
| td3_zlb               |                17.5375 |               5.80489 |      -22.1448 |          0.802036 |    0        |              10.1788   |

## Approximate Policy Coefficients

| policy                |   intercept |   inflation_gap |   output_gap |   lagged_policy_rate_gap |   fit_rmse |
|:----------------------|------------:|----------------:|-------------:|-------------------------:|-----------:|
| riccati_rule_with_zlb |    0        |        1.0891   |     1.07844  |                 0.434052 |   0        |
| empirical_taylor      |    0.216573 |        0.193173 |     0.265769 |                 0.8692   |   0        |
| linear_policy_search  |    0        |        1.28079  |     1.10031  |                 0.287039 |   0        |
| ppo_zlb               |    0.030362 |        0.56169  |     0.564681 |                 0.143889 |   0.2562   |
| sac_zlb               |    0.072414 |        0.519133 |     0.601045 |                 0.276336 |   0.383272 |
| td3_zlb               |    0.253109 |        0.703543 |     0.716584 |                 0.224013 |   0.466118 |

## Training Snapshot

| algo    |   seed |   update |   mean_episode_reward_in_rollout |   rollout_clip_rate |   rollout_mean_abs_action |   eval_mean_reward |   eval_mean_discounted_loss |   eval_clip_rate |   step |
|:--------|-------:|---------:|---------------------------------:|--------------------:|--------------------------:|-------------------:|----------------------------:|-----------------:|-------:|
| ppo_zlb |     43 |      160 |                         -37.4699 |                   0 |                  0.851032 |           -27.5163 |                     21.8494 |                0 |    nan |
| ppo_zlb |     43 |      170 |                         -39.1895 |                   0 |                  0.870433 |           -26.9323 |                     21.5645 |                0 |    nan |
| ppo_zlb |     43 |      179 |                         -34.8713 |                   0 |                  0.884828 |           -25.781  |                     20.6932 |                0 |    nan |
| sac_zlb |     43 |      nan |                         nan      |                 nan |                nan        |           -23.0611 |                     18.5155 |                0 |  10000 |
| sac_zlb |     43 |      nan |                         nan      |                 nan |                nan        |           -21.8374 |                     17.7119 |                0 |  12000 |
| sac_zlb |     43 |      nan |                         nan      |                 nan |                nan        |           -21.9165 |                     17.3113 |                0 |  14000 |
| td3_zlb |     43 |      nan |                         nan      |                 nan |                nan        |           -24.7065 |                     19.7553 |                0 |  10000 |
| td3_zlb |     43 |      nan |                         nan      |                 nan |                nan        |           -22.0091 |                     17.4944 |                0 |  12000 |
| td3_zlb |     43 |      nan |                         nan      |                 nan |                nan        |           -22.6035 |                     18.1469 |                0 |  14000 |

## Notes

- This environment imposes a ZLB by constraining the policy-rate gap to stay above -neutral_rate, i.e. actual nominal rate remains non-negative.
- The Riccati rule shown here is the unconstrained linear benchmark rule executed through the ZLB-constrained environment, not a constrained optimum.
- Clip rates now matter economically in this environment because hitting the lower bound is part of the policy behavior.
