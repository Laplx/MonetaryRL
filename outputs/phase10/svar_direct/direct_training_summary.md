# Phase 10 Direct Empirical RL Summary

## Training Evaluation

| policy_name     | training_env   | algo   | policy_parameterization   |   mean_discounted_loss |   std_discounted_loss |   mean_reward |   clip_rate |   explosion_rate |
|:----------------|:---------------|:-------|:--------------------------|-----------------------:|----------------------:|--------------:|------------:|-----------------:|
| sac_svar_direct | svar           | sac    | standard_nonlinear        |                89.1856 |               37.901  |      -124.89  |           0 |                0 |
| td3_svar_direct | svar           | td3    | standard_nonlinear        |                96.7765 |               40.9132 |      -135.14  |           0 |                0 |
| ppo_svar_direct | svar           | ppo    | linear_policy             |               107.215  |               55.0904 |      -156.743 |           0 |                0 |

## Approximate Policy Coefficients

| policy          | training_env   | algo   | policy_parameterization   |   intercept |   inflation_gap |   output_gap |   lagged_policy_rate_gap |   fit_rmse |
|:----------------|:---------------|:-------|:--------------------------|------------:|----------------:|-------------:|-------------------------:|-----------:|
| ppo_svar_direct | svar           | ppo    | linear_policy             |     3.23413 |        0.252398 |     0.150362 |                 0.281657 |   0.028201 |
| sac_svar_direct | svar           | sac    | standard_nonlinear        |     2.49943 |        1.52863  |     0.572243 |                 0.226216 |   1.03569  |
| td3_svar_direct | svar           | td3    | standard_nonlinear        |     3.22999 |        1.59613  |     1.16731  |                -0.058738 |   1.35078  |

## Notes

- `Phase 8/9` transfer baseline is preserved; this directory adds direct-trained empirical RL rules only.
- `PPO` here follows the current main setting (`linear_policy=True`); nonlinear PPO is handled separately in `phase10_ppo_policy_variants.py`.
- Benchmark and empirical environments remain strictly separated.