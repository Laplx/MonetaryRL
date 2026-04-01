# Phase 10 Direct Empirical RL Summary

## Training Evaluation

| policy_name    | training_env   | algo   | policy_parameterization   |   mean_discounted_loss |   std_discounted_loss |   mean_reward |   clip_rate |   explosion_rate |
|:---------------|:---------------|:-------|:--------------------------|-----------------------:|----------------------:|--------------:|------------:|-----------------:|
| td3_ann_direct | ann            | td3    | standard_nonlinear        |                64.1953 |               11.113  |      -90.8561 |           0 |                0 |
| ppo_ann_direct | ann            | ppo    | linear_policy             |                64.3898 |               19.8198 |      -92      |           0 |                0 |
| sac_ann_direct | ann            | sac    | standard_nonlinear        |                74.4977 |               13.5313 |     -106.802  |           0 |                0 |

## Approximate Policy Coefficients

| policy         | training_env   | algo   | policy_parameterization   |   intercept |   inflation_gap |   output_gap |   lagged_policy_rate_gap |   fit_rmse |
|:---------------|:---------------|:-------|:--------------------------|------------:|----------------:|-------------:|-------------------------:|-----------:|
| ppo_ann_direct | ann            | ppo    | linear_policy             |    2.60858  |       -0.959321 |     0.547218 |                 0.108079 |   0.043582 |
| sac_ann_direct | ann            | sac    | standard_nonlinear        |    0.503693 |       -1.20278  |     1.12529  |                 0.528418 |   1.11326  |
| td3_ann_direct | ann            | td3    | standard_nonlinear        |    0.926137 |       -1.08262  |     0.967379 |                 0.430589 |   0.908876 |

## Notes

- `Phase 8/9` transfer baseline is preserved; this directory adds direct-trained empirical RL rules only.
- `PPO` here follows the current main setting (`linear_policy=True`); nonlinear PPO is handled separately in `phase10_ppo_policy_variants.py`.
- Benchmark and empirical environments remain strictly separated.