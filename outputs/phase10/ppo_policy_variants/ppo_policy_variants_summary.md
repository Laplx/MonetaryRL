# Phase 10 PPO Policy Variants

## Variant Evaluation

| policy_name               | training_env   | policy_parameterization   |   mean_discounted_loss |   std_discounted_loss |   mean_reward |   clip_rate |   explosion_rate |
|:--------------------------|:---------------|:--------------------------|-----------------------:|----------------------:|--------------:|------------:|-----------------:|
| ppo_ann_direct_linear     | ann            | linear_policy             |                64.3898 |               19.8198 |       -92     |           0 |                0 |
| ppo_svar_direct_nonlinear | svar           | nonlinear_policy          |                91.9932 |               42.7811 |      -131.532 |           0 |                0 |
| ppo_svar_direct_linear    | svar           | linear_policy             |               107.215  |               55.0904 |      -156.743 |           0 |                0 |
| ppo_ann_direct_nonlinear  | ann            | nonlinear_policy          |               134.862  |               27.043  |      -197.445 |           0 |                0 |

## Approximate Coefficients

| policy                    | training_env   | policy_parameterization   |   intercept |   inflation_gap |   output_gap |   lagged_policy_rate_gap |   fit_rmse |
|:--------------------------|:---------------|:--------------------------|------------:|----------------:|-------------:|-------------------------:|-----------:|
| ppo_svar_direct_linear    | svar           | linear_policy             |     3.23413 |        0.252398 |     0.150362 |                 0.281657 |   0.028201 |
| ppo_svar_direct_nonlinear | svar           | nonlinear_policy          |     4.65654 |        1.24664  |     0.507447 |                -0.105742 |   0.403751 |
| ppo_ann_direct_linear     | ann            | linear_policy             |     2.60858 |       -0.959321 |     0.547218 |                 0.108079 |   0.043582 |
| ppo_ann_direct_nonlinear  | ann            | nonlinear_policy          |     2.94133 |        0.599941 |     0.599015 |                 0.049123 |   0.264656 |

## Notes

- Linear-policy PPO reuses the main direct-training artifact to avoid duplicating the same run.
- Nonlinear-policy PPO is newly trained here and kept separate from the main Phase 10 unified rule set.