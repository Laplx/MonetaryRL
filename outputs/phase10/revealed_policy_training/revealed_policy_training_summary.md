# Phase 10 Revealed-Welfare RL Training

## Revealed Loss Weights

|   inflation |   output_gap |   rate_smoothing |
|------------:|-------------:|-----------------:|
|           1 |     0.881713 |          20.0855 |

## Training Evaluation

| policy_name              | training_env   | algo   | policy_parameterization   |   mean_discounted_loss |   std_discounted_loss |   mean_reward |   clip_rate |   explosion_rate |
|:-------------------------|:---------------|:-------|:--------------------------|-----------------------:|----------------------:|--------------:|------------:|-----------------:|
| sac_svar_revealed_direct | svar           | sac    | standard_nonlinear        |                174.136 |               76.9165 |      -248.905 |           0 |                0 |
| sac_ann_revealed_direct  | ann            | sac    | standard_nonlinear        |                220.458 |               72.0281 |      -313.325 |           0 |                0 |
| td3_ann_revealed_direct  | ann            | td3    | standard_nonlinear        |                260.012 |               64.9779 |      -375.076 |           0 |                0 |
| td3_svar_revealed_direct | svar           | td3    | standard_nonlinear        |                337.446 |              101.698  |      -486.211 |           0 |                0 |
| ppo_svar_revealed_direct | svar           | ppo    | linear_policy             |                556.195 |              185.619  |      -763.597 |           0 |                0 |
| ppo_ann_revealed_direct  | ann            | ppo    | linear_policy             |                604.095 |              190.096  |      -801.166 |           0 |                0 |

## Approximate Policy Coefficients

| policy                   | training_env   | algo   | policy_parameterization   |   intercept |   inflation_gap |   output_gap |   lagged_policy_rate_gap |   fit_rmse |
|:-------------------------|:---------------|:-------|:--------------------------|------------:|----------------:|-------------:|-------------------------:|-----------:|
| ppo_svar_revealed_direct | svar           | ppo    | linear_policy             |    3.76074  |       -0.33602  |     0.177204 |                 0.61337  |   0.279514 |
| sac_svar_revealed_direct | svar           | sac    | standard_nonlinear        |    0.195235 |        0.067874 |     0.050975 |                 0.930723 |   0.163919 |
| td3_svar_revealed_direct | svar           | td3    | standard_nonlinear        |   -0.061685 |       -0.002512 |    -0.148787 |                 0.947788 |   0.403334 |
| ppo_ann_revealed_direct  | ann            | ppo    | linear_policy             |    3.59209  |        0.53686  |    -0.250778 |                 0.558573 |   0.303964 |
| sac_ann_revealed_direct  | ann            | sac    | standard_nonlinear        |    0.137515 |       -0.142674 |     0.092529 |                 0.968365 |   0.269911 |
| td3_ann_revealed_direct  | ann            | td3    | standard_nonlinear        |    0.466979 |       -0.151141 |     0.147203 |                 0.887135 |   0.25774  |