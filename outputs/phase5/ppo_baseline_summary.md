# Phase 5 PPO Baseline Summary

## Goal

Train a continuous-action PPO agent on the exact same LQ benchmark solved analytically in Phase 4.

## PPO Configuration

| Item | Value |
|---|---|
| total_updates | 120 |
| rollout_steps | 512 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_ratio | 0.2 |
| policy_lr | 0.0003 |
| value_lr | 0.001 |
| train_epochs | 8 |
| minibatch_size | 256 |
| entropy_coef | 0.001 |
| value_coef | 0.5 |
| max_grad_norm | 0.5 |
| hidden_size | 64 |
| linear_policy | True |
| state_scale | [2.5, 2.5, 3.0] |
| eval_episodes | 24 |
| seed | 7 |

## Evaluation Comparison

| Policy | Mean discounted loss | Std. discounted loss | Mean reward |
|---|---:|---:|---:|
| Zero policy | 36.063537 | 21.590574 | -46.413329 |
| Riccati optimal | 15.266706 | 4.084527 | -19.032021 |
| PPO baseline | 32.456691 | 18.117278 | -41.598772 |

## PPO Relative Performance

| Item | Value |
|---|---:|
| Improvement over zero policy (%) | 10.001364 |
| Distance above Riccati optimum (%) | 112.597860 |

## Final Training Log Snapshot

|   update |   mean_episode_reward_in_rollout |   eval_mean_reward |   eval_mean_discounted_loss |
|---------:|---------------------------------:|-------------------:|----------------------------:|
|      110 |                         -34.4966 |           -43.5105 |                     32.835  |
|      111 |                         -47.9653 |           -42.9564 |                     32.3966 |
|      112 |                         -49.9398 |           -43.2249 |                     32.694  |
|      113 |                         -49.5615 |           -43.6414 |                     32.9188 |
|      114 |                         -51.0923 |           -44.4538 |                     33.4609 |
|      115 |                         -32.2597 |           -44.7982 |                     33.6193 |
|      116 |                         -39.0622 |           -45.4262 |                     33.8313 |
|      117 |                         -52.46   |           -46.7974 |                     35.1183 |
|      118 |                         -42.8032 |           -45.965  |                     34.6337 |
|      119 |                         -42.7258 |           -45.4933 |                     34.3481 |

## Riccati Benchmark Reference

| Item | Value |
|---|---|
| Largest closed-loop eigenvalue modulus | 0.651192 |
| Feedback matrix K | [[1.089101, 1.078439, 0.434052]] |

## PPO First Evaluation Trajectory

|   inflation_gap |   output_gap |   lagged_policy_rate_gap |   action |    loss |
|----------------:|-------------:|-------------------------:|---------:|--------:|
|        0.107158 |     1.60799  |                -1.90536  | 0.087591 | 1.70149 |
|        0.535773 |     1.63253  |                 0.087591 | 0.133791 | 1.61985 |
|        0.694465 |     1.1659   |                 0.133791 | 0.116089 | 1.16198 |
|        1.02248  |     1.23086  |                 0.116089 | 0.142526 | 1.80305 |
|        1.10554  |     1.60189  |                 0.142526 | 0.171462 | 2.50532 |
|        1.16621  |     1.07761  |                 0.171462 | 0.143315 | 1.94076 |
|        0.9921   |     0.860366 |                 0.143315 | 0.117653 | 1.35444 |
|        1.11094  |     0.831302 |                 0.117653 | 0.12382  | 1.57972 |
|        1.07849  |     0.886177 |                 0.12382  | 0.125042 | 1.55579 |
|        1.03608  |     0.687871 |                 0.125042 | 0.10983  | 1.31007 |
|        1.43335  |     0.374231 |                 0.10983  | 0.117548 | 2.12451 |
|        1.53364  |     0.402666 |                 0.117548 | 0.126264 | 2.43314 |

## Notes

- This is a baseline PPO run, not the final RL result of the thesis.
- Phase 6 will compare PPO and Riccati solutions more systematically.
- ANN environment tuning remains a separate parallel task and does not affect this benchmark PPO result.
