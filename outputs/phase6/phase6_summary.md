# Phase 6 Benchmark Comparison Summary

## Benchmark Protocol

| Item | Value |
|---|---|
| Horizon | 60 |
| Evaluation episodes per policy | 48 |
| PPO seeds | [7, 29, 43] |
| Action bounds | [-6.0, 6.0] |
| Riccati K | [[1.089101, 1.078439, 0.434052]] |

## Policy Performance

| policy               |   mean_discounted_loss |   std_discounted_loss |   mean_reward |   mean_abs_action |   clip_rate |   loss_gap_vs_riccati_pct |
|:---------------------|-----------------------:|----------------------:|--------------:|------------------:|------------:|--------------------------:|
| zero_policy          |                42.5528 |              24.6998  |      -55.5669 |          0        |           0 |                 166.945   |
| riccati_optimal      |                15.9407 |               4.57099 |      -19.9961 |          0.893118 |           0 |                   0       |
| empirical_taylor     |                32.3667 |               9.82    |      -41.9459 |          0.808122 |           0 |                 103.045   |
| linear_policy_search |                16.152  |               4.84391 |      -20.2784 |          0.811888 |           0 |                   1.32568 |
| ppo_best_seed_43     |                20.7039 |               7.85082 |      -26.2495 |          0.488255 |           0 |                  29.8809  |

## PPO Seed Stability

|   seed |   total_updates |   mean_discounted_loss |   std_discounted_loss |   mean_reward |   clip_rate |   mean_abs_action |   final_eval_mean_discounted_loss |
|-------:|----------------:|-----------------------:|----------------------:|--------------:|------------:|------------------:|----------------------------------:|
|     43 |             120 |                20.7039 |               7.85082 |      -26.2495 |           0 |          0.488255 |                           20.0827 |
|     29 |             120 |                21.2528 |               8.21318 |      -26.9689 |           0 |          0.45948  |                           20.6149 |
|      7 |             120 |                21.7128 |               8.52432 |      -27.555  |           0 |          0.435855 |                           21.1232 |

## Approximate Policy Coefficients

| policy               |   intercept |   inflation_gap |   output_gap |   lagged_policy_rate_gap |   fit_rmse |
|:---------------------|------------:|----------------:|-------------:|-------------------------:|-----------:|
| riccati_optimal      |    0        |        1.0891   |     1.07844  |                 0.434052 |   0        |
| empirical_taylor     |    0.216573 |        0.193173 |     0.265769 |                 0.8692   |   0        |
| linear_policy_search |    0        |        1.15759  |     0.923679 |                 0.379484 |   0        |
| ppo_best_seed_43     |   -0.090496 |        0.489349 |     0.509024 |                 0.263188 |   0.017642 |

## Common-shock Trajectory Excerpt

| policy          |   period |   inflation_gap |   output_gap |   lagged_policy_rate_gap |    action |     loss |
|:----------------|---------:|----------------:|-------------:|-------------------------:|----------:|---------:|
| zero_policy     |        0 |        1        |    -1        |                 0        |  0        | 1.5      |
| zero_policy     |        1 |        0.595384 |    -0.596403 |                 0        |  0        | 0.532331 |
| zero_policy     |        2 |        0.122191 |     0.079113 |                 0        |  0        | 0.01806  |
| zero_policy     |        3 |        0.123352 |     0.188749 |                 0        |  0        | 0.033029 |
| zero_policy     |        4 |        0.246559 |     0.615976 |                 0        |  0        | 0.250504 |
| zero_policy     |        5 |        0.243557 |    -0.175391 |                 0        |  0        | 0.074701 |
| zero_policy     |        6 |        0.091982 |    -0.584728 |                 0        |  0        | 0.179414 |
| zero_policy     |        7 |        0.018071 |     0.581926 |                 0        |  0        | 0.169645 |
| zero_policy     |        8 |        0.193268 |     0.168319 |                 0        |  0        | 0.051518 |
| zero_policy     |        9 |        0.123611 |    -0.230024 |                 0        |  0        | 0.041735 |
| zero_policy     |       10 |        0.053552 |     0.283034 |                 0        |  0        | 0.042922 |
| zero_policy     |       11 |        0.074536 |     0.547045 |                 0        |  0        | 0.155185 |
| zero_policy     |       12 |        0.136984 |     0.146899 |                 0        |  0        | 0.029554 |
| zero_policy     |       13 |        0.051302 |     0.122193 |                 0        |  0        | 0.010097 |
| zero_policy     |       14 |       -0.116466 |     0.189646 |                 0        |  0        | 0.031547 |
| zero_policy     |       15 |        0.01931  |     0.393259 |                 0        |  0        | 0.077699 |
| zero_policy     |       16 |        0.075325 |     0.088453 |                 0        |  0        | 0.009586 |
| zero_policy     |       17 |        0.027255 |     0.740731 |                 0        |  0        | 0.275084 |
| zero_policy     |       18 |        0.371845 |     0.449125 |                 0        |  0        | 0.239125 |
| zero_policy     |       19 |        0.342145 |     0.805105 |                 0        |  0        | 0.44116  |
| riccati_optimal |        0 |        1        |    -1        |                 0        |  0.010662 | 1.50001  |
| riccati_optimal |        1 |        0.594532 |    -0.597682 |                 0.010662 |  0.007569 | 0.532081 |
| riccati_optimal |        2 |        0.120647 |     0.077164 |                 0.007569 |  0.217898 | 0.021957 |
| riccati_optimal |        3 |        0.104296 |     0.161004 |                 0.217898 |  0.381801 | 0.026525 |
| riccati_optimal |        4 |        0.19522  |     0.547566 |                 0.381801 |  0.968852 | 0.222488 |
| riccati_optimal |        5 |        0.111296 |    -0.34758  |                 0.968852 |  0.166901 | 0.137105 |
| riccati_optimal |        6 |       -0.061617 |    -0.745677 |                 0.166901 | -0.79883  | 0.375077 |
| riccati_optimal |        7 |       -0.073091 |     0.544566 |                -0.79883  |  0.160943 | 0.245735 |
| riccati_optimal |        8 |        0.099991 |     0.115306 |                 0.160943 |  0.303109 | 0.018667 |
| riccati_optimal |        9 |        0.014138 |    -0.31241  |                 0.303109 | -0.189952 | 0.073311 |
| riccati_optimal |       10 |       -0.035307 |     0.236093 |                -0.189952 |  0.133709 | 0.039592 |
| riccati_optimal |       11 |       -0.016637 |     0.489943 |                 0.133709 |  0.568291 | 0.139185 |
| riccati_optimal |       12 |        0.007163 |     0.029606 |                 0.568291 |  0.286398 | 0.008436 |
| riccati_optimal |       13 |       -0.098926 |    -0.010154 |                 0.286398 |  0.00562  | 0.017722 |
| riccati_optimal |       14 |       -0.263567 |     0.078229 |                 0.00562  | -0.200246 | 0.076766 |
| riccati_optimal |       15 |       -0.104634 |     0.323028 |                -0.200246 |  0.147492 | 0.075214 |
| riccati_optimal |       16 |       -0.049677 |     0.009777 |                 0.147492 |  0.02046  | 0.004129 |
| riccati_optimal |       17 |       -0.090119 |     0.670659 |                 0.02046  |  0.633997 | 0.270656 |
| riccati_optimal |       18 |        0.213212 |     0.31252  |                 0.633997 |  0.84443  | 0.098722 |
| riccati_optimal |       19 |        0.120363 |     0.58929  |                 0.84443  |  1.13313  | 0.196453 |

## Notes

- Phase 6 compares policies under one unified benchmark evaluation protocol instead of mixing Phase 4 fixed-state examples with Phase 5 random-start evaluation.
- Empirical Taylor rule is treated as an external rule estimated in Phase 2 and converted into benchmark gap form before simulation.
- Taylor gap-form intercept used in benchmark simulations: 0.216573.
- PPO now uses a squashed Gaussian action parameterization so the optimized action distribution matches the bounded action actually executed by the environment.
- Linear policy search is included as an additional continuous-control baseline; its best coefficients are [1.15759, 0.923679, 0.379484].
- The benchmark remains a stylized LQ model calibrated to realistic magnitudes, not the empirical SVAR environment itself.
