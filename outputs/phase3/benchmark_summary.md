# Phase 3 Benchmark Summary

## Benchmark Role

This benchmark is a stylized 3-state LQ model. It is not the empirical SVAR environment itself; instead, it is the theoretical benchmark that matches the Phase 0/1 specification and is calibrated to realistic magnitudes informed by Phase 2.

## Core Calibration

| Item | Value |
|---|---|
| Name | baseline_lq_benchmark |
| Discount factor | 0.9900 |
| Inflation target | 2.0000 |
| Neutral rate | 2.0000 |
| Loss weight: inflation | 1.0000 |
| Loss weight: output gap | 0.5000 |
| Loss weight: rate smoothing | 0.1000 |

## State Transition

State vector:

$$
s_t = [\tilde{\pi}_t, x_t, \tilde{i}_{t-1}]^\top
$$

Transition:

$$
s_{t+1} = A s_t + B a_t + \Sigma \varepsilon_{t+1}
$$

### A Matrix

|                        |   inflation_gap |   output_gap |   lagged_policy_rate_gap |
|:-----------------------|----------------:|-------------:|-------------------------:|
| inflation_gap          |            0.8  |         0.2  |                        0 |
| output_gap             |            0.05 |         0.78 |                        0 |
| lagged_policy_rate_gap |            0    |         0    |                        0 |

### B Matrix

|                        |   policy_rate_gap |
|:-----------------------|------------------:|
| inflation_gap          |             -0.08 |
| output_gap             |             -0.12 |
| lagged_policy_rate_gap |              1    |

### Sigma Matrix

|                        |   inflation_gap |   output_gap |   lagged_policy_rate_gap |
|:-----------------------|----------------:|-------------:|-------------------------:|
| inflation_gap          |          0.1719 |       0      |                        0 |
| output_gap             |          0      |       0.4587 |                        0 |
| lagged_policy_rate_gap |          0      |       0      |                        0 |

## LQ Loss Representation

Single-period loss:

$$
\ell_t = \lambda_\pi \tilde{\pi}_t^2 + \lambda_x x_t^2 + \lambda_i(\tilde{i}_t-\tilde{i}_{t-1})^2
$$

### Q Matrix

|                        |   inflation_gap |   output_gap |   lagged_policy_rate_gap |
|:-----------------------|----------------:|-------------:|-------------------------:|
| inflation_gap          |               1 |          0   |                      0   |
| output_gap             |               0 |          0.5 |                      0   |
| lagged_policy_rate_gap |               0 |          0   |                      0.1 |

### N Matrix

|                        |   policy_rate_gap |
|:-----------------------|------------------:|
| inflation_gap          |               0   |
| output_gap             |               0   |
| lagged_policy_rate_gap |              -0.1 |

### R Matrix

|                 |   policy_rate_gap |
|:----------------|------------------:|
| policy_rate_gap |               0.1 |

## Example Simulation

| Item | Value |
|---|---|
| Policy | zero-gap policy: $a_t = 0$ |
| Initial state | $[1.0, -1.0, 0.0]^\top$ |
| Horizon | 40 |
| Total discounted loss | 20.261789 |

|   period |   inflation_gap |   output_gap |   lagged_policy_rate_gap |
|---------:|----------------:|-------------:|-------------------------:|
|        0 |        1        |    -1        |                        0 |
|        1 |        0.652381 |    -1.20704  |                        0 |
|        2 |        0.44218  |    -1.80381  |                        0 |
|        3 |        0.014957 |    -1.52992  |                        0 |
|        4 |       -0.440658 |    -0.789214 |                        0 |
|        5 |       -0.499018 |    -0.120554 |                        0 |
|        6 |       -0.571038 |     0.050163 |                        0 |
|        7 |       -0.295792 |    -0.012326 |                        0 |
|        8 |       -0.356151 |     0.536376 |                        0 |
|        9 |       -0.251275 |     0.239042 |                        0 |

## Notes

- Shock scales are anchored to Phase 2 empirical SVAR residual standard deviations.
- Dynamic coefficients are stylized and chosen to provide a stable, interpretable, 3-state LQ benchmark.
- This benchmark is intentionally simpler than the empirical SVAR environment, which uses additional lags.
- ANN environment tuning remains a parallel task and does not block this benchmark track.
