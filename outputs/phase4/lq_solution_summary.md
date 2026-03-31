# Phase 4 LQ Solution Summary

## Core Result

The Phase 3 benchmark now has an infinite-horizon discounted LQ solution. The optimal policy is a linear state-feedback rule obtained from the generalized discounted discrete algebraic Riccati equation.

## Problem

$$
V(s)=\min_a \left\{ \ell(s,a)+\beta \mathbb{E}[V(s')\mid s,a] \right\}
$$

$$
\ell(s_t,a_t)=s_t^\top Q s_t + 2 s_t^\top N a_t + a_t^\top R a_t
$$

## Q / N / R

### Q

|                        |   inflation_gap |   output_gap |   lagged_policy_rate_gap |
|:-----------------------|----------------:|-------------:|-------------------------:|
| inflation_gap          |               1 |          0   |                      0   |
| output_gap             |               0 |          0.5 |                      0   |
| lagged_policy_rate_gap |               0 |          0   |                      0.1 |

### N

|                        |   policy_rate_gap |
|:-----------------------|------------------:|
| inflation_gap          |               0   |
| output_gap             |               0   |
| lagged_policy_rate_gap |              -0.1 |

### R

|                 |   policy_rate_gap |
|:----------------|------------------:|
| policy_rate_gap |               0.1 |

## Riccati Solution

### P Matrix

|                        |   inflation_gap |   output_gap |   lagged_policy_rate_gap |
|:-----------------------|----------------:|-------------:|-------------------------:|
| inflation_gap          |        2.04264  |     0.243438 |                -0.10891  |
| output_gap             |        0.243438 |     0.975988 |                -0.107844 |
| lagged_policy_rate_gap |       -0.10891  |    -0.107844 |                 0.056595 |

### Feedback Matrix F in $a_t=-F s_t$

|                 |   inflation_gap |   output_gap |   lagged_policy_rate_gap |
|:----------------|----------------:|-------------:|-------------------------:|
| policy_rate_gap |         -1.0891 |     -1.07844 |                -0.434052 |

### Equivalent Policy Matrix K in $a_t=K s_t$

|                 |   inflation_gap |   output_gap |   lagged_policy_rate_gap |
|:----------------|----------------:|-------------:|-------------------------:|
| policy_rate_gap |          1.0891 |      1.07844 |                 0.434052 |

## Closed-loop Stability

| Item | Value |
|---|---|
| Stable (all eigenvalue moduli < 1) | True |
| Largest eigenvalue modulus | 0.651192 |
| Value-function constant | 26.305552 |

|   eigenvalue_real |   eigenvalue_imag |   modulus |
|------------------:|------------------:|----------:|
|          0.62848  |          0        |  0.62848  |
|          0.584516 |          0.287042 |  0.651192 |
|          0.584516 |         -0.287042 |  0.651192 |

## Stationary Covariance

|                        |   inflation_gap |   output_gap |   lagged_policy_rate_gap |
|:-----------------------|----------------:|-------------:|-------------------------:|
| inflation_gap          |        0.061162 |     0.020764 |                 0.129198 |
| output_gap             |        0.020764 |     0.337231 |                 0.268266 |
| lagged_policy_rate_gap |        0.129198 |     0.268266 |                 1.0927   |

## Example Simulation Comparison

| Item | Zero policy | Optimal policy |
|---|---:|---:|
| Total discounted loss | 20.261789 | 10.096854 |
| Improvement (%) | 0.000000 | 50.168003 |

|   period |   zero_inflation_gap |   opt_inflation_gap |   zero_output_gap |   opt_output_gap |   zero_action |   opt_action |
|---------:|---------------------:|--------------------:|------------------:|-----------------:|--------------:|-------------:|
|        0 |             1        |            1        |         -1        |        -1        |             0 |     0.010662 |
|        1 |             0.652381 |            0.651528 |         -1.20704  |        -1.20832  |             0 |    -0.588892 |
|        2 |             0.44218  |            0.488353 |         -1.80381  |        -1.73419  |             0 |    -1.59396  |
|        3 |             0.014957 |            0.193337 |         -1.52992  |        -1.28203  |             0 |    -1.86389  |
|        4 |            -0.440658 |           -0.099264 |         -0.789214 |        -0.363272 |             0 |    -1.3089   |
|        5 |            -0.499018 |           -0.036003 |         -0.120554 |         0.385818 |             0 |    -0.19126  |
|        6 |            -0.571038 |           -0.08405  |          0.050163 |         0.491235 |             0 |     0.355211 |
|        7 |            -0.295792 |            0.153596 |         -0.012326 |         0.313435 |             0 |     0.659482 |
|        8 |            -0.356151 |            0.015753 |          0.536376 |         0.733801 |             0 |     1.09477  |
|        9 |            -0.251275 |           -0.001848 |          0.239042 |         0.280257 |           nan |   nan        |

## Interpretation

- The optimal policy reacts positively to inflation and output-gap pressures because the policy matrix $K$ maps adverse states into corrective interest-rate moves.
- The lagged-rate state enters with a positive coefficient in $K$, which implies policy smoothing through the state-augmentation channel.
- This is the theoretical benchmark against which RL will be judged in Phase 5-6.
