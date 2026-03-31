# Phase 2 Estimation Summary

## Sample

| Item | Value |
|---|---|
| Start | 1987Q3 |
| End | 2007Q2 |
| Observations | 80 |

## Variable Summary

|      |   inflation |   output_gap |   policy_rate |
|:-----|------------:|-------------:|--------------:|
| mean |      2.433  |       0.1837 |        4.8373 |
| std  |      0.7798 |       1.2672 |        2.1719 |
| min  |      1.0817 |      -2.2681 |        0.9967 |
| max  |      4.1461 |       2.4516 |        9.7267 |

## Environment Fit Comparison

|            |   SVAR_MSE |   ANN_MSE |   ANN_improvement_pct |
|:-----------|-----------:|----------:|----------------------:|
| output_gap |   0.210391 |  0.18223  |                 13.38 |
| inflation  |   0.02954  |  0.043424 |                -47    |

## SVAR Output Gap Equation

| Item | Value |
|---|---|
| nobs | 78 |
| R2 | 0.8702 |
| Adj. R2 | 0.8631 |
| RMSE | 0.4587 |

|                  |      coef |   stderr |   pvalue |
|:-----------------|----------:|---------:|---------:|
| const            |  0.306568 | 0.183697 | 0.099424 |
| output_gap_lag1  |  0.917556 | 0.063049 | 0        |
| inflation_lag1   | -0.058219 | 0.084187 | 0.49142  |
| policy_rate_lag1 |  0.208595 | 0.152926 | 0.176751 |
| policy_rate_lag2 | -0.237499 | 0.136509 | 0.086109 |

## SVAR Inflation Equation

| Item | Value |
|---|---|
| nobs | 78 |
| R2 | 0.9518 |
| Adj. R2 | 0.9477 |
| RMSE | 0.1719 |

|                  |      coef |   stderr |   pvalue |
|:-----------------|----------:|---------:|---------:|
| const            |  0.125399 | 0.073194 | 0.09103  |
| output_gap       | -0.063915 | 0.044089 | 0.151556 |
| output_gap_lag1  |  0.204966 | 0.064407 | 0.002168 |
| output_gap_lag2  | -0.107045 | 0.045986 | 0.02278  |
| inflation_lag1   |  1.2922   | 0.105753 | 0        |
| inflation_lag2   | -0.31592  | 0.109213 | 0.005068 |
| policy_rate_lag1 | -0.016365 | 0.014856 | 0.274368 |

## Taylor Rule Regression

| Item | Value |
|---|---|
| nobs | 79 |
| R2 | 0.9738 |
| Adj. R2 | 0.9727 |
| RMSE | 0.3498 |

|                  |     coef |   stderr |   pvalue |
|:-----------------|---------:|---------:|---------:|
| const            | 0.091828 | 0.138683 | 0.50991  |
| inflation        | 0.193173 | 0.05956  | 0.001765 |
| output_gap       | 0.265769 | 0.036139 | 0        |
| policy_rate_lag1 | 0.8692   | 0.023817 | 0        |

## ANN Output Gap Equation

| Item | Value |
|---|---|
| Hidden units | 3 |
| Random state | 3 |
| Train MSE | 0.101253 |
| Validation MSE | 0.499148 |
| Test MSE | 0.229710 |
| Full-sample MSE | 0.182230 |
| Residual std (full sample) | 0.429505 |

## ANN Inflation Equation

| Item | Value |
|---|---|
| Hidden units | 2 |
| Random state | 0 |
| Train MSE | 0.016222 |
| Validation MSE | 0.041755 |
| Test MSE | 0.167500 |
| Full-sample MSE | 0.043424 |
| Residual std (full sample) | 0.201736 |

## Taylor Rule Long-run Responses

| Item | Value |
|---|---|
| Long-run inflation response | 1.4769 |
| Long-run output-gap response | 2.0319 |

## Notes

- Inflation is measured as four-quarter log growth of the GDP deflator.
- Output gap is measured as 100 * (log real GDP - log potential GDP).
- The empirical linear environment follows Hinterlang's recursive SVAR structure.
- The empirical ANN environment uses the same input sets as the linear environment for a fair comparison.
- The empirical SVAR environment is not identical to the theoretical 3-state LQ benchmark and will require state augmentation in implementation.
