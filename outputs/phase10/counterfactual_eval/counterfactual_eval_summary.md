# Phase 10 Unified Counterfactual Evaluation

## Unified Policy Registry

| policy_name                        | rule_family          | source_env   | policy_parameterization   | callable_type   |
|:-----------------------------------|:---------------------|:-------------|:--------------------------|:----------------|
| ppo_ann_direct                     | ann_direct           | ann          | linear_policy             | checkpoint      |
| ppo_ann_direct_nonlinear           | ann_direct           | ann          | nonlinear_policy          | checkpoint      |
| sac_ann_direct                     | ann_direct           | ann          | standard_nonlinear        | checkpoint      |
| td3_ann_direct                     | ann_direct           | ann          | standard_nonlinear        | checkpoint      |
| ppo_ann_revealed_direct            | ann_revealed_direct  | ann          | linear_policy             | checkpoint      |
| ppo_ann_revealed_direct_nonlinear  | ann_revealed_direct  | ann          | nonlinear_policy          | checkpoint      |
| sac_ann_revealed_direct            | ann_revealed_direct  | ann          | standard_nonlinear        | checkpoint      |
| td3_ann_revealed_direct            | ann_revealed_direct  | ann          | standard_nonlinear        | checkpoint      |
| linear_policy_search_transfer      | benchmark_transfer   | benchmark    | fixed_rule                | linear          |
| ppo_benchmark_transfer             | benchmark_transfer   | benchmark    | linear_surrogate          | linear          |
| sac_benchmark_transfer             | benchmark_transfer   | benchmark    | linear_surrogate          | linear          |
| td3_benchmark_transfer             | benchmark_transfer   | benchmark    | linear_surrogate          | linear          |
| empirical_taylor_rule              | empirical_rule       | svar         | fixed_rule                | linear          |
| historical_actual_policy           | historical_actual    | historical   | fixed_rule                | historical      |
| ppo_svar_direct                    | svar_direct          | svar         | linear_policy             | checkpoint      |
| ppo_svar_direct_nonlinear          | svar_direct          | svar         | nonlinear_policy          | checkpoint      |
| sac_svar_direct                    | svar_direct          | svar         | standard_nonlinear        | checkpoint      |
| td3_svar_direct                    | svar_direct          | svar         | standard_nonlinear        | checkpoint      |
| ppo_svar_revealed_direct           | svar_revealed_direct | svar         | linear_policy             | checkpoint      |
| ppo_svar_revealed_direct_nonlinear | svar_revealed_direct | svar         | nonlinear_policy          | checkpoint      |
| sac_svar_revealed_direct           | svar_revealed_direct | svar         | standard_nonlinear        | checkpoint      |
| td3_svar_revealed_direct           | svar_revealed_direct | svar         | standard_nonlinear        | checkpoint      |
| riccati_reference                  | theory_reference     | benchmark    | fixed_rule                | linear          |

## SVAR Historical Counterfactual

| evaluation_env   | policy_name                        |   total_discounted_loss |   mean_period_loss |   mean_sq_inflation_gap |   mean_sq_output_gap |   mean_sq_rate_change |   mean_policy_rate |   std_policy_rate |   improvement_vs_actual_pct |   improvement_vs_taylor_pct |
|:-----------------|:-----------------------------------|------------------------:|-------------------:|------------------------:|---------------------:|----------------------:|-------------------:|------------------:|----------------------------:|----------------------------:|
| svar             | ppo_ann_direct_nonlinear           |                 87.3361 |            1.44818 |                0.751651 |              1.36936 |              0.118458 |           5.28829  |          0.901747 |                    0.659613 |                     7.58266 |
| svar             | historical_actual_policy           |                 87.916  |            1.54727 |                0.723281 |              1.60241 |              0.227843 |           4.77892  |          2.18807  |                    0        |                     6.96902 |
| svar             | ppo_svar_direct_nonlinear          |                 90.7985 |            1.56307 |                0.782447 |              1.52977 |              0.15742  |           6.14034  |          1.18277  |                   -3.27868  |                     3.91883 |
| svar             | sac_svar_direct                    |                 92.1016 |            1.55643 |                0.708636 |              1.57203 |              0.617792 |           5.34625  |          1.5752   |                   -4.76086  |                     2.53994 |
| svar             | empirical_taylor_rule              |                 94.5018 |            1.64709 |                0.693088 |              1.88343 |              0.122907 |           4.73712  |          1.99524  |                   -7.49107  |                     0       |
| svar             | ppo_svar_direct                    |                100.998  |            1.77717 |                1.15267  |              1.24748 |              0.007613 |           6.47433  |          0.399381 |                  -14.88     |                    -6.87398 |
| svar             | td3_svar_direct                    |                103.644  |            1.8112  |                0.86723  |              1.6905  |              0.987259 |           5.21452  |          2.09872  |                  -17.8897   |                    -9.67398 |
| svar             | sac_svar_revealed_direct           |                105.451  |            1.8813  |                1.31262  |              1.13371 |              0.018313 |           6.69631  |          0.526392 |                  -19.9448   |                   -11.5859  |
| svar             | sac_ann_revealed_direct            |                110.788  |            1.99489 |                1.39011  |              1.20615 |              0.017107 |           6.4702   |          0.940572 |                  -26.0162   |                   -17.2341  |
| svar             | sac_benchmark_transfer             |                114.455  |            1.99224 |                1.00172  |              1.86482 |              0.581034 |           4.27898  |          2.34381  |                  -30.1866   |                   -21.1139  |
| svar             | linear_policy_search_transfer      |                120.575  |            2.10817 |                1.17866  |              1.76384 |              0.475911 |           4.02609  |          2.08302  |                  -37.1484   |                   -27.5905  |
| svar             | riccati_reference                  |                121.072  |            2.10878 |                1.09218  |              1.90192 |              0.656402 |           4.15918  |          2.46192  |                  -37.7134   |                   -28.1161  |
| svar             | td3_benchmark_transfer             |                129.555  |            2.25366 |                1.24125  |              1.88213 |              0.713523 |           3.98097  |          2.44256  |                  -47.3624   |                   -37.0927  |
| svar             | td3_ann_revealed_direct            |                140.989  |            2.43737 |                1.68849  |              1.48428 |              0.067341 |           3.93823  |          1.66303  |                  -60.3682   |                   -49.1921  |
| svar             | ppo_ann_direct                     |                150.19   |            2.47015 |                1.87964  |              1.13904 |              0.209855 |           3.73454  |          0.968385 |                  -70.8331   |                   -58.9277  |
| svar             | ppo_benchmark_transfer             |                171.875  |            3.02802 |                2.37379  |              1.26636 |              0.210448 |           2.89154  |          0.772566 |                  -95.4995   |                   -81.8751  |
| svar             | ppo_ann_revealed_direct            |                195.085  |            3.76635 |                3.11478  |              1.28508 |              0.0903   |           8.44125  |          1.06835  |                 -121.9      |                  -106.435   |
| svar             | td3_svar_revealed_direct           |                195.8    |            3.58652 |                2.98347  |              1.18981 |              0.081539 |           2.33346  |          0.998335 |                 -122.712    |                  -107.192   |
| svar             | td3_ann_direct                     |                270.191  |            4.90445 |                4.33277  |              1.06019 |              0.415838 |           1.65881  |          0.770966 |                 -207.329    |                  -185.911   |
| svar             | ppo_svar_revealed_direct_nonlinear |                293.712  |            6.03514 |                5.37578  |              1.3     |              0.093579 |           9.58368  |          0.01801  |                 -234.083    |                  -210.801   |
| svar             | ppo_svar_revealed_direct           |                293.956  |            6.0744  |                5.43124  |              1.27748 |              0.04423  |           9.58647  |          0.208574 |                 -234.361    |                  -211.059   |
| svar             | ppo_ann_revealed_direct_nonlinear  |                313.572  |            6.46767 |                5.79808  |              1.3179  |              0.1065   |           9.78031  |          4e-06    |                 -256.672    |                  -231.816   |
| svar             | sac_ann_direct                     |                407.733  |            7.76666 |                7.24039  |              1.00497 |              0.237839 |           0.345926 |          0.496773 |                 -363.775    |                  -331.455   |

## ANN Historical Counterfactual

| evaluation_env   | policy_name                        |   total_discounted_loss |   mean_period_loss |   mean_sq_inflation_gap |   mean_sq_output_gap |   mean_sq_rate_change |   mean_policy_rate |   std_policy_rate |   improvement_vs_actual_pct |   improvement_vs_taylor_pct |
|:-----------------|:-----------------------------------|------------------------:|-------------------:|------------------------:|---------------------:|----------------------:|-------------------:|------------------:|----------------------------:|----------------------------:|
| ann              | ppo_ann_direct                     |                 69.2297 |            1.23336 |                0.571374 |             1.27638  |              0.237965 |            4.85375 |          0.986362 |                    58.8507  |                     46.9126 |
| ann              | td3_ann_direct                     |                 71.239  |            1.25295 |                0.305417 |             1.5083   |              1.93379  |            3.91172 |          2.27815  |                    57.6564  |                     45.3718 |
| ann              | sac_ann_direct                     |                 86.01   |            1.54357 |                0.38708  |             1.81328  |              2.49849  |            3.72476 |          2.46862  |                    48.8766  |                     34.0449 |
| ann              | td3_svar_direct                    |                111.444  |            2.09229 |                1.1918   |             1.37749  |              2.11749  |            5.3367  |          2.16721  |                    33.759   |                     14.5415 |
| ann              | td3_ann_revealed_direct            |                114.154  |            2.03117 |                1.46768  |             1.11815  |              0.04409  |            5.0828  |          0.912082 |                    32.1479  |                     12.4629 |
| ann              | empirical_taylor_rule              |                130.407  |            2.33591 |                1.55388  |             1.54077  |              0.116428 |            5.21069 |          2.06386  |                    22.4876  |                      0      |
| ann              | td3_benchmark_transfer             |                155.226  |            2.84425 |                2.09077  |             1.3472   |              0.79874  |            4.94194 |          2.55687  |                     7.73555 |                    -19.0318 |
| ann              | riccati_reference                  |                157.011  |            2.8811  |                2.106    |             1.39958  |              0.753141 |            5.14884 |          2.70275  |                     6.67434 |                    -20.4009 |
| ann              | sac_svar_direct                    |                157.371  |            2.94634 |                1.79042  |             2.09482  |              1.08511  |            6.45629 |          2.05647  |                     6.46045 |                    -20.6769 |
| ann              | linear_policy_search_transfer      |                159.9    |            2.9363  |                2.20673  |             1.33939  |              0.598787 |            5.02625 |          2.47133  |                     4.95706 |                    -22.6164 |
| ann              | sac_benchmark_transfer             |                161.911  |            2.97455 |                2.16364  |             1.47022  |              0.757954 |            5.34278 |          2.71145  |                     3.76219 |                    -24.1579 |
| ann              | ppo_ann_direct_nonlinear           |                164.815  |            3.00957 |                2.47498  |             1.05121  |              0.089841 |            6.17196 |          0.898698 |                     2.03579 |                    -26.3852 |
| ann              | historical_actual_policy           |                168.24   |            3.34058 |                1.55898  |             3.51721  |              0.229996 |            4.75412 |          2.19169  |                     0       |                    -29.0116 |
| ann              | ppo_svar_direct_nonlinear          |                171.841  |            3.16242 |                2.26858  |             1.74375  |              0.219641 |            7.46712 |          0.989891 |                    -2.14054 |                    -31.7731 |
| ann              | sac_svar_revealed_direct           |                172.599  |            3.1435  |                2.53649  |             1.2109   |              0.015621 |            6.30121 |          0.888595 |                    -2.59064 |                    -32.3538 |
| ann              | ppo_svar_direct                    |                172.814  |            3.16724 |                2.58718  |             1.15851  |              0.008004 |            7.00066 |          0.234108 |                    -2.7189  |                    -32.5193 |
| ann              | sac_ann_revealed_direct            |                178.441  |            3.29042 |                2.8279   |             0.923449 |              0.007912 |            5.02788 |          0.350264 |                    -6.06304 |                    -36.8336 |
| ann              | ppo_svar_revealed_direct           |                224.232  |            4.28739 |                1.79815  |             4.96733  |              0.055745 |            9.11424 |          0.129813 |                   -33.2811  |                    -71.9481 |
| ann              | ppo_benchmark_transfer             |                225.277  |            4.38216 |                0.335505 |             8.06437  |              0.144733 |            3.64916 |          0.525939 |                   -33.9023  |                    -72.7495 |
| ann              | ppo_svar_revealed_direct_nonlinear |                254.855  |            4.96396 |                1.60861  |             6.68801  |              0.113455 |            9.59918 |          0.004045 |                   -51.4828  |                    -95.4303 |
| ann              | ppo_ann_revealed_direct            |                260.248  |            5.08305 |                1.60659  |             6.93982  |              0.065439 |            9.65003 |          0.126661 |                   -54.6884  |                    -99.566  |
| ann              | ppo_ann_revealed_direct_nonlinear  |                268.895  |            5.26645 |                1.52792  |             7.45149  |              0.127837 |            9.78032 |          0        |                   -59.828   |                   -106.197  |
| ann              | td3_svar_revealed_direct           |                364.344  |            7.30166 |                0.399159 |            13.7938   |              0.056    |            2.83661 |          0.546522 |                  -116.562   |                   -179.39   |

## SVAR Long-Run Stochastic Evaluation

| policy_name                        | evaluation_env   |   mean_discounted_loss |   std_discounted_loss |   mean_reward |   mean_abs_action |   clip_rate |   explosion_rate |
|:-----------------------------------|:-----------------|-----------------------:|----------------------:|--------------:|------------------:|------------:|-----------------:|
| ppo_ann_direct_nonlinear           | svar             |                 98.318 |               31.9575 |      -164.537 |          3.37109  |    0        |                0 |
| sac_svar_direct                    | svar             |                100.116 |               33.269  |      -171.227 |          3.30937  |    0        |                0 |
| ppo_svar_direct_nonlinear          | svar             |                107.706 |               35.7623 |      -181.34  |          4.10631  |    0        |                0 |
| td3_svar_direct                    | svar             |                109.47  |               33.459  |      -185.762 |          3.37363  |    0        |                0 |
| empirical_taylor_rule              | svar             |                111.711 |               34.4294 |      -191.834 |          2.99697  |    0.007378 |                0 |
| ppo_svar_direct                    | svar             |                126.817 |               45.7506 |      -222.811 |          4.46166  |    0        |                0 |
| linear_policy_search_transfer      | svar             |                134.453 |               38.3968 |      -230.234 |          2.33767  |    0.031424 |                0 |
| sac_benchmark_transfer             | svar             |                135.267 |               39.5613 |      -233.464 |          2.70708  |    0.040538 |                0 |
| riccati_reference                  | svar             |                135.843 |               37.7737 |      -234.367 |          2.58518  |    0.061198 |                0 |
| sac_svar_revealed_direct           | svar             |                138.545 |               51.2718 |      -246.014 |          3.31939  |    0        |                0 |
| td3_benchmark_transfer             | svar             |                150.971 |               40.3576 |      -260.451 |          2.56505  |    0.072656 |                0 |
| sac_ann_revealed_direct            | svar             |                183.48  |              121.427  |      -335.759 |          4.09283  |    0        |                0 |
| td3_ann_revealed_direct            | svar             |                184.35  |              145.333  |      -338.65  |          3.81608  |    0        |                0 |
| ppo_ann_direct                     | svar             |                206.105 |              145.304  |      -375.859 |          2.50719  |    0        |                0 |
| ppo_benchmark_transfer             | svar             |                238.086 |               59.8023 |      -425.302 |          1.14342  |    0.001997 |                0 |
| ppo_ann_revealed_direct            | svar             |                245.594 |               76.8752 |      -442.688 |          6.3248   |    0        |                0 |
| td3_svar_revealed_direct           | svar             |                271.028 |               81.6177 |      -500.788 |          0.863713 |    0        |                0 |
| td3_ann_direct                     | svar             |                371.643 |              191.146  |      -709.851 |          1.71778  |    0        |                0 |
| ppo_svar_revealed_direct_nonlinear | svar             |                410.589 |              148.663  |      -783.197 |          7.57538  |    0        |                0 |
| ppo_svar_revealed_direct           | svar             |                414.916 |              145.486  |      -795.094 |          7.64231  |    0        |                0 |
| ppo_ann_revealed_direct_nonlinear  | svar             |                421.698 |              128.268  |      -793.999 |          7.78031  |    0        |                0 |
| sac_ann_direct                     | svar             |                439.546 |              227.914  |      -840.284 |          2.88501  |    0        |                0 |

## ANN Long-Run Stochastic Evaluation

| policy_name                        | evaluation_env   |   mean_discounted_loss |   std_discounted_loss |   mean_reward |   mean_abs_action |   clip_rate |   explosion_rate |
|:-----------------------------------|:-----------------|-----------------------:|----------------------:|--------------:|------------------:|------------:|-----------------:|
| ppo_ann_direct                     | ann              |                78.4765 |               18.1763 |      -131.711 |          3.1319   |    0        |                0 |
| td3_ann_direct                     | ann              |                81.2336 |               12.4516 |      -137.097 |          2.69522  |    0        |                0 |
| sac_ann_direct                     | ann              |                91.7854 |               13.3482 |      -156.207 |          2.73318  |    0        |                0 |
| td3_svar_direct                    | ann              |               137.659  |               26.8201 |      -236.581 |          3.83055  |    0        |                0 |
| td3_ann_revealed_direct            | ann              |               142.358  |               22.8806 |      -248.286 |          3.26348  |    0        |                0 |
| td3_benchmark_transfer             | ann              |               148.608  |               16.1487 |      -254.559 |          2.85768  |    0.042622 |                0 |
| riccati_reference                  | ann              |               152.73   |               19.1538 |      -261.885 |          2.98896  |    0.03316  |                0 |
| empirical_taylor_rule              | ann              |               159.656  |               31.6506 |      -281.573 |          3.80139  |    0        |                0 |
| sac_benchmark_transfer             | ann              |               159.817  |               19.8193 |      -276.151 |          3.12522  |    0.028299 |                0 |
| ppo_ann_direct_nonlinear           | ann              |               173.805  |               33.6903 |      -308.405 |          4.10015  |    0        |                0 |
| linear_policy_search_transfer      | ann              |               179.944  |               21.0006 |      -313.739 |          2.79321  |    0.013455 |                0 |
| sac_svar_direct                    | ann              |               184.183  |               24.0986 |      -320.41  |          4.6468   |    0        |                0 |
| sac_ann_revealed_direct            | ann              |               193.071  |               25.2651 |      -336.445 |          3.26439  |    0        |                0 |
| ppo_svar_direct_nonlinear          | ann              |               194.144  |               17.3268 |      -339.396 |          5.42917  |    0        |                0 |
| ppo_svar_direct                    | ann              |               202.591  |               18.9013 |      -354.234 |          4.98253  |    0        |                0 |
| sac_svar_revealed_direct           | ann              |               222.126  |               34.4356 |      -392.997 |          3.09964  |    0        |                0 |
| ppo_svar_revealed_direct           | ann              |               267.592  |               28.693  |      -473.252 |          7.1026   |    0        |                0 |
| ppo_ann_revealed_direct            | ann              |               311.405  |               48.4617 |      -559.268 |          7.64104  |    0        |                0 |
| ppo_svar_revealed_direct_nonlinear | ann              |               317.321  |               46.9151 |      -566.158 |          7.60029  |    0        |                0 |
| ppo_ann_revealed_direct_nonlinear  | ann              |               323.101  |               43.1943 |      -577.644 |          7.78032  |    0        |                0 |
| ppo_benchmark_transfer             | ann              |               327.15   |               34.6617 |      -588.786 |          1.75737  |    0.000781 |                0 |
| td3_svar_revealed_direct           | ann              |               693.439  |              219.163  |     -1361.51  |          0.717517 |    0        |                0 |

## Cross-Transfer Summary

| policy_name                   | rule_family        | source_env   | evaluation_env   | policy_parameterization   |   mean_discounted_loss |   std_discounted_loss |   clip_rate |   explosion_rate |
|:------------------------------|:-------------------|:-------------|:-----------------|:--------------------------|-----------------------:|----------------------:|------------:|-----------------:|
| td3_svar_direct               | svar_direct        | svar         | ann              | standard_nonlinear        |                137.659 |               26.8201 |    0        |                0 |
| td3_benchmark_transfer        | benchmark_transfer | benchmark    | ann              | linear_surrogate          |                148.608 |               16.1487 |    0.042622 |                0 |
| sac_benchmark_transfer        | benchmark_transfer | benchmark    | ann              | linear_surrogate          |                159.817 |               19.8193 |    0.028299 |                0 |
| linear_policy_search_transfer | benchmark_transfer | benchmark    | ann              | fixed_rule                |                179.944 |               21.0006 |    0.013455 |                0 |
| sac_svar_direct               | svar_direct        | svar         | ann              | standard_nonlinear        |                184.183 |               24.0986 |    0        |                0 |
| ppo_svar_direct_nonlinear     | svar_direct        | svar         | ann              | nonlinear_policy          |                194.144 |               17.3268 |    0        |                0 |
| ppo_svar_direct               | svar_direct        | svar         | ann              | linear_policy             |                202.591 |               18.9013 |    0        |                0 |
| ppo_benchmark_transfer        | benchmark_transfer | benchmark    | ann              | linear_surrogate          |                327.15  |               34.6617 |    0.000781 |                0 |
| ppo_ann_direct_nonlinear      | ann_direct         | ann          | svar             | nonlinear_policy          |                 98.318 |               31.9575 |    0        |                0 |
| linear_policy_search_transfer | benchmark_transfer | benchmark    | svar             | fixed_rule                |                134.453 |               38.3968 |    0.031424 |                0 |
| sac_benchmark_transfer        | benchmark_transfer | benchmark    | svar             | linear_surrogate          |                135.267 |               39.5613 |    0.040538 |                0 |
| td3_benchmark_transfer        | benchmark_transfer | benchmark    | svar             | linear_surrogate          |                150.971 |               40.3576 |    0.072656 |                0 |
| ppo_ann_direct                | ann_direct         | ann          | svar             | linear_policy             |                206.105 |              145.304  |    0        |                0 |
| ppo_benchmark_transfer        | benchmark_transfer | benchmark    | svar             | linear_surrogate          |                238.086 |               59.8023 |    0.001997 |                0 |
| td3_ann_direct                | ann_direct         | ann          | svar             | standard_nonlinear        |                371.643 |              191.146  |    0        |                0 |
| sac_ann_direct                | ann_direct         | ann          | svar             | standard_nonlinear        |                439.546 |              227.914  |    0        |                0 |

## Notes

- `Phase 8/9` remains the benchmark-transfer baseline; this file adds the direct-trained empirical rules under the same evaluator.
- `benchmark transfer` and empirical direct-trained rules are kept distinct in all tables.
- Lucas critique still applies because both empirical environments hold reduced-form transitions fixed under policy changes.