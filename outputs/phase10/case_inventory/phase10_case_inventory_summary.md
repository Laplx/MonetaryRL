# Phase 10 Case Inventory

- Total cases: `432`
- CSV: `outputs\phase10\case_inventory\phase10_case_inventory.csv`

## Count By Group

| case_group             | evaluation_type     | loss_function   |   case_count |
|:-----------------------|:--------------------|:----------------|-------------:|
| empirical_unified      | cross_transfer      | artificial      |           16 |
| empirical_unified      | historical_shock    | artificial      |           46 |
| empirical_unified      | historical_shock    | revealed        |           50 |
| empirical_unified      | long_run_stochastic | artificial      |           44 |
| empirical_unified      | long_run_stochastic | revealed        |           44 |
| external_models        | external_model_eval | artificial      |          101 |
| external_models        | external_model_eval | revealed        |          101 |
| revealed_trained_rules | cross_transfer      | revealed        |            6 |
| revealed_trained_rules | historical_shock    | revealed        |           12 |
| revealed_trained_rules | long_run_stochastic | revealed        |           12 |

## Count By Environment

| environment   | evaluation_type     | loss_function   |   case_count |
|:--------------|:--------------------|:----------------|-------------:|
| NK_CW09       | external_model_eval | artificial      |           21 |
| NK_CW09       | external_model_eval | revealed        |           21 |
| US_CCTW10     | external_model_eval | artificial      |           21 |
| US_CCTW10     | external_model_eval | revealed        |           21 |
| US_KS15       | external_model_eval | artificial      |           21 |
| US_KS15       | external_model_eval | revealed        |           21 |
| US_SW07       | external_model_eval | artificial      |           21 |
| US_SW07       | external_model_eval | revealed        |           21 |
| ann           | cross_transfer      | artificial      |            8 |
| ann           | historical_shock    | artificial      |           23 |
| ann           | historical_shock    | revealed        |           31 |
| ann           | long_run_stochastic | artificial      |           22 |
| ann           | long_run_stochastic | revealed        |           22 |
| ann_revealed  | cross_transfer      | revealed        |            3 |
| ann_revealed  | long_run_stochastic | revealed        |            6 |
| pyfrbus       | external_model_eval | artificial      |           17 |
| pyfrbus       | external_model_eval | revealed        |           17 |
| svar          | cross_transfer      | artificial      |            8 |
| svar          | historical_shock    | artificial      |           23 |
| svar          | historical_shock    | revealed        |           31 |
| svar          | long_run_stochastic | artificial      |           22 |
| svar          | long_run_stochastic | revealed        |           22 |
| svar_revealed | cross_transfer      | revealed        |            3 |
| svar_revealed | long_run_stochastic | revealed        |            6 |

## Fields

| 字段 | 含义 |
|---|---|
| `environment` | 经验环境或外部模型名 |
| `policy_name` | 规则名 |
| `loss_function` | `artificial` 或 `revealed` |
| `evaluation_type` | `historical_shock` / `long_run_stochastic` / `cross_transfer` / `external_model_eval` |
| `result_metric` | 主结果指标名 |
| `result_value` | 主结果指标值 |
| `comparison_metric` | 相对 baseline 或历史政策的比较列 |
| `comparison_value` | 比较值 |

## Preview

| case_id   | case_group        | evaluation_type   | loss_function   | environment   | policy_name                   | rule_family        | source_env   | training_env   | policy_parameterization   | algo   | source_file                                                    | result_metric         |   result_value | secondary_metric    |   secondary_value | comparison_metric         |   comparison_value | solver_status   |
|:----------|:------------------|:------------------|:----------------|:--------------|:------------------------------|:-------------------|:-------------|:---------------|:--------------------------|:-------|:---------------------------------------------------------------|:----------------------|---------------:|:--------------------|------------------:|:--------------------------|-------------------:|:----------------|
| case_0001 | empirical_unified | cross_transfer    | artificial      | ann           | linear_policy_search_transfer | benchmark_transfer | benchmark    | benchmark      | fixed_rule                | nan    | outputs\phase10\counterfactual_eval\cross_transfer_summary.csv | mean_discounted_loss  |       179.944  | std_discounted_loss |          21.0006  |                           |          nan       |                 |
| case_0002 | empirical_unified | cross_transfer    | artificial      | ann           | ppo_benchmark_transfer        | benchmark_transfer | benchmark    | benchmark      | linear_surrogate          | ppo    | outputs\phase10\counterfactual_eval\cross_transfer_summary.csv | mean_discounted_loss  |       327.15   | std_discounted_loss |          34.6617  |                           |          nan       |                 |
| case_0003 | empirical_unified | cross_transfer    | artificial      | ann           | sac_benchmark_transfer        | benchmark_transfer | benchmark    | benchmark      | linear_surrogate          | sac    | outputs\phase10\counterfactual_eval\cross_transfer_summary.csv | mean_discounted_loss  |       159.817  | std_discounted_loss |          19.8193  |                           |          nan       |                 |
| case_0004 | empirical_unified | cross_transfer    | artificial      | ann           | td3_benchmark_transfer        | benchmark_transfer | benchmark    | benchmark      | linear_surrogate          | td3    | outputs\phase10\counterfactual_eval\cross_transfer_summary.csv | mean_discounted_loss  |       148.608  | std_discounted_loss |          16.1487  |                           |          nan       |                 |
| case_0005 | empirical_unified | cross_transfer    | artificial      | ann           | ppo_svar_direct               | svar_direct        | svar         | svar           | linear_policy             | ppo    | outputs\phase10\counterfactual_eval\cross_transfer_summary.csv | mean_discounted_loss  |       202.591  | std_discounted_loss |          18.9013  |                           |          nan       |                 |
| case_0006 | empirical_unified | cross_transfer    | artificial      | ann           | ppo_svar_direct_nonlinear     | svar_direct        | svar         | svar           | nonlinear_policy          | ppo    | outputs\phase10\counterfactual_eval\cross_transfer_summary.csv | mean_discounted_loss  |       194.144  | std_discounted_loss |          17.3268  |                           |          nan       |                 |
| case_0007 | empirical_unified | cross_transfer    | artificial      | ann           | sac_svar_direct               | svar_direct        | svar         | svar           | standard_nonlinear        | sac    | outputs\phase10\counterfactual_eval\cross_transfer_summary.csv | mean_discounted_loss  |       184.183  | std_discounted_loss |          24.0986  |                           |          nan       |                 |
| case_0008 | empirical_unified | cross_transfer    | artificial      | ann           | td3_svar_direct               | svar_direct        | svar         | svar           | standard_nonlinear        | td3    | outputs\phase10\counterfactual_eval\cross_transfer_summary.csv | mean_discounted_loss  |       137.659  | std_discounted_loss |          26.8201  |                           |          nan       |                 |
| case_0009 | empirical_unified | cross_transfer    | artificial      | svar          | ppo_ann_direct                | ann_direct         | ann          | ann            | linear_policy             | ppo    | outputs\phase10\counterfactual_eval\cross_transfer_summary.csv | mean_discounted_loss  |       206.105  | std_discounted_loss |         145.304   |                           |          nan       |                 |
| case_0010 | empirical_unified | cross_transfer    | artificial      | svar          | ppo_ann_direct_nonlinear      | ann_direct         | ann          | ann            | nonlinear_policy          | ppo    | outputs\phase10\counterfactual_eval\cross_transfer_summary.csv | mean_discounted_loss  |        98.318  | std_discounted_loss |          31.9575  |                           |          nan       |                 |
| case_0011 | empirical_unified | cross_transfer    | artificial      | svar          | sac_ann_direct                | ann_direct         | ann          | ann            | standard_nonlinear        | sac    | outputs\phase10\counterfactual_eval\cross_transfer_summary.csv | mean_discounted_loss  |       439.546  | std_discounted_loss |         227.914   |                           |          nan       |                 |
| case_0012 | empirical_unified | cross_transfer    | artificial      | svar          | td3_ann_direct                | ann_direct         | ann          | ann            | standard_nonlinear        | td3    | outputs\phase10\counterfactual_eval\cross_transfer_summary.csv | mean_discounted_loss  |       371.643  | std_discounted_loss |         191.146   |                           |          nan       |                 |
| case_0013 | empirical_unified | cross_transfer    | artificial      | svar          | linear_policy_search_transfer | benchmark_transfer | benchmark    | benchmark      | fixed_rule                | nan    | outputs\phase10\counterfactual_eval\cross_transfer_summary.csv | mean_discounted_loss  |       134.453  | std_discounted_loss |          38.3968  |                           |          nan       |                 |
| case_0014 | empirical_unified | cross_transfer    | artificial      | svar          | ppo_benchmark_transfer        | benchmark_transfer | benchmark    | benchmark      | linear_surrogate          | ppo    | outputs\phase10\counterfactual_eval\cross_transfer_summary.csv | mean_discounted_loss  |       238.086  | std_discounted_loss |          59.8023  |                           |          nan       |                 |
| case_0015 | empirical_unified | cross_transfer    | artificial      | svar          | sac_benchmark_transfer        | benchmark_transfer | benchmark    | benchmark      | linear_surrogate          | sac    | outputs\phase10\counterfactual_eval\cross_transfer_summary.csv | mean_discounted_loss  |       135.267  | std_discounted_loss |          39.5613  |                           |          nan       |                 |
| case_0016 | empirical_unified | cross_transfer    | artificial      | svar          | td3_benchmark_transfer        | benchmark_transfer | benchmark    | benchmark      | linear_surrogate          | td3    | outputs\phase10\counterfactual_eval\cross_transfer_summary.csv | mean_discounted_loss  |       150.971  | std_discounted_loss |          40.3576  |                           |          nan       |                 |
| case_0017 | empirical_unified | historical_shock  | artificial      | ann           | ppo_ann_direct                | ann_direct         | ann          | ann            | linear_policy             | ppo    | outputs\phase10\counterfactual_eval\ann_historical_summary.csv | total_discounted_loss |        69.2297 | mean_period_loss    |           1.23336 | improvement_vs_actual_pct |           58.8507  |                 |
| case_0018 | empirical_unified | historical_shock  | artificial      | ann           | ppo_ann_direct_nonlinear      | ann_direct         | ann          | ann            | nonlinear_policy          | ppo    | outputs\phase10\counterfactual_eval\ann_historical_summary.csv | total_discounted_loss |       164.815  | mean_period_loss    |           3.00957 | improvement_vs_actual_pct |            2.03579 |                 |
| case_0019 | empirical_unified | historical_shock  | artificial      | ann           | sac_ann_direct                | ann_direct         | ann          | ann            | standard_nonlinear        | sac    | outputs\phase10\counterfactual_eval\ann_historical_summary.csv | total_discounted_loss |        86.01   | mean_period_loss    |           1.54357 | improvement_vs_actual_pct |           48.8766  |                 |
| case_0020 | empirical_unified | historical_shock  | artificial      | ann           | td3_ann_direct                | ann_direct         | ann          | ann            | standard_nonlinear        | td3    | outputs\phase10\counterfactual_eval\ann_historical_summary.csv | total_discounted_loss |        71.239  | mean_period_loss    |           1.25295 | improvement_vs_actual_pct |           57.6564  |                 |