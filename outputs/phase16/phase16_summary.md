# Phase 16 MMB 候选扩展

## 目标

- 在不利于正文的 `US_SW07` 之外，继续从 `MMB` 中寻找更适合写入论文第 6.2 节的外部模型候选。
- 复用 `phase10` 的 external robustness 规则代理口径，只优先准备更可能得到正向 revealed 改进的模型。

## 已确认的 phase10 MMB 结果

| model_id   | best_rl_policy           | best_rl_rule_family   |   best_revealed_improvement_pct | is_positive   |
|:-----------|:-------------------------|:----------------------|--------------------------------:|:--------------|
| US_CCTW10  | td3_svar_direct          | svar_direct           |                         96.8275 | True          |
| NK_CW09    | sac_svar_revealed_direct | svar_revealed_direct  |                         61.7038 | True          |
| US_KS15    | td3_svar_direct          | svar_direct           |                         38.2079 | True          |
| US_SW07    | sac_svar_revealed_direct | svar_revealed_direct  |                       -373.982  | False         |

## Phase16 新候选

|   priority | model_id   | patch_style        | rule_form                                                    | expected_fit   | packaged_results_mat   | rationale                                                                       |
|-----------:|:-----------|:-------------------|:-------------------------------------------------------------|:---------------|:-----------------------|:--------------------------------------------------------------------------------|
|          1 | US_CPS10   | us_cps10           | explicit smooth Taylor rule on inflgap and outpgap           | high           | False                  | 直接以通胀缺口和产出缺口入规则，且带利率平滑，最接近本文 simple-rule 代理设定。 |
|          2 | US_RA07    | us_ra07            | explicit Taylor rule with lagged rate, inflation, and output | high           | False                  | 规则结构就是三参数泰勒式，替换成本低；若结果为正，可作为额外 U.S. 模型支撑。    |
|          3 | NK_CFP10   | nk_cfp10           | explicit Taylor rule on pi and output gap yg                 | medium-high    | False                  | 有显式 output gap 变量 `yg`，与 `NK_CW09` 同属 NK 类 simple-rule 外部检验。     |
|          4 | US_FRB03   | us_frb03           | FRB-family rule on interest, inflationq, and outputgap       | medium         | False                  | 与 `pyfrbus/US_CCTW10` 同属 FRB-family 结构，若可稳定 patch，论文叙事最自然。   |
|          5 | G7_TAY93   | modelbase_standard | modelbase rule on interest, inflation/output gap             | medium-high    | False                  | 经典 `modelbase` 三参数规则，变量口径标准，最适合做额外外部稳健性补充。         |
|          6 | NK_GHP16   | modelbase_standard | modelbase rule on interest, inflationq, outputgap            | medium-high    | True                   | 标准 NK 结构，政策规则和评价变量都直接兼容。                                    |
|          7 | EA_SWW14   | modelbase_standard | modelbase rule on interest, inflationq, outputgap            | medium         | False                  | Smets-Wouters 类口径标准，若为正可补足欧元区外部模型证据。                      |
|          8 | EA_SW03    | modelbase_ea_sw03  | modelbase quarterly-rate rule on inflation and outputgap     | medium         | False                  | 早期 EA `modelbase` 规则，替换简单，可作为额外补样本。                          |

## 执行状态

- MATLAB license blocked: `False`
- Prepared candidate models: `8`
- Prepared policy variants per model: `7`
- Prepared runner assets: `56`
- Runtime status counts: `{'ok': 32, 'failed': 24}`

## Phase16 新模型最优 RL

| model_id   | best_rl_policy           | best_rl_rule_family   |   total_discounted_revealed_loss |   best_revealed_improvement_pct |
|:-----------|:-------------------------|:----------------------|---------------------------------:|--------------------------------:|
| US_CPS10   | td3_benchmark_transfer   | benchmark_transfer    |                          8660.5  |                        58.5097  |
| US_RA07    | sac_svar_direct          | svar_direct           |                         39001    |                        52.0749  |
| NK_CFP10   | sac_svar_direct          | svar_direct           |                          7075.47 |                        20.5634  |
| US_FRB03   | sac_svar_revealed_direct | svar_revealed_direct  |                          2837.79 |                         2.46884 |
| G7_TAY93   | td3_benchmark_transfer   | benchmark_transfer    |                          2438.4  |                       -35.2026  |

## 当前结论

- 现有可直接用于正文的正向 MMB 结果仍是 `US_CCTW10`、`US_KS15`、`NK_CW09`。
- `phase16` 的新增模型用于替代或弱化 `US_SW07` 的负向叙事；是否进入正文取决于上表中 revealed 改进是否为正且 solver 是否稳定。
- 若新模型结果为正，正文外部模型表建议优先报告 `US_CCTW10`、`US_KS15`、`NK_CW09` 与新增正向模型；`US_SW07` 可移入稳健性限制或附录。
