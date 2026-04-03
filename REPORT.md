# （AI 生成）中期报告清单

## 1. 研究目标与当前完成度

| 项目 | 当前状态 |
|---|---|
| 研究目标 | 以 RL 作为货币政策规则求解器，与 benchmark 最优控制、经验 Taylor、历史政策、外部模型基线做系统比较 |
| 已完成主线 | benchmark LQ、经验 `SVAR/ANN`、direct training、revealed welfare、统一反事实、长期随机、交叉迁移、外部模型、非 benchmark 扩展、`pyfrbus` 原生优化 |
| 当前口径 | benchmark 与经验环境严格区分；经验结果始终提示 `Lucas critique`，不作结构最优政策的过度解释 |
| 中期判断 | 除论文写作外，核心实验模块已完成；后续主要是收敛为论文叙事与取舍 |

## 2.  核心结果

> 本报告只保留当前最终矩阵与可写入论文主体的结果，不再重复 phase7 之后各轮扩容中的早期中间版本。

| 模块 | 最终保留口径 | 对应产物 |
|---|---|---|
| Benchmark 基准 | `LQ + Riccati + benchmark transfer (PPO/TD3/SAC surrogate)` | `outputs/phase4/`、`outputs/phase7/matrix/` |
| 经验 direct 训练 | `SVAR/ANN` 中直接训练 `PPO/TD3/SAC`，其中 `PPO` 区分 `linear/nonlinear` | `outputs/phase10/svar_direct/`、`outputs/phase10/ann_direct/`、`outputs/phase10/ppo_policy_variants/` |
| 经验 revealed 训练 | `SVAR/ANN revealed direct`，与人工损失口径分开报告 | `outputs/phase10/revealed_policy_training/` |
| 统一评价 | 历史反事实、长期随机、交叉迁移；人工损失与反推损失两套评分 | `outputs/phase10/counterfactual_eval/`、`outputs/phase10/revealed_policy_eval/`、`outputs/phase10/revealed_welfare/` |
| 外部模型 | `pyfrbus + US_SW07 + US_CCTW10 + US_KS15 + NK_CW09` | `outputs/phase10/external_model_robustness/` |
| **非 benchmark 扩展** | phase11 最终保留 `v2` 六环境：`nonlinear` 两档、`zlb trap` 两档、`threshold asymmetric` 两档 | `outputs/phase11/extreme_matrix_v2/`、`outputs/phase11/extreme_numerical_compare_v2/` |
| `pyfrbus` 原生优化 | phase12 保留 `warm-start nonlinear PPO` 相对 `fine_05` 的最终搜索结果 | `outputs/phase12/pyfrbus_warmstart_ppo/`、`outputs/phase12/pyfrbus_nonlinear_search/` |

## 3. 进一步结果与稳健性

### 3.1 经验环境：人工损失下的统一比较

| 维度 | 最终观察 |
|---|---|
| `SVAR` 历史反事实 | 最优规则不是 benchmark transfer，而是经验环境训练出的规则；**`ppo_ann_direct_nonlinear` 在人工损失下略优于历史实际政策**，说明迁移规则并非主导 |
| `SVAR` 长期随机 | `ppo_ann_direct_nonlinear`、`sac_svar_direct`、`ppo_svar_direct_nonlinear` 位于前列；经验训练优于 benchmark transfer |
| `ANN` 长期随机 | `ppo_ann_direct`、`td3_ann_direct`、`sac_ann_direct` 排名靠前；ANN 内 direct 规则在本环境内最有竞争力 |
| 交叉迁移 | `SVAR → ANN` 与 `ANN → SVAR` 都存在明显退化；经验 direct 规则优势高度依赖训练环境 |
| 结论 | 人工损失下不能简单说“RL 全面优于历史政策/基准规则”，更准确的表述是：**经验环境内训练的 RL 规则通常优于 benchmark transfer，但优势具有环境依赖性** |

### 3.2 反推福利：排序会显著改变

| 项目 | 结果 |
|---|---|
| 反推权重 | 识别得到的权重大致为 `inflation = 1`、`output_gap ≈ 0.882`、`rate_smoothing ≈ 20.09` |
| `SVAR revealed` 历史反事实 | **`sac_svar_revealed_direct` 最优，明显优于经验 Taylor 与历史实际政策** |
| `SVAR revealed` 长期随机 | `sac_svar_revealed_direct` 仍最优 |
| `ANN revealed` 历史反事实 | **`td3_ann_revealed_direct` 与 `sac_ann_revealed_direct` 最优，明显优于经验 Taylor 与历史实际政策** |
| `ANN revealed` 长期随机 | `sac_ann_revealed_direct` 最优 |
| 结论 | **福利标尺改变会重排政策优劣**；因此人工损失与反推损失必须分开报告，不能混用 |

### 3.2A 经济变量波动

| 口径 | 补充观察 |
|---|---|
| 说明 | 既有 phase10 汇总以损失为主；历史反事实虽已保存二阶矩，但不是中心化方差。本次据逐期路径与长期随机再模拟，补算了 `inflation gap`、`output gap`、`policy rate`、`rate change` 的标准差。 |
| 人工损失，`SVAR` 历史 | `ppo_ann_direct_nonlinear` 相比历史实际政策，`output gap` 波动更低（`1.17 < 1.26`），利率波动显著更低（`0.90 < 2.19`），但通胀波动略高（`0.80 > 0.75`）；因此其优势不是“所有变量都更平滑”，而是**更偏向产出与利率平滑**。 |
| 人工损失，`ANN` 长期随机 | `ppo_ann_direct` 相比经验 Taylor 同时降低通胀与产出波动（`0.72/0.98` vs `0.93/1.05`），但利率调整更频繁（`rate-change std 0.52` vs `0.25`）；ANN 环境中的 **RL 优势部分来自更积极的政策反应**。 |
| 反推损失，`SVAR` | `sac_svar_revealed_direct` 在历史与长期随机下都明显压低产出与利率波动，但通胀波动高于经验 Taylor；**这与反推福利中的高利率平滑权重一致**，说明其改进依赖福利口径而非传统稳通胀标准。 |
| 反推损失，`ANN` | `sac_ann_revealed_direct`/`td3_ann_revealed_direct` 不仅损失更低，也显著压低目标变量与利率波动；**其中 `sac_ann_revealed_direct` 在长期随机下对 Taylor 同时实现更低的通胀、产出与利率变动标准差**。 |
| 交叉迁移解释 | 跨环境后损失排序会明显退化，说明“更平滑”主要是环境内性质；经验环境中的波动改进不能直接外推为结构稳健结论，这一点也再次对应 `Lucas critique`。 |

### 3.3 外部模型：迁移稳健性高度异质

| 外部模型 | 当前可保留结论 |
|---|---|
| `pyfrbus` | 绝大多数迁移 RL 规则都不如 model-native baseline；这说明外部结构变化下，经验 RL 规则存在明显失配，**主要是参数量级等原因导致的，进一步调整即可**。 |
| `US_SW07` | baseline 最稳，迁移 RL 普遍较差 |
| `US_CCTW10` | **`td3_svar_direct` 表现最好**，说明部分外部模型上 RL 规则可以优于 baseline |
| `US_KS15` | **`td3_svar_direct` 也最优**，说明经验 direct 规则在部分模型上具备跨模型竞争力 |
| `NK_CW09` | `empirical_taylor_rule` 最优，说明简单经验规则在某些 NK 外部模型中更稳健 |
| 总结 | 外部模型结果不支持“RL 普遍支配”；更准确的结论是：**RL 规则存在模型选择性优势，而不是统一优势** |

### 3.4 `pyfrbus` 局部改进

| 阶段 | 结果 |
|---|---|
| 直接迁移 RL | 基本都明显差于 `pyfrbus_baseline` |
| 本地线性微调 | 找到 `fine_05`，人工损失 `0.012596`、反推损失 `0.015951`，优于 `pyfrbus_baseline` |
| 原生深度 RL（早期 SAC） | 结果很差，说明“直接 native RL”当时训练法不稳 |
| phase12 warm-start PPO | 先把 `fine_05` 作为 warm start，再做 residual-style nonlinear PPO，最终 `residual_u2_lr5e6` 反推损失 `0.015942`，**略优于 `fine_05`** |
| 结论 | `pyfrbus` 的关键信息不是“RL 一开始就好”，而是：**直接迁移失败，但在 model-native warm-start 之后，nonlinear PPO 可以小幅超过最优线性规则** |

### 3.5 phase11 非 benchmark 扩展：最终保留 `v2`

| 环境 | 最优原始 RL | 相对 Riccati 外推 |
|---|---:|---:|
| `nonlinear_extreme_v2` | `TD3` | `+74.19%` |
| `nonlinear_hyper` | `TD3` | `+3.50%` |
| `zlb_trap_very_strong` | `TD3` | `+0.16%` |
| `zlb_trap_extreme` | `TD3` | `+9.46%` |
| `asymmetric_threshold_very_strong` | `PPO` | `+12.19%` |
| `asymmetric_threshold_extreme` | `SAC` | `+17.38%` |

| 解释 | 结论 |
|---|---|
| 原始 RL vs Riccati | 六个 `v2` 环境里，原始 RL 全部压过 `Riccati` 外推 |
| RL surrogate vs Riccati | 并非所有环境都成立；`nonlinear` 与 `zlb` 的优势主要依赖非线性/状态依赖行为 |
| 数值解对照 | 大多数环境中有限期 DP 仍优于 RL surrogate；说明 RL 优势主要体现在“外推 Riccati 失灵后仍能找到可行改进” |
| 报告口径 | phase11 的最终结论应写成：**在真正非 benchmark 扩展下，RL 能突破 Riccati 外推，但与环境内数值最优相比仍有差距** |

## 4. 与开题报告及 Hinterlang 的关系

| 对照 | 当前判断 |
|---|---|
| 开题报告 | 原计划要求的 benchmark、经验环境、历史反事实、外部比较都已完成；实际工作已扩展到双福利、双经验环境、交叉迁移、非 benchmark 扩展、`pyfrbus` 原生优化 |
| 相对 Hinterlang | 当前项目不止做经验环境 RL 与 DSGE 稳健性，还额外加入 benchmark/empirical 严格分离、人工/反推双福利、长期随机、交叉迁移、非 benchmark 扩展，比较维度更全 |
| 需要如实表述之处 | 结果不支持“RL 在所有环境都优于传统规则”；真正更强的地方在于**比较框架更完整、边界条件更清楚、何时有效何时失效更可识别** |

## 5. 论文主体建议保留与弱化内容

| 类别 | 建议 |
|---|---|
| 主体保留 | benchmark 基准、`SVAR/ANN` direct 与 revealed、统一历史反事实、长期随机、交叉迁移、外部模型总表、phase11 `v2`、phase12 `pyfrbus` |
| 主体弱化 | phase7 之后早期扩容中的中间版本、已被 `v2` 取代的非 benchmark 环境、未能形成稳健结论的早期原生 SAC 尝试 |
| 附录保留 | 全量 case inventory、早期 phase11 `v1`、各类失败或退化的中间搜索 |
| 写作主轴 | **RL 是有条件的数值求解优势，而不是无条件支配优势**；benchmark、经验环境与外部模型必须分别解释 |

## 6. 当前总体结论

| 主题 | 最终表述 |
|---|---|
| Benchmark | RL 可以逼近或改进 benchmark 外推规则，但结论依赖扩展环境类型 |
| Empirical | direct/revealed 训练能够学到有竞争力规则，但人工损失与反推损失下的最优规则不同 |
| External | 外部模型稳健性高度异质，不能把经验环境内的优势直接外推为结构稳健优势 |
| `pyfrbus` | model-native warm-start 是有效方向；直接迁移和粗糙 native RL 都不够好 |
| 整体 | 本文最可靠的贡献不是“证明 RL 永远更好”，而是**建立了一套更完整的比较框架，明确了 RL 优势出现的条件与边界** |

## 7. 主要对应文件

| 模块 | 文件 |
|---|---|
| 中期总汇总 | `REPORT.md` |
| phase10 总结 | `outputs/phase10/phase10_summary.md` |
| 统一反事实 | `outputs/phase10/counterfactual_eval/counterfactual_eval_summary.md` |
| 反推福利 | `outputs/phase10/revealed_welfare/revealed_welfare_summary.md` |
| 外部模型 | `outputs/phase10/external_model_robustness/external_model_robustness_summary.md` |
| phase11 最终扩展 | `outputs/phase11/extreme_v2_overview.md` |
| phase12 `pyfrbus` | `outputs/phase12/pyfrbus_warmstart_ppo/summary.md`、`outputs/phase12/pyfrbus_nonlinear_search/summary.md` |

一句话概括：截至目前，项目已完成 benchmark、经验环境、双福利、统一反事实、外部模型与非 benchmark 扩展的完整矩阵；最终结论不是“RL 全面胜出”，而是“RL 在特定环境与目标下可显著改进传统规则，但这种优势具有明确边界”。 
