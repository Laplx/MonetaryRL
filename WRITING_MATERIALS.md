# 写作材料总汇

> 目标：把开题报告、早期理论写作提醒、`phase13` 图表配套分析与最终矩阵口径合并成一份“写初稿时可直接查”的材料总汇。

## 0. 写作总原则

| 主题 | 必须坚持的口径 |
|---|---|
| benchmark / empirical | 两者必须严格区分，不能把经验环境结果写成理论最优政策定理。 |
| RL 定位 | RL 是 Bellman/动态规划问题的数值求解器，不是对理论框架的替代。 |
| 福利口径 | 人工损失与反推损失必须分开报告，不能混用。 |
| 经验边界 | 经验环境反事实始终要提醒 `Lucas critique`。 |
| 正文矩阵 | 只写最终矩阵：`phase11 v2`、`phase10 unified eval`、`phase12 pyfrbus warm-start`。 |
| 叙事主线 | 主线写 RL 的显著优势；边界与失败写在稳健性/讨论，而不是反客为主。 |

## 1. 与开题报告对齐的主线

### 1.1 开题报告中的原始目标

| 开题报告原话 | 写作落地 |
|---|---|
| 比较经典最优控制与 RL 在央行规则求解中的表现 | 第 2–3 章完成：benchmark、Riccati、RL、强扭曲扩展 |
| 重点考察 RL 在可解析与不可解析情形下对最优政策的近似能力及其结构特征 | 第 3 章与第 4 章完成：非 benchmark 扩展、规则切片、波动与反事实 |
| 进行历史政策的反事实比较 | 第 4 章完成：历史反事实、长期随机、交叉迁移 |
| 关注强化学习政策的结构特征与经济学内涵而非单纯性能 | 第 4–5 章完成：规则切片、路径图、外部模型、`pyfrbus` 原生优化 |

### 1.2 对初稿最重要的三句话

| 编号 | 建议句式 |
|---|---|
| 1 | 本文将 RL 定位为复杂货币政策动态规划问题的数值求解器，并在统一框架下系统比较其与经典最优控制规则的关系。 |
| 2 | 在非线性、约束与经验福利口径下，RL 学得的政策规则可以产生显著优势，而且这些优势具有可解释的状态依赖结构。 |
| 3 | 本文不仅比较平均损失，还比较历史反事实、长期随机、交叉迁移、外部模型稳健性与规则结构，因此比较维度明显扩展。 |

## 2. 理论与文献综述写作提醒

### 2.1 必须记住的文献口径

| 文献 | 写作要点 | 禁忌 |
|---|---|---|
| `Svensson (1997)` | 要区分 `target rule` 与 `instrument rule`；强调 inflation forecast targeting | 不要写成“证明最优政策就是 Taylor rule” |
| `Clarida, Galí, Gertler (1999)` | canonical NK 规范分析框架 | 不要把本文 reduced-form LQ benchmark 写成完整 NK 结构复现 |
| `Walsh (2017)` | 福利目标、instrument rules、ELB 与政策框架 | 不要只剩机械 LQ 控制而忽略其微观基础背景 |
| `Hinterlang and Tänzer (2021)` | 可作为经验环境与 RL 规则学习的参照 | 不要照搬其 ANN/PD 图逻辑，更不要把它当成本文模板 |

### 2.2 理论章节必须交代的三个点

| 点 | 必须写明 |
|---|---|
| benchmark 的定位 | 本文构造的是一个 reduced-form、可计算的 LQ benchmark，是对经典最优货币政策问题的规范性实现 |
| 福利函数来源 | 经典 NK 文献常由家庭效用近似推出福利目标，而本文 benchmark 先采用显式二次损失函数 |
| RL 的角色 | RL 用于近似求解 Bellman 问题，尤其在非线性、约束和状态依赖情况下替代显式 HJB 数值求解 |

## 3. 最终矩阵：正文只写这些

### 3.1 各模块最终保留口径

| 模块 | 正文保留口径 | 结果位置 |
|---|---|---|
| 理论 benchmark | `LQ + Riccati + benchmark transfer (PPO/TD3/SAC surrogate)` | `outputs/phase7/matrix/` |
| 非 benchmark 扩展 | `phase11 v2` 六环境：`nonlinear` 两档、`zlb trap` 两档、`threshold asymmetric` 两档 | `outputs/phase11/extreme_matrix_v2/` |
| 经验环境 direct | `SVAR/ANN` 中直接训练 `PPO/TD3/SAC`，其中 `PPO` 区分 `linear/nonlinear` | `outputs/phase10/svar_direct/`、`outputs/phase10/ann_direct/`、`outputs/phase10/ppo_policy_variants/` |
| 经验环境 revealed | `SVAR/ANN revealed direct` | `outputs/phase10/revealed_policy_training/` |
| 统一评价 | 历史反事实、长期随机、交叉迁移；人工损失与反推损失两套评分 | `outputs/phase10/counterfactual_eval/`、`outputs/phase10/revealed_policy_eval/`、`outputs/phase10/revealed_welfare/` |
| 外部模型 | `pyfrbus + US_SW07 + US_CCTW10 + US_KS15 + NK_CW09` | `outputs/phase10/external_model_robustness/` |
| `pyfrbus` 原生优化 | `fine_05`、warm-start linear PPO、warm-start nonlinear PPO、residual search | `outputs/phase12/` |

### 3.2 应放附录或垃圾块的内容

| 类别 | 位置 | 备注 |
|---|---|---|
| `phase7` 之后早期扩容的中间版本 | `outputs/phase7/` 与若干中间总结 | 只保留最终 `phase11 v2` 结论 |
| `phase11 v1` 与旧数值对照 | `outputs/phase11/extreme_matrix/`、`outputs/phase11/extreme_numerical_compare/` | 已被 `v2` 覆盖 |
| 早期原生 SAC 与失败搜索 | `outputs/phase10/pyfrbus_native/` | 仅作方法迭代记录 |
| 全量 case inventory | `outputs/phase10/case_inventory/phase10_case_inventory.csv` | 不进正文，可作附录索引 |

## 4. 各章写作材料

### 4.1 导论与文献综述

| 小节 | 可直接写的内容 |
|---|---|
| 研究问题 | 现实政策问题往往包含非线性动态、非对称偏好和政策约束，此时解析最优控制与传统数值方法都受限，RL 因而成为有吸引力的数值求解器。 |
| 本文动机 | 不满足于“RL 能不能学出规则”，而是问：RL 与理论基准关系如何、在什么环境下优势更大、规则结构有什么经济学特征、外部稳健性如何。 |
| 本文贡献 | 相比 Hinterlang，本项目增加了 benchmark/empirical 严格分离、人工/反推双福利、长期随机、交叉迁移、外部模型、`pyfrbus` 原生优化与 phase11 强扭曲扩展。 |

### 4.2 理论基准

| 小节 | 可直接写的内容 |
|---|---|
| benchmark 定位 | 本文 benchmark 是对经典最优货币政策问题的可计算化实现，不是 canonical NK 模型的逐项复现。 |
| RL 方法定位 | RL 是近似求解 Bellman 问题的数值工具，因此 benchmark 章节的意义是先验证 RL 在可对照情形中的有效性。 |
| benchmark 结果 | benchmark 下 RL 能学到有效规则，说明方法是可信的；后续真正的价值在不可解析或难解析环境中展开。 |

### 4.3 人工损失 benchmark 与三类非 benchmark 扩展

| 小节 | 可直接写的内容 |
|---|---|
| 早期全矩阵 vs 最终矩阵 | 早期 `(1+3+3+3)×3` 全矩阵用于发现结构规律；正文只保留 `phase11 v2` 六个强扭曲环境。 |
| 主结果 | 六个 `v2` 环境里，原始 RL 全部压过 Riccati 外推。可写成：当线性外推依赖的局部二次近似被明显扭曲时，RL 的优势迅速放大。 |
| 机制解释 | 共同冲击轨迹说明 RL 的优势不是静态系数微调，而是在受约束区间、阈值区域和强非线性区间采取了不同的状态依赖利率路径。 |
| 保守口径 | 需要如实写：phase11 的亮点是“突破 Riccati 外推”，而不是“处处达到环境内数值最优”。 |

### 4.4 经验环境、反推福利与统一评价

| 小节 | 可直接写的内容 |
|---|---|
| 经验环境主线 | 正文主环境建议以 `SVAR` 为核心；`ANN` 放在稳健性章作为替代状态转移。 |
| 反推福利 | 反推权重显示央行 revealed 偏好对利率平滑赋予很高权重，因此人工损失与反推损失下最优规则会系统重排。 |
| 历史反事实 | `SVAR` 人工损失下 `ppo_ann_direct_nonlinear` 略优于历史政策；反推损失下 `sac_svar_revealed_direct` 明显优于 Taylor 与历史政策。 |
| 长期随机 | RL 优势不仅是平均损失下降，还包括更好的通胀—产出波动组合；但不同规则通过不同路径实现改进。 |
| 交叉迁移 | 环境不匹配时 direct-trained 规则退化明显；这应写成经验环境的适用边界，并与 `Lucas critique` 一起放到讨论。 |

### 4.5 稳健性：ANN 与外部模型

| 小节 | 可直接写的内容 |
|---|---|
| ANN 环境 | ANN 拟合优于 SVAR，是经验非线性的一个替代刻画；`ppo_ann_direct` 与 `sac/td3_ann_revealed_direct` 提供了 RL 优势的另一组支持证据。 |
| ANN-native 数值对照 | `phase14` 表明：`ANN-native affine search` 在人工损失下优于 ANN-RL，但在反推福利历史反事实下仍被 `td3_ann_revealed_direct` 压过；若按训练步数计，`TD3/SAC` 的成本仅为数值搜索的约 `1/24`，`PPO` 也低于数值搜索。这一节应写成 ANN 稳健性与计算效率边界，而非主结论。 |
| 外部模型 | 正文不需要堆满所有模型，只需保留最有力的 RL 案例：如 `US_CCTW10`、`US_KS15` 中 `td3_svar_direct` 的较强表现。 |
| `pyfrbus` | 这是外部模型部分最重要的亮点：先有本地线性搜索 `fine_05`，再有 warm-start nonlinear PPO 的边际改进，说明 RL 在复杂模型中可作为结构化初值之上的继续优化器。 |

### 4.6 讨论与总结

| 小节 | 可直接写的内容 |
|---|---|
| 主结论 | RL 在复杂货币政策环境中具有显著且可解释的数值求解优势。 |
| 方法边界 | 经验环境结果不能被外推为结构稳健最优政策定理；`Lucas critique` 必须保留。 |
| 贡献定位 | 本文最强的贡献不只是“某些 case 上 RL 更好”，而是把 benchmark、非 benchmark、经验环境、反推福利、外部模型和规则解释串成一个统一比较框架。 |

## 5. 图表与图组说明

### 5.1 正文主图（推荐顺序）

| 顺序 | 图 | 用途 | 推荐章节 |
|---|---|---|---|
| 1 | `outputs/phase13/figures/phase13_theory_heatmap.png` | 给出 benchmark 与早期扩展的总览 | 第 2–3 章过渡 |
| 2 | `outputs/phase13/figures/phase13_theory_strength_curves.png` | 显示扭曲增强时 RL 优势扩大 | 第 3 章 |
| 3 | `outputs/phase13/figures/phase13_extreme_v2_advantage.png` | 展示 `phase11 v2` 六环境中 RL 全面压过 Riccati 外推 | 第 3 章 |
| 4 | `outputs/phase13/figures/phase13_extreme_mechanism_paths.png` | 解释第 3 章的机制 | 第 3 章 |
| 5 | `outputs/phase13/figures/phase13_empirical_revealed_histories.png` | 经验环境主图：反推福利历史反事实 | 第 4 章 |
| 6 | `outputs/phase13/figures/phase13_stochastic_tradeoffs.png` | 长期随机评价 | 第 4 章 |
| 7 | `outputs/phase13/figures/phase13_policy_slices_core.png` | 规则级经济学解释 | 第 4 章 |
| 8 | `outputs/phase9/plots/phase9_ann_fit_comparison.png` | ANN 稳健性入口图 | 第 5 章 |
| 9 | `outputs/phase13/figures/phase13_external_models_best_rules.png` | 外部模型稳健性总览 | 第 5 章 |
| 10 | `outputs/phase13/figures/phase13_pyfrbus_progression.png` | `pyfrbus` 原生优化亮点图 | 第 5 章 |
| 11 | `outputs/phase14/figures/ann_numerical_search_vs_rl_matrix.png` | ANN-native 数值搜索与 RL 的损失/波动对照 | 第 5 章 |
| 12 | `outputs/phase14/figures/ann_numerical_search_compute_cost.png` | ANN-native 数值搜索与 RL 的训练成本对照 | 第 5 章 |

### 5.2 图组短分析（可直接放文稿）

| 图组 | 可直接用的解释 |
|---|---|
| `phase13_theory_heatmap` + `phase13_theory_strength_curves` | 在 benchmark 及第一轮扩展中，RL 已表现出稳定竞争力；随着非线性、有效下界和目标扭曲增强，RL 相对线性参考规则的优势明显扩大。 |
| `phase13_extreme_v2_advantage` + `phase13_extreme_mechanism_paths` | 六个最终强扭曲环境里，RL 全部压过 Riccati 外推；共同冲击路径说明，RL 的优势来自对特定状态区域的不同政策反应。 |
| `phase13_empirical_revealed_histories` + `phase13_stochastic_tradeoffs` | 在经验环境中，RL 的优势不仅体现在历史反事实总损失上，还体现在长期随机下更优的波动权衡。 |
| `phase13_policy_slices_core` | 最优 RL 规则不是没有结构的黑箱，而是对 Taylor 型规则的状态依赖修正。 |
| `ann_numerical_search_vs_rl_matrix` + `ann_numerical_search_compute_cost` | ANN 经验环境内的传统数值搜索并不总被 RL 压过，因此 ANN 章应承担“稳健性与计算成本边界”功能：一方面说明 RL 并非处处支配环境内数值搜索，另一方面强调 `TD3/SAC` 以远低于数值搜索的训练步数就能逼近甚至在反推福利历史评价下超过其表现。 |
| `phase13_external_models_best_rules` + `phase13_pyfrbus_progression` | 外部模型稳健性不完全统一，但 `pyfrbus` 表明 native warm-start RL 可以在强线性基准上继续挖出边际增益。 |

## 6. 规则级经济学解释（可直接转化为正文段落）

### 6.1 历史反事实与波动

| case | 简洁解读 |
|---|---|
| `SVAR artificial` / `ppo_ann_direct_nonlinear` | 相比历史政策，它主要通过压低产出缺口波动和利率波动取得改进，更像一种“温和但持续”的稳定化反馈。 |
| `ANN artificial` / `ppo_ann_direct` | 它同时压低通胀与产出波动，但伴随更高的利率变动频率，适合解释为“更积极使用政策工具换稳定”。 |
| `SVAR revealed` / `sac_svar_revealed_direct` | 它在反推福利下显著压低利率与产出波动，但通胀波动高于 Taylor；这与高利率平滑权重一致。 |
| `ANN revealed` / `sac_ann_revealed_direct` | 它在长期随机下对 Taylor 同时实现更低的通胀、产出和利率变动标准差，是“revealed 平滑型规则”的强证据。 |

### 6.2 规则切片

| case | 可写解释 |
|---|---|
| `PPO ANN direct nonlinear` 的通胀切片 | 若其斜率较 Taylor 更平缓，可解释为避免对短期价格偏离过度反应，更依赖跨期平滑。 |
| `PPO ANN direct` 的通胀切片 | 若出现反向或更弱的线性结构，要结合环境拟合与多变量交互解释，不能机械套 Taylor 系数含义。 |
| `SAC SVAR revealed` / `SAC ANN revealed` 的产出切片 | 若整体更平，可解释为高利率平滑权重把政策最优点推向低频率、小幅度调节。 |

## 7. 哪些文件写作时最该打开

| 用途 | 文件 |
|---|---|
| 细纲 | `OUTLINE.md` |
| 理论写作边界 | `docs/theory_writing_notes.md` |
| 最终结果总表 | `REPORT.md` |
| 图表目录 | `outputs/phase13/phase13_figure_catalog.md` |
| 图表配套分析 | `outputs/phase13/phase13_writing_materials.md` |
| 最终扩展结论 | `outputs/phase11/extreme_v2_overview.md` |
| `pyfrbus` 最终结论 | `outputs/phase12/pyfrbus_warmstart_ppo/summary.md`、`outputs/phase12/pyfrbus_nonlinear_search/summary.md` |

## 8. 你写初稿时可直接按的顺序

| 步骤 | 建议做法 |
|---|---|
| 1 | 先按 `OUTLINE.md` 写每节 1–2 段骨架，不急着填满细节。 |
| 2 | 理论与文献综述优先参考本文件第 1–2 节，避免理论定位失误。 |
| 3 | 结果章节直接用本文件第 3–6 节与 `outputs/phase13/phase13_writing_materials.md`。 |
| 4 | 作图时优先从 `phase13` 图中挑正文图，旧图只作补充。 |
| 5 | 写完整稿后，再让我帮你逐章润色、挑问题、压缩和加强经济学表达。 |

## 9. 最后一条提醒

| 主题 | 提醒 |
|---|---|
| 主线 | 主线要写 RL 的显著优势，但表达应建立在“最终保留的最强证据”上，而不是所有 case 一视同仁。 |
| 边界 | `Lucas critique`、外部模型异质性、`phase11` 数值最优对照，都应保留，但应放在稳健性与讨论部分。 |
| 相对 Hinterlang | 不要把“我们也做了 SVAR/ANN + RL”当主要卖点；真正更强的是比较框架更全、福利口径更完整、规则解释更深、`pyfrbus` 原生优化更进一步。 |
