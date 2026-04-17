# 论文细纲（最终矩阵版）

> 口径：本提纲只整理当前应进入正文或附录的**最终矩阵**与最终结论。`phase7` 之后被覆盖的中间版本、早期失败搜索、被 `phase11 v2` 取代的旧环境，统一放到附录或“垃圾块”。

## 0. 全文总结构

| 章 | 建议标题 | 主任务 | 主图 |
|---|---|---|---|
| 1 | 导论与文献综述 | 提出问题、定位文献、概括贡献 | 无必需主图 |
| 2 | 理论基准与方法框架 | 建立 reduced-form LQ benchmark，说明 Riccati 与 RL 的比较关系 | `outputs/phase13/figures/phase13_theory_heatmap.png` |
| 3 | 人工损失下的 benchmark 与三类非 benchmark 扩展 | 展示 RL 在 benchmark 与强扭曲环境中的优势 | `outputs/phase13/figures/phase13_theory_strength_curves.png`、`outputs/phase13/figures/phase13_extreme_v2_advantage.png` |
| 4 | 经验环境、反推福利与统一评价 | 展示经验环境中的历史反事实、长期随机、交叉迁移与福利重排 | `outputs/phase13/figures/phase13_empirical_revealed_histories.png`、`outputs/phase13/figures/phase13_stochastic_tradeoffs.png` |
| 5 | 稳健性：ANN 与外部模型 | 展示 ANN 替代状态转移与外部模型迁移/原生优化证据 | `outputs/phase9/plots/phase9_ann_fit_comparison.png`、`outputs/phase13/figures/phase13_external_models_best_rules.png` |
| 6 | 讨论与总结 | 收束经济含义、边界与贡献 | 无必需主图 |
| A | 附录与垃圾块 | 收纳全量 case、被替代版本、失败搜索 | 不进正文 |

## 1. 导论与文献综述

### 1.1 研究问题与现实动机

| 内容 | 建议标题 | 本节要点 |
|---|---|---|
| 研究问题 | `1.1 为什么要把 RL 作为货币政策规则求解器` | 对齐开题报告：当动态线性二次条件成立时，Riccati/HJB 可给出线性最优反馈；当引入非线性、约束、非对称目标时，解析解或稳定数值解变难，RL 可视为近似 Bellman 问题的数值求解器。 |
| 现实意义 | `1.2 为什么只比较性能不够` | 强调本文不只比较总损失，还比较规则结构、波动、外部稳健性与反事实政策路径。 |

### 1.2 经典最优货币政策文献

| 内容 | 建议标题 | 本节要点 |
|---|---|---|
| 经典规范分析 | `1.3 经典最优货币政策：从 target rule 到 instrument rule` | 明确区分 `target rule` 与 `instrument rule`；引用 `Svensson (1997)`、`Clarida, Galí and Gertler (1999)`、`Walsh (2017)`。 |
| 本文 benchmark 定位 | `1.4 本文 benchmark 的理论定位` | 不能把本文 reduced-form LQ benchmark 写成 canonical NK 完整复现；应写成“对经典最优货币政策问题的可计算化表达”。 |

### 1.3 非线性、约束与 RL 文献

| 内容 | 建议标题 | 本节要点 |
|---|---|---|
| 非线性最优货币政策 | `1.5 非线性、ZLB 与非对称目标下的数值难题` | 连接 `Woodford`、`Adam and Billi` 等文献，说明传统数值方法的维度和稳定性问题。 |
| RL 与经济政策 | `1.6 RL 在动态政策问题中的应用与局限` | 引 `Sutton and Barto`、`Han et al.`、`Buehler et al.`、`Charpentier et al.`、`Hinterlang and Tänzer (2021)`；强调本文不是机械复刻 Hinterlang。 |

### 1.4 本文贡献与结构

| 内容 | 建议标题 | 主要观察/结论 |
|---|---|---|
| 本文贡献 | `1.7 本文的四点贡献` | 1）benchmark/empirical 严格区分；2）人工损失与反推福利双口径；3）历史反事实、长期随机、交叉迁移、外部模型一体化；4）非 benchmark 扩展与 `pyfrbus` 原生优化。 |
| 结构说明 | `1.8 论文结构安排` | 简要预告第 2–6 章。 |

## 2. 理论基准与方法框架

### 2.1 基准模型与福利目标

| 内容 | 建议标题 | 主要观察/结论 | 图表 |
|---|---|---|---|
| benchmark 设定 | `2.1 Reduced-form LQ benchmark` | 状态变量围绕通胀缺口、产出缺口与利率平滑展开；规范目标先采用显式二次损失。 | 无必需图 |
| 解析参考 | `2.2 Riccati reference 与线性反馈结构` | Riccati 解是唯一规范 benchmark；后续 RL 与其他规则都以它为理论参照。 | 可选：`outputs/phase7/matrix/plots/benchmark_all_policies.png` |

### 2.2 RL 的方法定位

| 内容 | 建议标题 | 主要观察/结论 | 图表 |
|---|---|---|---|
| 方法定位 | `2.3 RL 作为 Bellman 问题的近似数值求解器` | 这是全文方法主轴；不要写成“RL 替代理论”。 | 无必需图 |
| 规则族比较 | `2.4 政策类：Riccati、benchmark transfer、经验 Taylor 与 RL` | benchmark transfer 是理论环境中的 RL 近似规则，不等于经验环境最优规则。 | 无必需图 |

### 2.3 benchmark 数值结果

| 内容 | 建议标题 | 主要观察/结论 | 图表 |
|---|---|---|---|
| benchmark 主结果 | `2.5 benchmark 下的数值对照` | RL 在 benchmark 中学到有效规则，至少能接近强基线；这为后续扩展环境提供出发点。 | `outputs/phase13/figures/phase13_theory_heatmap.png` |
| 稳定性 | `2.6 多 seed 与强基线比较` | 不宜只报单次最好结果；benchmark 章节可补强多 seed 稳定性与 `linear policy search` 的强度。 | 可选：`outputs/phase7/matrix/plots/benchmark_seed_boxplots.png` |

## 3. 人工损失下的 benchmark 与三类非 benchmark 扩展

### 3.1 实验矩阵与正文保留口径

| 内容 | 建议标题 | 主要观察/结论 | 图表 |
|---|---|---|---|
| 早期扩展矩阵 | `3.1 从 phase7 全矩阵到最终正文矩阵` | 早期全矩阵是 `(1+3+3+3)×3`；正文不需要把所有中间轮都堆进去。 | `outputs/phase13/figures/phase13_theory_heatmap.png` |
| 正文最终口径 | `3.2 正文只保留 phase11 v2 六环境` | 最终保留六个强扭曲环境：`nonlinear` 两档、`zlb trap` 两档、`asymmetric threshold` 两档。 | `outputs/phase13/figures/phase13_extreme_v2_advantage.png` |

### 3.2 benchmark 与轻中度扭曲下的主结果

| 内容 | 建议标题 | 主要观察/结论 | 图表 |
|---|---|---|---|
| 排序变化 | `3.3 从 benchmark 到扭曲环境：RL 排序如何变化` | 需要讲清“扭曲变强时 RL 优势扩大”，而不是机械列算法名次。 | `outputs/phase13/figures/phase13_theory_strength_curves.png` |
| 解释口径 | `3.4 为什么线性参考规则在轻扭曲下仍然很强` | 这不是 RL 失败，而是当前扭曲在大部分状态区间内仍未完全改写最优反馈结构。 | 可选：`outputs/phase7/matrix/plots/best_rl_coefficients_heatmap.png` |

### 3.3 三类非 benchmark 扩展的最终结果

| 内容 | 建议标题 | 主要观察/结论 | 图表 |
|---|---|---|---|
| nonlinear | `3.5 强非线性 Phillips 曲线下的 RL 优势` | `nonlinear_extreme_v2` 与 `nonlinear_hyper` 中 RL 均压过 Riccati 外推；极端档优势更大。 | `outputs/phase13/figures/phase13_extreme_v2_advantage.png` |
| ZLB/ELB | `3.6 有效下界陷阱环境下的 RL 优势` | `zlb_trap_very_strong` 与 `zlb_trap_extreme` 中 RL 仍优于 Riccati 外推，说明约束区间内状态依赖行为变得关键。 | `outputs/phase13/figures/phase13_extreme_v2_advantage.png` |
| 非对称目标 | `3.7 阈值型非对称目标下的 RL 优势` | `asymmetric_threshold_*` 中 RL 优势也成立，但其机制更多来自损失函数扭曲而非单纯状态转移扭曲。 | `outputs/phase13/figures/phase13_extreme_v2_advantage.png` |

### 3.4 机制解释

| 内容 | 建议标题 | 主要观察/结论 | 图表 |
|---|---|---|---|
| 共同冲击路径 | `3.8 为什么 RL 在强扭曲区间更优` | 共同冲击路径应服务于一个机制：RL 在强非线性、约束或阈值附近给出不同于 Riccati 的状态依赖政策路径。 | `outputs/phase13/figures/phase13_extreme_mechanism_paths.png` |
| 数值解对照 | `3.9 RL、Riccati 外推与有限期数值解` | 应如实写：phase11 的核心亮点是“RL 超过 Riccati 外推”；但对环境内数值最优，RL 并非处处支配。 | 可选：`outputs/phase11/extreme_numerical_compare_v2/summary.csv`（表） |

## 4. 经验环境、反推福利与统一评价

### 4.1 经验环境与福利识别

| 内容 | 建议标题 | 主要观察/结论 | 图表 |
|---|---|---|---|
| 经验环境核心 | `4.1 经验环境：SVAR 作为正文主环境` | 正文主线建议把 `SVAR` 放在经验环境核心，`ANN` 放到稳健性章。 | 无必需图 |
| 反推福利 | `4.2 反推福利权重的识别与经济含义` | 反推权重大致为 `inflation=1`、`output_gap≈0.882`、`rate_smoothing≈20.09`；这意味着央行 revealed 偏好更重视利率平滑。 | 可选：`outputs/phase10/revealed_welfare/revealed_weight_grid.csv`（表） |

### 4.2 历史反事实

| 内容 | 建议标题 | 主要观察/结论 | 图表 |
|---|---|---|---|
| 人工损失 | `4.3 人工损失下的历史反事实` | `SVAR` 中 `ppo_ann_direct_nonlinear` 略优于历史实际政策；说明经验环境内 direct-trained RL 已可超越 benchmark transfer。 | `outputs/phase13/figures/phase13_empirical_artificial_histories.png` |
| 反推损失 | `4.4 反推福利下的历史反事实` | `SVAR revealed` 中 `sac_svar_revealed_direct` 最优，明显优于经验 Taylor 与历史政策。 | `outputs/phase13/figures/phase13_empirical_revealed_histories.png` |

### 4.3 长期随机与波动评价

| 内容 | 建议标题 | 主要观察/结论 | 图表 |
|---|---|---|---|
| 长期随机 | `4.5 长期随机评价：损失与波动的联合比较` | RL 优势不只来自总损失下降，还体现在通胀—产出波动组合改善；但不同福利口径下“为什么更优”的机制不同。 | `outputs/phase13/figures/phase13_stochastic_tradeoffs.png` |
| 波动分解 | `4.6 波动、平滑与利率调整频率` | 可直接用 `phase13` 补算结果：有些 RL 规则通过更积极的利率调整换取更低目标变量波动；revealed 规则则更倾向压低利率变动波动。 | `outputs/phase13/phase13_writing_materials.md` 中第 3 节表格 |

### 4.4 规则级解释与交叉迁移

| 内容 | 建议标题 | 主要观察/结论 | 图表 |
|---|---|---|---|
| 规则切片 | `4.7 最优 RL 规则的局部反应函数` | 要直接解释 RL 规则为何更优：更平缓的通胀响应、更强/更弱的产出缺口响应、revealed 规则的平滑偏好。 | `outputs/phase13/figures/phase13_policy_slices_core.png` |
| 交叉迁移 | `4.8 交叉迁移：经验规则的环境依赖性` | `SVAR → ANN` 与 `ANN → SVAR` 的退化要写，但只作为边界与稳健性讨论，不作为主结论。 | `outputs/phase13/figures/phase13_cross_transfer.png` |

## 5. 稳健性：ANN 与外部模型

### 5.1 ANN 经验环境

| 内容 | 建议标题 | 主要观察/结论 | 图表 |
|---|---|---|---|
| ANN 状态转移 | `5.1 ANN 环境作为经验非线性稳健性检验` | ANN 在拟合上优于 SVAR，是经验非线性的一个替代刻画。 | `outputs/phase9/plots/phase9_ann_fit_comparison.png` |
| ANN 中的 RL 结果 | `5.2 ANN 环境中的 direct 与 revealed 规则` | `ppo_ann_direct` 在人工损失长期随机评价下表现很强；`sac_ann_revealed_direct`/`td3_ann_revealed_direct` 在反推福利下表现很强。 | `outputs/phase13/figures/phase13_empirical_artificial_histories.png`、`outputs/phase13/figures/phase13_empirical_revealed_histories.png` |
| ANN-native 数值对照 | `5.3 ANN 环境内数值搜索与 RL 的比较` | `ANN-native affine search` 在人工损失下优于 ANN-RL，但在反推福利历史反事实下不如 `td3_ann_revealed_direct`；若按训练步数计，`TD3/SAC` 显著更省。正文应把这一节写成“ANN 稳健性与计算效率边界”，而非全文主结论。 | `outputs/phase14/figures/ann_numerical_search_vs_rl_matrix.png`、`outputs/phase14/figures/ann_numerical_search_compute_cost.png` |

### 5.2 外部模型稳健性

| 内容 | 建议标题 | 主要观察/结论 | 图表 |
|---|---|---|---|
| 外部模型总表 | `5.4 外部模型：哪些 RL 规则具备跨模型竞争力` | 正文建议只保留可写的正面案例：`US_CCTW10`、`US_KS15` 中 `td3_svar_direct` 较强；少量负面案例用于提示边界。 | `outputs/phase13/figures/phase13_external_models_best_rules.png` |
| 外部模型边界 | `5.5 为什么外部模型稳健性不应被过度表述` | `US_SW07` 与部分 `NK` 结果说明外部稳健性高度异质；但这应写成稳健性边界，而不是推翻主线。 | 同上 |

### 5.3 `pyfrbus` 原生优化

| 内容 | 建议标题 | 主要观察/结论 | 图表 |
|---|---|---|---|
| 线性本地搜索 | `5.5 从线性本地搜索到 native warm-start` | `fine_05` 已优于 `pyfrbus_baseline`；说明参数量级校准很关键。 | `outputs/phase13/figures/phase13_pyfrbus_progression.png` |
| nonlinear PPO | `5.6 warm-start nonlinear PPO 的边际增益` | `residual_u2_lr5e6` 在反推损失上略优于 `fine_05`；这是 RL 在复杂外部模型中的可保留亮点。 | `outputs/phase13/figures/phase13_pyfrbus_progression.png` |

## 6. 讨论与总结

### 6.1 主要经济学含义

| 内容 | 建议标题 | 主要观察/结论 |
|---|---|---|
| 主结果 | `6.1 RL 在何处具有显著优势` | 主线写法：RL 在 benchmark 扩展、经验环境内训练、部分外部模型与 `pyfrbus` native warm-start 下具有清晰优势。 |
| 规则结构 | `6.2 RL 规则的经济学特征` | 结合切片图与波动表：RL 规则不是纯黑箱替代，而是可解释的状态依赖修正。 |

### 6.2 方法边界

| 内容 | 建议标题 | 主要观察/结论 |
|---|---|---|
| Lucas critique | `6.3 为什么经验反事实必须保留 Lucas critique` | 经验环境下私人部门行为与状态转移被固定，因此不能把经验环境优越性外推为结构稳健最优政策定理。 |
| 非 benchmark 数值解 | `6.4 为什么 RL 优势不等于环境内全局最优` | 在 phase11 中，RL 重点是突破 Riccati 外推，而不是在所有环境中压过有限期 DP。 |

### 6.3 结论与未来工作

| 内容 | 建议标题 | 主要观察/结论 |
|---|---|---|
| 结论 | `6.5 结论` | 收束为：RL 是复杂货币政策动态规划问题中的有力数值求解器，其优势在非线性、约束和经验福利口径下更明显。 |
| 未来工作 | `6.6 后续方向` | 可提 Lucas critique 下的结构再估计、更多外部模型、更多内生状态变量。 |

## A. 附录与垃圾块

### A.1 建议保留到附录的内容

| 类别 | 建议保留内容 | 位置 |
|---|---|---|
| 全量 case | phase10 全量 case inventory | `outputs/phase10/case_inventory/phase10_case_inventory.csv` |
| 早期矩阵 | `phase7` 全矩阵完整热图、boxplot、clip rate 图 | `outputs/phase7/matrix/plots/` |
| 旧扩展版本 | `phase11 v1`、`numerical_control_compare`、被后续加大扭曲替代的环境 | `outputs/phase11/extreme_matrix/`、`outputs/phase11/numerical_control_compare/` |
| 失败搜索 | 早期 `pyfrbus` native SAC、被 `warm-start PPO` 覆盖的旧搜索 | `outputs/phase10/pyfrbus_native/` |

### A.2 不建议进入正文的内容

| 类别 | 原因 |
|---|---|
| `phase7` 之后早期中间轮的重复结果 | 已被 `phase11 v2` 和 `phase13` 整理图覆盖 |
| 没有形成稳健结论的单次搜索 | 会冲淡正文主线 |
| 过多外部模型负结果细节 | 正文应保留边界，但不宜让负例主导叙事 |

## B. 写作时优先引用的文件

| 用途 | 文件 |
|---|---|
| 总体结果总表 | `REPORT.md` |
| 图表归档 | `outputs/phase13/phase13_figure_catalog.md` |
| 规则级经济学解释 | `outputs/phase13/phase13_writing_materials.md` |
| phase10 总结 | `outputs/phase10/phase10_summary.md` |
| phase11 最终扩展 | `outputs/phase11/extreme_v2_overview.md` |
| phase12 `pyfrbus` | `outputs/phase12/pyfrbus_warmstart_ppo/summary.md`、`outputs/phase12/pyfrbus_nonlinear_search/summary.md` |
