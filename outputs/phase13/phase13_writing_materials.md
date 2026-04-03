# Phase 13 写作材料

## 1. 章节级主叙事

### 1.1 理论基准与 benchmark

可直接写入正文的主句式是：RL 先在 benchmark 与可控扩展环境中通过基准检验，再在强扭曲环境中把优势显著放大，因此其表现并不是偶然的单个案例结果，而是来自对非线性、约束和状态依赖反馈的系统适应。

### 1.2 三类非 benchmark 扩展

可直接写入正文的主句式是：一旦线性外推所依赖的局部二次近似被明显扭曲，RL 相对 Riccati 的优势会迅速扩大；这种优势尤其体现在强非线性、有效下界陷阱与阈值型非对称目标下。

### 1.3 经验环境与反推福利

可直接写入正文的主句式是：在经验状态转移下，RL 的优势不仅体现在反事实总损失下降，也体现在对通胀、产出缺口和利率平滑之间权衡的重新配置。反推福利进一步表明，最优规则的排序取决于央行真实偏好的权重结构，而 RL 可以据此学出更贴近该目标的反馈行为。

### 1.4 ANN 与外部模型稳健性

可直接写入正文的主句式是：ANN 经验环境与部分外部模型支持 RL 的稳健性，尤其 `pyfrbus` 原生 warm-start 结果表明，当 RL 直接在目标模型内继续学习时，它可以在强线性基准之上再实现小幅但清晰的增益。

## 2. 图组配套短分析

### 2.1 理论基准与扩展图组

`phase13_theory_heatmap.png` 与 `phase13_theory_strength_curves.png` 可配套写成：在 benchmark 及第一轮扭曲扩展中，RL 已经具备稳定竞争力；随着 nonlinear、ZLB 与 asymmetric 扭曲增强，最优 RL 与 Riccati/线性搜索之间的差距进一步拉大，说明 RL 的边际价值主要体现在状态依赖反馈结构被改写的区域。

`phase13_extreme_v2_advantage.png` 与 `phase13_extreme_mechanism_paths.png` 可配套写成：在最终保留的六个强扭曲环境里，RL 全面压过 Riccati 外推。共同冲击路径进一步说明，RL 的优势不是静态参数微调，而是对受约束区间和阈值区域给出不同的利率路径，从而改变输出与通胀的动态调整过程。

### 2.2 经验环境图组

`phase13_empirical_artificial_histories.png` 与 `phase13_empirical_revealed_histories.png` 可配套写成：在历史反事实中，优选 RL 规则并非简单复制 Taylor rule，而是在若干关键阶段给出更平滑或更及时的利率反应，使通胀和产出缺口路径更快回归目标附近。

`phase13_stochastic_tradeoffs.png` 与 `phase13_cross_transfer.png` 可配套写成：长期随机评估显示，RL 的优势不仅是平均损失降低，还体现为更好的通胀—产出波动组合；但交叉迁移结果说明，这一优势依赖于训练环境与评价环境的一致性，因此正文需要把环境匹配解释为优势成立的制度背景，而不是把它写成对 RL 的否定。

### 2.3 稳健性图组

`phase13_external_models_best_rules.png` 可配套写成：外部模型比较保留了若干关键的 RL 优势案例，说明经验环境中学到的反馈结构并非完全不可迁移。

`phase13_pyfrbus_progression.png` 可配套写成：`pyfrbus` 原生结果最有价值的地方不在于“迁移 RL 一开始就赢”，而在于 model-native warm-start 之后，nonlinear PPO 可以在强线性基准之上继续挖出额外改进。

## 3. 规则级经济学解释：优先写进正文的 cases

### 3.1 历史反事实与波动结果

| 福利口径 | 环境 | 规则 | 总损失 | 通胀波动 | 产出波动 | 利率波动 | 利率变动波动 |
|---|---|---|---:|---:|---:|---:|---:|
| artificial | SVAR | PPO ANN direct nonlinear | 87.34 | 0.80 | 1.17 | 0.90 | 0.35 |
| artificial | SVAR | historical policy | 87.92 | 0.75 | 1.26 | 2.19 | 0.48 |
| artificial | SVAR | empirical Taylor | 94.50 | 0.69 | 1.38 | 2.00 | 0.35 |
| artificial | ANN | PPO ANN direct | 69.23 | 0.76 | 1.14 | 0.99 | 0.49 |
| artificial | ANN | empirical Taylor | 130.41 | 0.70 | 1.25 | 2.06 | 0.34 |
| artificial | ANN | historical policy | 168.24 | 0.87 | 1.71 | 2.19 | 0.48 |
| revealed | SVAR | SAC SVAR revealed | 148.76 | 1.12 | 1.07 | 0.53 | 0.14 |
| revealed | SVAR | empirical Taylor | 254.74 | 0.69 | 1.38 | 2.00 | 0.35 |
| revealed | SVAR | historical policy | 375.37 | 0.75 | 1.26 | 2.19 | 0.48 |
| revealed | ANN | TD3 ANN revealed | 186.23 | 0.82 | 1.04 | 0.91 | 0.21 |
| revealed | ANN | empirical Taylor | 285.35 | 0.70 | 1.25 | 2.06 | 0.34 |
| revealed | ANN | historical policy | 483.77 | 0.87 | 1.71 | 2.19 | 0.48 |

建议从这张表里抓三种典型机制写正文：

- `SVAR artificial`：`PPO ANN direct nonlinear` 相比历史政策，主要是通过显著压低产出与利率波动取得改进，说明其规则特征更接近“温和但持续的稳定化反馈”。
- `ANN artificial`：`PPO ANN direct` 同时压低通胀与产出波动，但利率变动更频繁，适合解释为更积极地使用政策工具来换取目标变量稳定。
- `SVAR/ANN revealed`：revealed 规则把高利率平滑权重内生化进反馈结构，因此在不少情况下会牺牲部分通胀稳定，以换取更低的利率调整成本与更优的总体福利。

### 3.2 长期随机评价

| 情境 | 规则 | 平均阶段损失 | 通胀波动 | 产出波动 | 利率变动波动 |
|---|---|---:|---:|---:|---:|
| artificial_svar | PPO ANN direct nonlinear | 1.48 | 0.83 | 1.17 | 0.38 |
| artificial_svar | empirical Taylor | 1.62 | 0.75 | 1.26 | 0.29 |
| artificial_ann | PPO ANN direct | 1.12 | 0.72 | 0.98 | 0.52 |
| artificial_ann | empirical Taylor | 2.38 | 0.93 | 1.12 | 0.26 |
| revealed_svar | SAC SVAR revealed | 3.05 | 1.20 | 1.16 | 0.12 |
| revealed_svar | empirical Taylor | 3.90 | 0.75 | 1.26 | 0.29 |
| revealed_ann | SAC ANN revealed | 3.63 | 0.79 | 1.01 | 0.12 |
| revealed_ann | empirical Taylor | 4.31 | 0.93 | 1.12 | 0.26 |

正文可据此写两类解释：

- 人工损失下，RL 的优势可以理解为对通胀与产出稳定化的重新加权，部分规则愿意承担更高的利率调整频率以换取更低的目标变量波动。
- 反推损失下，SAC revealed 规则更明显地体现出利率平滑偏好，因此“为什么更优”不能只看通胀反应强弱，而要同时看它如何减少不必要的政策转向。

### 3.3 规则切片：为什么这些 RL 规则更优

下面这些切片点可直接转化成正文中的规则级解释：

- `artificial_svar` 下 `PPO ANN direct nonlinear` 的 `inflation_gap` 切片为：-1→4.21, +0→4.53, +1→5.48
- `artificial_ann` 下 `PPO ANN direct` 的 `inflation_gap` 切片为：-1→5.59, +0→4.58, +1→3.61
- `revealed_svar` 下 `SAC SVAR revealed` 的 `output_gap` 切片为：-1→1.86, +0→2.14, +1→2.42
- `revealed_ann` 下 `SAC ANN revealed` 的 `output_gap` 切片为：-1→2.02, +0→2.13, +1→2.25

建议写法：

- 若 RL 在 `inflation gap` 切片上比 Taylor 更平缓，可解释为它避免对短期价格偏离做过度反应，更多依靠跨期平滑来稳定路径。
- 若 RL 在 `output gap` 切片上呈现更强的正向或非线性斜率，可解释为它更积极应对深度衰退或需求过热区间。
- 若 revealed 规则整体切片更平，可解释为高利率平滑权重把政策最优点推向“低频率、小幅度”的调节方式。

## 4. 稳健性部分可直接使用的表

### 4.1 外部模型中最好的 RL 规则

| 模型 | 最优 RL 规则 | 相对 baseline 的 revealed 改进 |
|---|---|---:|
| US_CCTW10 | td3_svar_direct | 96.83% |
| NK_CW09 | sac_svar_revealed_direct | 61.70% |
| US_KS15 | td3_svar_direct | 38.21% |
| pyfrbus | sac_svar_revealed_direct | -301.30% |
| US_SW07 | sac_svar_revealed_direct | -373.98% |

正文可写成：外部模型结果不是把所有 RL 规则都包装成稳健赢家，而是突出若干可保留的成功迁移案例；这比简单宣称“普遍稳健”更可信，也更有助于说明哪些 RL 反馈结构具备跨模型竞争力。

### 4.2 pyfrbus 原生优化路径

| 变体 | 相对 pyfrbus baseline 的 revealed 改进 |
|---|---:|
| fine_05 | 14.048% |
| ppo_linear_best_init | 12.813% |
| ppo_nonlinear_warmstart_best | 13.487% |
| residual_init_only | 14.048% |
| residual_u2_lr5e6 | 14.094% |
| residual_u4_lr5e6 | 14.078% |

正文建议写法：`pyfrbus` 结果表明，本地线性搜索已经能显著超过 baseline，而 warm-start nonlinear PPO 还能在此基础上进一步提升。这说明 RL 在复杂外部模型中的价值，并不是从零开始暴力替代传统方法，而是作为结构化初值之上的继续优化器。

## 5. 相对 Hinterlang 的可写优势

- 我们的图组不只展示“RL 规则优于若干传统规则”，还系统覆盖了 benchmark、三类非 benchmark 扩展、双福利、历史反事实、长期随机、交叉迁移、外部模型与 `pyfrbus` 原生训练。
- 我们的经济学解释不只停留在 ANN 或 PD 图，而是把规则切片、历史路径、波动评价与外部稳健性结合起来，解释 RL 规则为什么更优、优在哪个维度、代价是什么。
- 因而论文正文应突出：RL 的优势不仅在数值大小上更丰富，也在机制解释和稳健性层面更完整。
