# Phase 13 总结

本阶段目标不是新增训练，而是把已有 benchmark、非 benchmark、经验环境与外部模型结果整理成可直接进入论文写作的图表与解释材料。

## 交付内容

- 图表统一放在 `outputs/phase13/figures/`
- 计算中间表统一放在 `outputs/phase13/tables/`
- 图表目录：`outputs/phase13/phase13_figure_catalog.md`
- 经济学解释材料：`outputs/phase13/phase13_writing_materials.md`

## 本阶段生成的核心图

- `phase13_theory_heatmap.png`：用一张图展示 benchmark 与三类早期扩展里 RL 排序如何变化，建立“RL 在核心环境中具有系统竞争力”的总览。
- `phase13_theory_strength_curves.png`：把 benchmark 外推、线性搜索和最优 RL 放到同一扭曲强度轴上，直接展示 RL 优势如何随非线性/约束增强而扩大。
- `phase13_extreme_v2_advantage.png`：六个最终保留的强扭曲环境里，原始 RL 全部压过 Riccati 外推，说明 RL 的优势在真正非 benchmark 场景中是系统性的而不是个例。
- `phase13_extreme_mechanism_paths.png`：共同冲击轨迹图帮助解释为何 RL 更优：它在强扭曲区域给出不同于 Riccati 外推的状态依赖利率路径，从而改写产出缺口动态。
- `phase13_empirical_artificial_histories.png`：用历史路径把‘更低损失’转化成更直观的经济动态：比较 RL、Taylor 与历史政策在通胀、产出缺口和利率路径上的差异。
- `phase13_empirical_revealed_histories.png`：用历史路径把‘更低损失’转化成更直观的经济动态：比较 RL、Taylor 与历史政策在通胀、产出缺口和利率路径上的差异。
- `phase13_stochastic_tradeoffs.png`：长期随机图不只比较总损失，还直接看通胀与产出波动的联合表现，突出 RL 的动态稳定化特征。
- `phase13_policy_slices_core.png`：把最佳 RL 规则直接画成状态切片，展示它们如何改变对通胀缺口和产出缺口的局部反应，这一图是规则级经济学解释的核心入口。
- `phase13_cross_transfer.png`：把 direct-trained 规则放到对方环境中，展示 RL 优势主要来自正确匹配状态转移与福利权重，而不是任意环境下都同样有效。
- `phase13_external_models_best_rules.png`：外部模型部分不堆满所有规则，而是直接展示每个外部模型上最有竞争力的 RL 规则，突出可保留的稳健性证据。
- `phase13_pyfrbus_progression.png`：pyfrbus 图组强调‘native warm-start 之后 nonlinear PPO 可以再往前推一步’，这是论文中最适合讲原生 RL 增益的外部模型证据。
