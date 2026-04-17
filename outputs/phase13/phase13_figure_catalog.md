# Phase 13 图表目录

| 图名 | 论文位置 | 核心信息 | 数据来源 |
|:--|---|---|---|
| `phase13_theory_heatmap.png` | 理论基准与人工损失 benchmark | 用一张图展示 benchmark 与三类早期扩展里 RL 排序如何变化，建立“RL 在核心环境中具有系统竞争力”的总览。 | `outputs/phase7/matrix/rl_summary.csv` |
| `phase13_theory_strength_curves.png` | 理论基准与人工损失 benchmark | 把 benchmark 外推、线性搜索和最优 RL 放到同一扭曲强度轴上，直接展示 RL 优势如何随非线性/约束增强而扩大。 | `outputs/phase7/matrix/all_policy_summary.csv + outputs/phase7/matrix/rl_summary.csv` |
| `phase13_extreme_v2_advantage.png` | 三类非线性扩展 | 六个最终保留的强扭曲环境里，原始 RL 全部压过 Riccati 外推，说明 RL 的优势在真正非 benchmark 场景中是系统性的而不是个例。 | `outputs/phase11/extreme_matrix_v2/riccati_vs_best_rl.csv` |
| `phase13_extreme_mechanism_paths.png` | 三类非线性扩展 | 共同冲击轨迹图帮助解释为何 RL 更优：它在强扭曲区域给出不同于 Riccati 外推的状态依赖利率路径，从而改写产出缺口动态。 | `outputs/phase11/extreme_numerical_compare_v2/*/common_shock_trajectories.csv` |
| `phase13_empirical_artificial_histories.png` | 经验环境：历史反事实 | 用历史路径把‘更低损失’转化成更直观的经济动态：比较 RL、Taylor 与历史政策在通胀、产出缺口和利率路径上的差异。 | `outputs/phase10/counterfactual_eval/*_historical_paths.csv or outputs/phase10/revealed_policy_eval/*_historical_paths.csv` |
| `phase13_empirical_revealed_histories.png` | 经验环境：历史反事实 | 用历史路径把‘更低损失’转化成更直观的经济动态：比较 RL、Taylor 与历史政策在通胀、产出缺口和利率路径上的差异。 | `outputs/phase10/counterfactual_eval/*_historical_paths.csv or outputs/phase10/revealed_policy_eval/*_historical_paths.csv` |
| `phase13_stochastic_tradeoffs.png` | 经验环境：长期随机评价 | 长期随机图不只比较总损失，还直接看通胀与产出波动的联合表现，突出 RL 的动态稳定化特征。 | `phase13 recomputation from phase10 policy maps` |
| `phase13_policy_slices_core.png` | 规则机制与经济学解释 | 把最佳 RL 规则直接画成状态切片，展示它们如何改变对通胀缺口和产出缺口的局部反应，这一图是规则级经济学解释的核心入口。 | `phase13 evaluation of phase10 checkpoints and benchmark rules` |
| `phase13_cross_transfer.png` | 经验环境：交叉迁移 | 把 direct-trained 规则放到对方环境中，展示 RL 优势主要来自正确匹配状态转移与福利权重，而不是任意环境下都同样有效。 | `outputs/phase10/counterfactual_eval/cross_transfer_summary.csv` |
| `phase13_external_models_best_rules.png` | 稳健性：外部模型 | 外部模型部分不堆满所有规则，而是直接展示每个外部模型上最有竞争力的 RL 规则，突出可保留的稳健性证据。 | `outputs/phase10/external_model_robustness/all_external_summary.csv` |
| `phase13_pyfrbus_progression.png` | 稳健性：pyfrbus 原生接口 | pyfrbus 图组强调‘native warm-start 之后 nonlinear PPO 可以再往前推一步’，这是论文中最适合讲原生 RL 增益的外部模型证据。 | `outputs/phase12/pyfrbus_warmstart_ppo/comparison.csv + outputs/phase12/pyfrbus_nonlinear_search/comparison.csv` |
