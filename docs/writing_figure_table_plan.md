# 写作图表与表格计划

## 0. 文档定位

| 项目 | 说明 |
|---|---|
| 目标 | 为论文正文与附录准备图表清单，至少做到不弱于 `Hinterlang and Tänzer (2021)` |
| 参考文献 | `literature/hinterlang2021rl_optim_i_reaction.pdf` |
| 当前基础 | Phase 6 与 Phase 7 已经有一批可直接使用或可二次加工的图表 |

## 1. Hinterlang 可参考的图表类型

| 类别 | Hinterlang 中出现的类型 | 本项目建议 |
|---|---|---|
| 环境拟合 | `fit / MSE` 图表、模型比较表 | 若写经验环境，可保留 Phase 2 的 `SVAR vs ANN` 拟合结果 |
| 规则解释 | `partial dependence` 面图与线图 | Phase 7 与 Phase 8 都建议补政策面图和切片图 |
| 反事实 | `actual vs counterfactual series` | Phase 8 必做，而且应比文献更系统 |
| 福利表 | `target deviation / loss` 表 | Phase 8 主表必须有 |
| 模型比较 | `relative variances / loss` | 可扩展为 benchmark、经验环境、扩展环境三层比较 |

## 2. 当前已经有的图表资产

| 路径 | 内容 |
|---|---|
| `outputs/phase7/matrix/plots/rl_mean_loss_heatmap.png` | RL 平均损失热图 |
| `outputs/phase7/matrix/plots/rl_loss_gap_heatmap.png` | 相对最佳 RL 的损失差热图 |
| `outputs/phase7/matrix/plots/distortion_strength_curves.png` | 扭曲强度上升下的损失曲线 |
| `outputs/phase7/matrix/plots/algorithm_win_counts.png` | 各算法胜出次数 |
| `outputs/phase7/matrix/plots/best_rl_coefficients_heatmap.png` | 最佳 RL 近似系数热图 |
| `outputs/phase7/matrix/plots/benchmark_seed_boxplots.png` | benchmark 多 seed 箱线图 |
| `outputs/phase7/matrix/plots/nonlinear_seed_boxplots.png` | nonlinear 多 seed 箱线图 |
| `outputs/phase7/matrix/plots/zlb_seed_boxplots.png` | zlb 多 seed 箱线图 |
| `outputs/phase7/matrix/plots/asymmetric_seed_boxplots.png` | asymmetric 多 seed 箱线图 |
| `outputs/phase7/matrix/plots/zlb_clip_rates.png` | ZLB/ELB 环境裁剪或约束相关图 |
| `outputs/phase7/matrix/plots/benchmark_all_policies.png` | benchmark 下全规则对照图 |

## 3. 论文正文建议最少保留的主表

| 表编号建议 | 内容 | 来源 |
|---|---|---|
| 表 1 | benchmark 对照：Riccati、linear policy、PPO、SAC、TD3、empirical Taylor、zero policy | Phase 6 |
| 表 2 | Phase 7 全矩阵 RL 汇总表 | `outputs/phase7/matrix/rl_summary.csv` |
| 表 3 | Phase 7 含外部对照规则的摘要表 | `all_policy_summary.csv` |
| 表 4 | 不同环境下最佳 RL 的近似线性系数 | `policy_coefficients.csv` |
| 表 5 | Phase 8 经验环境反事实福利表 | 待 Phase 8 生成 |
| 表 6 | Phase 8 目标偏离与利率波动分解表 | 待 Phase 8 生成 |

## 4. 论文正文建议最少保留的主图

| 图编号建议 | 内容 | 当前状态 |
|---|---|---|
| 图 1 | benchmark 下各规则综合对比图 | 已有基础图 |
| 图 2 | Phase 7 RL 平均损失热图 | 已有 |
| 图 3 | 扭曲强度上升下的算法损失曲线 | 已有 |
| 图 4 | 各环境组的多 seed 箱线图 | 已有 |
| 图 5 | 最佳 RL 近似线性系数热图 | 已有 |
| 图 6 | 代表性环境的政策函数切片图 | 待补 |
| 图 7 | 代表性环境的政策面图 | 待补 |
| 图 8 | Phase 8 历史与反事实路径图 | 待补 |
| 图 9 | Phase 8 福利损失对比图 | 待补 |

## 5. 为了“至少比 Hinterlang 多”的推荐补强项

| 补强项 | 原因 |
|---|---|
| 多 seed 分布图 | Hinterlang 的随机性稳健展示不如我们现在完整 |
| 扭曲强度曲线 | 可以清楚展示环境强度上升时算法排序变化 |
| 最佳 RL 系数热图 | 有助于把 RL 翻译回经济学语言 |
| benchmark 与扩展环境并列展示 | 更系统地强调环境依赖性 |
| 经验环境反事实的多指标分解 | 不只报告总 loss，更报告 inflation/output/rate 的分项 |

## 6. 当前最值得优先补的图

| 图 | 用途 |
|---|---|
| benchmark / nonlinear / zlb / asymmetric 的政策切片图 | 解释为何算法排序变化 |
| benchmark 与代表性扩展环境的政策面图 | 可视化非线性与约束影响 |
| Phase 8 历史与反事实三联图 | 直接进入正文 |
| Phase 8 福利分解条形图 | 让结果更容易写作与答辩 |

## 7. 写作时的表述提醒

| 问题 | 建议措辞 |
|---|---|
| ZLB 三档 | 写作 `ZLB/ELB-tightness tiers` 更准确 |
| PPO 与单次 tuned 结果不一致 | 明确区分单次强化与多 seed 固定预算稳健性 |
| nonlinear 中 Riccati 外推仍强 | 解释为当前扭曲尚不足以完全改变反馈结构 |
| ANN 环境 | 当前可报告“已跑通但仍待调优”，不要写成最终主结果环境 |
