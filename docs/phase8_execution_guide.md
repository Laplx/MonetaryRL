# Phase 8 执行指引

## 0. 文档定位

| 项目 | 说明 |
|---|---|
| 目标 | 固定 `Phase 8` 的执行口径，并为 `Phase 9` 接手提供结果入口 |
| 最终权威 | `Thesis Proposal.docx` |
| 依赖前置 | `Phase 2`、`Phase 6`、`Phase 7` |
| 当前假设 | `Phase 8` 已按该口径完成；`ANN` 与 `DSGE` 继续留在 `Phase 9` |

## 0.1 当前完成状态

| 项目 | 状态 |
|---|---|
| `SVAR` 历史反事实 | 已完成 |
| 长期随机评估 | 已完成 |
| policy registry | 已生成于 `outputs/phase8/policy_registry.csv` |
| 主总结 | 已生成于 `outputs/phase8/phase8_summary.md` |
| 主图 | 已生成于 `outputs/phase8/plots/` |

## 1. Phase 8 的任务定义

| 项目 | 内容 |
|---|---|
| 阶段名称 | 经验规则与历史反事实 |
| 核心目标 | 将理论规则、RL 规则、经验 Taylor rule 与历史实际政策放到统一经验环境中比较 |
| 主环境 | 先以 `SVAR` 经验环境作为主结果环境 |
| ANN 环境 | 当前只作为候选补充，不应抢占主线 |
| 后续衔接 | `Phase 9` 负责 ANN 补充与 DSGE/model uncertainty 扩展 |

## 2. 必须坚持的原则

| 原则 | 说明 |
|---|---|
| benchmark 与经验环境严格区分 | 不要把 Phase 3/4 的 benchmark 与 Phase 2 的经验环境混写 |
| empirical Taylor 是外部规则 | 它不是 benchmark 的内生部分 |
| Phase 8 先用 SVAR 做主结果 | 因为当前 ANN inflation 方程仍未优于 SVAR |
| Lucas critique 必须写 | 这是 Phase 8 的方法边界核心 |
| 规则比较要统一损失口径 | 否则对照失效 |
| 不在 Phase 8 做 DSGE 扩展 | 避免主线被 model uncertainty 模块分流 |

## 3. 建议直接执行的工作顺序

| 顺序 | 任务 | 目标产物 |
|---|---|---|
| 1 | 核对 `Phase 2` 环境对象与接口 | 明确如何把不同规则注入经验环境 |
| 2 | 明确待比较规则集合 | 形成统一 policy registry |
| 3 | 先完成 `SVAR` 环境下的动态反事实 | 形成主结果表与主路径图 |
| 4 | 生成福利损失、目标偏离、波动、利率路径等表图 | 形成正文主图主表 |
| 5 | 对 ANN 做进入 `Phase 9` 的质量门槛判断 | 决定其是补充结果还是仅保留局限说明 |
| 6 | 为 `Phase 9` 预留规则导出与图表接口 | 避免后续 ANN/DSGE 模块重复搭管线 |
| 7 | 同步更新写作材料与 handoff | 避免实验和写作脱节 |

## 4. 建议优先比较的规则

| 规则 | 来源 | 备注 |
|---|---|---|
| `historical_actual_policy` | 数据 | 历史对照线 |
| `empirical_taylor_rule` | Phase 2 | 经验规则基线 |
| `riccati_reference` | Phase 4 | 理论规范参考 |
| `linear_policy_search` | Phase 6 | 第二轮补充的强线性基线 |
| `best_benchmark_rl` | Phase 6/7 | 第一轮建议先用 `SAC`，第二轮再补 `PPO/TD3` |
| `best_extension_rl` | Phase 7 | Phase 8 不抢主表，作为补充展示环境依赖性 |

## 5. 建议优先完成的结果表

| 表格 | 内容 |
|---|---|
| 表 1 | 各规则在经验 `SVAR` 环境中的总福利损失、平均单期损失 |
| 表 2 | 通胀偏离、产出缺口偏离、利率波动的分项指标 |
| 表 3 | 相对历史政策和相对经验 Taylor 的改进百分比 |
| 表 4 | 规则系数或近似线性系数对照表 |
| 表 5 | 是否进入 `Phase 9 ANN` 与 `Phase 9 DSGE` 模块的门槛判断表 |

## 6. 建议优先完成的图

| 图 | 内容 |
|---|---|
| 图 1 | 历史实际路径与各规则反事实路径对比：`inflation/output gap/policy rate` |
| 图 2 | 各规则福利损失条形图或点图 |
| 图 3 | 通胀偏离与产出缺口偏离的分项对照 |
| 图 4 | 代表性时期的政策利率路径对照 |
| 图 5 | 若可行，规则在状态空间上的切片图或热图 |
| 图 6 | 第一轮核心规则与第二轮补充规则的分层展示图 |

## 7. 与 Phase 7 的衔接方式

| 问题 | 建议做法 |
|---|---|
| 是否把所有 Phase 7 扩展环境规则都带入经验环境 | 不建议一开始全带入，先做主结果集合 |
| 是否继续补跑 RL | Phase 8 不是继续做算法矩阵，而是做经验反事实连接 |
| 是否现在恢复 ANN 调优 | 不建议，除非 Phase 8 主结果已经完成 |
| 是否在此阶段直接做 DSGE 比较 | 不建议，放到 `Phase 9` 的 model uncertainty 模块 |
| Phase 7 哪些结论要延续 | 算法环境依赖性、PPO 非全局最优、线性参考规则鲁棒性 |

## 8. 当前最推荐的 Phase 8 实施策略

| 方案 | 推荐度 | 说明 |
|---|---|---|
| 先做 `SVAR` 经验环境主结果 | 高 | 最稳，最符合当前项目成熟度 |
| `ANN` 经验环境仅做补充 | 高 | 仅在 `Phase 8` 主结果完成后进入 `Phase 9` |
| 只带 `SAC + Riccati + Taylor + Actual` 四条主线先跑 | 高 | 先把论文核心结果跑扎实 |
| 再考虑补 `PPO/TD3/linear_policy_search` | 高 | 第二轮丰富表图时再加 |
| `DSGE` 稳健性扩展后置 | 高 | 放到 `Phase 9`，以免打乱主线 |

## 9. ANN 进入 Phase 9 的门槛

| 检查项 | 要求 |
|---|---|
| 主结果完成 | `SVAR` 主表主图已完成 |
| 拟合质量 | inflation 方程不应继续明显弱于 `SVAR` |
| 动态稳定性 | ANN 反事实路径不能出现明显失稳 |
| 写作定位 | 明确是补充结果还是仅作局限讨论 |

## 10. Phase 9 的接口预留

| 项目 | 要求 |
|---|---|
| 规则导出 | `Phase 8` 应保存统一 policy registry 与近似系数表 |
| 图表接口 | `Phase 8` 图表脚本应可扩展到 ANN/DSGE |
| 规则形式 | `DSGE` 第一轮优先迁移线性或可压缩为线性系数的规则 |
| 资源来源 | 未来 `DSGE` 模型优先来自 `MMB` |

## 11. 进入 Phase 8 前的最低检查清单

| 检查项 | 要求 |
|---|---|
| 已读 `docs/agent_handoff.md` | 必须 |
| 已读 `outputs/phase6/phase6_summary.md` | 必须 |
| 已读 `outputs/phase7/matrix/phase7_matrix_summary.md` | 必须 |
| 已明白 benchmark 与经验环境区别 | 必须 |
| 已确认 ANN 暂不抢主线 | 必须 |
| 已确认 `DSGE` 扩展后置到 `Phase 9` | 必须 |
| 已准备在写作中说明 Lucas critique | 必须 |
