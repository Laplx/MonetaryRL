# Agent 交接指引

## 0. 文档目的

本文件用于在上下文切换时，把当前项目的状态、关键结论、文件位置、后续优先级和注意事项一次性交代清楚。新的 agent 读完本文件后，应能直接进入后续阶段而不需要重新摸索。

## 1. 项目总目标

| 项目 | 内容 |
|---|---|
| 研究目标 | 在一个结构清晰、参数具现实量级的随机宏观货币政策模型中，系统比较经典最优控制与强化学习在央行利率规则求解中的表现 |
| 最终权威 | `Thesis Proposal.docx` |
| 核心定位 | RL 是最优控制问题的数值求解器，不是 L3 学习主体路线 |
| 关键参照 | `Hinterlang and Tänzer (2021)` 是经验环境设计的最重要参照 |

## 2. 已完成阶段

| 阶段 | 状态 | 主要产物 |
|---|---|---|
| Phase 0 | 完成 | `docs/model_spec.md` |
| Phase 1 | 完成 | `docs/theory_notes.md`、`docs/theory_writing_notes.md` |
| Phase 2 | 完成 | `docs/data_spec.md`、`docs/calibration_plan.md`、`scripts/phase2_estimation.py`、`data/processed/`、`outputs/phase2/` |
| Phase 3 | 完成 | `src/monetary_rl/config/benchmark_lq.json`、`src/monetary_rl/models/lq_benchmark.py`、`scripts/phase3_build_benchmark.py`、`outputs/phase3/` |
| Phase 4 | 完成 | `src/monetary_rl/solvers/riccati.py`、`scripts/phase4_solve_lq.py`、`outputs/phase4/` |
| Phase 5 | 完成基础版 | `src/monetary_rl/envs/benchmark_env.py`、`src/monetary_rl/agents/ppo.py`、`scripts/phase5_train_ppo.py`、`outputs/phase5/` |

## 3. 关键文档先读顺序

新 agent 开始前，建议按如下顺序阅读：

| 顺序 | 文件 | 原因 |
|---|---|---|
| 1 | `Thesis Proposal.docx` | 最终权威 |
| 2 | `ROADMAP.md` | 全局路线图 |
| 3 | `docs/model_spec.md` | Phase 0 冻结的结构性决策 |
| 4 | `docs/theory_notes.md` | 理论框架 |
| 5 | `docs/data_spec.md` | 数据口径与样本 |
| 6 | `docs/calibration_plan.md` | 环境估计与规则估计分工 |
| 7 | `docs/ann_tuning_plan.md` | ANN 后续调优顺序 |
| 8 | `outputs/phase2/phase2_summary.md` | 当前经验环境估计结果 |
| 9 | `outputs/phase4/lq_solution_summary.md` | 理论最优 benchmark 结果 |
| 10 | `outputs/phase5/ppo_baseline_summary.md` | 当前 RL baseline 状态 |

## 4. 当前最重要的理论/实现边界

| 事项 | 结论 |
|---|---|
| Phase 4 的 Riccati 最优解 | 来自理论 LQ benchmark，不是直接把经验 SVAR 环境代入后求得 |
| benchmark 状态 | $[\tilde{\pi}_t, x_t, \tilde{i}_{t-1}]^\top$ |
| benchmark 损失 | $\lambda_\pi \tilde{\pi}_t^2+\lambda_x x_t^2+\lambda_i(\tilde{i}_t-\tilde{i}_{t-1})^2$ |
| 经验环境 | 与 benchmark 分开，Phase 2 已估计 `SVAR + ANN` 两套 |
| 反事实比较对象 | 理论最优、RL 规则、经验 Taylor rule、历史实际政策 |
| 写作边界 | 必须说明 Lucas critique |

补充说明：

| 容易混淆的问题 | 正确认知 |
|---|---|
| `Phase 4` 的具体数值是否来自经验 SVAR 环境 | 不是；它来自 `Phase 3` 的理论 LQ benchmark |
| `Phase 4` 为什么还能给出具体反馈矩阵和损失值 | 因为 `Phase 3` 已固定一个可计算的 benchmark 配置文件 |
| benchmark 是否内生包含经验 Taylor rule | 不包含；经验 Taylor rule 是外部比较规则，可被带入 benchmark 环境做对照 |

## 5. 当前已有数据与估计结果

### 5.1 数据

| 路径 | 内容 |
|---|---|
| `data/raw/` | 用户手动下载的原始 FRED CSV |
| `data/processed/macro_quarterly_full.csv` | 清洗后的完整季度数据 |
| `data/processed/macro_quarterly_sample_1987Q3_2007Q2.csv` | 主样本数据 |

### 5.2 Phase 2 经验结果

| 项目 | 当前结果 |
|---|---|
| 主样本 | `1987Q3-2007Q2`，80 个季度 |
| 线性环境 | 已按 Hinterlang 的递归 SVAR 结构估计 |
| 经验 Taylor rule | 已估计，长期通胀响应约 `1.4769` |
| ANN 环境 | 已跑通初版，但通胀方程尚未优于 SVAR |

## 6. 当前 benchmark 与理论最优结果

### 6.1 Phase 3 benchmark

| 文件 | 作用 |
|---|---|
| `src/monetary_rl/config/benchmark_lq.json` | benchmark 参数配置 |
| `src/monetary_rl/models/lq_benchmark.py` | 状态转移、损失函数、仿真 |

### 6.2 Phase 4 关键结果

| 项目 | 当前数值 |
|---|---|
| 反馈矩阵 $K$ | `[[1.089101, 1.078439, 0.434052]]` |
| 闭环最大特征值模 | `0.651192` |
| 零政策折现总损失 | `20.261789`（单个固定初值例子） |
| 最优政策折现总损失 | `10.096854`（单个固定初值例子） |

注意：这些是 benchmark 世界中的规范基准，不是经验环境结果。

## 7. 当前 RL baseline 状态

### 7.1 已完成的内容

| 模块 | 文件 |
|---|---|
| RL 环境 | `src/monetary_rl/envs/benchmark_env.py` |
| PPO 实现 | `src/monetary_rl/agents/ppo.py` |
| PPO 配置 | `src/monetary_rl/config/benchmark_ppo.json` |
| 训练脚本 | `scripts/phase5_train_ppo.py` |

### 7.2 当前 baseline 结果

以 `outputs/phase5/ppo_baseline_summary.md` 为准，当前 PPO baseline：

| 指标 | 当前结果 |
|---|---|
| 相对零政策改善 | 约 `10%` |
| 距 Riccati 最优差距 | 仍很大 |
| 结论 | PPO 已跑通且优于零政策，但还不能视为最终可发表的 benchmark RL 结果 |

### 7.3 正确解读

| 事项 | 解读 |
|---|---|
| Phase 5 是否完成 | 完成了“环境与 baseline 训练”这一步 |
| PPO 是否已达到理想 benchmark 水平 | 没有 |
| 对 Phase 6 的含义 | Phase 6 需要把“系统比较”与“RL 继续调优”部分结合起来推进 |

### 7.4 对 PPO 当前结果的具体判断

| 问题 | 当前判断 |
|---|---|
| PPO 是否跑通 | 是 |
| PPO 是否显著优于零政策 | 是，但改善幅度有限 |
| PPO 是否逼近 Riccati 最优 | 远未达到 |
| 结论 | 当前 PPO 只能算可运行 baseline，不是最终 benchmark RL 结果 |

### 7.5 RL 后续扩展建议

用户已明确同意：可以多跑几个算法、多调超参数，时间充足。

建议优先级：

| 优先级 | 任务 |
|---|---|
| 1 | 继续强化 PPO：更长训练、更多超参数搜索、reward/状态尺度调整 |
| 2 | 新增 `SAC` baseline |
| 3 | 新增 `TD3` baseline |
| 4 | 新增线性策略参数化的连续控制基线 |
| 5 | 若需要，再考虑 DDPG 作为与 Hinterlang 的方法对照 |

说明：

| 算法 | 建议 |
|---|---|
| PPO | 保留，但不要假定它是最适合当前 LQ benchmark 的算法 |
| SAC | 连续控制优先推荐 |
| TD3 | 可作为稳定的连续控制补充基线 |
| 线性策略基线 | 对当前 LQ benchmark 很贴题，信息量高 |

## 8. ANN 环境调优待办

此项非常重要，不能丢。

| 文档 | `docs/ann_tuning_plan.md` |
|---|---|

调优优先级已冻结为：

| 优先级 | 任务 |
|---|---|
| 1 | 在相同输入集下继续调 MLP |
| 2 | 在相同输入集下换更稳的训练策略 |
| 3 | 增加 1 期额外滞后 |
| 4 | 改成 RNN / LSTM / GRU |

注意：用户明确接受这个优先级顺序。

## 9. 下一阶段最合理的工作顺序

建议下一位 agent 按如下顺序推进：

| 顺序 | 任务 | 说明 |
|---|---|---|
| 1 | 阅读本文件与关键文档 | 建立上下文 |
| 2 | 复核 `Phase 5` PPO baseline | 确认为何距离最优仍远 |
| 3 | 进入 `Phase 6` | 做 benchmark 下的系统对照实验 |
| 4 | 在 `Phase 6` 内并行做 PPO 强化调优 | 时间充足，可系统搜索超参数 |
| 5 | 新增 `SAC/TD3/线性策略` 几条 benchmark RL 线 | 不要只押 PPO |
| 6 | 把经验 Taylor rule 带入 benchmark 环境做规则对照 | 这是 benchmark 比较的一部分 |
| 7 | 保持 ANN 环境调优并行推进 | 不阻塞 benchmark 主线 |

### 9.1 benchmark 中如何比较经验 Taylor rule

必须明确：

| 问题 | 回答 |
|---|---|
| benchmark 模型里有没有经验 Taylor rule | 没有，它不是 benchmark 内生部分 |
| 那怎么比较 | 用 `Phase 2` 估计出的 Taylor 系数构造一个外部规则，再把它代入 benchmark 环境模拟 |
| 这样做的意义 | 可以在同一 benchmark 世界里比较“理论最优 / PPO / 经验 Taylor / 零政策” |

可比较的经验 Taylor 规则基准形式：

$$
i_t=\alpha+\phi_\pi \pi_t+\phi_x x_t+\phi_i i_{t-1}
$$

在 benchmark 中，需要先转成 gap 形式或保证与中心化状态定义一致后再代入。

## 10. 推荐直接运行的命令

| 目的 | 命令 |
|---|---|
| 重跑 Phase 2 | `C:\Users\Laplace\anaconda3\envs\tor\python.exe scripts\phase2_estimation.py` |
| 重跑 Phase 3 | `C:\Users\Laplace\anaconda3\envs\tor\python.exe scripts\phase3_build_benchmark.py` |
| 重跑 Phase 4 | `C:\Users\Laplace\anaconda3\envs\tor\python.exe scripts\phase4_solve_lq.py` |
| 重跑 Phase 5 | `C:\Users\Laplace\anaconda3\envs\tor\python.exe scripts\phase5_train_ppo.py` |

## 11. 需要避免的误解

| 误解 | 正确认知 |
|---|---|
| “Phase 4 最优规则是用 SVAR 直接算出来的” | 错。它是理论 LQ benchmark 的 Riccati 解 |
| “Phase 5 PPO 已经学到理论最优” | 错。当前只是一个能优于零政策的 baseline |
| “ANN 现在只输入当期变量” | 错。当前 ANN 已使用与 SVAR 对齐的信息集 |
| “ANN 调优会阻塞主线” | 不会。用户明确允许并行推进 |

## 12. 用户偏好与协作注意事项

| 事项 | 说明 |
|---|---|
| 输出语言 | 中文 |
| 表格偏好 | 对总结、Plan、Task、长内容优先用表格 |
| 文件引用 | 用户要求用相对路径表达即可 |
| 协作方式 | 若需要用户补数据或文章，应立刻明确提出 |

---

一句话总结：

主线已经推进到“理论 benchmark 可解、经验环境已估、PPO baseline 已跑通但仍需增强”的状态；下一位 agent 应在不打断主线的前提下，一边做 `Phase 6` benchmark 对照，一边继续改进 PPO 和 ANN 两条学习线。
