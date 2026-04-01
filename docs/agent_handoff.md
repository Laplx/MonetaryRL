# Agent 交接指引

## 0. 文档定位

| 项目 | 说明 |
|---|---|
| 作用 | 为新 agent 提供当前项目真实状态、已完成成果、关键边界、下一阶段任务与常见误区 |
| 最终权威 | `Thesis Proposal.docx` |
| 次级权威 | `ROADMAP.md` |
| 辅助文档 | `docs/*.md`、`outputs/*/*_summary.md` |
| 当前更新时间 | `2026-04-01` |

## 1. 新 agent 必读顺序

| 顺序 | 文件 | 目的 |
|---|---|---|
| 1 | `Thesis Proposal.docx` | 核对最终研究边界与开题要求 |
| 2 | `docs/agent_handoff.md` | 先建立当前真实项目状态 |
| 3 | `ROADMAP.md` | 理解总路线与 Phase 8 之后的安排 |
| 4 | `docs/model_spec.md` | 核对 benchmark、扩展顺序与结构性冻结决策 |
| 5 | `docs/theory_notes.md` | 核对理论框架与写作边界 |
| 6 | `docs/data_spec.md` | 核对样本、变量与数据定义 |
| 7 | `docs/calibration_plan.md` | 核对经验环境与 benchmark 的区分 |
| 8 | `docs/ann_tuning_plan.md` | 明确 ANN 仍然延后且调优顺序已冻结 |
| 9 | `outputs/phase6/phase6_summary.md` | 理解 benchmark 系统对照结果 |
| 10 | `outputs/phase7/matrix/phase7_matrix_summary.md` | 理解 Phase 7 主矩阵结果 |
| 11 | `docs/phase7_writing_findings.md` | 直接接收当前可写入论文的主要结论 |
| 12 | `docs/phase8_execution_guide.md` | 进入下一阶段前先看执行指引 |
| 13 | `outputs/phase9/phase9_summary.md` | 理解 ANN 补充与 local model uncertainty 已完成到什么程度 |
| 14 | `docs/phase10_enhancement_guide.md` | 当前冻结的增强轮总指引 |
| 15 | `docs/writing_figure_table_plan.md` | 写作与图表组织参考 |

## 2. 当前阶段状态

| 阶段 | 状态 | 主要产物 |
|---|---|---|
| Phase 0 | 完成 | `docs/model_spec.md` |
| Phase 1 | 完成 | `docs/theory_notes.md`、`docs/theory_writing_notes.md` |
| Phase 2 | 完成 | `scripts/phase2_estimation.py`、`data/processed/`、`outputs/phase2/` |
| Phase 3 | 完成 | `scripts/phase3_build_benchmark.py`、`src/monetary_rl/models/lq_benchmark.py`、`outputs/phase3/` |
| Phase 4 | 完成 | `scripts/phase4_solve_lq.py`、`src/monetary_rl/solvers/riccati.py`、`outputs/phase4/` |
| Phase 5 | 完成基础版 | `scripts/phase5_train_ppo.py`、`src/monetary_rl/envs/benchmark_env.py`、`src/monetary_rl/agents/ppo.py`、`outputs/phase5/` |
| Phase 6 | 完成 | `scripts/phase6_benchmark_compare.py`、`src/monetary_rl/agents/linear_policy.py`、`outputs/phase6/` |
| Phase 7 | 完成 | `scripts/phase7_matrix_experiments.py`、`src/monetary_rl/agents/sac.py`、`src/monetary_rl/agents/td3.py`、`src/monetary_rl/models/asymmetric_benchmark.py`、`outputs/phase7/matrix/` |
| Phase 8 | 完成 | `SVAR` 经验环境反事实、长期随机评估、主表主图、policy registry |
| Phase 9 | 完成 | `ANN` 调优、ANN 补充反事实、本地 model uncertainty 汇总 |
| Phase 10A | 已冻结待执行 | 增强轮：`SVAR direct`、`ANN direct`、`PPO` 双版本、统一反事实/随机/交叉迁移、外部模型、反推福利模块 |
| Phase 10B | 后续阶段 | 论文组装与定稿 |

## 3. 当前主线与用户已冻结的优先级

| 主题 | 当前结论 |
|---|---|
| 当前主线 | `Phase 9` 已完成；下一步不是直接写作，而是先执行 `Phase 10A` 增强轮 |
| ANN 调优 | 已完成本轮调优，但仅通过补充结果门槛 |
| RL 主线 | 已完成 benchmark 与扩展矩阵，也已完成 transfer-based 经验比较；下一步是补上 `SVAR/ANN` 直接训练规则 |
| 当前增强轮 | 先做 `SVAR direct`，再做 `ANN direct`，然后只对 `PPO` 补 `linear-policy` 与 `nonlinear-policy` 双版本 |
| PPO 双版本 | 已冻结为仅 `PPO` 拆成 `linear-policy + nonlinear-policy`；`TD3/SAC` 不拆 |
| 反推福利模块 | 已冻结为本轮附加报告口径，基于 `SVAR + empirical Taylor` 反推权重 |
| 扩展环境矩阵 | 已冻结并完成为 `1 + 3 + 3 + 3` 个环境，即 `benchmark + 3 nonlinear + 3 ZLB/ELB-tightness + 3 asymmetric` |
| RL 算法矩阵 | 已冻结并完成为 `PPO + TD3 + SAC` |
| 重要参照 | `Hinterlang and Tänzer (2021)` 必须始终作为最重要相似工作记在脑中 |

## 4. 必须严格记住的边界

| 容易混淆的问题 | 正确认知 |
|---|---|
| Phase 4 的最优规则是否直接来自经验 SVAR 环境 | 不是。它来自 `Phase 3` 理论 LQ benchmark 的 Riccati 解 |
| benchmark 中是否内生包含经验 Taylor rule | 没有。经验 Taylor rule 是外部比较规则 |
| 经验 Taylor rule 在 benchmark 中如何比较 | 先转成与 benchmark gap 状态一致的形式，再代入 benchmark 环境仿真 |
| 当前 PPO 是否已经逼近理论最优 | 没有。Phase 5 只是可运行 baseline；Phase 6/7 也未支持“PPO 全局最好”的说法 |
| 当前 ZLB 三档是否都是纯结构性的 ZLB | 严格说不是，应写作 `ZLB/ELB-tightness tiers` 或“逐步收紧的有效下界环境” |
| 当前 nonlinear 环境下 Riccati 外推仍强是否意味着 RL 没价值 | 不是。更合理的解释是当前 nonlinear 扭曲尚未完全压倒线性参考结构 |
| 单次 tuned PPO 与 Phase 7 多 seed 结果是否矛盾 | 不是。应区分“单次强化结果”和“多 seed 固定预算稳健性结果” |
| 经验环境与 benchmark 的关系 | 两者必须严格分开；Phase 8 才进入经验环境反事实比较 |
| Lucas critique | 必须在写作中明确说明，是经验转移环境的核心方法边界 |
| 是否在 Phase 8 先重启 ANN 调优 | 不应。应先完成 `SVAR` 主结果，再在 `Phase 9` 判断 ANN 是否达到补充结果门槛 |
| 是否在 Phase 8 直接做 11 个 DSGE 比较 | 不应。应放入 `Phase 9`，作为 model uncertainty robustness extension |

## 5. 已完成的关键结果

### 5.1 Phase 2

| 模块 | 当前结果 |
|---|---|
| 样本 | `1987Q3-2007Q2`，`80` 个季度 |
| 经验 Taylor rule | 已估计，长期通胀响应约 `1.4769` |
| 经验环境 | `SVAR + ANN` 两套都已跑通 |
| ANN 当前状态 | `Phase 2` 初版中 output gap 优于 `SVAR`、inflation 弱于 `SVAR`；最新口径已由 `Phase 9` 更新，见 `outputs/phase9/phase9_summary.md` |

### 5.2 Phase 4

| 项目 | 当前结果 |
|---|---|
| Riccati 最优反馈矩阵 `K` | `[[1.089101, 1.078439, 0.434052]]` |
| benchmark 定位 | 理论规范基准，不是经验环境结果 |

### 5.3 Phase 6

| 事项 | 当前结果 |
|---|---|
| benchmark 统一评价协议 | 已完成，避免了 Phase 4 固定初值例子与 Phase 5 随机起点评价口径混杂 |
| PPO 关键修复 | 已改为 squashed Gaussian，优化分布与环境实际执行动作一致 |
| 新增基线 | 已加入 `linear_policy_search` |
| benchmark 对照结论 | `riccati_optimal` 最好，`linear_policy_search` 非常接近，`empirical_taylor` 明显弱于理论基准，PPO 仍与最优有较大差距 |
| 结果文件 | `outputs/phase6/phase6_summary.md`、`policy_evaluation.csv`、`policy_coefficients.csv` |

### 5.4 Phase 7

| 事项 | 当前结果 |
|---|---|
| 扩展环境 | 已完成 `nonlinear` 三档、`zlb` 三档、`asymmetric` 三档 |
| RL 算法 | 已完成 `PPO`、`TD3`、`SAC` 三算法矩阵 |
| 多 seed | 每个环境-算法对运行 `3` 个 seeds：`7/29/43` |
| 共享工具 | 已加入 `evaluation.py`、`experiment_utils.py`、`replay_buffer.py` |
| 主矩阵脚本 | `scripts/phase7_matrix_experiments.py` |
| 主结果汇总 | `outputs/phase7/matrix/phase7_matrix_summary.md` |
| 图表产物 | `outputs/phase7/matrix/plots/` |

## 6. 当前最值得记住的实证结论

| 环境 | 当前最强 RL | 需要怎么表述 |
|---|---|---|
| `benchmark` | `SAC` | 仅指当前多 seed / 当前固定预算设置下的 Phase 7 主矩阵 |
| `nonlinear_*` | `SAC` | 随非线性增强，`PPO` 相对掉队更明显 |
| `zlb_mild` | `SAC` | mild tier 下 `SAC` 略优 |
| `zlb_medium` | `TD3` | tighter bound 下 `TD3` 开始占优 |
| `zlb_strong` | `TD3` | stronger bound 下 `TD3` 最强，`PPO` 最弱 |
| `asymmetric_*` | `SAC` | 三档 asymmetric 中 `SAC` 都最稳、最强 |

更高层的结论：

| 结论 | 当前支持程度 |
|---|---|
| RL 算法优劣排序具有明显环境依赖性 | 强支持 |
| PPO 不是所有环境下最合适的算法 | 强支持 |
| 约束型环境更偏向 off-policy 连续控制算法 | 强支持 |
| 线性参考规则在当前 nonlinear 与 asymmetric 扩展下仍很鲁棒 | 强支持 |
| 当前扩展环境已经完全推翻线性反馈结构 | 不支持 |

## 7. 重要文件索引

### 7.1 代码

| 路径 | 作用 |
|---|---|
| `src/monetary_rl/agents/ppo.py` | 已修复为 squashed Gaussian PPO |
| `src/monetary_rl/agents/sac.py` | SAC baseline |
| `src/monetary_rl/agents/td3.py` | TD3 baseline |
| `src/monetary_rl/agents/linear_policy.py` | benchmark 线性策略搜索基线 |
| `src/monetary_rl/agents/replay_buffer.py` | off-policy 经验回放 |
| `src/monetary_rl/envs/benchmark_env.py` | benchmark 与扩展环境执行层；已加入状态爆炸保护 |
| `src/monetary_rl/models/asymmetric_benchmark.py` | 非对称损失扩展 |
| `src/monetary_rl/evaluation.py` | 统一评估工具 |
| `src/monetary_rl/experiment_utils.py` | 脚本辅助工具 |

### 7.2 配置

| 路径 | 作用 |
|---|---|
| `src/monetary_rl/config/benchmark_lq.json` | benchmark 配置 |
| `src/monetary_rl/config/nonlinear_phillips_mild.json` | nonlinear mild |
| `src/monetary_rl/config/nonlinear_phillips_medium.json` | nonlinear medium |
| `src/monetary_rl/config/nonlinear_phillips_strong.json` | nonlinear strong |
| `src/monetary_rl/config/asymmetric_loss_mild.json` | asymmetric mild |
| `src/monetary_rl/config/asymmetric_loss_medium.json` | asymmetric medium |
| `src/monetary_rl/config/asymmetric_loss_strong.json` | asymmetric strong |

### 7.3 结果

| 路径 | 作用 |
|---|---|
| `outputs/phase6/phase6_summary.md` | Phase 6 benchmark 对照摘要 |
| `outputs/phase7/matrix/phase7_matrix_summary.md` | Phase 7 主矩阵摘要 |
| `outputs/phase7/matrix/rl_summary.csv` | RL 汇总表 |
| `outputs/phase7/matrix/all_policy_summary.csv` | 含外部对照规则的汇总表 |
| `outputs/phase7/matrix/policy_coefficients.csv` | 近似线性系数 |
| `outputs/phase7/matrix/training_logs.csv` | 训练日志 |
| `outputs/phase7/matrix/plots/` | 热图、箱线图、强度曲线、ZLB 图等 |
| `docs/phase7_writing_findings.md` | 当前最值得写进论文的发现 |

## 8. 当前不应该做什么

| 不应该做的事 | 原因 |
|---|---|
| 把 `Phase 7` 说成未完成 | 已完成，矩阵结果已生成 |
| 把 `PPO` 说成当前最强算法 | 当前主矩阵结果不支持 |
| 把 `ZLB` 三档写成纯粹结构性 ZLB 约束 | 当前实现更准确地说是逐步收紧的 `ELB-tightness` 环境 |
| 现在就重启 ANN 调优主线 | 用户明确要求延后 |
| 在 `Phase 8` 中直接插入 `DSGE` 稳健性比较 | 会打断主线，应后置到 `Phase 9` |
| 在经验环境里混淆 benchmark 规则与经验环境规则 | 会破坏论文主线结构 |
| 用“RL 已经全面击败线性规则”做结论 | 当前结果不支持，且与实际产出不一致 |

## 9. Phase 8 的正确进入方式

| 项目 | 要求 |
|---|---|
| 下一阶段名称 | 经验规则与历史反事实 |
| 首要任务 | 把 `benchmark` 中形成的主要规则与 `Phase 2` 经验环境连接起来 |
| 经验环境首选 | 先用 `SVAR` 环境完成主结果 |
| ANN 经验环境 | 在当前阶段不作为主线；如要用，应明确其 inflation 方程尚未优于 SVAR |
| ANN 的正确位置 | `Phase 8` 主结果完成后，再在 `Phase 9` 按门槛决定是否作为补充结果 |
| DSGE 扩展 | 放入 `Phase 9`，不属于当前阶段的首要任务 |
| 比较对象 | `Riccati reference`、`best RL rules`、`empirical Taylor rule`、`historical actual policy` |
| 写作要求 | 必须同时报告福利损失、路径图、目标偏离、规则含义，并说明 Lucas critique |

详细执行顺序见 `docs/phase8_execution_guide.md`。

## 10. 当前冻结的增强轮（Phase 10A）

| 项目 | 要求 |
|---|---|
| 核心目标 | 按固定顺序完成 `SVAR direct`、`ANN direct`、`PPO` 双版本、统一历史反事实、长期随机与交叉迁移，再接外部模型与反推福利 |
| PPO 双版本 | 仅 `PPO` 拆成 `linear-policy` 与 `nonlinear-policy`；`TD3/SAC` 保持标准参数化 |
| 福利口径 | 主线仍用当前基准损失；新增 `反推福利模块` 作为第二套报告口径 |
| 外部模型模块 | 严格后置在统一反事实、长期随机与交叉迁移之后；优先模型按 `pyfrbus`、`US_FRB03`、`US_SW07`、`US_CCTW10`、`US_CPS10`、`US_KS15`、`US_RA07` 执行 |
| 详细指引 | 见 `docs/phase10_enhancement_guide.md` |
| 写作要求 | 必须写清 `transfer vs direct-trained` 区别，并继续明确 `Lucas critique` |
| 若需继续经验调优 | 当 `SVAR` 样本内历史政策仍优于固定反馈规则时，不立即重开经验训练；先完成外部模型评估，再补 `SVAR direct SAC/TD3` 的更多 seeds / 更长预算 / `state_scale` 与 `horizon` 调整，以及必要时的 `revealed welfare` 附加训练实验 |
| `revealed welfare` 新规则状态 | 已完成 `SVAR/ANN` 环境下 `PPO/TD3/SAC` 重新训练，并已生成对应历史反事实、长期随机、交叉迁移与 `pyfrbus` 外部检验结果，输出位于 `outputs/phase10/revealed_policy_training/`、`outputs/phase10/revealed_policy_eval/` |
| 外部模型已完成批次 | 已完成 `pyfrbus`、`US_SW07`、`US_CCTW10`、`US_KS15`、`NK_CW09`；其中 `Dynare/MMB` 口径统一用规则的线性 surrogate 系数接 simple-rule 接口，见 `outputs/phase10/external_model_robustness/mmb_summary.csv` |
| 外部模型当前剩余阻塞 | `US_FRB03`、`US_CPS10`、`US_RA07` 仍受 legacy `Dynare/MMB` 兼容问题阻塞，不再是 MATLAB 许可证本身；当前状态见 `outputs/phase10/external_model_robustness/runtime_status.csv` |

## 11. 可直接运行的命令

| 目的 | 命令 |
|---|---|
| 重跑 Phase 2 | `C:\Users\Laplace\anaconda3\envs\tor\python.exe scripts/phase2_estimation.py` |
| 重跑 Phase 3 | `C:\Users\Laplace\anaconda3\envs\tor\python.exe scripts/phase3_build_benchmark.py` |
| 重跑 Phase 4 | `C:\Users\Laplace\anaconda3\envs\tor\python.exe scripts/phase4_solve_lq.py` |
| 重跑 Phase 5 | `C:\Users\Laplace\anaconda3\envs\tor\python.exe scripts/phase5_train_ppo.py` |
| 重跑 Phase 6 | `C:\Users\Laplace\anaconda3\envs\tor\python.exe scripts/phase6_benchmark_compare.py` |
| 重跑 Phase 7 全矩阵 | `C:\Users\Laplace\anaconda3\envs\tor\python.exe scripts/phase7_matrix_experiments.py --group all` |
| 只跑 benchmark 组 | `C:\Users\Laplace\anaconda3\envs\tor\python.exe scripts/phase7_matrix_experiments.py --group benchmark` |
| 只跑 nonlinear 组 | `C:\Users\Laplace\anaconda3\envs\tor\python.exe scripts/phase7_matrix_experiments.py --group nonlinear` |
| 只跑 zlb 组 | `C:\Users\Laplace\anaconda3\envs\tor\python.exe scripts/phase7_matrix_experiments.py --group zlb` |
| 只跑 asymmetric 组 | `C:\Users\Laplace\anaconda3\envs\tor\python.exe scripts/phase7_matrix_experiments.py --group asymmetric` |

## 12. 一句话交接总结

当前项目已经完成理论 benchmark、Phase 6/7 稳健矩阵、Phase 8 的 `SVAR` transfer-based 经验反事实与 Phase 9 的 `ANN` 补充模块。下一位 agent 的当前主任务不是直接写论文，而是执行 `docs/phase10_enhancement_guide.md` 中冻结的正式顺序：先做 `SVAR direct`，再做 `ANN direct`，随后只对 `PPO` 做 `linear-policy / nonlinear-policy` 双版本，再统一做历史反事实、长期随机与交叉迁移，之后接外部模型，最后加入反推福利模块。
