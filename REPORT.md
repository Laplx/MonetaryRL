# 中期报告

## 1. 项目定位与当前状态

| 项目 | 内容 |
|---|---|
| 研究主线 | 将强化学习视为货币政策最优控制的数值求解器，并与经典最优控制方法系统对照 |
| benchmark 主线 | 已完成 LQ benchmark、Riccati 规范解、`PPO/TD3/SAC` 多环境矩阵 |
| empirical 主线 | 已完成 `SVAR` 与 `ANN` 两类经验环境、历史反事实、长期随机、交叉迁移 |
| 增强模块 | 已完成 `SVAR/ANN direct`、`PPO linear/nonlinear`、反推福利、外部模型稳健性 |
| 当前阶段 | 除论文写作与少量可选增强外，核心实验工作已基本完成 |

## 2. 已完成实验框架

| 模块 | 已完成内容 | 主要产物 |
|---|---|---|
| 理论基准 | LQ 模型、Riccati 最优反馈、benchmark 统一评价协议 | `outputs/phase4/`、`outputs/phase6/` |
| RL 主矩阵 | `benchmark + nonlinear + ELB-tightness + asymmetric` × `PPO/TD3/SAC` | `outputs/phase7/matrix/` |
| 经验环境 | `SVAR` 主结果、`ANN` 补充结果、经验 Taylor、历史实际政策对照 | `outputs/phase8/`、`outputs/phase9/` |
| direct training | `SVAR direct`、`ANN direct` 三算法；`PPO` 线性/非线性双版本 | `outputs/phase10/svar_direct/`、`outputs/phase10/ann_direct/`、`outputs/phase10/ppo_policy_variants/` |
| 统一评价 | 历史反事实、长期随机、交叉迁移；人工损失与反推损失双口径 | `outputs/phase10/counterfactual_eval/`、`outputs/phase10/revealed_welfare/`、`outputs/phase10/revealed_policy_eval/` |
| 外部模型 | `pyfrbus`、`US_SW07`、`US_CCTW10`、`US_KS15`、`NK_CW09` | `outputs/phase10/external_model_robustness/` |
| case 管理 | 已汇总全量 case inventory，共 `432` 个 cases | `outputs/phase10/case_inventory/phase10_case_inventory.csv` |

## 3. 核心实验 cases 与主要观察

| 维度 | cases 概览 | 主要观察 |
|---|---|---|
| benchmark/扩展环境 | `PPO/TD3/SAC` 在理论与扩展环境系统对照 | 算法优劣显著依赖环境；`SAC/TD3` 在约束或非线性环境更稳，`PPO` 并非普遍最优 |
| SVAR direct | `PPO/TD3/SAC` 直接在 `SVAR` 训练 | 在人工损失下，`PPO nonlinear` 与 `SAC` 可形成有竞争力规则，但不支持“RL 全面压过历史政策” |
| ANN direct | `PPO/TD3/SAC` 直接在 `ANN` 训练 | `ANN` 环境下 direct rules 有时样本内更强，但跨环境迁移更脆弱 |
| PPO 双版本 | `linear-policy` vs `nonlinear-policy` | `PPO nonlinear` 在部分经验任务上优于线性版，但提升不稳定，不足以单独支撑“非线性一定更优” |
| 反推福利 | 基于 `SVAR + empirical Taylor` 识别第二套福利权重 | 排序发生明显变化，说明结果对福利标尺敏感；经验偏好识别有信息量，但仍受固定转移与 Lucas critique 限制 |
| 外部模型 | 五个外部模型、人工/反推双评分 | 迁移表现高度异质；部分 legacy 模型对激进规则极敏感，稳定性筛选本身就是结果 |

## 4. 相对开题报告与 Hinterlang 的完成度判断

| 对照对象 | 计划要求 | 当前判断 |
|---|---|---|
| 开题报告 | 理论 benchmark、RL 对照、非线性/约束扩展、经验校准、历史反事实 | 已完成，且超出原开题：新增多算法、多环境矩阵、双经验环境、反推福利、外部模型 |
| Hinterlang (2021) | `SVAR/ANN` 经验环境、RL 规则、历史反事实、DSGE 稳健性 | 核心结构已覆盖，且扩展更多：不仅有经验环境 direct/transfer 区分，还有 `PPO` 双版本、人工/反推双福利、长期随机与交叉迁移 |
| “比他更多更好” | 不只复刻其样本内损失比较，而要提供更完整稳健性框架 | 已基本做到“更多”；“更好”体现在比较维度更全，但并非所有模型上 RL 都优于 baseline，这一点需要如实表述 |

## 5. 当前结论

| 主题 | 结论 |
|---|---|
| 总体结论 | 项目已不能简单表述为“RL 全面优于传统规则”，更准确表述应是：RL 在不同环境与福利口径下能学到有竞争力规则，但优势依赖环境、损失函数与迁移目标 |
| benchmark vs empirical | 二者已被严格区分：benchmark 用于规范基准与算法矩阵，经验环境用于历史反事实与经验比较 |
| Lucas critique | 已在经验环境、反推福利与外部模型模块中持续保留为核心边界 |
| 外部模型 | `pyfrbus` 原 baseline 对多数迁移规则很强；但在 `pyfrbus` 原生局部调优下，已找到优于 baseline 的 tuned linear rule |

## 6. 除写作外是否还有应做事项

| 优先级 | 事项 | 是否“基本工作”必须 |
|---|---|---|
| 高 | 无。按开题报告与当前增强轮主线，核心实验已基本齐备 | 否 |
| 中 | 若坚持“外部稳健性要明显超过 Hinterlang”，可继续补 `US_FRB03`、`US_CPS10`、`US_RA07` 或再加 1–2 个 NK 备选模型 | 可选增强 |
| 中 | 将 `pyfrbus` 上优于 baseline 的 tuned rule 纳入主比较叙述 | 可选增强 |
| 低 | 围绕 `SVAR revealed direct` 或 `pyfrbus native` 做进一步局部优化 | 可选增强 |

## 7. 交付说明

| 项目 | 路径 |
|---|---|
| 阶段总总结 | `outputs/phase10/phase10_summary.md` |
| 全量 case 清单 | `outputs/phase10/case_inventory/phase10_case_inventory.csv` |
| 外部模型汇总 | `outputs/phase10/external_model_robustness/all_external_summary.csv` |
| 反推福利汇总 | `outputs/phase10/revealed_welfare/revealed_welfare_summary.md` |

一句话概括：截至目前，除论文写作与少量可选增强外，本文的核心研究与实验工作已基本完成。
