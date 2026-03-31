# Phase 5 RL 基线说明

## 0. 目标

在与 `Phase 4` 理论最优解完全相同的 LQ benchmark 上训练一个 PPO 连续控制基线，为 `Phase 6` 的系统对照实验做准备。

## 1. 本阶段完成内容

| 内容 | 说明 |
|---|---|
| RL 环境 | `src/monetary_rl/envs/benchmark_env.py` |
| PPO 实现 | `src/monetary_rl/agents/ppo.py` |
| PPO 配置 | `src/monetary_rl/config/benchmark_ppo.json` |
| 训练脚本 | `scripts/phase5_train_ppo.py` |
| 结果摘要 | `outputs/phase5/ppo_baseline_summary.md` |

## 2. 设计原则

| 原则 | 说明 |
|---|---|
| 同一 benchmark | 必须与 `Phase 4` 共用完全相同的环境和损失函数 |
| 连续动作 | 利率为连续控制变量 |
| 最小实现 | 先得到稳定可运行的 PPO baseline，再做更复杂比较 |
| 可交接 | 代码结构要便于下一位 agent 继续扩展 |

## 3. 与后续阶段关系

| 下一阶段 | 衔接 |
|---|---|
| Phase 6 | 系统比较 PPO 与 Riccati 最优解 |
| ANN 调优 | 与当前 benchmark PPO 主线独立 |

---

一句话总结：

`Phase 5` 的任务是先让 RL 在理论 benchmark 上跑起来，并产生一个可比较的 PPO 基线。
