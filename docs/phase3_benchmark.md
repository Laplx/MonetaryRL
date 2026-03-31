# Phase 3 基准 LQ 模型说明

## 0. 目标

`Phase 3` 的目标不是求解最优控制，而是把 `Phase 0/1` 冻结的理论 benchmark 变成一个可运行的模型层。

## 1. 本阶段完成内容

| 内容 | 说明 |
|---|---|
| benchmark 配置文件 | `src/monetary_rl/config/benchmark_lq.json` |
| benchmark 模型类 | `src/monetary_rl/models/lq_benchmark.py` |
| 构建脚本 | `scripts/phase3_build_benchmark.py` |
| 输出摘要 | `outputs/phase3/benchmark_summary.md` |

## 2. benchmark 的定位

| 对象 | 性质 |
|---|---|
| 理论 LQ benchmark | 本文规范分析的可解基准 |
| 经验 SVAR 环境 | 基于数据估计的线性环境 |

这两者相关，但不相同。`Phase 3` 明确走第一条线。

## 3. 当前校准原则

| 项目 | 当前做法 |
|---|---|
| 动态系数 | 使用稳定、可解释的 stylized 系数 |
| 冲击尺度 | 用 `Phase 2` 的 SVAR 残差量级锚定 |
| 目标通胀与中性利率 | 先固定为 `2%` |
| 损失函数权重 | 先给出基准值，后续可在 `Phase 4-9` 做稳健性分析 |

## 4. 为什么这样做

如果直接把经验 SVAR 当作理论 benchmark，会把“规范 benchmark”和“经验环境”混在一起。当前做法是先让 benchmark 成为一个干净、三维、可解的 LQ 系统，之后再在 `Phase 4` 上求解，在 `Phase 8` 中与经验规则和历史政策比较。

## 5. 后续衔接

| 下一阶段 | 衔接点 |
|---|---|
| Phase 4 | 对本 benchmark 求 Riccati 解和理论最优反馈规则 |
| Phase 5 | 用完全相同的 benchmark 环境训练 RL |
| ANN 调优并行待办 | 不阻塞当前 benchmark 线 |

---

一句话总结：

`Phase 3` 的 benchmark 已被实现为一个独立、可运行、可求解、可与 RL 一一对照的 LQ 模型层。
