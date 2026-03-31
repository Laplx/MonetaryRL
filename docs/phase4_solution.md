# Phase 4 经典最优控制求解说明

## 0. 目标

`Phase 4` 的目标是在 `Phase 3` 的 LQ benchmark 上求出无限期贴现最优控制解。

## 1. 本阶段完成内容

| 内容 | 说明 |
|---|---|
| Riccati 求解器 | `src/monetary_rl/solvers/riccati.py` |
| solver 导出 | `src/monetary_rl/solvers/__init__.py` |
| 求解脚本 | `scripts/phase4_solve_lq.py` |
| 说明文档 | 本文件 |
| 输出结果 | `outputs/phase4/` |

## 2. 求解问题

本阶段解决的是带状态-动作交叉项的折现离散 LQ 问题：

$$
V(s)=\min_a \left\{\ell(s,a)+\beta \mathbb{E}[V(s')\mid s,a]\right\}
$$

其中

$$
\ell(s,a)=s^\top Q s + 2 s^\top N a + a^\top R a
$$

## 3. 方法

采用广义离散代数 Riccati 方程。由于问题带折现因子 $\beta$，求解时将系统做

$$
\tilde{A}=\sqrt{\beta}A,\quad \tilde{B}=\sqrt{\beta}B
$$

的等价变换，再用标准离散 Riccati 求解器得到 $P$。

## 4. 输出对象

| 对象 | 含义 |
|---|---|
| $P$ | 值函数二次型矩阵 |
| $F$ | 在 $a_t=-F s_t$ 中的反馈矩阵 |
| $K$ | 等价写法 $a_t=K s_t$ 中的策略矩阵，$K=-F$ |
| $A_{cl}$ | 闭环矩阵 |
| 特征值 | 用于稳定性检查 |
| 平稳协方差 | 用于长期波动分析 |

## 5. 与后续阶段的关系

| 下一阶段 | 作用 |
|---|---|
| Phase 5 | RL 要在同一个 benchmark 上训练 |
| Phase 6 | RL 与理论最优将逐项对照 |

---

一句话总结：

`Phase 4` 把 `Phase 3` 的理论 benchmark 从“可运行模型”推进成了“可解析求解的最优控制基准”。
