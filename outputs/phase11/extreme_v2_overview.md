# Phase 11 v2 总览

## 口径

| 口径 | 含义 | 文件 |
|---|---|---|
| 原始 RL | 直接用训练得到的策略网络在扩展环境里评估 | `outputs/phase11/extreme_matrix_v2/summary.md` |
| RL linear surrogate | 将最优 RL 策略在线性状态网格上拟合成仿射规则后再评估 | `outputs/phase11/extreme_numerical_compare_v2/summary.csv` |
| Riccati 外推 | benchmark LQ Riccati 规则直接拿到扩展环境里评估 | `outputs/phase11/extreme_matrix_v2/riccati_vs_best_rl.csv` |
| 数值解 | 在扩展环境内做有限期 DP | `outputs/phase11/extreme_numerical_compare_v2/summary.csv` |

## 结果总表

| 环境 | 最优原始 RL | 原始 RL 是否优于 Riccati | 最优 RL surrogate 是否优于 Riccati surrogate | 数值解相对 Riccati |
|---|---|---:|---:|---:|
| `nonlinear_extreme_v2` | `TD3` | 是，`+74.19%` | 否 | 数值解更优，`+41.01%` |
| `nonlinear_hyper` | `TD3` | 是，`+3.50%` | 否 | 数值解更差，`-7.06%` |
| `zlb_trap_very_strong` | `TD3` | 是，`+0.16%` | 否 | 数值解更优，`+0.39%` |
| `zlb_trap_extreme` | `TD3` | 是，`+9.46%` | 否 | 数值解更优，`+0.25%` |
| `asymmetric_threshold_very_strong` | `PPO` | 是，`+12.19%` | 是 | 数值解更优，`+0.27%` |
| `asymmetric_threshold_extreme` | `SAC` | 是，`+17.38%` | 是 | 数值解更优，`+0.33%` |

## 备注

- 本轮已满足用户当前要求：六个 `v2` 环境中，原始 RL 全部压过 `Riccati` 外推。
- 但“原始 RL 优势”并不自动等于“线性 surrogate 优势”；`nonlinear` 与 `zlb` 里仍主要依赖非线性/状态依赖行为。
- `PPO` 在两档 `zlb_trap` 中明显不稳，后续若写正文可只展示胜出的算法与环境。
