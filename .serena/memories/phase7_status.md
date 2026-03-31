# Phase 7 当前状态

- 已完成 benchmark RL 强化：新增 tuned PPO、SAC、TD3 与线性策略搜索的统一对照，输出在 `outputs/phase7/benchmark_rl/`。
- benchmark 结果：Riccati 15.9407；线性策略搜索 16.1520；tuned PPO 16.1662；TD3 17.1134；SAC 18.0677。说明 benchmark 主线已经很强，PPO 不再是主要瓶颈。
- 已完成非线性 Phillips 扩展，输出在 `outputs/phase7/nonlinear/`。当前校准下，线性 Riccati 外推规则仍然最强，线性策略搜索和 PPO 非常接近。
- 已完成 ZLB 扩展，输出在 `outputs/phase7/zlb/`。当前 benchmark 约束设定下，Riccati 规则经过 ZLB 环境执行仍最强，TD3/SAC 在该扩展里优于 PPO。
- ANN 调优按用户要求继续延后，不作为当前主线。
- 重要边界：非线性 Phillips 环境使用带符号的二次非线性项，并在环境中加入状态爆炸保护；ZLB 结果不是受约束解析最优，而是各规则在约束环境下的可执行比较。