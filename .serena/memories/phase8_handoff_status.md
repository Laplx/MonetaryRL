# Phase 8 接手状态（2026-03-31）

- Phase 6 已完成：benchmark 统一比较协议、squashed Gaussian PPO 修复、linear policy search 基线、empirical Taylor 外部规则 gap-form benchmark 对照均已完成。
- Phase 7 已完成：环境矩阵为 1 benchmark + 3 nonlinear + 3 ZLB/ELB-tightness + 3 asymmetric；算法矩阵为 PPO/TD3/SAC；主脚本为 `scripts/phase7_matrix_experiments.py`；主结果位于 `outputs/phase7/matrix/`。
- 当前多 seed 固定预算主矩阵下：SAC 在 benchmark、nonlinear_*、asymmetric_* 中最强；TD3 在 zlb_medium 和 zlb_strong 中最强；PPO 不是全局最优。
- 重要边界：benchmark 与经验环境严格分开；Riccati 最优规则来自理论 LQ benchmark；empirical Taylor 是外部规则；ZLB 三档应写作 ZLB/ELB-tightness tiers；Lucas critique 必须写。
- ANN 调优继续延后，不作为当前主线。
- 下一阶段是 Phase 8：先用经验 SVAR 环境完成历史政策、经验 Taylor、Riccati reference、best RL rules 的反事实福利比较，再决定是否加入 ANN 作为补充结果。
- 当前关键交接文档：`docs/agent_handoff.md`、`docs/phase8_execution_guide.md`、`docs/writing_figure_table_plan.md`。