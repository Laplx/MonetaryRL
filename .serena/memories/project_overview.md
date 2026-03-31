# MonetaryRL 项目概览

- 项目目标：在随机宏观货币政策模型中，系统比较经典最优控制与强化学习在央行利率规则求解中的表现。
- 核心定位：RL 是最优控制问题的数值求解器，不是多主体学习或 L3 学习主体路线。
- 最终权威：`Thesis Proposal.docx`。
- 关键参照：`Hinterlang and Tänzer (2021)`，尤其用于经验环境（SVAR + ANN）设计。
- 当前阶段：Phase 0-4 完成，Phase 5 完成基础版，下一主线阶段是 Phase 6 benchmark 对照实验。
- 已知关键结果：经验 Taylor rule 长期通胀响应约 1.4769；Phase 4 Riccati 最优反馈矩阵 `[[1.089101, 1.078439, 0.434052]]`；PPO baseline 相对零政策约改善 10%，但距离 Riccati 最优仍远。
- 重要边界：Phase 4 最优规则来自理论 LQ benchmark，而非经验 SVAR 环境；benchmark 不内生包含经验 Taylor rule；经验 Taylor rule 需要作为外部规则代入 benchmark 环境做对照；写作必须说明 Lucas critique。
- 项目结构：`docs/` 放研究设计与理论/数据文档；`scripts/` 放 phase 入口脚本；`src/monetary_rl/` 下分为 `config/`, `models/`, `solvers/`, `envs/`, `agents/`；`data/processed/` 放处理后数据；`outputs/phase2-5/` 放阶段性结果。