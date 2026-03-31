# MonetaryRL 任务完成检查

- 若任务涉及论文主线推进，先核对 `Thesis Proposal.docx`、`ROADMAP.md` 和 `docs/agent_handoff.md` 是否与当前计划一致。
- 若任务涉及 benchmark 或 RL，对照必须保持：相同状态定义、相同损失函数、相同冲击设定。
- 若任务涉及 ANN，必须遵守既定调优顺序，不得直接跳到 RNN。
- 若任务涉及经验 Taylor rule，对照时必须先转换到与 benchmark gap 状态一致的形式。
- 完成实验后应保存配置、随机种子、结果摘要和输出文件到对应 `outputs/phase*`。
- 当前仓库未发现正式测试/格式化流水线；若无新增说明，至少需要重跑相关 phase 脚本或相关评估脚本验证结果。