# MonetaryRL 风格与约定

- 用户偏好：默认中文输出；总结、计划、任务、长内容优先使用表格整理。
- 文档层级权威顺序：`Thesis Proposal.docx` > `ROADMAP.md` > 各 phase 文档与 handoff。
- 工作方式：先 benchmark，后扩展；未完成 benchmark 对照前，不跳到复杂非线性或约束扩展。
- ANN 调优顺序已经冻结：先在相同输入集下调 MLP，再换更稳训练策略，再考虑增加 1 期滞后，最后才考虑 RNN/LSTM/GRU。
- 代码风格（基于现有 Python 源码）：使用类型标注、`@dataclass` 配置类、类与方法命名清晰，配置多由 JSON 读取；实现偏简洁直接，少注释，`numpy` 为核心数值依赖。
- 研究约定：理论 benchmark 与经验环境严格区分；理论、RL、反事实比较必须共用同一损失定义；经验 Taylor rule 是外部比较规则而不是 benchmark 内生部分。