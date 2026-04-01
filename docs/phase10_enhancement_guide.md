# Phase 10 增强方案执行与交接指引

## 0. 文档定位

| 项目 | 说明 |
|---|---|
| 作用 | 固定下一轮增强工作的执行口径，供新 agent 直接接手 |
| 权威顺序 | `Thesis Proposal.docx` > `ROADMAP.md` > 本文档 > 既有 Phase 8/9 文档 |
| 当前更新时间 | `2026-04-01` |
| 当前阶段基础 | `Phase 8` 与 `Phase 9` 已完成；本轮是在此基础上的增强轮，而不是回滚重做 |
| 与写作关系 | 本轮完成后再进入大规模正文组装；现阶段不应直接跳到纯写作模式 |

## 1. 当前真实状态与本轮目标

| 项目 | 当前状态 |
|---|---|
| 已完成经验转移结果 | `Phase 8` 已完成 `SVAR` 环境下的 historical actual / empirical Taylor / Riccati / benchmark-transfer RL 比较 |
| 已完成 ANN 补充 | `Phase 9` 已完成 `ANN` 调优、ANN 补充反事实与本地 model uncertainty 汇总 |
| 当前缺口 | 还没有在 `SVAR` 和 `ANN` 环境中直接训练 `PPO/TD3/SAC`，也没有把“直接经验训练规则”带入统一反事实与跨模型稳健性框架 |
| 本轮核心目标 | 新增 `SVAR direct` 与 `ANN direct` 经验训练规则，并与现有 `benchmark transfer` 规则统一做历史反事实、长期随机、交叉迁移、外部模型与反推福利比较 |

## 2. 本轮冻结边界

| 主题 | 冻结要求 |
|---|---|
| benchmark 与经验环境 | 必须严格区分；`Riccati reference` 仍然只来自理论 LQ benchmark |
| empirical Taylor | 始终是外部经验规则，不是 benchmark 或经验环境的内生解 |
| Phase 8/9 既有结果 | 不覆盖、不改写，作为 transfer baseline 保留 |
| 新增 RL 规则 | 必须明确标注是在 `SVAR` 或 `ANN` 环境中直接训练得到 |
| PPO 双版本 | 只对 `PPO` 保留 `linear-policy` 与 `nonlinear-policy` 两版；`TD3/SAC` 维持标准策略参数化 |
| 外部模型扩展 | 写作为 `model uncertainty robustness extension`，不是机械复刻 `Hinterlang` |
| 方法边界 | 必须明确写 `Lucas critique` |
| 损失函数 | 先沿用当前主线损失；`反推福利模块` 作为附加报告口径，不在第一轮替代主损失训练 |
| 后续调优顺序 | 若 `SVAR` 下历史实际政策仍优于固定反馈规则，则将 `SVAR direct SAC/TD3` 增加 seeds / 训练预算、单独调 `state_scale`/`horizon`/平滑惩罚、以及 `revealed welfare` 附加训练口径，统一后置到外部模型模块完成之后再做 |
| 当前执行状态 | `revealed welfare` 口径下的 `SVAR/ANN` 重新训练、统一反事实、长期随机、交叉迁移与 `pyfrbus` 外部检验已完成；新增完成 `US_SW07`、`US_CCTW10`、`US_KS15`、`NK_CW09` 的 `Dynare/MMB` 外部批次；`US_FRB03`、`US_CPS10`、`US_RA07` 仍受 legacy 兼容问题阻塞 |

## 3. PPO 双版本口径

| 版本 | 定义 | 用途 |
|---|---|---|
| `linear-policy PPO` | 在经验环境中直接训练、但策略头保持线性反馈形式的 `PPO` | 检查 `PPO` 在受限表达力下的表现 |
| `nonlinear-policy PPO` | 在经验环境中直接训练、使用标准非线性策略网络的 `PPO` | 检查 `PPO` 是否因表达力受限而掉队 |

补充说明：

| 项目 | 当前已知实现 |
|---|---|
| `PPO` 默认网络 | `src/monetary_rl/agents/ppo.py` 中默认 `hidden_size=64`；支持 `linear_policy=True` |
| `SAC` 默认网络 | `src/monetary_rl/agents/sac.py` 中 actor / critic 默认 `hidden_size=64` |
| `TD3` 默认网络 | `src/monetary_rl/agents/td3.py` 中 actor / critic 默认 `hidden_size=64` |
| `ANN` 当前最优规模 | `Phase 9` 最优 ANN 两个方程的 `hidden_layer_sizes` 都是 `(3,)` |

## 4. 六步增强方案

### 4.1 总览

| 步骤 | 任务 | 核心产物 |
|---|---|---|
| 1 | 在 `SVAR` 环境直接训练 `PPO/TD3/SAC` | `SVAR direct` 训练结果 |
| 2 | 在 `ANN` 环境直接训练 `PPO/TD3/SAC` | `ANN direct` 训练结果 |
| 3 | 对 `PPO` 单独补跑 `linear-policy` 与 `nonlinear-policy` 两版 | `PPO` 表达力对照 |
| 4 | 统一做三类规则的历史反事实 | `benchmark transfer / SVAR direct / ANN direct` 主表主图 |
| 5 | 统一做长期随机评估与交叉迁移 | 稳定随机表图与 cross-transfer 结果 |
| 6 | 将优选规则接到外部模型接口 | 外部模型稳健性结果 |
| 7 | 加入 `反推福利模块` | 第二套福利口径与排序变化表 |

### 4.2 详细要求

| 步骤 | 具体要求 | 必须输出 |
|---|---|---|
| 1 | 在 `SVAR` 环境中从头训练 `PPO/TD3/SAC`；其中 `PPO` 先按现有主设定训练一版 direct rule，并保留训练日志、checkpoint、近似系数 | `SVAR direct` 训练日志、政策系数、评估输入 |
| 2 | 在 `ANN` 环境中从头训练 `PPO/TD3/SAC`；若出现动态失稳，必须单列记录，不可静默丢弃 | `ANN direct` 训练日志、政策系数、稳定性记录 |
| 3 | 对 `PPO` 单独在 `SVAR` 与 `ANN` 两环境中补跑 `linear-policy` 与 `nonlinear-policy` 两版；`TD3/SAC` 不做此拆分 | `PPO` 双版本对照表与排序变化说明 |
| 4 | 冻结统一比较集合：`historical actual`、`empirical Taylor`、`Riccati reference`、`linear policy search transfer`、`benchmark-transfer PPO/TD3/SAC`、`SVAR direct PPO/TD3/SAC`、`ANN direct PPO/TD3/SAC`；对三类规则统一做历史反事实 | 历史反事实表图、三类规则总表 |
| 5 | 在统一评价器里做 `长期稳定随机评估` 与 `交叉迁移`：至少比较 `benchmark transfer -> SVAR/ANN`、`SVAR direct -> ANN`、`ANN direct -> SVAR` | 随机评估表图、cross-transfer 汇总 |
| 6 | 将优选规则集合迁移到外部模型接口；优先顺序固定为 `pyfrbus`、`US_FRB03`、`US_SW07`、`US_CCTW10`、`US_CPS10`、`US_KS15`、`US_RA07`，备选 `NK_CW09`、`NK_CFP10`、`NK_GLSV07`、`NK_GK13` | 外部模型稳健性表、模型映射说明 |
| 7 | 加入 `反推福利模块`：基于 `SVAR + empirical Taylor` 反推权重，并用该口径重评 `benchmark transfer / SVAR direct / ANN direct` | 反推权重表、重评分表、排序变化表 |

## 5. 反推福利模块

### 5.1 目标

| 项目 | 内容 |
|---|---|
| 模块名称 | `反推福利模块` / `revealed welfare module` |
| 目的 | 基于 `SVAR` 转移方程与经验 Taylor rule，反推出一组与经验政策更一致的福利权重，作为第二套报告口径 |
| 主张边界 | 它是附加福利标尺，不自动替代当前主线福利；除非后续结果非常稳健，否则不作为第一轮训练目标 |

### 5.2 建议实现顺序

| 顺序 | 任务 | 说明 |
|---|---|---|
| 1 | 设定归一化 | 固定通胀项系数为 `1`，反推 `output gap` 与 `rate smoothing` 权重 |
| 2 | 使用 `Phase 2` 经验 Taylor 系数与 `SVAR` 状态转移 | 将经验 Taylor 规则视为某个二次损失下的近似最优线性反馈 |
| 3 | 采用逆向 `LQ` / 最小距离估计 | 若无法解析反解，则最小化“模型 implied feedback 与经验 Taylor feedback 的距离” |
| 4 | 加入约束 | 权重非负、闭环稳定、反馈系数在经验置信区间内 |
| 5 | 生成第二套评分 | 用这组 `revealed welfare weights` 重算 `Phase 8/9` 与本轮增强规则的福利排序 |
| 6 | 判断是否值得二次训练 | 只有当排序变化显著且解释清晰时，才考虑把该损失作为附录中的附加训练实验 |

### 5.3 报告方式

| 项目 | 要求 |
|---|---|
| 主文定位 | 作为“经验偏好识别”补充分析，不替代 benchmark/LQ welfare |
| 表格 | 至少给出反推权重表、基线福利排序、反推福利排序、排序变化表 |
| 图形 | 建议给出权重敏感性图或排序变化条形图 |
| 写作提醒 | 必须说明这是一种依赖经验 Taylor 与固定 `SVAR` 转移的识别练习，仍受 `Lucas critique` 限制 |

## 6. 统一评价框架

| 维度 | 必须内容 |
|---|---|
| 历史反事实 | 使用历史 shock 或历史观测初始化，比较 `historical actual` 与各规则路径 |
| 长期随机 | 在稳定随机模拟下比较长期平均折现损失、波动、clip rate、失稳率 |
| 规则来源区分 | 必须区分 `transfer` 与 `direct-trained` |
| 环境区分 | 必须区分 `SVAR evaluation` 与 `ANN evaluation` |
| 福利口径区分 | 必须区分 `baseline welfare` 与 `revealed welfare` |

## 7. 推荐目录与产物规划

| 路径 | 建议用途 |
|---|---|
| `docs/phase10_enhancement_guide.md` | 本轮总指引 |
| `outputs/phase10/svar_direct/` | `SVAR direct` 训练与评估结果 |
| `outputs/phase10/ann_direct/` | `ANN direct` 训练与评估结果 |
| `outputs/phase10/ppo_policy_variants/` | `PPO linear-policy` 与 `PPO nonlinear-policy` 双版本对照 |
| `outputs/phase10/revealed_welfare/` | 反推福利模块输出 |
| `outputs/phase10/revealed_policy_training/` | `revealed welfare` 口径下 `SVAR/ANN` 的 `PPO/TD3/SAC` 重新训练结果 |
| `outputs/phase10/revealed_policy_eval/` | `revealed welfare` 新规则的历史反事实、长期随机与交叉迁移结果 |
| `outputs/phase10/external_model_robustness/` | 外部模型稳健性输出 |
| `outputs/phase10/phase10_summary.md` | 本轮总总结 |

## 8. 脚本复用与建议新增入口

| 类型 | 路径 | 用法 |
|---|---|---|
| 现有可复用 | `scripts/phase8_empirical_counterfactual.py` | 复用 `policy_registry`、历史反事实、随机评估、作图框架 |
| 现有可复用 | `scripts/phase9_ann_model_uncertainty.py` | 复用 `ANN` 环境构造、local uncertainty、phase9 registry 逻辑 |
| 建议新增 | `scripts/phase10_train_empirical_rl.py` | 负责 `SVAR/ANN` 直接训练 |
| 建议新增 | `scripts/phase10_ppo_policy_variants.py` | 负责 `PPO linear-policy` 与 `PPO nonlinear-policy` 双版本 |
| 建议新增 | `scripts/phase10_counterfactual_eval.py` | 负责 `historical + stochastic` 统一评估 |
| 建议新增 | `scripts/phase10_revealed_welfare.py` | 负责反推福利权重与重评分 |
| 建议新增 | `scripts/phase10_external_model_robustness.py` | 负责外部模型映射与稳健性汇总 |

## 9. 外部模型部分的现状与用户接口

| 项目 | 当前现状 |
|---|---|
| 主入口目录 | `external_models/` |
| 已有压缩包 | `external_models/mmb-rep-master.zip`、`external_models/pyfrbus.zip` |
| 已有解压内容 | `external_models/mmb_extracted/`、`external_models/frbus_extracted/` |
| 已有筛选摘要 | `external_models/screening_summary.md` |
| `mmb-electron` | 只有快捷方式 `external_models/mmb-electron.lnk`，尚无额外下载 manifest |

| 用户若继续补模型 | 建议放置位置 |
|---|---|
| 新下载的压缩包 / PDF / 说明材料 | `external_models/` 根目录 |
| 新解压的 MMB 模型 | `external_models/mmb_extracted/` |
| 新解压的 FRB-US 或其他美国模型 | `external_models/frbus_extracted/` |

| 用户已接受的优先外部模型 | 说明 |
|---|---|
| `pyfrbus` | 美国政策模型，高优先级 |
| `US_FRB03` | 直接的美国模型候选 |
| `US_SW07` | 常见美国 DSGE 参考系 |
| `US_CCTW10` | 美国货币模型候选 |
| `US_CPS10` | 美国货币模型候选 |
| `US_KS15` | 映射较清晰的美国候选 |
| `US_RA07` | 美国候选模型 |

| 备选 NK 基准 | 说明 |
|---|---|
| `NK_CW09` | NK fallback |
| `NK_CFP10` | NK fallback |
| `NK_GLSV07` | NK fallback |
| `NK_GK13` | NK fallback |

## 10. 写作与交接时必须重复强调的话

| 主题 | 必须表述 |
|---|---|
| 与既有结果关系 | `Phase 8/9` 结果是 `benchmark transfer` baseline，本轮新增的是 `SVAR direct` 与 `ANN direct` |
| 研究贡献定位 | 不是证明 RL 普遍碾压传统方法，而是比较不同求解器与规则在不同环境、不同福利口径、不同模型族中的表现 |
| 相似工作关系 | 参考 `Hinterlang`，但本项目强调三类规则比较、`PPO` 双版本、双经验环境 direct training、双福利口径与更清楚的 transfer/direct 区分 |
| 方法边界 | 必须明确 `Lucas critique`，外部模型迁移只是部分缓解，不是彻底解决 |

## 11. 新 agent 接手时的最低检查清单

| 检查项 | 要求 |
|---|---|
| 已读 `docs/agent_handoff.md` | 必须 |
| 已读 `ROADMAP.md` | 必须 |
| 已读 `outputs/phase8/phase8_summary.md` | 必须 |
| 已读 `outputs/phase9/phase9_summary.md` | 必须 |
| 已读 `docs/phase10_enhancement_guide.md` | 必须 |
| 已理解 `transfer` 与 `direct-trained` 的区别 | 必须 |
| 已理解只有 `PPO` 做双版本 | 必须 |
| 已确认 `反推福利模块` 先做报告口径，不自动替代训练目标 | 必须 |
