# `REPORT_v2` 后续图表与可视化准备清单

下阶段图表准备应严格服务于论文主线，而不是把所有已有图片机械堆叠出来。建议优先围绕三类问题组织：一是 benchmark 与扩展环境的比较逻辑；二是 RL 规则的结构可解释性；三是经验反事实与经济学含义。

| 优先级 | 图表 | 目的 | 直接数据来源 |
| --- | --- | --- | --- |
| 高 | Phase 7 RL 平均损失热图 | 一图说明不同算法在 `10` 个环境中的排序变化 | `outputs/phase7/matrix/rl_summary.csv` |
| 高 | 扭曲强度曲线图 | 展示 nonlinear / ZLB / asymmetric 强化时损失如何变化 | `outputs/phase7/matrix/all_policy_summary.csv` |
| 高 | benchmark 与各组多 seed 箱线图 | 体现算法稳定性，而不只看单次最优值 | `outputs/phase7/matrix/training_logs.csv` 或现成 `plots/` |
| 高 | 历史实际与反事实路径图 | 连接经验环境，展示政策路径、通胀、产出缺口的动态差异 | `outputs/phase8/historical_counterfactual_paths.csv` |
| 高 | 历史/随机福利分解图 | 展示各规则在福利损失、波动、平滑项上的差异 | `outputs/phase8/historical_welfare_summary.csv`、`outputs/phase8/stochastic_welfare_summary.csv` |
| 中 | 最佳 RL 近似系数热图 | 解释不同环境下规则系数为何变化 | `outputs/phase7/matrix/policy_coefficients.csv` |
| 中 | ZLB/ELB 约束命中率图 | 解释 tighter lower-bound 环境下算法排序变化 | `outputs/phase7/matrix/plots/zlb_clip_rates.png` 或原始汇总表 |
| 中 | benchmark 全规则对照图 | 在正文中保留理论 benchmark 的规范地位 | `outputs/phase7/matrix/all_policy_summary.csv` |
| 中 | transfer vs direct-trained 对照图 | 为后续经验环境扩展预留接口 | 后续 direct-trained 输出 |

## 一、最值得新增或强化的可视化

### 1. RL 规则的“切片图”或“响应面图”

目前最需要补的不是更多总损失表，而是规则本身的形状解释图。建议在 `benchmark`、`nonlinear_strong`、`zlb_strong`、`asymmetric_strong` 四个代表环境下，固定一个或两个状态变量，只画政策利率对通胀缺口、产出缺口的局部响应切片。这样可以直接回答两个问题：第一，RL 学到的规则是否仍近似 Taylor 型；第二，不同环境下非线性或约束是否改变了局部斜率和弯折位置。

若后续能稳定生成二维响应面图，可以借鉴 `Hinterlang and Tänzer (2021)` 的 partial dependence 展示思路，但表达必须改写成项目自己的口径：我们不是为了证明“黑箱 ANN 一定更优”，而是为了说明“RL 规则在何种状态区域出现更强响应、平滑或约束扭曲”。

### 2. 环境异质性的“结构图”

建议把 `phase7` 的结果做成两层图：

- 第一层是总览图，如 heatmap 或 win-count，回答“谁在什么环境里更好”。
- 第二层是机制图，如 distortion-strength curves、ZLB clip-rate 图，回答“为什么排序发生变化”。

这样能避免正文只剩下大量表格数字，也能把“环境依赖性”这一核心发现讲清楚。

### 3. 经验反事实的“三联图”

经验部分最适合采用三联图：`inflation`、`output gap`、`policy rate` 各一幅，把历史实际路径与若干关键规则的反事实路径放在一起。这里不建议一次放太多规则，正文优先保留：

- `historical_actual_policy`
- `empirical_taylor_rule`
- `riccati_reference`
- `sac_benchmark_transfer`

其余规则可以放附录。这样图面更干净，也更符合当前阶段的核心叙事：经验规则、理论规则、最优迁移 RL 之间到底差在哪。

## 二、建议重点准备的经济学解释

### 1. 为什么线性参考规则目前仍然很强

这应当被解释为：当前引入的 nonlinear、asymmetric 与 lower-bound 扭曲是“有效但有限”的，而不是已经把最优反馈结构完全改写。换言之，基准模型所刻画的稳定通胀—产出权衡逻辑仍然主导大部分状态空间，因此 Riccati reference 和 linear policy search 仍保持竞争力。这个解释比“RL 失败了”更准确。

### 2. 为什么 tighter lower-bound 环境更偏向 TD3 / SAC

这里建议把经济学解释和算法解释分开写。经济学上，可以说有效下界会改变最优反应的局部形状，使政策在衰退区间更依赖状态区域划分和边际调整。算法上，则表现为 off-policy 方法在这类连续控制与约束并存的问题上更有优势。正文中不要把算法优势直接解释成经济结构结论，但可以把两者并列展示。

### 3. 为什么经验环境里 transfer 规则没有直接胜出

这一点非常关键。建议明确写成：benchmark 中训练得到的规则迁入经验 `SVAR` 环境后，并未普遍优于经验 Taylor 或历史实际政策，这说明规则优劣具有环境依赖性，也说明 transfer-based 结果只能视为过渡证据。这里必须同时引出 Lucas critique：政策规则改变后，私人部门行为与状态转移本身可能随之变化，因此固定转移方程下的反事实比较有方法边界。

### 4. 为什么规则可解释性仍然是本文优势

从当前近似系数看，较优 RL 规则并不是完全无结构的黑箱；它们大多仍表现为对通胀缺口、产出缺口的正向响应，并带有一定利率平滑。因此，后续图表应强调“可解释的非线性修正”而不是“完全替代 Taylor rule”。这会让论文更稳，也更容易与开题报告衔接。

## 三、推荐的出图顺序

建议按下面顺序准备图表，而不是同时分散推进：

1. `phase7` 总览：heatmap、strength curves、seed boxplots  
2. `phase7` 结构解释：最佳 RL 系数热图、规则切片图、ZLB 局部图  
3. `phase8` 经验结果：历史路径图、福利分解图  
4. direct-trained 经验结果完成后，再补 transfer vs direct-trained 对照图  

这样安排的好处是：先把现有最成熟的证据链画完整，再为后续 direct-trained 扩展预留接口，避免重复返工。
