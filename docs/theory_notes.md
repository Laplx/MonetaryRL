# Phase 1 理论笔记

## 0. 文档定位

| 项目 | 说明 |
|---|---|
| 阶段 | `Phase 1` |
| 文档用途 | 将本文需要用到的理论框架系统化，供后续正文、附录、实验实现和结果解释直接使用。 |
| 上游约束 | 完全遵守 `docs/model_spec.md` 中冻结的 benchmark 规格。 |
| 最终权威 | `Thesis Proposal.docx`。如与本文件冲突，以开题报告为准。 |
| 目标 | 回答“本文为什么可以这样建模、为什么可以这样比较、为什么 RL 在这里是一个合理的最优控制数值方法”。 |

## 1. 本文理论主线

### 1.1 一句话概括

本文的理论主线是：把中央银行货币政策问题写成一个带随机状态转移的动态最优控制问题，在可解析的线性二次型 benchmark 下用 Bellman 方程和 Riccati 方程得到理论最优反馈规则，再把强化学习视为对该动态规划问题的近似数值求解器，并将其推广到非线性和约束环境。

### 1.2 理论结构总览

| 层次 | 核心问题 | 本文作用 |
|---|---|---|
| LQ 动态最优控制 | 可解析 benchmark 的理论基准是什么 | 给出真值或准真值对照 |
| Bellman 动态规划 | 央行问题如何写成递归优化问题 | 连接经典控制与 RL |
| Riccati 方程 | 线性二次型下为何能得到线性反馈规则 | 给出可比较的解析结构 |
| RL-MDP 映射 | 为什么 RL 可以求解本文问题 | 赋予 RL 方法论基础 |
| 非线性与约束 | 为什么传统方法会变难 | 说明 RL 的扩展价值 |
| 经验反事实边界 | 为什么需要讨论 Lucas critique | 约束结果解释的外推范围 |

## 2. 符号与对象

### 2.1 基本符号

| 符号 | 含义 |
|---|---|
| $\pi_t$ | 当期通胀率 |
| $\pi^\ast$ | 目标通胀率 |
| $\tilde{\pi}_t$ | 通胀偏离，$\tilde{\pi}_t=\pi_t-\pi^\ast$ |
| $x_t$ | 产出缺口 |
| $i_t$ | 名义政策利率 |
| $i^\ast$ | 稳态或中性名义利率 |
| $\tilde{i}_t$ | 利率偏离，$\tilde{i}_t=i_t-i^\ast$ |
| $s_t$ | 状态向量 |
| $a_t$ | 控制变量，在本文中定义为 $a_t=\tilde{i}_t$ |
| $\ell_t$ | 单期损失 |
| $\beta$ | 折现因子 |

### 2.2 benchmark 状态

根据 `Phase 0` 冻结结果，基准状态为

$$
s_t=
\begin{bmatrix}
\tilde{\pi}_t \\
x_t \\
\tilde{i}_{t-1}
\end{bmatrix}
$$

之所以使用增广状态，是因为利率平滑项 $(\tilde{i}_t-\tilde{i}_{t-1})^2$ 使得上期利率成为当前决策的必要状态变量。

## 3. 中央银行问题的动态最优控制表示

### 3.1 随机动态系统

本文 benchmark 的状态转移为

$$
s_{t+1}=A s_t + B a_t + \Sigma \varepsilon_{t+1}
$$

其中

$$
\varepsilon_{t+1}\sim \mathcal{N}(0,I)
$$

这是一个标准离散时间随机线性状态空间系统。

### 3.2 央行损失函数

中央银行的单期损失为

$$
\ell_t=\lambda_\pi \tilde{\pi}_t^2+\lambda_x x_t^2+\lambda_i(\tilde{i}_t-\tilde{i}_{t-1})^2
$$

长期目标是最小化折现总损失：

$$
V(s_t)=\min_{\{a_{t+k}\}_{k\ge 0}}\mathbb{E}_t\left[\sum_{k=0}^{\infty}\beta^k \ell_{t+k}\right]
$$

这里的 $V(s_t)$ 是从状态 $s_t$ 出发的最优值函数。

### 3.3 经济含义

| 项目 | 含义 |
|---|---|
| $\lambda_\pi \tilde{\pi}_t^2$ | 反映稳定通胀目标 |
| $\lambda_x x_t^2$ | 反映稳定产出缺口目标 |
| $\lambda_i(\tilde{i}_t-\tilde{i}_{t-1})^2$ | 反映政策渐进调整与利率平滑偏好 |

## 4. Bellman 方程与动态规划

### 4.1 递归表示

由于系统满足马尔可夫性质，最优控制问题可以写成 Bellman 方程：

$$
V(s)=\min_a \left\{\ell(s,a)+\beta \mathbb{E}\left[V(s')\mid s,a\right]\right\}
$$

其中 $s'$ 由状态转移方程给出。

### 4.2 Bellman 方程的意义

Bellman 方程说明：从当前状态出发的最优问题，等于“当前损失”与“下一期继续按最优策略行动所产生的期望未来损失”的加总。这一递归结构既是经典动态规划的核心，也是 RL 理论的出发点。

### 4.3 与开题报告的关系

开题报告提到 HJB 方程。对于本文的离散时间设定，严格对应的是 Bellman 方程；HJB 是连续时间控制问题的对应对象。两者本质上都表达了动态最优性的递归条件。

## 5. benchmark 的 LQ 表示

### 5.1 矩阵写法

由于状态为

$$
s_t=
\begin{bmatrix}
\tilde{\pi}_t \\
x_t \\
\tilde{i}_{t-1}
\end{bmatrix}
$$

且动作为 $a_t=\tilde{i}_t$，单期损失可以写成

$$
\ell(s_t,a_t)=s_t^\top Q s_t + 2 s_t^\top N a_t + a_t^\top R a_t
$$

其中

$$
Q=
\begin{bmatrix}
\lambda_\pi & 0 & 0 \\
0 & \lambda_x & 0 \\
0 & 0 & \lambda_i
\end{bmatrix},
\quad
N=
\begin{bmatrix}
0 \\
0 \\
-\lambda_i
\end{bmatrix},
\quad
R=\lambda_i
$$

因为

$$
\lambda_i(\tilde{i}_t-\tilde{i}_{t-1})^2
=\lambda_i a_t^2 -2\lambda_i \tilde{i}_{t-1}a_t + \lambda_i \tilde{i}_{t-1}^2
$$

这使得损失函数不仅含有状态与动作的平方项，也含有一个状态-动作交叉项。

### 5.2 为什么要显式写出交叉项

很多教材直接给出 $\ell=s^\top Qs+a^\top Ra$ 的形式，但本文的利率平滑项天然会产生交叉项。后续 Riccati 方程与最优反馈规则的推导必须保留这一项，否则理论和实现会不一致。

## 6. 值函数猜测与 Riccati 方程

### 6.1 二次型值函数猜测

在线性状态转移与二次损失下，标准猜测是

$$
V(s)=s^\top P s + c
$$

其中 $P$ 是对称半正定矩阵，$c$ 是常数项。

### 6.2 将猜测代入 Bellman 方程

由

$$
s'=As+Ba+\Sigma\varepsilon'
$$

可得

$$
\mathbb{E}[V(s')\mid s,a]
=
\mathbb{E}\left[(As+Ba+\Sigma\varepsilon')^\top P(As+Ba+\Sigma\varepsilon')+c\right]
$$

利用 $\mathbb{E}[\varepsilon']=0$，得到

$$
\mathbb{E}[V(s')\mid s,a]
=
(As+Ba)^\top P(As+Ba)+\mathrm{tr}(P\Sigma\Sigma^\top)+c
$$

于是 Bellman 方程右侧关于 $a$ 的部分变为

$$
s^\top Q s + 2 s^\top N a + a^\top R a
+ \beta (As+Ba)^\top P(As+Ba)
+ \beta \mathrm{tr}(P\Sigma\Sigma^\top)
+ \beta c
$$

### 6.3 一阶条件

将上式对 $a$ 求导并令其为零：

$$
2N^\top s + 2Ra + 2\beta B^\top P(As+Ba)=0
$$

整理可得

$$
\left(R+\beta B^\top P B\right)a
=
-\left(N^\top+\beta B^\top P A\right)s
$$

因此最优反馈规则为

$$
a_t=-F s_t
$$

其中

$$
F=\left(R+\beta B^\top P B\right)^{-1}\left(N^\top+\beta B^\top P A\right)
$$

### 6.4 广义离散 Riccati 方程

将最优反馈代回 Bellman 方程，可得 $P$ 满足

$$
P
=
Q+\beta A^\top P A
-\left(N+\beta A^\top P B\right)
\left(R+\beta B^\top P B\right)^{-1}
\left(N^\top+\beta B^\top P A\right)
$$

这就是含交叉项的广义离散代数 Riccati 方程。

### 6.5 常数项

由于存在随机冲击，值函数中的常数项满足

$$
c=\beta c + \beta \mathrm{tr}(P\Sigma\Sigma^\top)
$$

因此

$$
c=\frac{\beta}{1-\beta}\mathrm{tr}(P\Sigma\Sigma^\top)
$$

如果只关心最优反馈规则，常数项不是必须；但若要精确比较理论总福利水平，则常数项有用。

## 7. 闭环系统与稳定性

### 7.1 闭环动态

最优反馈规则代入状态转移后，闭环系统为

$$
s_{t+1}=(A-BF)s_t+\Sigma\varepsilon_{t+1}
$$

### 7.2 稳定性条件

若闭环矩阵 $A-BF$ 的特征值都落在单位圆内，则系统在二阶矩意义下稳定。

这是本文 benchmark 理论部分必须检查的内容，因为：

| 原因 | 说明 |
|---|---|
| 经济合理性 | 最优规则不应导致通胀、产出或利率路径爆炸 |
| 数值合理性 | 后续 RL 对照应在可稳定 benchmark 上进行 |
| 福利可比性 | 若理论规则不稳定，则无限期损失不可比较 |

### 7.3 无条件二阶矩

当闭环系统稳定时，状态协方差矩阵 $\Omega$ 满足离散 Lyapunov 方程：

$$
\Omega=(A-BF)\Omega(A-BF)^\top+\Sigma\Sigma^\top
$$

这为后续报告无条件方差、政策波动与长期福利提供理论基础。

## 8. 理论最优规则与 Taylor rule 的关系

### 8.1 形式关系

由于状态向量包含 $\tilde{i}_{t-1}$，理论最优规则可写为

$$
a_t = -f_\pi \tilde{\pi}_t - f_x x_t - f_i \tilde{i}_{t-1}
$$

等价地，写回原始利率可得

$$
i_t = i^\ast -f_\pi(\pi_t-\pi^\ast) - f_x x_t - f_i(i_{t-1}-i^\ast)
$$

整理后就是一种包含利率惯性的线性反应函数。

### 8.2 与经验 Taylor rule 的本质差异

| 对比项 | 理论最优反馈规则 | 经验 Taylor rule |
|---|---|---|
| 来源 | 动态最优控制问题的解 | 数据回归或政策规则设定 |
| 系数决定因素 | 状态转移、冲击方差、损失权重、折现因子 | 样本统计关系与制度背景 |
| 含义 | 规范性政策基准 | 经验比较对象 |

因此，本文后续并不是要证明理论最优规则等于 Taylor rule，而是比较它们在形式、福利与动态效果上的关系。

## 9. RL 作为最优控制数值求解器的理论基础

### 9.1 MDP 映射

本文将央行问题写成 MDP：

| 元素 | 定义 |
|---|---|
| 状态 | $s_t=[\tilde{\pi}_t,x_t,\tilde{i}_{t-1}]^\top$ |
| 动作 | $a_t=\tilde{i}_t$ |
| 即时奖励 | $r_t=-\ell(s_t,a_t)$ |
| 转移 | 由状态方程给出 |
| 目标 | 最大化 $\mathbb{E}[\sum_{t\ge 0}\gamma^t r_t]$ |

取 $\gamma=\beta$ 时，RL 的目标与最优控制问题等价。

### 9.2 价值函数与动作价值函数

给定策略 $\pi$，定义状态价值函数

$$
V^\pi(s)=\mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0=s\right]
$$

动作价值函数

$$
Q^\pi(s,a)=\mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r_t \mid s_0=s,a_0=a\right]
$$

最优价值函数满足

$$
V^\ast(s)=\max_a\left\{r(s,a)+\gamma \mathbb{E}[V^\ast(s')\mid s,a]\right\}
$$

这与 Bellman 最优方程完全同构，只是本文在最优控制里写成“最小化损失”，在 RL 里写成“最大化负损失”。

### 9.3 为什么 RL 是本文的合理方法

| 理由 | 说明 |
|---|---|
| 与 Bellman 方程同源 | RL 本质上是在近似求 Bellman 最优解 |
| 不需要显式解析解 | 适合本文后续不可解析或难解析的扩展环境 |
| 直接产出反馈规则 | 训练得到的策略网络本身就是状态到利率的映射 |
| 易于处理非线性与约束 | 只需改环境与奖励，不必手工重推解析解 |

### 9.4 为什么 PPO 可作为主算法

PPO 是一种策略梯度类 actor-critic 方法。其核心思想不是直接求解 Bellman 方程的闭式解，而是在与环境交互时，通过优势函数近似提升当前策略，并通过裁剪目标限制每次更新步长，提升训练稳定性。

PPO 的实际优化目标可写为

$$
L^{\mathrm{clip}}(\theta)
=
\mathbb{E}\left[
\min\left(
r_t(\theta)\hat{A}_t,
\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right]
$$

其中

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
$$

$\hat{A}_t$ 为优势函数估计。对本文而言，PPO 的价值在于数值稳定性，而不是提供新的经济理论。

## 10. 为什么传统方法在扩展环境中变难

### 10.1 非线性状态转移

若状态转移变为

$$
s_{t+1}=f(s_t,a_t)+\Sigma(s_t,a_t)\varepsilon_{t+1}
$$

或者分量形式如

$$
\tilde{\pi}_{t+1}=f_\pi(\tilde{\pi}_t,x_t,a_t)+\sigma_\pi \varepsilon^\pi_{t+1}
$$

$$
x_{t+1}=f_x(\tilde{\pi}_t,x_t,a_t)+\sigma_x \varepsilon^x_{t+1}
$$

则二次型值函数与线性反馈规则通常不再成立。

### 10.2 约束

若存在零利率下界

$$
i_t \ge 0
$$

等价地对偏离量写成

$$
a_t \ge -i^\ast
$$

则最优策略可能呈现分段、折角或强烈状态依赖的非线性结构。

### 10.3 非对称损失

若损失函数改为

$$
\ell_t=\lambda_\pi \tilde{\pi}_t^2+\lambda_x x_t^2+\lambda_i(\tilde{i}_t-\tilde{i}_{t-1})^2+\lambda_{\pi,+}\max(\tilde{\pi}_t,0)^2
$$

或加入其他分段项，解析结构同样会被破坏。

### 10.4 维数灾难

传统数值动态规划在处理 Bellman 方程时常需要对状态空间做网格离散。若状态维度上升，网格点数量会指数增长，这就是所谓维数灾难。RL 的优势并不是完全消除困难，而是通过函数逼近避免对整个状态空间进行显式穷举。

## 11. 非线性与约束环境下的理论表述

### 11.1 一般 Bellman 方程

在更一般情形下，Bellman 方程仍然成立：

$$
V(s)=\min_a \left\{\ell(s,a)+\beta \mathbb{E}[V(s')\mid s,a]\right\}
$$

变化的是：

| 项目 | LQ benchmark | 非线性/约束环境 |
|---|---|---|
| 状态转移 | 线性 | 一般为非线性 |
| 损失函数 | 二次 | 可非二次、可分段 |
| 策略形式 | 线性反馈 | 一般为非线性反馈 |
| 解法 | Riccati 或半解析方法 | 数值近似为主 |

### 11.2 连续时间的 HJB 说明

如果未来某一步需要写连续时间版本，则值函数满足 HJB 方程：

$$
\rho V(s)=\min_a \left\{\ell(s,a)+\mathcal{L}^a V(s)\right\}
$$

其中 $\mathcal{L}^a$ 是受控扩散过程对应的生成元。本文当前主线采用离散时间，因此真正实现时不需要直接数值求解 HJB，但在写作中可以把 HJB 作为与开题报告一致的连续时间参照。

## 12. 反事实模拟与福利比较的理论基础

### 12.1 反事实路径

给定初始状态 $s_0$ 与同一组冲击序列 $\{\varepsilon_t\}$，不同政策规则会生成不同路径：

$$
\{s_t^\mathrm{rule}, a_t^\mathrm{rule}\}_{t=0}^{T}
$$

比较反事实时必须固定初始条件和冲击序列，否则比较会混入环境差异。

### 12.2 福利指标

本文后续将使用

$$
L_T=\mathbb{E}\left[\sum_{t=0}^{T}\beta^t \ell_t\right]
$$

或其无穷期近似作为主福利指标。

同时，为了增强经济解释，还应报告：

| 指标 | 理论意义 |
|---|---|
| $\mathrm{Var}(\tilde{\pi}_t)$ | 通胀稳定性 |
| $\mathrm{Var}(x_t)$ | 产出稳定性 |
| $\mathrm{Var}(\Delta i_t)$ | 政策平滑性 |

### 12.3 不能只看一个数字

即便某个规则在总损失上更低，也必须解释：

| 问题 | 说明 |
|---|---|
| 它是通过压低通胀波动还是压低产出波动实现的 | 明确损失分项来源 |
| 是否导致利率过度波动 | 避免“福利改善”只是通过不现实的剧烈调息获得 |
| 是否在特定区域才更优 | 尤其是 ZLB 或高通胀区域的状态依赖性 |

## 13. 经验规则、历史数据与 Lucas critique

### 13.1 经验 Taylor rule

经验规则可写为

$$
i_t=\alpha+\phi_\pi \pi_t+\phi_x x_t+\phi_i i_{t-1}+u_t
$$

它的用途是构造现实比较基准，而不是替代理论最优规则。

### 13.2 reduced-form 转移与反事实

若用历史数据估计状态转移，再把不同政策规则带入做反事实，其理论前提近似为：

$$
f_{\text{estimated}} \text{ 在政策改变后仍然保持不变}
$$

这是一个很强的假设。

### 13.3 Lucas critique

Lucas critique 的核心思想是：政策规则改变时，私人部门行为和整体结构关系可能随之改变，因此在旧政策下估计得到的 reduced-form 关系未必能用于新政策下的反事实。

对本文而言，这意味着：

| 结论 | 解释边界 |
|---|---|
| 基于历史估计转移的反事实是有价值的 | 可以比较“给定估计环境下”的政策表现 |
| 但它不是完整结构政策评估 | 不能把结论过度解释为真实世界中的结构性福利定理 |

正文中必须明确写出这个边界。

## 14. 本文理论贡献的准确表述

### 14.1 不宜夸大的说法

以下说法不准确，应避免：

| 不准确表述 | 原因 |
|---|---|
| RL 证明真实央行应采用神经网络政策 | 本文只是数值求解与政策比较，不是现实制度设计结论 |
| RL 替代了宏观经济理论 | 本文依然依赖宏观模型、损失函数和政策目标设定 |
| RL 比传统方法全面更优 | 在 LQ benchmark 中，传统方法恰恰提供更强理论基准 |

### 14.2 准确表述

更准确的理论定位是：

| 表述 | 含义 |
|---|---|
| RL 是动态规划问题的近似数值方法 | 它的理论合法性来自 Bellman 递归结构 |
| benchmark 中经典方法提供真值基准 | 用于检验 RL 是否真正学到最优控制 |
| 非线性与约束环境体现 RL 的相对价值 | 创新点在这里，而不是否定经典控制 |

## 15. Phase 1 的直接输出要求

本阶段完成后，后续正文至少可以直接使用以下内容：

| 内容 | 用途 |
|---|---|
| 第 3 章模型设定的符号和方程 | 正文主文 |
| Bellman 方程、Riccati 推导、闭环系统说明 | 理论部分或附录 |
| RL-MDP 映射与 PPO 理论说明 | 方法部分 |
| 非线性、ZLB、Lucas critique 的边界说明 | 扩展章节与局限讨论 |

## 16. Phase 1 验收清单

| 验收项 | 状态 | 说明 |
|---|---|---|
| 已给出动态最优控制主线 | 完成 | 从状态转移到长期目标完整闭合 |
| 已给出 Bellman 方程 | 完成 | 与离散时间设定一致 |
| 已给出含交叉项的 Riccati 推导 | 完成 | 与利率平滑项一致 |
| 已给出最优反馈规则形式 | 完成 | $a_t=-F s_t$ |
| 已给出 RL-MDP 理论映射 | 完成 | 解释了 RL 的方法论地位 |
| 已给出非线性与 ZLB 的理论边界 | 完成 | 服务于扩展实验 |
| 已给出 Lucas critique 讨论 | 完成 | 服务于经验反事实解释 |

## 17. 下一步

`Phase 1` 完成后，应进入 `Phase 2`：

| 顺序 | 任务 | 说明 |
|---|---|---|
| 1 | 数据来源与变量定义冻结 | 通胀、产出缺口、政策利率 |
| 2 | 参数校准方案制定 | 明确哪些来自文献，哪些来自数据估计 |
| 3 | 经验 Taylor rule 规格确定 | 为后续反事实做准备 |
| 4 | benchmark 参数表草拟 | 供 `Phase 3` 实现使用 |

---

一句话总结：

本文的理论框架已经明确为“离散时间随机动态最优控制 + LQ benchmark 解析基准 + RL 作为 Bellman 问题近似求解器 + 非线性和约束扩展 + 经验反事实的结构边界说明”。
