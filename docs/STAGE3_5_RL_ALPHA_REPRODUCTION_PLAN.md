# 阶段 3-5：强化学习公式因子挖掘复现计划

## 目标

复现论文 **Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning** 的主线方法：

RL 公式生成器 -> RPN 公式因子 -> 因子求值 -> alpha pool 线性组合 -> 组合 IC 奖励 -> PPO 继续生成新因子。

本阶段暂不实现 XGBoost，不接入真实交易，不做实盘、模拟盘、券商 API 或自动下单。

## 当前默认决策

- 阶段 3-5 一体推进，但分阶段实现。
- 第一版实现论文全量算子。
- 外部论文、Alpha101、Qlib 等算子只做候选记录，不进入首轮实现。
- 数据规模按 toy data -> 小样本 smoke -> Top500 pilot -> Top1500 long run 放大。
- alpha pool 权重优化按论文 Algorithm 1 使用梯度下降。
- 阶段 3-4 不引入 PyTorch；阶段 5 PPO 前再确认并加入 PyTorch。
- 代码必须保持简洁紧凑，不写大量无用兜底，不把简单表达式拆成很多碎行。

## 阶段 3：表达式系统

实现 RPN、token、operator、parser、safe evaluator、action mask。

核心接口建议：

- `Token`：表示 `BEG/SEP`、特征、常数、时间窗口、算子。
- `Expression`：表达式树节点，可渲染为可读公式。
- `parse_rpn(tokens) -> Expression`
- `evaluate(expr, daily_panel) -> factor_values`
- `valid_actions(tokens) -> list[Token]`
- `is_semantically_valid(values) -> bool`

第一版算子：

- 特征：`open/close/high/low/volume/vwap`
- 常数：沿用 `config/expression.yml`
- 时间窗口：`10d/20d/30d/40d/50d`
- 截面算子：`Abs/Log/Add/Sub/Mul/Div/Greater/Less`
- 时序算子：`Ref/Mean/Med/Sum/Std/Var/Max/Min/Mad/Delta/WMA/EMA/Cov/Corr`

关键规则：

- `Ref(x, t)` 只能表示过去 `t` 日。
- `SEP` 只在 stack 中刚好剩一个完整、非纯常数表达式时允许。
- token 长度上限为 20。
- 时序算子的最后一个参数必须是时间窗口。
- 表达式系统维护最小量纲约束，拒绝价格和成交量直接加减、价格乘价格、价格与纯常数直接比较等明显无意义公式。
- 语义非法表达式不能静默吞掉，应明确标记 invalid。

## 阶段 4：因子评估与 Alpha Pool

实现论文的组合优先逻辑，不做单因子 IC 贪心筛选。

核心接口建议：

- `daily_ic(factor, label) -> Series`
- `mean_ic(factor, label) -> float`
- `mean_rank_ic(factor, label) -> float`
- `mutual_ic(factor_a, factor_b) -> float`
- `AlphaPool.add(expr, values) -> PoolUpdateResult`
- `AlphaPool.optimize_weights() -> weights`

计算口径：

- 每日截面去均值并 L2 归一化，贴近论文 Equation 7。
- 单因子 IC：每日 Pearson IC 的时间均值。
- mutual IC：两个因子每日截面相关的时间均值。
- pool loss：`L(w) = 1 - 2 * w @ ic + w.T @ mutual_ic @ w`
- 新因子加入后优化权重。
- 超过容量后删除 `abs(weight)` 最小的因子。
- 不用 mutual IC 阈值提前拒绝因子。
- 不按单因子 IC 删除因子。

## 阶段 5：PPO 公式生成器

阶段 3-4 稳定后再实现 PPO，并在此时确认 PyTorch 依赖。

核心接口建议：

- `AlphaMiningEnv.reset() -> state`
- `AlphaMiningEnv.step(action) -> next_state, reward, done, info`
- `PolicyValueNet(tokens) -> logits, value`
- `PPOTrainer.train_smoke()/train_pilot()/train_long()`

MDP 口径：

- 状态：当前 token 序列，从 `BEG` 开始。
- 动作：下一个 token。
- 必须在 logits 上做 invalid action mask。
- 中间奖励：`0`
- 有效终止奖励：新因子加入 alpha pool 后的组合 IC。
- 语义非法奖励：`-1`
- 折扣因子：`gamma = 1`
- alpha pool 不在 episode 间重置。

网络默认：

- 共享 2 层 LSTM，hidden size 128，dropout 0.1。
- policy/value head 各为两层 64 维 MLP。
- PPO clip epsilon 0.2。
- 单元测试不依赖 GPU；long run 才使用 GPU。

## 测试要求

阶段 3：

- 至少 10 个手写 RPN 公式可解析、渲染、求值。
- 非法 RPN 被 parser/action mask 拒绝。
- `Log` 非正、除零、全 NaN、短窗口等返回语义非法。
- action mask 不允许缺操作数、错误时间窗口、纯常数表达式终止。
- `Ref`、rolling、Cov/Corr 均不使用未来数据。

阶段 4：

- IC、RankIC、mutual IC 与手算 toy case 一致。
- 截面归一化均值接近 0，L2 norm 接近 1。
- pool 加入新因子后更新 IC/mutual IC 缓存。
- 权重优化能降低 toy loss。
- pool 超容量时删除绝对权重最小因子。
- invalid alpha 不进入 pool。

阶段 5：

- env `reset/step` 行为确定。
- logits mask 后不会采样非法动作。
- 终止有效公式能触发 pool 更新和组合 IC 奖励。
- 语义非法公式奖励为 `-1`。
- smoke run 能完成少量 episode，并产生至少一个合法公式。
- PPO 测试只验证链路和参数更新，不要求策略收益好看。

## 子代理建议

允许多开子代理，但子代理必须只负责明确边界内的任务，并由主代理复核后合并。

建议拆分：

- 子代理 A：论文算法和官方实现只读审查。
- 子代理 B：表达式系统和 action mask 审查。
- 子代理 C：IC、RankIC、mutual IC、alpha pool 数学口径审查。
- 子代理 D：代码简洁性和重复逻辑审查。
- 子代理 E：无未来函数和数据窗口审查。

子代理不得接入真实交易，不得安装大型依赖，不得改变项目边界。

## 验收标准

- 阶段 3 完成后，可以用手写公式在全量缓存的小样本上求值。
- 阶段 4 完成后，可以把固定手写公式加入 alpha pool 并优化权重。
- 阶段 5 smoke 完成后，可以生成合法公式、计算奖励、完成 PPO 参数更新。
- 所有测试通过：`ruff check src tests` 和 `pytest`。
- 不保留临时 smoke/debug 文件。

## 给另一个对话的 Prompt

```text
你将接手 E:\codex_projects\quant 项目的阶段 3-5：复现论文中的强化学习公式因子挖掘方法。

开始前必须阅读：

1. AGENTS.md
2. PROJECT_PLAN.md
3. docs/STAGE3_5_RL_ALPHA_REPRODUCTION_PLAN.md
4. docs/DATA_CLEANING.md
5. docs/UNIVERSE_AND_LABELS.md
6. 本地论文：Automatic formulaic alpha generation with reinforcement learning.pdf

当前任务不是接入交易，也不是做 XGBoost，而是按计划推进论文复现主线：

阶段 3：表达式系统
阶段 4：因子评估与 alpha pool
阶段 5：PPO 公式生成器

请严格遵守：

- 代码保持简洁、紧凑、直接，不写大量无用兜底。
- 不要把简单表达式拆成很多只有一两个字符的短行。
- 不要引入 FinRL、Qlib、RD-Agent 等大型框架作为项目依赖。
- 阶段 3-4 不要提前引入 PyTorch；阶段 5 前再确认 PyTorch 安装方式。
- 所有实验配置进入 config/*.yml。
- 不允许未来函数。
- 不接入真实交易、券商 API、模拟盘或自动下单。
- 可以创建多个子代理辅助，但必须给每个子代理明确只读/写入边界，并由主代理复核。
- 子代理适合负责：论文细节审查、action mask 审查、alpha pool 数学口径审查、无未来函数审查、代码简洁性审查。

请先做只读探索，确认当前 expression/alpha/rl 目录状态、配置文件和数据产物，然后给出执行计划。若进入实现，请从阶段 3 表达式系统开始，不要跳到 PPO。
```
