# 强化学习公式因子挖掘 + XGBoost 截面预测项目计划

## 0. 如何使用这份计划

这份文档是后续实现项目的主执行说明。推荐按阶段推进，每个阶段都包含：

- 阶段目标：这一阶段要解决什么问题。
- 主要交付物：完成后仓库里应该出现什么。
- 执行步骤：建议按顺序实现的工作。
- 验收标准：什么情况可以进入下一阶段。

后续开发时不要一开始就写 RL 或 XGBoost。这个项目的核心风险在数据、无未来函数、表达式求值和实验口径上，所以必须先把数据和研究框架打稳，再进入模型训练。

## 1. 项目概述

本项目是一个个人量化研究项目，方向是：

1. 使用强化学习生成可解释的公式因子。
2. 方法级复现论文 **Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning**。
3. 在论文方法基础上，把生成出来的公式因子输入 XGBoost，做 A股多因子截面预测。
4. 通过 IC、RankIC、分组收益和多头组合回测验证信号有效性。

当前工作区已有参考论文：

```text
Automatic formulaic alpha generation with reinforcement learning.pdf
```

论文官方参考实现：

```text
https://github.com/RL-MLDM/alphagen
```

本项目不直接 fork 官方仓库，而是参考论文和官方实现，在当前仓库中自建核心模块。这样做的目的不是最快跑出结果，而是让你真正掌握数据、表达式系统、RL 训练、alpha pool、XGBoost 和回测之间的关系。

项目周期按 **6-8 周研究版** 规划。

## 2. 最终目标

项目最终应能完成以下闭环：

1. 从 AKShare 下载 A股日频行情数据。
2. 本地缓存并清洗 OHLCV 数据。
3. 构建每月流动性 Top1500 股票池。
4. 实现公式因子的 RPN 表达、解析和安全求值。
5. 使用 PPO 强化学习生成公式因子。
6. 按论文方法维护一个协同 alpha pool。
7. 选出最终 30 个公式因子。
8. 对因子做 IC、RankIC、mutual IC、分组收益分析。
9. 将最终 30 个公式因子输入 XGBoost，预测未来 20 日截面收益排名。
10. 构建每月 Top10% 等权多头组合，并做真实约束下的回测。
11. 输出中文 Notebook 研究报告和可复现实验说明。

## 3. 项目边界

### 3.1 第一版包含

- A股日频数据。
- 价量类公式因子。
- 强化学习公式生成器。
- 论文方法级复现。
- XGBoost 截面预测扩展。
- 未来 20 个交易日收益排名预测。
- 月度 Top10% 多头组合回测。
- 研究级交易约束。
- 中文 README 和中文 Notebook。

### 3.2 第一版不包含

- 真实交易。
- 券商接口。
- 模拟盘下单。
- 分钟级或 tick 数据。
- 基本面因子。
- 另类数据。
- 行业中性化。
- 市值中性化。
- 精确复现论文 CSI300/CSI500 数值。
- 历史指数成份股重建。
- 生产级实盘信号服务。

这些内容可以作为第二阶段或长期扩展，不放进第一版。

## 4. 核心研究设计

### 4.1 主线方法

论文主线是：

```text
RL 公式生成器 -> 公式因子 -> 线性 alpha pool -> 组合 IC 奖励
```

也就是说，强化学习生成器的目标不是找单个 IC 最高的因子，而是找一个能改善当前 alpha pool 组合效果的新因子。

第一版必须复现这些关键点：

- RPN 公式表达。
- 合法动作 mask。
- PPO token 生成器。
- 公式求值引擎。
- alpha pool 线性组合权重。
- 用组合模型表现作为 RL 奖励。
- IC 和 RankIC 评估。

### 4.2 XGBoost 的位置

XGBoost 是本项目的扩展线，不是论文主线的替代品。

项目结构应分成两条实验线：

1. 论文复现线：RL 生成公式因子，线性 alpha pool 做组合。
2. XGBoost 扩展线：冻结最终 30 个公式因子，用它们作为 XGBoost 特征做截面预测。

这样能回答两个问题：

- 论文方法本身是否能生成有协同效果的公式因子。
- 这些公式因子作为多因子特征，是否能帮助 XGBoost 做截面排序。

### 4.3 数据源

第一版使用 AKShare。

选择理由：

- 无需个人 token。
- 适合个人研究项目起步。
- 可以获取 A股日频历史行情。
- 易于接入自建项目。

风险：

- 免费数据源稳定性不保证。
- 字段名称和单位可能变化。
- 接口可能失效。
- 数据质量必须本地校验。

应对方式：

- 所有下载数据必须本地缓存。
- 原始数据和处理后数据分开保存。
- 下载器支持重试和断点续传。
- 生成数据质量报告。
- 数据源适配层保持独立，方便未来替换为 Tushare、BaoStock、Qlib 或商业数据。

### 4.4 股票池

第一版使用 A股流动性股票池，不使用最初讨论过的 ETF 池。

每月股票池构建规则：

1. 从 AKShare 可获取的 A股代码开始。
2. 第一版默认排除北交所股票，除非后续确认数据质量稳定。
3. 排除 ST 和特殊处理股票。
4. 排除上市不足 250 个交易日的新股。
5. 排除调仓日前后停牌、零成交或关键价格缺失的股票。
6. 对剩余股票计算过去 20 个交易日的平均成交额。
7. 选取平均成交额最高的 Top1500。

使用 Top1500 的理由：

- 截面样本足够大，适合多因子预测。
- 避免极端小票流动性问题。
- 不依赖历史指数成份股数据，工程更可控。

### 4.5 时间切分

论文复现线采用固定切分：

| 区间 | 用途 |
| --- | --- |
| 2015-01-01 至 2020-12-31 | RL 因子挖掘训练 |
| 2021-01-01 至 2021-12-31 | 验证集，选择 seed、模型和最终公式池 |
| 2022-01-01 至最新可得交易日 | 样本外测试 |

XGBoost 扩展线在测试期内使用逐月 walk-forward：

- 每个月预测前，只使用过去 3 年样本训练。
- 预测下一个调仓月的截面分数。
- 不允许预测月之后的数据进入训练。

### 4.6 标签定义

第一版统一研究未来 20 个交易日收益。

论文复现线使用 close-to-close 标签：

```text
label_t = close_{t+20} / close_t - 1
```

组合回测使用更真实的执行口径：

- 月末收盘后生成信号。
- 下一交易日开盘调仓。
- 净值用 open-to-open 收益计算。

这样既能保留论文 IC 口径，又能让回测执行假设更合理。

### 4.7 因子处理

每个交易日对因子做截面处理：

1. 在当日股票池内计算公式因子值。
2. 非法值、无穷值、明显异常值设为缺失。
3. 必要时对极端值做 winsorize。
4. 对有效因子值做截面标准化：

```text
z_i = (x_i - mean(x)) / std(x)
```

第一版不做行业中性化或市值中性化。原因是这会引入额外数据源和历史口径问题，容易让第一版过重。

### 4.8 最终公式因子池

最终保留：

```text
30 个公式因子
```

必须遵守：

- 因子筛选不能使用未来数据。
- 训练、验证和测试边界必须严格。
- 最终测试结果不能反过来影响公式选择。

## 5. 推荐仓库结构

后续实现时建议创建以下结构：

```text
quant/
  PROJECT_PLAN.md
  README.md
  requirements.txt
  pyproject.toml
  config/
    data.yml
    universe.yml
    expression.yml
    rl.yml
    xgboost.yml
    backtest.yml
  data/
    raw/
    interim/
    processed/
    features/
    models/
    reports/
  notebooks/
    01_data_quality.ipynb
    02_expression_engine.ipynb
    03_rl_alpha_mining.ipynb
    04_xgboost_cross_section.ipynb
    05_backtest_report.ipynb
  src/
    quant_rl_alpha/
      __init__.py
      data/
        __init__.py
        akshare_client.py
        schema.py
        cache.py
        quality.py
        universe.py
      expression/
        __init__.py
        tokens.py
        operators.py
        rpn.py
        parser.py
        evaluator.py
        action_mask.py
      alpha/
        __init__.py
        metrics.py
        pool.py
        selection.py
      rl/
        __init__.py
        env.py
        policy.py
        ppo.py
        trainer.py
      model/
        __init__.py
        xgboost_ranker.py
        dataset.py
      backtest/
        __init__.py
        engine.py
        costs.py
        constraints.py
        performance.py
      reporting/
        __init__.py
        plots.py
        tables.py
      utils/
        __init__.py
        calendar.py
        logging.py
        paths.py
        seed.py
  tests/
    test_rpn.py
    test_action_mask.py
    test_operators.py
    test_metrics.py
    test_alpha_pool.py
    test_backtest.py
```

## 6. 环境规划

### 6.1 Python 版本

推荐使用 Python 3.12。

当前 Codex 内置运行时有 Python 3.12，但系统 PATH 中暂未暴露 `python`。为了后续能在普通终端中独立运行，建议安装标准 Python 并创建项目本地虚拟环境。

### 6.2 创建虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 6.3 依赖包

初始依赖建议：

```text
akshare
pandas
numpy
pyarrow
pyyaml
jupyterlab
matplotlib
plotly
scikit-learn
xgboost
torch
tqdm
joblib
pytest
ruff
```

### 6.4 GPU 策略

本机有 NVIDIA RTX 5070 Ti 16GB 显存，因此第一版可以直接规划 CUDA 版 PyTorch。

要求：

- 实施时确认 CUDA 版 PyTorch 能正常调用 GPU。
- smoke test 必须支持 CPU，方便快速调试。
- 长时间 RL 训练使用 GPU。
- 单元测试不得依赖 GPU。

## 7. 分阶段执行计划

下面是后续实现时最重要的部分。建议严格按阶段推进。

## 阶段 0：项目骨架与工程约定

### 阶段目标

创建项目基础结构，让后续所有模块有明确位置，避免 Notebook、脚本、数据和配置混在一起。

### 主要交付物

- `README.md`
- `requirements.txt`
- `pyproject.toml`
- `config/*.yml`
- `src/quant_rl_alpha/`
- `tests/`
- 基础日志、路径和随机种子工具。

### 执行步骤

1. 创建推荐仓库结构。
2. 创建 Python 虚拟环境。
3. 安装基础依赖。
4. 写入配置文件草案。
5. 实现 `utils/paths.py`，统一管理项目根目录和数据目录。
6. 实现 `utils/logging.py`，统一日志格式。
7. 实现 `utils/seed.py`，统一随机种子设置。
8. 配置 `pytest` 和 `ruff`。

### 验收标准

- `pytest` 可以运行，即使暂时只有空测试或 smoke 测试。
- 能从任意模块稳定定位项目根目录。
- 配置文件能被 Python 正确加载。
- README 中说明项目目标和当前阶段。

## 阶段 1：数据下载、缓存与质量检查

### 阶段目标

建立可靠的数据底座。只有数据层稳定后，后面的因子、RL、XGBoost 和回测才有意义。

### 主要交付物

- AKShare 下载器。
- 原始数据 Parquet 缓存。
- 标准化日频行情 schema。
- 数据质量报告。
- `01_data_quality.ipynb`。

### 标准行情字段

统一日频行情表字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| date | date | 交易日期 |
| symbol | string | 股票代码 |
| name | string | 股票名称 |
| open | float | 前复权开盘价 |
| high | float | 前复权最高价 |
| low | float | 前复权最低价 |
| close | float | 前复权收盘价 |
| volume | float | 成交量 |
| amount | float | 成交额 |
| vwap | float | 近似成交均价 |
| turnover | float | 换手率，可选 |
| source | string | 数据源 |
| adjust | string | 复权方式 |

### 数据存储

主格式使用 Parquet：

```text
data/raw/akshare/daily/{symbol}.parquet
data/interim/daily_panel.parquet
data/processed/universe_monthly.parquet
data/features/alpha_values.parquet
data/features/xgboost_dataset.parquet
```

CSV 只用于小样本调试和人工查看。

### 执行步骤

1. 实现 `data/akshare_client.py`。
2. 获取 A股股票代码列表。
3. 对每只股票下载日频行情。
4. 每只股票单独缓存为 Parquet。
5. 下载器支持断点续传，已存在文件默认跳过。
6. 失败代码写入日志。
7. 实现重试机制。
8. 实现 `data/schema.py`，统一字段名称和类型。
9. 实现 `data/quality.py`，生成数据质量报告。
10. 编写 `01_data_quality.ipynb` 展示数据覆盖、缺失、异常和样例。

### 数据质量检查

至少检查：

- 下载成功股票数。
- 每只股票日期范围。
- OHLCV 缺失数量。
- 零价格或负价格。
- 零成交量。
- 重复日期。
- 异常大涨跌幅。
- 样本太短的股票。
- 被剔除股票及剔除原因。

### 验收标准

- 能下载并缓存一小批股票。
- 能对缓存数据生成标准字段表。
- 能输出数据质量报告。
- 能发现并记录缺失、重复、异常数据。

## 阶段 2：股票池与标签构建

### 阶段目标

构建每月 Top1500 股票池和未来 20 日收益标签，为因子评估和模型训练提供标准样本。

### 主要交付物

- 月度股票池构建器。
- 未来收益标签生成器。
- 交易日历工具。
- 股票池和标签数据缓存。

### 执行步骤

1. 实现 `utils/calendar.py`，处理交易日、月末交易日和下一交易日。
2. 实现上市天数过滤。
3. 实现 ST 股票过滤。
4. 实现停牌、零成交和价格缺失过滤。
5. 计算过去 20 日平均成交额。
6. 每月选取 Top1500。
7. 生成 close-to-close 未来 20 日收益标签。
8. 生成 XGBoost 用的未来 20 日截面 rank 标签。
9. 缓存 `universe_monthly.parquet`。
10. 缓存 `labels.parquet`。

### 标签口径

论文复现标签：

```text
future_20d_return = close_{t+20} / close_t - 1
```

XGBoost 标签：

```text
future_20d_rank = rank_pct(future_20d_return)
```

### 验收标准

- 能为每个月生成 Top1500 股票池。
- 标签没有使用未来不可见的信息参与因子计算或股票池选择。
- 每个样本都有清晰的 `date`、`symbol`、`future_20d_return`、`future_20d_rank`。
- 可以抽查某个月股票池和标签，逻辑解释得通。

## 阶段 3：公式表达式系统

### 阶段目标

实现论文中的公式因子表达系统，让公式可以被生成、解析、检查、求值和展示。

### 主要交付物

- token 定义。
- RPN parser。
- 表达式树。
- 公式字符串渲染。
- 算子库。
- 安全求值器。
- action mask。
- `02_expression_engine.ipynb`。

### RPN 表达方式

公式示例：

```text
Mean(close, 20) / close - 1
```

对应 RPN 可能是：

```text
BEG close 20d Mean close Div 1 Sub SEP
```

表达式系统必须能完成：

1. token 序列转表达式树。
2. 表达式树转可读公式字符串。
3. 公式在日频行情面板上求值。
4. 非法表达式拒绝或返回 invalid。

### token 类型

| 类型 | 示例 |
| --- | --- |
| 序列标记 | BEG, SEP |
| 原始特征 | open, close, high, low, volume, vwap |
| 常数 | -30, -10, -5, -2, -1, -0.5, -0.01, 0.01, 0.5, 1, 2, 5, 10, 30 |
| 时间窗口 | 10d, 20d, 30d, 40d, 50d |
| 截面一元算子 | Abs, Log |
| 截面二元算子 | Add, Sub, Mul, Div, Greater, Less |
| 时序一元算子 | Ref, Mean, Med, Sum, Std, Var, Max, Min, Mad, Delta, WMA, EMA |
| 时序二元算子 | Cov, Corr |

### 算子语义

| 算子 | 含义 |
| --- | --- |
| Abs(x) | 绝对值 |
| Log(x) | 自然对数，只允许正数输入 |
| Add/Sub/Mul/Div | 加减乘除 |
| Greater(x, y) | 元素级较大值 |
| Less(x, y) | 元素级较小值 |
| Ref(x, t) | t 日前的值 |
| Mean/Med/Sum(x, t) | 滚动均值、中位数、求和 |
| Std/Var(x, t) | 滚动标准差、方差 |
| Max/Min(x, t) | 滚动最大值、最小值 |
| Mad(x, t) | 滚动平均绝对偏差 |
| Delta(x, t) | x - Ref(x, t) |
| WMA/EMA(x, t) | 加权移动平均、指数移动平均 |
| Cov/Corr(x, y, t) | 滚动协方差、相关系数 |

### 安全规则

求值器必须处理：

- 除零。
- Log 非正数。
- 滚动窗口历史不足。
- 全部为 NaN 的输出。
- 纯常数表达式。
- 截面方差过低。
- 无穷值。
- 数据缺失。

语义非法表达式在 RL 中奖励为：

```text
-1
```

### action mask 规则

动作 mask 必须保证形式合法：

1. 时序算子最后一个参数必须是时间窗口。
2. 一元算子需要一个表达式操作数。
3. 二元算子需要两个表达式操作数。
4. `SEP` 只能在当前 stack 中刚好有一个完整非纯常数表达式时允许。
5. 多 token 表达式不能等价于纯常数。
6. token 序列长度不得超过 20。

### 验收标准

- 至少 10 个手写公式可以正确求值。
- 非法公式能被 parser 或 evaluator 拒绝。
- action mask 不允许明显非法 RPN。
- Notebook 能清楚展示公式、RPN、求值结果和 IC 示例。

## 阶段 4：因子评估与 alpha pool

### 阶段目标

实现论文中的因子评估和线性 alpha pool，为 RL 奖励提供核心评估器。

### 主要交付物

- IC 计算。
- RankIC 计算。
- mutual IC 计算。
- 因子值缓存。
- alpha pool 权重优化。
- pool cap 删除逻辑。

### IC

每日截面 Pearson IC：

```text
IC_t = corr(alpha_t, future_return_t)
```

总 IC：

```text
IC = mean(IC_t)
```

### RankIC

每日截面 Spearman RankIC：

```text
RankIC_t = corr(rank(alpha_t), rank(future_return_t))
```

总 RankIC：

```text
RankIC = mean(RankIC_t)
```

### mutual IC

两个因子之间：

```text
MutualIC = mean_t corr(alpha_a_t, alpha_b_t)
```

mutual IC 需要缓存，因为 alpha pool 反复用到。

### alpha pool 更新

当 RL 生成一个新因子：

1. 计算该因子值。
2. 计算或读取该因子的 IC。
3. 计算或读取它与已有 pool 因子的 mutual IC。
4. 将新因子加入 pool。
5. 优化线性组合权重。
6. 如果 pool 大小超过 30，删除绝对权重最小的因子。
7. 使用新组合的 IC 作为 RL episode return。

### 缓存内容

每个因子应缓存：

- RPN token。
- 可读公式字符串。
- 训练期 IC。
- 训练期 RankIC。
- 验证期 IC。
- 验证期 RankIC。
- 与 pool 因子的 mutual IC。
- 组合权重。
- 使用的数据区间。

### 验收标准

- 固定手写公式列表可以进入 alpha pool。
- pool 权重优化能运行。
- pool 超过 30 时能正确删除最小绝对权重因子。
- 组合 IC 能被计算和记录。
- 所有评估只使用指定训练区间。

## 阶段 5：RL 公式生成器

### 阶段目标

实现论文中的 PPO 公式生成器，让模型能生成合法公式，并通过 alpha pool 奖励学习。

### 主要交付物

- RL 环境。
- LSTM policy/value 网络。
- PPO 算法。
- invalid action masking。
- smoke run。
- pilot run。
- long run 训练脚本。
- `03_rl_alpha_mining.ipynb`。

### MDP 定义

状态：

```text
当前 token 序列
```

动作：

```text
下一个 token
```

状态转移：

```text
把动作 token 追加到当前序列
```

奖励：

- 中间步骤：`0`
- 终止且公式有效：新 alpha pool 的组合 IC
- 终止但公式语义非法：`-1`

折扣因子：

```text
gamma = 1
```

### PPO 网络结构

按论文 appendix 设置：

- 共享 LSTM 特征提取器。
- LSTM 层数：2。
- hidden size：128。
- dropout：0.1。
- policy head：两层 MLP，每层 64。
- value head：两层 MLP，每层 64。
- PPO clip epsilon：0.2。

### 三段式训练

#### 1. Smoke run

目的：验证工程链路。

规模：

- 小股票池。
- 短时间区间。
- 极少 episode。

必须验证：

- 环境能 reset 和 step。
- action mask 生效。
- 能生成合法公式。
- 能计算奖励。
- PPO 参数能更新。

#### 2. Pilot run

目的：观察生成质量。

规模：

- 中等样本。
- 足够生成若干有效公式。

必须检查：

- 公式是否过于单一。
- invalid ratio 是否过高。
- reward 是否有明显变化。
- alpha pool 是否正常更新。

#### 3. Long run

目的：生成最终候选因子。

规模：

- Top1500 股票池。
- 2015-2020 训练区间。
- 使用 GPU。

输出：

- 候选公式。
- 最终 30 因子池。
- 训练和验证指标。

### 可复现记录

每次训练必须保存：

- 随机种子。
- 配置文件。
- 数据区间。
- 股票池定义。
- 包版本。
- 使用设备。
- 生成公式。
- 训练指标。
- 验证指标。

### 验收标准

- RL 能稳定生成合法公式。
- invalid 公式不会进入 alpha pool。
- 至少能完成 pilot run。
- long run 后能得到 30 个可用公式因子。
- Notebook 能展示公式、权重、IC、RankIC 和训练过程。

## 阶段 6：XGBoost 截面预测扩展

### 阶段目标

将最终 30 个 RL 公式因子作为特征，训练 XGBoost 做未来 20 日截面 rank 预测。

### 主要交付物

- 因子矩阵。
- XGBoost 数据集构建器。
- walk-forward 训练器。
- 预测分数。
- 特征重要性。
- 分组收益分析。
- `04_xgboost_cross_section.ipynb`。

### 特征

第一版只使用：

```text
30 个 RL 生成公式因子
```

不加入原始 OHLCV。这样可以清楚判断生成公式因子的价值。

### 目标

每个交易日内对未来 20 日收益做截面百分位排名：

```text
future_20d_rank = rank_pct(future_20d_return)
```

模型预测的是截面排名分数，而不是原始收益。

### 初始 XGBoost 参数

```yaml
objective: reg:squarederror
max_depth: 4
learning_rate: 0.03
n_estimators: 500
subsample: 0.8
colsample_bytree: 0.8
reg_lambda: 5
tree_method: hist
```

如果 XGBoost GPU 支持顺利，可以开启 GPU；否则先用 CPU，保证实验正确性。

### walk-forward 训练

对测试期每个月：

1. 取预测月之前 3 年样本作为训练集。
2. 训练 XGBoost。
3. 预测下月调仓日的股票分数。
4. 选出 Top10%。
5. 输出给回测模块。

### 验收标准

- 训练窗口不包含预测月未来数据。
- 每个月都有预测分数。
- 能计算预测 IC 和 RankIC。
- 能输出特征重要性。
- 能做分组收益分析。

## 阶段 7：组合回测与结果评估

### 阶段目标

把 XGBoost 预测分数转换成可执行的月度多头组合，并评估策略表现。

### 主要交付物

- 回测引擎。
- 交易约束模块。
- 成本模型。
- 绩效指标。
- 净值图和回撤图。
- `05_backtest_report.ipynb`。

### 策略规则

每月调仓：

1. 月末根据 XGBoost 分数排序。
2. 选取当前 Top1500 股票池中分数最高的 Top10%。
3. 等权配置。
4. 下一交易日开盘成交。
5. 持有到下一个月度调仓。

### 交易约束

第一版使用研究级真实约束：

- 涨停不可买。
- 跌停不可卖。
- 停牌不可交易。
- 开盘价缺失不可交易。
- 无法卖出的旧持仓继续保留。
- 无法买入的资金保留为现金。

### 成本模型

默认参数：

```yaml
buy_cost_bps: 15
sell_cost_bps: 20
```

含义：

- 买入成本近似包含佣金和滑点。
- 卖出成本近似包含佣金、滑点和税费。
- 成本参数必须可配置。

### 绩效指标

至少报告：

- 累计收益。
- 年化收益。
- 年化波动。
- 最大回撤。
- Sharpe。
- Calmar。
- 月度胜率。
- 换手率。
- 持仓数量。
- 买入失败次数。
- 卖出失败次数。
- 总交易成本。
- 含成本与不含成本差异。

### 基准

初始基准：

- Top1500 等权。
- CSI300 指数或 ETF 代理。
- 线性 alpha pool Top10% 组合。

### 验收标准

- 能生成连续净值曲线。
- 能展示回撤曲线。
- 能输出完整绩效表。
- 能解释交易成本和换手对收益的影响。
- 所有交易都符合信号生成和下一开盘执行口径。

## 阶段 8：测试、文档与最终报告

### 阶段目标

把项目从“能跑”整理成“可复现、可检查、可继续扩展”的研究工程。

### 主要交付物

- 完整 README。
- 完整 Notebook 报告。
- 单元测试。
- smoke pipeline。
- 最终研究总结。

### 单元测试

表达式测试：

- RPN parser 接受合法公式。
- RPN parser 拒绝非法公式。
- 公式字符串渲染稳定。
- action mask 不允许形式非法动作。
- SEP 只在表达式完整时允许。

算子测试：

- Abs、Log、Add、Sub、Mul、Div。
- Greater、Less。
- Ref、Mean、Med、Sum。
- Std、Var、Max、Min、Mad。
- Delta、WMA、EMA。
- Cov、Corr。
- 除零、log 非正数、缺失值、短窗口。

指标测试：

- IC 与手算样例一致。
- RankIC 与手算样例一致。
- mutual IC 缓存对称。
- 截面标准化均值接近 0，标准差接近 1。

alpha pool 测试：

- 新因子加入后指标更新。
- pool cap 删除绝对权重最小因子。
- toy example 中权重优化能降低 loss 或提高组合 IC。
- invalid alpha 不进入 pool。

回测测试：

- 等权组合权重和为 1。
- 成本扣减正确。
- 买入失败后资金保留为现金。
- 卖出失败后持仓继续保留。
- 换手率计算正确。
- 当前组合不使用未来信号。

### 泄漏测试

必须防止：

- 未来 20 日标签进入当日因子计算。
- 股票池选择使用未来流动性。
- XGBoost 训练数据包含预测月之后的数据。
- RL 训练使用验证或测试标签。
- 测试集结果反向影响模型选择。

### smoke pipeline

提供一个小规模端到端命令：

```powershell
python -m quant_rl_alpha.cli smoke
```

smoke pipeline 应完成：

- 加载小样本数据。
- 评估几个公式。
- 计算 IC 和 RankIC。
- 更新一个小 alpha pool。
- 训练 PPO 几个 episode。
- 训练一个极小 XGBoost。
- 跑一个迷你回测。

smoke test 只验证工程链路，不要求策略表现。

### 验收标准

- README 能指导新环境复现 smoke pipeline。
- 核心模块有单元测试。
- Notebook 能按顺序解释数据、公式、RL、XGBoost 和回测。
- 最终报告说明哪些结果有效、哪些地方不可靠、下一步应怎么扩展。

## 8. 配置文件草案

### 8.1 `config/data.yml`

```yaml
provider: akshare
start_date: "2015-01-01"
end_date: null
adjust: qfq
cache_format: parquet
exclude_bj: true
retry_times: 3
sleep_seconds: 0.2
```

### 8.2 `config/universe.yml`

```yaml
min_listed_days: 250
liquidity_window: 20
top_n: 1500
exclude_st: true
exclude_zero_volume: true
rebalance_frequency: monthly
```

### 8.3 `config/expression.yml`

```yaml
max_tokens: 20
features:
  - open
  - close
  - high
  - low
  - volume
  - vwap
constants:
  - -30
  - -10
  - -5
  - -2
  - -1
  - -0.5
  - -0.01
  - 0.01
  - 0.5
  - 1
  - 2
  - 5
  - 10
  - 30
time_deltas:
  - 10
  - 20
  - 30
  - 40
  - 50
```

### 8.4 `config/rl.yml`

```yaml
algorithm: ppo
gamma: 1.0
clip_epsilon: 0.2
lstm_layers: 2
lstm_hidden_size: 128
dropout: 0.1
head_hidden_size: 64
pool_size: 30
invalid_reward: -1.0
device: cuda
```

### 8.5 `config/xgboost.yml`

```yaml
target: future_20d_rank
train_window_years: 3
rebalance_frequency: monthly
model:
  objective: reg:squarederror
  max_depth: 4
  learning_rate: 0.03
  n_estimators: 500
  subsample: 0.8
  colsample_bytree: 0.8
  reg_lambda: 5
  tree_method: hist
```

### 8.6 `config/backtest.yml`

```yaml
portfolio:
  selection: top_pct
  top_pct: 0.10
  weighting: equal
  rebalance_frequency: monthly
execution:
  signal_time: month_end_close
  trade_time: next_open
  cannot_buy_limit_up: true
  cannot_sell_limit_down: true
costs:
  buy_cost_bps: 15
  sell_cost_bps: 20
```

## 9. 主要风险与应对

### 9.1 数据质量风险

风险：

- AKShare 数据可能缺失、变动或接口不稳定。

应对：

- 缓存原始数据。
- 生成质量报告。
- 下载支持重试。
- 数据源适配层隔离。

### 9.2 计算资源风险

风险：

- RL 训练慢且不稳定。

应对：

- 分 smoke、pilot、long run 三步。
- 保存 checkpoint。
- 持续保存生成公式。
- CPU smoke test 和 GPU long run 分开。

### 9.3 过拟合风险

风险：

- 公式生成器可能过拟合训练期。
- XGBoost 可能过拟合噪声标签。

应对：

- 严格训练、验证、测试切分。
- 用验证集选择模型和 seed。
- 最终测试集只用于最终评估。
- 公式长度限制为 20。
- XGBoost 使用 walk-forward。

### 9.4 未来函数风险

风险：

- 未来收益可能意外进入因子、股票池、模型训练或回测。

应对：

- 单独写泄漏测试。
- 因子生成与标签生成分离。
- 所有 rolling 计算只用历史窗口。
- walk-forward 强制按日期切分。

### 9.5 可复现风险

风险：

- 不同 seed、依赖版本、数据日期会导致公式和结果变化。

应对：

- 保存配置、seed、包版本、公式和指标。
- 尽量设置随机种子。
- 不依赖单次训练结果下结论。

## 10. 第一版完成标准

第一版完成时，仓库应包含：

1. 可运行的项目环境和依赖文件。
2. AKShare 数据下载和缓存。
3. 月度 Top1500 股票池。
4. RPN 表达式引擎。
5. 安全公式求值器。
6. PPO 强化学习公式生成器。
7. 论文线性 alpha pool 复现。
8. 最终 30 个公式因子。
9. XGBoost 截面预测流程。
10. Top10% 多头组合回测。
11. 中文 Notebook 研究报告。
12. 可复现实验 README。
13. 核心逻辑单元测试。
14. 最终研究总结：结果、限制和下一步。

## 11. 下一步执行建议

下一步应从 **阶段 0：项目骨架与工程约定** 开始。

推荐第一批任务：

1. 创建仓库结构。
2. 创建 `.venv` 和依赖文件。
3. 写入 `config/*.yml`。
4. 实现路径、日志、随机种子工具。
5. 建立最小 `pytest` 测试。

完成这些后再进入阶段 1 的 AKShare 数据下载和质量检查。

最重要的原则：

```text
先保证数据和无未来函数，再追求模型效果。
```
