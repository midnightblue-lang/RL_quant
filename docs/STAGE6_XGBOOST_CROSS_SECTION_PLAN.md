# 阶段 6：XGBoost 截面预测实施计划

## 目标

在阶段 3-5 生成的 30 个 RL 公式因子基础上，构建月度截面特征矩阵，并用 XGBoost 预测未来 20 日截面收益排名。

本阶段只做研究预测和模型评估，不进入组合回测，不接入真实交易、模拟盘、券商 API 或自动下单。

## 当前默认决策

- 第一版只使用 30 个 RL 公式因子，不加入原始 OHLCV、行业、市值或基本面特征。
- 预测目标固定为 `future_20d_rank`。
- 模型使用 `reg:squarederror` 回归截面 rank，不使用排序学习 objective。
- walk-forward 训练窗口固定 3 年。
- 预测期从 2022-01-01 起，到当前标签可用的最后月份。
- XGBoost 第一版使用 CPU，`tree_method: hist`。
- 缺失因子值保留为 NaN，由 XGBoost 原生处理。
- 阶段 6 产物只用于研究复现，不构成投资建议。

## 前置条件

- 正式使用前，`data/features/rl_alpha_pool.parquet` 应替换为当前量纲规则下的阶段 5 产物。
- 阶段 6 代码必须在加载 alpha pool 时用当前 `parse_rpn` 校验 tokens；如存在无法解析或量纲非法公式，应 fail fast。
- 当前 `.venv` 中尚未安装 `xgboost`，实施阶段需安装 `requirements.txt` 中的 `xgboost`，并在 README 说明原因。
- 不要为了阶段 6 绕过阶段 3-5 的量纲约束、无未来函数约束或数据边界。

## 主要交付物

- `model/dataset.py`：XGBoost 特征矩阵构建器。
- `model/xgboost_ranker.py`：walk-forward 训练、预测和评估。
- CLI 命令：
  - `stage6-build-dataset`
  - `stage6-train`
- `notebooks/04_xgboost_cross_section.ipynb`
- 输出数据：
  - `data/features/xgboost_dataset.parquet`
  - `data/reports/xgboost_predictions.parquet`
  - `data/reports/xgboost_metrics.parquet`
  - `data/reports/xgboost_feature_importance.parquet`

## 配置变更

扩展 `config/xgboost.yml`，保留当前模型参数，并增加输入输出路径。

建议结构：

```yaml
target: future_20d_rank
train_window_years: 3
prediction_start: "2022-01-01"
rebalance_frequency: monthly
features:
  winsorize_lower: 0.01
  winsorize_upper: 0.99
  standardize: true
data:
  daily_panel: data/interim/daily_panel.parquet
  labels: data/processed/labels.parquet
  alpha_pool: data/features/rl_alpha_pool.parquet
outputs:
  dataset: data/features/xgboost_dataset.parquet
  predictions: data/reports/xgboost_predictions.parquet
  metrics: data/reports/xgboost_metrics.parquet
  feature_importance: data/reports/xgboost_feature_importance.parquet
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

## 阶段 6.1：构建 XGBoost 数据集

实现 `model/dataset.py`。

核心流程：

1. 读取 `daily_panel`、`labels` 和 `alpha_pool`。
2. 对 alpha pool 中每条 `tokens` 调用当前 `parse_rpn`，无法解析则直接报错。
3. 对每个公式在 `daily_panel` 上求值。
4. 只保留 `labels` 中已有的月度 `(date, symbol)` 样本。
5. 合并 `label_end_date`、`future_20d_return`、`future_20d_rank`。
6. 每个因子按交易日截面 winsorize 1%/99%。
7. 每个因子按交易日截面 z-score 标准化。
8. 输出宽表。

宽表字段：

- `date`
- `symbol`
- `label_end_date`
- `future_20d_return`
- `future_20d_rank`
- `alpha_00` 至 `alpha_29`

关键规则：

- 不在 notebook 中散落关键参数。
- 不填充缺失值。
- 不因部分因子缺失而删除整行，除非目标标签缺失。
- 不使用预测月之后的数据参与任何特征处理。
- 因子计算只允许使用当日及历史行情，沿用表达式系统中的 rolling/ref 约束。

## 阶段 6.2：Walk-Forward 训练与预测

实现 `model/xgboost_ranker.py`。

对每个预测月 `d`：

1. 预测集：`date == d`。
2. 训练集：`date < d`。
3. 训练集额外要求：`label_end_date <= d`，避免标签未来函数。
4. 训练集时间窗口：`date >= d - 3 years`。
5. 使用 `future_20d_rank` 作为目标训练 XGBoost。
6. 对预测集输出 `score`。
7. 在当月截面内计算 `score_rank_pct`。
8. 计算当月预测 IC、RankIC、样本数、训练样本数和特征覆盖率。
9. 汇总 feature importance。

预测结果字段：

- `date`
- `symbol`
- `score`
- `score_rank_pct`
- `future_20d_return`
- `future_20d_rank`

月度指标字段：

- `date`
- `prediction_count`
- `train_count`
- `ic`
- `rank_ic`
- `feature_coverage`

特征重要性字段：

- `feature`
- `gain`
- `weight`
- `cover`（如可得）

## 阶段 6.3：CLI 与 Notebook

新增 CLI：

```powershell
.venv\Scripts\python.exe -m quant_rl_alpha.cli stage6-build-dataset
.venv\Scripts\python.exe -m quant_rl_alpha.cli stage6-train
```

CLI 行为：

- `stage6-build-dataset`：构建并缓存 XGBoost 数据集。
- `stage6-train`：若数据集不存在则先构建，再执行 walk-forward 训练预测。
- 命令输出应打印 dataset、predictions、metrics、feature importance 的路径和行数。

新增 `notebooks/04_xgboost_cross_section.ipynb`：

- 展示数据集覆盖情况。
- 展示月度 IC/RankIC。
- 展示预测分数分布。
- 展示 feature importance。
- 做分组收益雏形分析，但不进入真实回测。

## 测试要求

- toy alpha pool 公式能生成宽表，列名和行数符合预期。
- stale 或非法 alpha pool tokens 会 fail fast。
- walk-forward 训练严格排除 `label_end_date > prediction_date` 的样本。
- 每个月预测结果只包含当月截面。
- 预测 IC、RankIC 与手算 toy case 一致。
- feature importance 输出包含训练中使用过的因子。
- CLI 的 `stage6-build-dataset` 和 `stage6-train` 能被 monkeypatch 测试覆盖。
- Notebook JSON 可解析。

验收命令：

```powershell
.venv\Scripts\ruff.exe check src tests
.venv\Scripts\pytest.exe
.venv\Scripts\python.exe -m quant_rl_alpha.cli stage6-build-dataset
.venv\Scripts\python.exe -m quant_rl_alpha.cli stage6-train
```

## 验收标准

- 可以从阶段 5 的公式因子池生成 XGBoost 宽表。
- 所有特征名、目标列和日期边界清晰可追踪。
- walk-forward 训练不包含预测月未来数据。
- 每个预测月都有预测分数和 IC/RankIC 指标。
- 能输出 feature importance。
- Notebook 能解释数据、模型、预测指标和主要限制。
- 不保留临时 smoke/debug 文件。
- 不进入阶段 7 回测。

## 给另一个对话的 Prompt

```text
你将接手 E:\codex_projects\quant 项目的阶段 6：XGBoost 截面预测扩展。

开始前必须阅读：

1. AGENTS.md
2. PROJECT_PLAN.md
3. docs/STAGE3_5_RL_ALPHA_REPRODUCTION_PLAN.md
4. docs/STAGE6_XGBOOST_CROSS_SECTION_PLAN.md
5. docs/DATA_CLEANING.md
6. docs/UNIVERSE_AND_LABELS.md
7. 当前阶段 3-5 相关代码：
   - src/quant_rl_alpha/expression/
   - src/quant_rl_alpha/alpha/
   - src/quant_rl_alpha/rl/

当前任务不是接入交易，也不是做组合回测，而是实现阶段 6：

1. 构建 XGBoost 宽表特征矩阵。
2. 用 30 个 RL 公式因子预测 `future_20d_rank`。
3. 做逐月 walk-forward 训练预测。
4. 输出预测 IC、RankIC、feature importance 和 Notebook。

请严格遵守：

- 代码保持简洁、紧凑、直接，不写大量无用兜底。
- 不要把简单表达式拆成很多只有一两个字符的短行。
- 不要引入 FinRL、Qlib、RD-Agent 等大型框架作为项目依赖。
- 不接入真实交易、券商 API、模拟盘或自动下单。
- 不进入阶段 7 回测。
- 所有实验配置进入 `config/*.yml`。
- 不允许未来函数。
- 不要为了模型效果绕过阶段 3-5 的量纲约束。
- 子代理可以辅助只读审查，但主代理必须复核后再合并。

当前重要事实：

- `labels.parquet` 已包含 `future_20d_return` 和 `future_20d_rank`。
- `model/` 目录目前基本是空壳。
- `config/xgboost.yml` 目前只有基础参数，需要扩展输入输出路径。
- 当前 `.venv` 里可能尚未安装 `xgboost`；如缺失，请说明原因后安装 `requirements.txt` 中的依赖。
- 正式使用前，`data/features/rl_alpha_pool.parquet` 应是当前量纲规则下的阶段 5 产物。阶段 6 加载时必须用当前 `parse_rpn` 校验 tokens，非法则 fail fast。

建议实现顺序：

1. 只读探索当前 `model/`、`cli.py`、`config/xgboost.yml`、labels schema 和 alpha pool schema。
2. 扩展 `config/xgboost.yml`。
3. 实现 `model/dataset.py`。
4. 实现 `model/xgboost_ranker.py`。
5. 增加 CLI 命令 `stage6-build-dataset` 和 `stage6-train`。
6. 增加单元测试。
7. 新增 `notebooks/04_xgboost_cross_section.ipynb`。
8. 更新 README。
9. 运行验收命令：
   - `.venv\Scripts\ruff.exe check src tests`
   - `.venv\Scripts\pytest.exe`
   - `.venv\Scripts\python.exe -m quant_rl_alpha.cli stage6-build-dataset`
   - `.venv\Scripts\python.exe -m quant_rl_alpha.cli stage6-train`

最终交付时请说明：

- 新增了哪些模块和 CLI。
- 是否安装或确认了 XGBoost。
- 是否生成了数据集、预测、指标和 feature importance。
- 是否通过所有测试。
- 是否仍存在阻塞进入阶段 7 的问题。
```
