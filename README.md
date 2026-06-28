# 强化学习公式因子挖掘项目

这是一个个人量化研究项目，目标是先搭建可靠的数据、标签、回测和评估基础，再逐步进入公式因子挖掘、RL alpha 生成和 XGBoost 截面预测。

当前阶段：**阶段 6 XGBoost 截面预测扩展验收**。阶段 1 数据链路、阶段 2 股票池/标签链路和阶段 2.5 全量数据审计入口已保留；阶段 3 表达式系统、阶段 4 alpha pool、阶段 5 PPO 公式生成器、训练入口和报告入口已实现；阶段 6 宽表数据集、逐月 walk-forward 预测和评估入口已实现。

## 当前边界

- 当前已搭建项目骨架、配置、基础工具、阶段 1-2.5 数据模块、阶段 3 表达式系统、阶段 4 alpha pool、阶段 5 PPO 训练入口和阶段 6 XGBoost 截面预测入口。
- 暂不接入真实交易、模拟盘、券商 API 或自动下单。
- 当前阶段 6 只推进 XGBoost 截面预测和研究评估，不进入阶段 7 组合回测。
- 当前产物只用于研究复现和工程验证，不构成投资建议。
- 数据、缓存、模型和报告输出不提交到仓库。

## 推荐执行顺序

1. 阶段 0：项目骨架、配置、路径、日志、随机种子、测试框架。
2. 阶段 1：AKShare 数据下载、缓存和质量检查。
3. 阶段 2：股票池和未来收益标签。
4. 阶段 3：表达式系统。
5. 阶段 4：因子评估和 alpha pool。
6. 阶段 5：RL 公式生成。
7. 阶段 6：XGBoost 截面预测。
8. 阶段 7：组合回测。
9. 阶段 8：测试、文档和最终报告。

详细计划见 [PROJECT_PLAN.md](PROJECT_PLAN.md)，协作规则见 [AGENTS.md](AGENTS.md)。

## 环境准备

建议使用 Python 3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
```

PyTorch CUDA 版本通常需要按本机显卡和驱动单独安装，阶段 0 不强制安装。

## 阶段 0 验证

```powershell
python -m pytest
```

## 阶段 1 小样本下载和质量报告

当前 `config/data.yml` 默认使用 AKShare 的 `daily_sina` 端点，因为本机网络下东方财富 `hist_em` 端点出现代理连接拒绝。执行：

```powershell
python -m quant_rl_alpha.cli stage1-sample
```

该命令会：

1. 下载 `config/data.yml` 中的 `sample_symbols`。
2. 保存 AKShare 原始返回到 `data/raw/akshare/hist/`。
3. 保存标准化日频数据到 `data/interim/akshare/daily/`。
4. 输出下载结果到 `data/reports/download_results.csv`。
5. 输出质量报告到 `data/reports/data_quality.md`。

数据清洗口径见 [docs/DATA_CLEANING.md](docs/DATA_CLEANING.md)。

## 阶段 2 股票池与标签

在阶段 1 标准化缓存存在后，执行：

```powershell
python -m quant_rl_alpha.cli stage2-build
```

该命令会：

1. 合并 `data/interim/akshare/daily/` 中的标准化日频缓存。
2. 输出 `data/interim/daily_panel.parquet`。
3. 构建月度股票池并输出 `data/processed/universe_monthly.parquet`。
4. 构建未来 20 个市场交易日收益标签并输出 `data/processed/labels.parquet`。

股票池和标签口径见 [docs/UNIVERSE_AND_LABELS.md](docs/UNIVERSE_AND_LABELS.md)。

## 阶段 2.5 全量数据接入与审计

如果要把 AKShare 当前可获取的 A 股日频数据完整缓存到本地，执行：

```powershell
python -m quant_rl_alpha.cli download-full
```

如果希望全量下载后立刻重建 daily panel、月度股票池和未来 20 日标签，执行：

```powershell
python -m quant_rl_alpha.cli stage25-full-data
```

全量下载支持断点续传：当 raw 和 standard 缓存都存在时会按 `skip_existing: true` 跳过该股票。输出包括股票列表、下载 manifest、失败列表、全市场质量报告和摘要报告。详细口径见 [docs/FULL_DATA_INGESTION.md](docs/FULL_DATA_INGESTION.md)。

## 阶段 3-5 公式因子挖掘

阶段 3-4 不依赖 PyTorch，可直接运行单元测试验证 RPN、算子、action mask、IC/RankIC 和 alpha pool。阶段 5 训练前需要按 PyTorch 官方 selector 为当前 `.venv` 安装合适的 CUDA 版 PyTorch，并确认：

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

正式训练入口：

```powershell
python -m quant_rl_alpha.cli stage5-train
```

该命令读取 `config/rl.yml`，使用 `data/interim/daily_panel.parquet` 和 `data/processed/labels.parquet` 的训练期样本训练 PPO 公式生成器，并输出：

1. `data/features/rl_alpha_pool.parquet`
2. `data/reports/rl_training_metrics.parquet`
3. `data/reports/rl_validation_metrics.parquet`
4. `data/reports/rl_training_config.yml`

训练标签要求 `label_end_date` 不超过训练期边界，验证指标要求 `label_end_date` 不超过验证期边界。

阶段 3 表达式系统维护一个最小量纲约束：`open/high/low/close/vwap` 视为价格，`volume` 视为成交量，常数只允许作为无量纲偏移或缩放使用。明显无意义的价格-成交量、价格乘价格、价格和常数直接比较等公式会在 parser/action mask 中被拒绝。阶段 4 的 mutual IC 只进入 pool loss、缓存和报告，不作为加入 pool 前的阈值筛选条件。

训练前或训练后都可以生成可视化报告：

```powershell
python -m quant_rl_alpha.cli stage5-report
```

训练产物缺失时，报告输出 pre-flight 视图；训练完成后，报告展示最终因子池、训练诊断、验证指标和 mutual IC 热力图。报告路径为 `data/reports/rl_factor_report.html`。

如果当前机器没有系统 Python，可以先使用 Codex 内置 Python 或后续安装标准 Python，再创建 `.venv`。

## 阶段 6 XGBoost 截面预测

阶段 6 只做研究预测和模型评估，不做组合回测，也不接入真实交易、模拟盘、券商 API 或自动下单。配置集中在 `config/xgboost.yml`，第一版只使用 `data/features/rl_alpha_pool.parquet` 中的 30 个 RL 公式因子预测 `future_20d_rank`。

构建 XGBoost 宽表数据集：

```powershell
python -m quant_rl_alpha.cli stage6-build-dataset
```

该命令会读取 `data/interim/daily_panel.parquet`、`data/processed/labels.parquet` 和 `data/features/rl_alpha_pool.parquet`，对 alpha pool 中每条 `tokens` 调用当前 `parse_rpn` 校验量纲和 RPN 合法性。若旧产物中存在当前规则下非法的公式，会直接失败，不会静默过滤或替换。

逐月 walk-forward 训练和预测：

```powershell
python -m quant_rl_alpha.cli stage6-train
```

该命令会在数据集缺失时先构建宽表，然后从 `prediction_start` 开始逐月训练 XGBoost。每个预测月只使用过去 3 年、`date < prediction_date` 且 `label_end_date <= prediction_date` 的样本，避免未来标签进入训练集。输出包括：

1. `data/features/xgboost_dataset.parquet`
2. `data/reports/xgboost_predictions.parquet`
3. `data/reports/xgboost_metrics.parquet`
4. `data/reports/xgboost_feature_importance.parquet`

Notebook 入口为 `notebooks/04_xgboost_cross_section.ipynb`，用于查看数据覆盖、月度 IC/RankIC、预测分数分布、feature importance 和非回测的分组收益分析。
