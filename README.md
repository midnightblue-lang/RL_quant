# 强化学习公式因子挖掘项目

这是一个个人量化研究项目，目标是先搭建可靠的数据、标签、回测和评估基础，再逐步进入公式因子挖掘、RL alpha 生成和 XGBoost 截面预测。

当前阶段：**阶段 2：股票池与标签基础构建**，阶段 1 数据下载、缓存与质量检查链路已保留。

## 当前边界

- 当前已搭建项目骨架、配置、基础工具、阶段 1 数据模块和阶段 2 股票池/标签模块。
- 暂不接入真实交易、模拟盘、券商 API 或自动下单。
- 暂不实现 RL 公式生成器和 XGBoost 训练。
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

如果当前机器没有系统 Python，可以先使用 Codex 内置 Python 或后续安装标准 Python，再创建 `.venv`。
