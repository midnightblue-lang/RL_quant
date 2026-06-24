# 阶段 2.5：全量数据接入与审计

本阶段的目标是把 AKShare 当前可获取的 A 股日频数据完整缓存到本地，并生成可复查的下载和质量报告。它是阶段 1 小样本数据链路的放大版，不改变清洗口径，也不提前进入真实交易或模型训练。

## 全量数据的边界

- “全量”指 AKShare 当前股票列表中可获取的 A 股日频历史数据。
- 这不是严格无幸存者偏差的历史全市场重建。已经退市且不在当前列表中的股票可能不会被纳入。
- 默认沿用 `config/data.yml` 中的 `symbol_list_endpoint: spot_sina` 和 `exclude_bj: true`，第一版排除北交所股票。
- 股票名称来自当前代码表，历史 ST 状态和退市状态仍可能存在误差，后续股票池和回测阶段需要继续审查。

## 执行命令

只下载全量数据并生成审计报告：

```powershell
python -m quant_rl_alpha.cli download-full
```

下载全量数据、生成审计报告，并继续重建阶段 2 的 daily panel、月度股票池和标签：

```powershell
python -m quant_rl_alpha.cli stage25-full-data
```

全量下载可能耗时很久。如果中途停止，重新执行命令会根据 `skip_existing: true` 跳过已经同时存在 raw 和 standard 缓存的股票。

`symbol_list_endpoint` 控制股票列表来源，当前支持：

- `spot_sina`：使用 AKShare 的 `stock_zh_a_spot()`，当前网络下可用。
- `code_name`：使用 AKShare 的 `stock_info_a_code_name()`，在当前代理下访问上交所列表可能失败。

## 输出文件

默认输出位置由 `config/data.yml` 的 `full_market.paths` 控制：

| 文件 | 用途 |
| --- | --- |
| `data/reports/full_a_stock_symbols.csv` | 本次使用的股票列表 |
| `data/reports/full_download_manifest.csv` | 每只股票的下载状态、行数、缓存路径和错误信息 |
| `data/reports/full_download_failures.csv` | 下载失败股票的子集 |
| `data/reports/full_data_quality.md` | 全市场标准化缓存的质量检查明细 |
| `data/reports/full_data_ingestion.md` | 全量接入摘要报告 |

行情缓存仍然使用阶段 1 的分层：

```text
data/raw/akshare/hist/{symbol}.parquet
data/interim/akshare/daily/{symbol}.parquet
```

## 清洗和质量规则

阶段 2.5 必须沿用 `docs/DATA_CLEANING.md` 的口径：

- 原始 AKShare 返回和标准化行情分开保存。
- 只做字段标准化、日期解析、数值类型转换、成交量单位统一和排序。
- `hist_em` 的成交量按“手”转为“股”，`daily_sina` 的成交量按“股”处理。
- `vwap = amount / volume`，但前复权 OHLC 与原始成交额/成交量可能口径不一致，因此 `vwap_outside_bar_count` 只作为提示，不自动修复。
- 不前向填充 OHLCV。
- 不伪造停牌交易日。
- 不自动删除异常涨跌幅样本。
- 同一股票同一日期重复行情直接报错。
- 缺少关键字段直接报错。
- 跳过已有缓存前，必须确认标准化文件中的 `source` 和 `adjust` 与本次配置一致；如果端点或复权口径不同，应重新下载。

质量报告至少关注：

- 行数过少。
- 日期范围。
- 重复交易日。
- 必需字段缺失。
- OHLCV 缺失。
- 非正价格。
- 负成交量或负成交额。
- OHLC 逻辑不一致。
- 单日收盘收益异常。
- VWAP 明显落在 OHLC 区间外。
- 成交量和成交额的组合矛盾，例如 `volume > 0` 但 `amount <= 0`。

## Manifest 重点字段

全量下载的 manifest 不只记录成功或失败，也记录本次运行和缓存口径：

- `run_id`、`provider`、`endpoint`、`adjust`、`start_date`、`end_date`。
- `retry_times`、`attempts`、`error_type`、`error`。
- `raw_rows`、`standard_rows`、`first_date`、`last_date`。
- `source_in_file`、`adjust_in_file`，用于发现端点或复权口径混用。
- `has_quality_issue`，用于把质量报告中的问题回写到股票级 manifest。
- `download_started_at`、`download_finished_at`、`updated_at`。

## 后续使用方式

全量缓存完成后，后续实验不需要重新下载数据。小规模实验应通过配置或读取层选择子集，例如：

- 使用少数股票做单元测试或调试。
- 使用 Top500/Top1500 做 pilot run。
- 使用全量缓存重建正式股票池和标签。

不要为了小实验删除全量缓存，也不要把全量缓存提交进 git。
