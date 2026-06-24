# 数据清洗口径

本项目第一版数据源为 AKShare，阶段 1 只做“字段标准化 + 质量检查 + 缓存”，不做复杂自动修复。数据质量问题应优先暴露出来，避免后续因子、回测和模型训练建立在悄悄修过的数据上。

## 参考资料

- AKShare 股票数据文档：https://akshare.akfamily.xyz/data/stock/stock.html
- Qlib 数据组件文档：https://qlib.readthedocs.io/en/latest/component/data.html
- 中国 A股涨跌停机制背景：https://zh.wikipedia.org/wiki/%E9%99%90%E4%BB%B7
- 中国市场涨跌停研究背景：https://arxiv.org/abs/1503.03548
- 中国市场停牌研究背景：https://arxiv.org/abs/1309.1138

## 接口选择

阶段 1 支持两个 AKShare 端点：

- `hist_em`：`stock_zh_a_hist`，东方财富历史行情接口，字段为中文，成交量常见单位为“手”。
- `daily_sina`：`stock_zh_a_daily`，新浪日频接口，字段多为英文，成交量为“股”。

当前环境中东方财富历史接口出现代理连接拒绝，因此 `config/data.yml` 默认使用 `daily_sina`。这不是隐式 fallback；如果要切换端点，应显式修改 `endpoint` 配置。

## 字段标准化

AKShare A股历史行情接口通常返回中文字段。项目内部统一为：

| 内部字段 | 含义 |
| --- | --- |
| date | 交易日期 |
| symbol | 6 位股票代码 |
| name | 股票名称，可为空 |
| open | 前复权开盘价 |
| high | 前复权最高价 |
| low | 前复权最低价 |
| close | 前复权收盘价 |
| volume | 成交量，统一转换为“股” |
| amount | 成交额，单位元 |
| vwap | 近似成交均价，按 `amount / volume` 计算 |
| turnover | 换手率，可为空 |
| source | 数据源 |
| adjust | 复权方式 |

注意：AKShare `stock_zh_a_hist` 文档说明成交量单位是“手”、成交额单位是“元”，项目内部会把 `hist_em` 成交量转换为“股”。`daily_sina` 成交量按“股”处理。`vwap` 使用成交额除以成交股数得到；如果 OHLC 使用前复权价格，`vwap` 与复权 OHLC 在除权除息附近可能存在口径不完全一致的问题。阶段 1 先记录并检查，不做额外复权因子推导。

## 缓存分层

- `data/raw/akshare/hist/{symbol}.parquet`：保存 AKShare 原始返回字段，不覆盖、不静默修正。
- `data/interim/akshare/daily/{symbol}.parquet`：保存标准化后的项目内部字段。
- `data/reports/data_quality.md`：保存质量报告。

## 清洗原则

- 不静默填充价格或成交量。
- 不把停牌日伪造成正常交易日。
- 不随意删除异常涨跌幅，只在质量报告中标记。
- 不对缺失 OHLCV 做前向填充。
- 只在字段解析阶段做必要的类型转换、排序和单位统一。
- 如果原始数据缺少关键字段，应直接报错。
- 如果同一股票同一日期出现重复行情，应直接报错。

## 质量检查项

阶段 1 至少检查：

- 行数是否低于 `min_rows`。
- 日期范围。
- 重复交易日。
- 必需字段缺失。
- OHLC 价格缺失。
- 零价格或负价格。
- 成交量为零或负数。
- 成交额为负数。
- `high < max(open, close, low)` 或 `low > min(open, close, high)`。
- 收盘价单日收益绝对值超过阈值。
- `vwap` 明显落在 OHLC 区间外。

## 与后续阶段的关系

阶段 1 不负责股票池过滤。ST、新股、停牌、低流动性、涨跌停不可交易等约束会在阶段 2 和回测阶段进一步处理。阶段 1 的任务是把问题暴露清楚，并提供可重复读取的标准行情缓存。
