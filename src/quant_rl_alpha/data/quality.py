from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from quant_rl_alpha.data.schema import DAILY_COLUMNS
from quant_rl_alpha.utils.paths import ensure_dir

ISSUE_COUNT_FIELDS: Final[tuple[str, ...]] = (
    "duplicate_date_count",
    "missing_required_count",
    "missing_ohlcv_count",
    "non_positive_price_count",
    "negative_volume_count",
    "negative_amount_count",
    "volume_amount_mismatch_count",
    "ohlc_inconsistent_count",
    "large_return_count",
    "vwap_outside_bar_count",
)


@dataclass(frozen=True)
class QualityConfig:
    min_rows: int = 250
    large_return_threshold: float = 0.35
    vwap_bar_tolerance: float = 0.02


@dataclass(frozen=True)
class SymbolQuality:
    symbol: str
    name: str
    rows: int
    start_date: str
    end_date: str
    duplicate_date_count: int
    missing_required_count: int
    missing_ohlcv_count: int
    non_positive_price_count: int
    zero_volume_count: int
    negative_volume_count: int
    negative_amount_count: int
    volume_amount_mismatch_count: int
    ohlc_inconsistent_count: int
    large_return_count: int
    vwap_outside_bar_count: int
    too_few_rows: bool

    @property
    def has_issue(self) -> bool:
        return self.too_few_rows or any(getattr(self, field) > 0 for field in ISSUE_COUNT_FIELDS)


def inspect_daily_bars(frame: pd.DataFrame, config: QualityConfig | None = None) -> SymbolQuality:
    config = config or QualityConfig()
    missing_columns = set(DAILY_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Cannot inspect frame missing standard columns: {sorted(missing_columns)}"
        )

    ordered = frame.sort_values("date").reset_index(drop=True)
    symbol = str(ordered["symbol"].dropna().iloc[0]) if not ordered.empty else ""
    name = str(ordered["name"].dropna().iloc[0]) if not ordered.empty else ""
    start_date = _format_date(ordered["date"].min()) if not ordered.empty else ""
    end_date = _format_date(ordered["date"].max()) if not ordered.empty else ""

    required = ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]
    ohlcv = ["open", "high", "low", "close", "volume", "amount"]
    prices = ["open", "high", "low", "close"]

    high_floor = ordered[["open", "close", "low"]].max(axis=1)
    low_ceiling = ordered[["open", "close", "high"]].min(axis=1)
    close_return = ordered["close"].pct_change()
    vwap_tolerance = config.vwap_bar_tolerance
    vwap_low = ordered["low"] * (1 - vwap_tolerance)
    vwap_high = ordered["high"] * (1 + vwap_tolerance)
    volume_amount_mismatch = ((ordered["volume"] > 0) & (ordered["amount"] <= 0)) | (
        (ordered["amount"] > 0) & (ordered["volume"] <= 0)
    )

    summary = SymbolQuality(
        symbol=symbol,
        name=name,
        rows=len(ordered),
        start_date=start_date,
        end_date=end_date,
        duplicate_date_count=_count_true(ordered["date"].duplicated()),
        missing_required_count=_count_true(ordered[required].isna()),
        missing_ohlcv_count=_count_true(ordered[ohlcv].isna()),
        non_positive_price_count=_count_true(ordered[prices] <= 0),
        zero_volume_count=_count_true(ordered["volume"] == 0),
        negative_volume_count=_count_true(ordered["volume"] < 0),
        negative_amount_count=_count_true(ordered["amount"] < 0),
        volume_amount_mismatch_count=_count_true(volume_amount_mismatch),
        ohlc_inconsistent_count=_count_true(
            (ordered["high"] < high_floor) | (ordered["low"] > low_ceiling)
        ),
        large_return_count=_count_true(close_return.abs() > config.large_return_threshold),
        vwap_outside_bar_count=_count_true(
            (ordered["vwap"] < vwap_low) | (ordered["vwap"] > vwap_high)
        ),
        too_few_rows=len(ordered) < config.min_rows,
    )
    return summary


def summarize_quality(
    frames: list[pd.DataFrame],
    config: QualityConfig | None = None,
) -> pd.DataFrame:
    summaries = [inspect_daily_bars(frame, config) for frame in frames]
    rows = [asdict(summary) | {"has_issue": summary.has_issue} for summary in summaries]
    return pd.DataFrame(rows)


def write_quality_report(summary: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    ensure_dir(output.parent)
    issue_count = int(summary["has_issue"].sum()) if "has_issue" in summary else 0
    zero_volume_symbols = int((summary.get("zero_volume_count", pd.Series(dtype=int)) > 0).sum())
    lines = [
        "# 数据质量报告",
        "",
        f"- 股票数：{len(summary)}",
        f"- 有质量提示的股票数：{issue_count}",
        f"- 出现零成交量的股票数：{zero_volume_symbols}",
        "",
        "## 字段说明",
        "",
        "- `large_return_count` 使用收盘价单日收益阈值标记异常，不自动删除。",
        "- `vwap_outside_bar_count` 只作为口径提示，"
        "前复权 OHLC 与原始成交额/成交量可能不完全一致。",
        "- `zero_volume_count` 通常代表停牌或不可交易日，后续股票池和回测阶段应继续处理。",
        "- `volume_amount_mismatch_count` 标记成交量和成交额口径明显矛盾的行。",
        "",
        "## 明细",
        "",
        "```csv",
        summary.to_csv(index=False) if not summary.empty else "",
        "```",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _format_date(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _count_true(mask: pd.Series | pd.DataFrame) -> int:
    total = mask.sum()
    if isinstance(total, pd.Series):
        total = total.sum()
    return int(total)
