from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

DAILY_COLUMNS: Final[list[str]] = [
    "date",
    "symbol",
    "name",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "turnover",
    "source",
    "adjust",
]

AKSHARE_DAILY_RENAME: Final[dict[str, str]] = {
    "日期": "date",
    "股票代码": "symbol",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover",
}

AKSHARE_SYMBOL_RENAME: Final[dict[str, str]] = {
    "code": "symbol",
    "代码": "symbol",
    "证券代码": "symbol",
    "name": "name",
    "名称": "name",
    "证券简称": "name",
}

NUMERIC_DAILY_COLUMNS: Final[list[str]] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "turnover",
]


@dataclass(frozen=True)
class NormalizationMeta:
    source: str = "akshare"
    adjust: str = "qfq"
    volume_unit: str = "lots"


def normalize_symbol(symbol: str | int) -> str:
    text = str(symbol).strip()
    if "." in text:
        text = text.split(".")[0]
    return text.zfill(6)


def standardize_symbol_list(frame: pd.DataFrame, *, exclude_bj: bool = True) -> pd.DataFrame:
    renamed = frame.rename(columns=AKSHARE_SYMBOL_RENAME).copy()
    required = {"symbol", "name"}
    _require_columns(renamed, required, "Symbol list")

    result = renamed.loc[:, ["symbol", "name"]].copy()
    result["symbol"] = result["symbol"].map(normalize_symbol)
    result["name"] = result["name"].astype(str).str.strip()
    result = result.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)

    if exclude_bj:
        result = result[~result["symbol"].str.startswith(("8", "4"))].reset_index(drop=True)
    return result


def standardize_daily_bars(
    frame: pd.DataFrame,
    *,
    symbol: str,
    name: str | None = None,
    meta: NormalizationMeta | None = None,
) -> pd.DataFrame:
    meta = meta or NormalizationMeta()
    renamed = frame.rename(columns=AKSHARE_DAILY_RENAME).copy()
    required = {"date", "open", "high", "low", "close", "volume", "amount"}
    _require_columns(renamed, required, "Daily bars")

    result = pd.DataFrame(index=renamed.index)
    result["date"] = pd.to_datetime(renamed["date"], errors="coerce").dt.normalize()
    result["symbol"] = normalize_symbol(symbol)
    result["name"] = "" if name is None else str(name)

    ohlcv_columns = ["open", "high", "low", "close", "volume", "amount"]
    result[ohlcv_columns] = renamed[ohlcv_columns].apply(pd.to_numeric, errors="coerce")

    if meta.volume_unit == "lots":
        result["volume"] = result["volume"] * 100.0
    elif meta.volume_unit != "shares":
        raise ValueError(f"Unsupported volume unit: {meta.volume_unit}")
    result["vwap"] = np.where(result["volume"] > 0, result["amount"] / result["volume"], np.nan)

    if "turnover" in renamed.columns:
        result["turnover"] = pd.to_numeric(renamed["turnover"], errors="coerce")
    else:
        result["turnover"] = np.nan

    result["source"] = meta.source
    result["adjust"] = meta.adjust
    result = result.loc[:, DAILY_COLUMNS].sort_values("date").reset_index(drop=True)

    if result["date"].isna().any():
        raise ValueError(f"{symbol} contains rows with invalid dates")
    duplicate_dates = result["date"].duplicated()
    if duplicate_dates.any():
        duplicates = result.loc[duplicate_dates, "date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"{symbol} contains duplicate dates: {duplicates[:5]}")
    return result


def assert_daily_schema(frame: pd.DataFrame) -> None:
    _require_columns(frame, set(DAILY_COLUMNS), "Daily frame")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        suffix = "standard columns" if label == "Daily frame" else "columns"
        raise ValueError(f"{label} missing {suffix}: {sorted(missing)}")
