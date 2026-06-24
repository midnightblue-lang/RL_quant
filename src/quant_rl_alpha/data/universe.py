from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_rl_alpha.utils.calendar import month_end_trading_days
from quant_rl_alpha.utils.paths import ensure_dir


@dataclass(frozen=True)
class UniverseConfig:
    min_listed_days: int = 250
    liquidity_window: int = 20
    top_n: int = 1500
    exclude_st: bool = True
    exclude_zero_volume: bool = True


def build_monthly_universe(
    daily_bars: pd.DataFrame,
    config: UniverseConfig | None = None,
) -> pd.DataFrame:
    config = config or UniverseConfig()
    prepared = _prepare_daily_bars(daily_bars)
    features = _with_history_features(prepared, config.liquidity_window)
    rebalance_dates = month_end_trading_days(features["date"])
    rows = [
        _select_universe_on_date(features, date, config)
        for date in rebalance_dates
    ]
    if not rows:
        return _empty_universe()
    return pd.concat(rows, ignore_index=True)


def write_universe(universe: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    ensure_dir(output.parent)
    universe.to_parquet(output, index=False)
    return output


def _prepare_daily_bars(daily_bars: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol", "name", "open", "high", "low", "close", "volume", "amount"}
    missing = required - set(daily_bars.columns)
    if missing:
        raise ValueError(f"Daily bars missing universe columns: {sorted(missing)}")
    frame = daily_bars.loc[:, sorted(required)].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["name"] = frame["name"].fillna("").astype(str)
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    if frame[["date", "symbol"]].duplicated().any():
        raise ValueError("Daily bars contain duplicate (date, symbol) rows")
    return frame


def _with_history_features(frame: pd.DataFrame, liquidity_window: int) -> pd.DataFrame:
    if liquidity_window <= 0:
        raise ValueError("liquidity_window must be positive")
    result = frame.copy()
    grouped = result.groupby("symbol", group_keys=False)
    result["listed_days"] = grouped.cumcount() + 1
    result["avg_amount"] = grouped["amount"].rolling(
        liquidity_window,
        min_periods=liquidity_window,
    ).mean().reset_index(level=0, drop=True)
    return result


def _select_universe_on_date(
    features: pd.DataFrame,
    date: pd.Timestamp,
    config: UniverseConfig,
) -> pd.DataFrame:
    current = features[features["date"] == date].copy()
    if current.empty:
        return _empty_universe()

    valid_price = current[["open", "high", "low", "close"]].notna().all(axis=1)
    valid_price &= (current[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid_liquidity = current["avg_amount"].notna() & (current["avg_amount"] > 0)
    listed_enough = current["listed_days"] >= config.min_listed_days
    mask = valid_price & valid_liquidity & listed_enough

    if config.exclude_zero_volume:
        mask &= (current["volume"] > 0) & (current["amount"] > 0)
    if config.exclude_st:
        mask &= ~current["name"].str.contains("ST|退", case=False, regex=True, na=False)

    selected = current.loc[mask].sort_values(
        ["avg_amount", "symbol"],
        ascending=[False, True],
    )
    selected = selected.head(config.top_n).copy()
    if selected.empty:
        return _empty_universe()

    selected["rank"] = range(1, len(selected) + 1)
    return selected.loc[
        :,
        ["date", "symbol", "name", "avg_amount", "listed_days", "rank"],
    ].reset_index(drop=True)


def _empty_universe() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "symbol", "name", "avg_amount", "listed_days", "rank"]
    )
