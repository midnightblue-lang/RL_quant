from __future__ import annotations

import pandas as pd


def trading_days(dates: pd.Series | pd.Index | list[object]) -> pd.DatetimeIndex:
    days = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().dropna().unique()
    return pd.DatetimeIndex(sorted(days))


def month_end_trading_days(dates: pd.Series | pd.Index | list[object]) -> pd.DatetimeIndex:
    days = trading_days(dates)
    if days.empty:
        return days
    series = pd.Series(days, index=days)
    month_ends = series.groupby(days.to_period("M")).max()
    return pd.DatetimeIndex(month_ends.to_list())


def next_trading_day(
    dates: pd.Series | pd.Index | list[object],
    date: object,
) -> pd.Timestamp | None:
    days = trading_days(dates)
    current = pd.Timestamp(date).normalize()
    position = days.searchsorted(current, side="right")
    if position >= len(days):
        return None
    return pd.Timestamp(days[position])


def previous_trading_days(
    dates: pd.Series | pd.Index | list[object],
    date: object,
    count: int,
) -> pd.DatetimeIndex:
    if count <= 0:
        raise ValueError("count must be positive")
    days = trading_days(dates)
    current = pd.Timestamp(date).normalize()
    end = days.searchsorted(current, side="right")
    start = max(0, end - count)
    return days[start:end]
