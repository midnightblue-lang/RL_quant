from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_rl_alpha.utils.calendar import trading_days
from quant_rl_alpha.utils.paths import ensure_dir


def build_forward_return_labels(
    daily_bars: pd.DataFrame,
    *,
    horizon: int = 20,
    universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    required = {"date", "symbol", "close"}
    missing = required - set(daily_bars.columns)
    if missing:
        raise ValueError(f"Daily bars missing label columns: {sorted(missing)}")

    frame = daily_bars.loc[:, ["date", "symbol", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    if frame[["date", "symbol"]].duplicated().any():
        raise ValueError("Daily bars contain duplicate (date, symbol) rows")

    calendar = _label_calendar(frame["date"], horizon)
    frame = frame.merge(calendar, on="date", how="left")
    future_close = frame[["symbol", "label_end_date"]].merge(
        frame.loc[:, ["date", "symbol", "close"]],
        left_on=["symbol", "label_end_date"],
        right_on=["symbol", "date"],
        how="left",
        suffixes=("", "_future"),
    )["close"]
    frame["future_20d_return"] = future_close / frame["close"] - 1
    labels = frame.dropna(subset=["future_20d_return"]).copy()
    if universe is not None:
        keys = universe.loc[:, ["date", "symbol"]].copy()
        keys["date"] = pd.to_datetime(keys["date"]).dt.normalize()
        keys["symbol"] = keys["symbol"].astype(str).str.zfill(6)
        labels = keys.merge(labels, on=["date", "symbol"], how="inner")

    labels["future_20d_rank"] = labels.groupby("date")["future_20d_return"].rank(
        pct=True,
        method="average",
    )
    labels = labels.loc[
        :,
        ["date", "symbol", "label_end_date", "future_20d_return", "future_20d_rank"],
    ]

    return labels.sort_values(["date", "symbol"]).reset_index(drop=True)


def write_labels(labels: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    ensure_dir(output.parent)
    labels.to_parquet(output, index=False)
    return output


def _label_calendar(dates: pd.Series, horizon: int) -> pd.DataFrame:
    days = trading_days(dates)
    label_dates = pd.Series(days).shift(-horizon)
    return pd.DataFrame({"date": days, "label_end_date": label_dates})
