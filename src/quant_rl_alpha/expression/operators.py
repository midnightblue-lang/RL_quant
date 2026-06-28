from __future__ import annotations

import numpy as np
import pandas as pd


def safe_log(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    mask = values > 0
    result.loc[mask] = np.log(values.loc[mask])
    return result


def safe_div(left: pd.Series, right: pd.Series) -> pd.Series:
    return pd.Series(np.where(right != 0, left / right, np.nan), index=left.index)


def rolling_unary(values: pd.Series, symbols: pd.Series, window: int, operator: str) -> pd.Series:
    grouped = values.groupby(symbols, sort=False)
    rolling = grouped.rolling(window, min_periods=window)
    if operator == "Mean":
        result = rolling.mean()
    elif operator == "Med":
        result = rolling.median()
    elif operator == "Sum":
        result = rolling.sum()
    elif operator == "Std":
        result = rolling.std(ddof=0)
    elif operator == "Var":
        result = rolling.var(ddof=0)
    elif operator == "Max":
        result = rolling.max()
    elif operator == "Min":
        result = rolling.min()
    elif operator == "Mad":
        result = rolling.apply(lambda item: float(np.mean(np.abs(item - np.mean(item)))), raw=True)
    elif operator == "WMA":
        weights = np.arange(1, window + 1, dtype=float)
        result = rolling.apply(lambda item: float(np.dot(item, weights) / weights.sum()), raw=True)
    else:
        raise ValueError(f"Unsupported rolling operator: {operator}")
    return result.reset_index(level=0, drop=True).sort_index()


def rolling_ema(values: pd.Series, symbols: pd.Series, window: int) -> pd.Series:
    return values.groupby(symbols, sort=False).transform(
        lambda item: item.ewm(span=window, adjust=False, min_periods=window).mean()
    )


def rolling_binary(
    left: pd.Series,
    right: pd.Series,
    symbols: pd.Series,
    window: int,
    operator: str,
) -> pd.Series:
    result = pd.Series(np.nan, index=left.index, dtype=float)
    grouped = left.groupby(symbols, sort=False)
    for _, index in grouped.indices.items():
        left_part = left.iloc[index]
        right_part = right.iloc[index]
        rolling = left_part.rolling(window, min_periods=window)
        if operator == "Cov":
            values = rolling.cov(right_part, ddof=0)
        elif operator == "Corr":
            values = rolling.corr(right_part)
        else:
            raise ValueError(f"Unsupported rolling binary operator: {operator}")
        result.iloc[index] = values.to_numpy()
    return result
