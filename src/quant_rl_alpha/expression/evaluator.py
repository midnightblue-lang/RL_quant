from __future__ import annotations

import numpy as np
import pandas as pd

from quant_rl_alpha.expression.operators import (
    rolling_binary,
    rolling_ema,
    rolling_unary,
    safe_div,
    safe_log,
)
from quant_rl_alpha.expression.rpn import Expression
from quant_rl_alpha.expression.tokens import (
    CS_BINARY_OPERATORS,
    CS_UNARY_OPERATORS,
    TS_BINARY_OPERATORS,
    TS_UNARY_OPERATORS,
    is_constant,
    is_feature,
    parse_constant,
)


def evaluate(
    expr: Expression,
    daily_panel: pd.DataFrame,
    value_name: str = "value",
) -> pd.DataFrame:
    frame = _prepare_panel(daily_panel)
    values = _evaluate_series(expr, frame).replace([np.inf, -np.inf], np.nan)
    result = frame.loc[:, ["date", "symbol"]].copy()
    result[value_name] = values.to_numpy()
    return result


def is_semantically_valid(values: pd.DataFrame, value_name: str = "value") -> bool:
    if value_name not in values.columns:
        raise ValueError(f"Factor values missing column: {value_name}")
    frame = values.loc[:, ["date", value_name]].copy()
    frame[value_name] = pd.to_numeric(frame[value_name], errors="coerce")
    frame[value_name] = frame[value_name].replace([np.inf, -np.inf], np.nan)
    finite = frame.dropna(subset=[value_name])
    if finite.empty or finite[value_name].nunique() <= 1:
        return False
    daily = finite.groupby("date")[value_name].agg(["count", "std"])
    return bool(((daily["count"] >= 2) & (daily["std"] > 1e-12)).any())


def _prepare_panel(daily_panel: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol"}
    missing = required - set(daily_panel.columns)
    if missing:
        raise ValueError(f"Daily panel missing expression columns: {sorted(missing)}")
    frame = daily_panel.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    if frame[["date", "symbol"]].duplicated().any():
        raise ValueError("Daily panel contains duplicate (date, symbol) rows")
    return frame


def _evaluate_series(expr: Expression, frame: pd.DataFrame) -> pd.Series:
    token = expr.token
    if is_feature(token):
        if token not in frame.columns:
            raise ValueError(f"Daily panel missing feature column: {token}")
        return pd.to_numeric(frame[token], errors="coerce")
    if is_constant(token):
        return pd.Series(parse_constant(token), index=frame.index, dtype=float)
    if token in CS_UNARY_OPERATORS:
        child = _evaluate_series(expr.children[0], frame)
        return child.abs() if token == "Abs" else safe_log(child)
    if token in CS_BINARY_OPERATORS:
        left = _evaluate_series(expr.children[0], frame)
        right = _evaluate_series(expr.children[1], frame)
        return _evaluate_cross_section_binary(token, left, right)
    if token in TS_UNARY_OPERATORS:
        child = _evaluate_series(expr.children[0], frame)
        return _evaluate_time_series_unary(token, child, frame["symbol"], _window(expr))
    if token in TS_BINARY_OPERATORS:
        left = _evaluate_series(expr.children[0], frame)
        right = _evaluate_series(expr.children[1], frame)
        return rolling_binary(left, right, frame["symbol"], _window(expr), token)
    raise ValueError(f"Unsupported expression operator: {token}")


def _evaluate_cross_section_binary(token: str, left: pd.Series, right: pd.Series) -> pd.Series:
    if token == "Add":
        return left + right
    if token == "Sub":
        return left - right
    if token == "Mul":
        return left * right
    if token == "Div":
        return safe_div(left, right)
    if token == "Greater":
        return pd.Series(np.maximum(left, right), index=left.index)
    if token == "Less":
        return pd.Series(np.minimum(left, right), index=left.index)
    raise ValueError(f"Unsupported cross-section binary operator: {token}")


def _evaluate_time_series_unary(
    token: str,
    child: pd.Series,
    symbols: pd.Series,
    window: int,
) -> pd.Series:
    if token == "Ref":
        return child.groupby(symbols, sort=False).shift(window)
    if token == "Delta":
        return child - child.groupby(symbols, sort=False).shift(window)
    if token == "EMA":
        return rolling_ema(child, symbols, window)
    return rolling_unary(child, symbols, window, token)


def _window(expr: Expression) -> int:
    if expr.window is None or expr.window <= 0:
        raise ValueError(f"{expr.token} requires a positive time window")
    return expr.window
