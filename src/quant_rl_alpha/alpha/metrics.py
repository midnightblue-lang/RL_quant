from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-12


@dataclass(frozen=True)
class AlphaCalculator:
    labels: pd.DataFrame
    label_col: str = "future_20d_return"

    def calc_single_IC_ret(self, factor: pd.DataFrame) -> float:
        return mean_ic(factor, self.labels, label_col=self.label_col)

    def calc_single_rIC_ret(self, factor: pd.DataFrame) -> float:
        return mean_rank_ic(factor, self.labels, label_col=self.label_col)

    def calc_single_all_ret(self, factor: pd.DataFrame) -> tuple[float, float]:
        return self.calc_single_IC_ret(factor), self.calc_single_rIC_ret(factor)

    def calc_mutual_IC(self, factor_a: pd.DataFrame, factor_b: pd.DataFrame) -> float:
        return mutual_ic(factor_a, factor_b)

    def calc_pool_IC_ret(
        self,
        factors: Sequence[pd.DataFrame],
        weights: Sequence[float],
    ) -> float:
        values = make_weighted_pool_values(factors, weights)
        return mean_ic(values, self.labels, label_col=self.label_col)

    def calc_pool_rIC_ret(
        self,
        factors: Sequence[pd.DataFrame],
        weights: Sequence[float],
    ) -> float:
        return mean_rank_ic(
            make_weighted_pool_values(factors, weights),
            self.labels,
            label_col=self.label_col,
        )

    def calc_pool_all_ret(
        self,
        factors: Sequence[pd.DataFrame],
        weights: Sequence[float],
    ) -> tuple[float, float]:
        values = make_weighted_pool_values(factors, weights)
        return (
            mean_ic(values, self.labels, label_col=self.label_col),
            mean_rank_ic(values, self.labels, label_col=self.label_col),
        )


def cross_section_l2_normalize(
    values: pd.DataFrame,
    *,
    value_col: str = "value",
    output_col: str = "normalized",
) -> pd.DataFrame:
    frame = _prepare_values(values, value_col)
    frame[output_col] = np.nan
    for _, group in frame.groupby("date", sort=True):
        valid = group[value_col].notna()
        if valid.sum() < 2:
            continue
        normalized = _center_l2(group.loc[valid, value_col].to_numpy(dtype=float))
        if normalized is not None:
            frame.loc[group.index[valid], output_col] = normalized
    return frame.loc[:, ["date", "symbol", output_col]]


def daily_ic(
    factor: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    factor_col: str = "value",
    label_col: str = "future_20d_return",
) -> pd.Series:
    aligned = align_factor_label(factor, labels, factor_col=factor_col, label_col=label_col)
    return _daily_corr(aligned, factor_col, label_col)


def mean_ic(
    factor: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    factor_col: str = "value",
    label_col: str = "future_20d_return",
) -> float:
    return _mean_or_nan(daily_ic(factor, labels, factor_col=factor_col, label_col=label_col))


def daily_rank_ic(
    factor: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    factor_col: str = "value",
    label_col: str = "future_20d_return",
) -> pd.Series:
    aligned = align_factor_label(factor, labels, factor_col=factor_col, label_col=label_col)
    aligned["factor_rank"] = aligned.groupby("date")[factor_col].rank(method="average")
    aligned["label_rank"] = aligned.groupby("date")[label_col].rank(method="average")
    return _daily_corr(aligned, "factor_rank", "label_rank")


def mean_rank_ic(
    factor: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    factor_col: str = "value",
    label_col: str = "future_20d_return",
) -> float:
    return _mean_or_nan(daily_rank_ic(factor, labels, factor_col=factor_col, label_col=label_col))


def mutual_ic(
    factor_a: pd.DataFrame,
    factor_b: pd.DataFrame,
    *,
    value_col: str = "value",
) -> float:
    left = _prepare_values(factor_a, value_col).rename(columns={value_col: "factor_a"})
    right = _prepare_values(factor_b, value_col).rename(columns={value_col: "factor_b"})
    aligned = left.merge(right, on=["date", "symbol"], how="inner")
    return _mean_or_nan(_daily_corr(aligned, "factor_a", "factor_b"))


def make_weighted_pool_values(
    factors: Sequence[pd.DataFrame],
    weights: Sequence[float],
) -> pd.DataFrame:
    if len(factors) != len(weights):
        raise ValueError("Factors and weights must have the same length")
    if not factors:
        return pd.DataFrame(columns=["date", "symbol", "value"])

    frames = []
    for index, (factor, weight) in enumerate(zip(factors, weights, strict=True)):
        normalized = cross_section_l2_normalize(factor).rename(columns={"normalized": f"f{index}"})
        normalized[f"f{index}"] = normalized[f"f{index}"] * float(weight)
        frames.append(normalized)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["date", "symbol"], how="outer")
    value_cols = [f"f{index}" for index in range(len(frames))]
    merged["value"] = merged[value_cols].sum(axis=1, min_count=1)
    return merged.loc[:, ["date", "symbol", "value"]]


def align_factor_label(
    factor: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    factor_col: str = "value",
    label_col: str = "future_20d_return",
) -> pd.DataFrame:
    factor_frame = _prepare_values(factor, factor_col)
    label_frame = _prepare_values(labels, label_col)
    return factor_frame.merge(label_frame, on=["date", "symbol"], how="inner")


def _prepare_values(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    required = {"date", "symbol", value_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Values missing columns: {sorted(missing)}")
    result = frame.loc[:, ["date", "symbol", value_col]].copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["symbol"] = result["symbol"].astype(str).str.zfill(6)
    result[value_col] = pd.to_numeric(result[value_col], errors="coerce")
    result[value_col] = result[value_col].replace([np.inf, -np.inf], np.nan)
    result = result.sort_values(["date", "symbol"]).reset_index(drop=True)
    if result[["date", "symbol"]].duplicated().any():
        raise ValueError("Values contain duplicate (date, symbol) rows")
    return result


def _daily_corr(frame: pd.DataFrame, left_col: str, right_col: str) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    for date, group in frame.groupby("date", sort=True):
        left = group[left_col]
        right = group[right_col]
        valid = left.notna() & right.notna()
        if valid.sum() < 2:
            continue
        left_normalized = _center_l2(left[valid].to_numpy(dtype=float))
        right_normalized = _center_l2(right[valid].to_numpy(dtype=float))
        if left_normalized is None or right_normalized is None:
            continue
        corr = float(np.dot(left_normalized, right_normalized))
        if np.isfinite(corr):
            values[pd.Timestamp(date)] = corr
    return pd.Series(values, dtype=float).rename("ic")


def _center_l2(values: np.ndarray) -> np.ndarray | None:
    finite = np.isfinite(values)
    if len(values) == 0 or not finite.all():
        return None
    scale = float(np.max(np.abs(values)))
    if not np.isfinite(scale) or scale <= EPS:
        return None
    centered = values / scale
    centered = centered - centered.mean()
    norm = float(np.linalg.norm(centered))
    if not np.isfinite(norm) or norm <= EPS:
        return None
    return centered / norm


def _mean_or_nan(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(values.mean())
