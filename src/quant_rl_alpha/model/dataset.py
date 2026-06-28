from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_rl_alpha.expression import Expression, evaluate, parse_rpn
from quant_rl_alpha.expression.rpn import RPNError
from quant_rl_alpha.utils.config import load_config
from quant_rl_alpha.utils.paths import ensure_dir, project_root

EPS = 1e-12


@dataclass(frozen=True)
class AlphaDefinition:
    name: str
    formula: str
    tokens: tuple[str, ...]
    expr: Expression
    feature_name: str


@dataclass(frozen=True)
class XGBoostDatasetResult:
    dataset_path: Path
    rows: int
    features: tuple[str, ...]
    alpha_count: int


def build_xgboost_dataset(config_name: str | Path = "xgboost") -> XGBoostDatasetResult:
    config = load_config(config_name)
    data_config = config["data"]
    daily_panel = pd.read_parquet(_project_path(data_config["daily_panel"]))
    labels = pd.read_parquet(_project_path(data_config["labels"]))
    alpha_pool = pd.read_parquet(_project_path(data_config["alpha_pool"]))
    dataset, features = make_xgboost_dataset(daily_panel, labels, alpha_pool, config)

    output = _project_path(config["outputs"]["dataset"])
    ensure_dir(output.parent)
    dataset.to_parquet(output, index=False)
    return XGBoostDatasetResult(
        dataset_path=output,
        rows=len(dataset),
        features=tuple(features),
        alpha_count=len(features),
    )


def make_xgboost_dataset(
    daily_panel: pd.DataFrame,
    labels: pd.DataFrame,
    alpha_pool: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    target = str(config["target"])
    prepared_labels = _prepare_labels(labels, target)
    alphas = parse_alpha_pool(
        alpha_pool,
        expected_count=config.get("features", {}).get("expected_alpha_count"),
    )
    panel = _prepare_daily_panel(daily_panel, max_date=prepared_labels["date"].max())
    dataset = prepared_labels.copy()
    keys = dataset.loc[:, ["date", "symbol"]]

    for alpha in alphas:
        values = evaluate(alpha.expr, panel, value_name=alpha.feature_name)
        aligned = keys.merge(values, on=["date", "symbol"], how="left")
        dataset[alpha.feature_name] = aligned[alpha.feature_name].to_numpy()

    feature_cols = [alpha.feature_name for alpha in alphas]
    dataset = _process_features(dataset, feature_cols, config.get("features", {}))
    return dataset.sort_values(["date", "symbol"]).reset_index(drop=True), feature_cols


def parse_alpha_pool(
    alpha_pool: pd.DataFrame,
    *,
    expected_count: int | None = None,
) -> list[AlphaDefinition]:
    if "tokens" not in alpha_pool.columns:
        raise ValueError("Alpha pool missing required column: tokens")
    if expected_count is not None and len(alpha_pool) != int(expected_count):
        raise ValueError(f"Alpha pool must contain {expected_count} rows, got {len(alpha_pool)}")

    width = max(2, len(str(max(len(alpha_pool) - 1, 0))))
    alphas: list[AlphaDefinition] = []
    for index, row in alpha_pool.reset_index(drop=True).iterrows():
        name = str(row["name"]) if "name" in alpha_pool.columns else f"alpha_{index}"
        tokens = _tokens_from_cell(row["tokens"])
        try:
            expr = parse_rpn(tokens)
        except (RPNError, ValueError) as error:
            joined = " ".join(tokens)
            message = (
                f"Invalid alpha pool tokens at row {index}, name={name}: "
                f"{error}; tokens={joined}"
            )
            raise ValueError(message) from error
        formula = str(row["formula"]) if "formula" in alpha_pool.columns else expr.to_formula()
        alphas.append(
            AlphaDefinition(
                name=name,
                formula=formula,
                tokens=tokens,
                expr=expr,
                feature_name=f"alpha_{index:0{width}d}",
            )
        )
    return alphas


def _prepare_labels(labels: pd.DataFrame, target: str) -> pd.DataFrame:
    required = {"date", "symbol", "label_end_date", "future_20d_return", target}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Labels missing XGBoost columns: {sorted(missing)}")
    frame = labels.loc[:, ["date", "symbol", "label_end_date", "future_20d_return", target]].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["label_end_date"] = pd.to_datetime(frame["label_end_date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["future_20d_return"] = pd.to_numeric(frame["future_20d_return"], errors="coerce")
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame.dropna(subset=[target]).sort_values(["date", "symbol"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("Labels are empty after dropping missing target values")
    if frame[["date", "symbol"]].duplicated().any():
        raise ValueError("Labels contain duplicate (date, symbol) rows")
    return frame


def _prepare_daily_panel(daily_panel: pd.DataFrame, *, max_date: pd.Timestamp) -> pd.DataFrame:
    required = {"date", "symbol"}
    missing = required - set(daily_panel.columns)
    if missing:
        raise ValueError(f"Daily panel missing XGBoost columns: {sorted(missing)}")
    frame = daily_panel.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame = frame[frame["date"] <= max_date].sort_values(["symbol", "date"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("Daily panel is empty after XGBoost date filtering")
    return frame


def _process_features(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    feature_config: dict[str, Any],
) -> pd.DataFrame:
    frame = dataset.copy()
    lower = feature_config.get("winsorize_lower")
    upper = feature_config.get("winsorize_upper")
    if lower is not None and upper is not None and float(lower) >= float(upper):
        raise ValueError("winsorize_lower must be smaller than winsorize_upper")

    for column in feature_cols:
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if lower is not None and upper is not None:
            values = values.groupby(frame["date"], sort=False).transform(
                lambda item: _winsorize_series(item, float(lower), float(upper))
            )
        if bool(feature_config.get("standardize", True)):
            values = values.groupby(frame["date"], sort=False).transform(_zscore_series)
        frame[column] = values
    return frame


def _winsorize_series(values: pd.Series, lower: float, upper: float) -> pd.Series:
    valid = values.dropna()
    if valid.empty:
        return values
    return values.clip(valid.quantile(lower), valid.quantile(upper))


def _zscore_series(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna()
    if len(valid) < 2:
        return result
    std = float(valid.std(ddof=0))
    if not np.isfinite(std) or std <= EPS:
        return result
    result.loc[valid.index] = (valid - float(valid.mean())) / std
    return result


def _tokens_from_cell(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        tokens = tuple(token for token in value.split() if token)
    elif isinstance(value, list | tuple):
        tokens = tuple(str(token) for token in value)
    else:
        raise ValueError(f"Alpha tokens must be a string or sequence, got {type(value).__name__}")
    if not tokens:
        raise ValueError("Alpha tokens are empty")
    return tokens


def _project_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else project_root() / resolved
