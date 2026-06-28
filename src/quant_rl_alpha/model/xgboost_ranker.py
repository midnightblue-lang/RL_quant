from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from quant_rl_alpha.alpha import mean_ic, mean_rank_ic
from quant_rl_alpha.model.dataset import build_xgboost_dataset
from quant_rl_alpha.utils.config import load_config
from quant_rl_alpha.utils.paths import ensure_dir, project_root


class RankModel(Protocol):
    def fit(self, x: pd.DataFrame, y: pd.Series) -> Any: ...

    def predict(self, x: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class XGBoostTrainingResult:
    dataset_path: Path
    predictions_path: Path
    metrics_path: Path
    feature_importance_path: Path
    prediction_rows: int
    metric_rows: int
    feature_importance_rows: int


def run_xgboost_training(config_name: str | Path = "xgboost") -> XGBoostTrainingResult:
    config = load_config(config_name)
    dataset_path = _project_path(config["outputs"]["dataset"])
    if not dataset_path.is_file():
        build_xgboost_dataset(config_name)

    dataset = pd.read_parquet(dataset_path)
    predictions, metrics, feature_importance = walk_forward_predict(dataset, config)
    predictions_path = _write_frame(predictions, config["outputs"]["predictions"])
    metrics_path = _write_frame(metrics, config["outputs"]["metrics"])
    importance_path = _write_frame(feature_importance, config["outputs"]["feature_importance"])
    return XGBoostTrainingResult(
        dataset_path=dataset_path,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        feature_importance_path=importance_path,
        prediction_rows=len(predictions),
        metric_rows=len(metrics),
        feature_importance_rows=len(feature_importance),
    )


def walk_forward_predict(
    dataset: pd.DataFrame,
    config: dict[str, Any],
    *,
    model_factory: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = str(config["target"])
    frame = _prepare_dataset(dataset, target)
    feature_cols = [column for column in frame.columns if column.startswith("alpha_")]
    if not feature_cols:
        raise ValueError("XGBoost dataset contains no alpha_* feature columns")

    prediction_dates = _prediction_dates(frame, config)
    if not prediction_dates:
        raise ValueError("No prediction dates are available after prediction_start")

    factory = model_factory or _make_model
    prediction_frames = []
    metric_rows = []
    importance_frames = []
    for prediction_date in prediction_dates:
        prediction = frame[frame["date"] == prediction_date].copy()
        train = _training_window(frame, prediction_date, int(config["train_window_years"]))
        if train.empty:
            raise ValueError(f"No training rows for prediction date {prediction_date.date()}")

        model = factory(config["model"])
        model.fit(train.loc[:, feature_cols], train[target])
        prediction["score"] = model.predict(prediction.loc[:, feature_cols])
        prediction["score_rank_pct"] = prediction["score"].rank(pct=True, method="average")

        prediction_frames.append(
            prediction.loc[
                :,
                ["date", "symbol", "score", "score_rank_pct", "future_20d_return", target],
            ]
        )
        metric_rows.append(
            _monthly_metrics(
                prediction,
                train_count=len(train),
                target=target,
                feature_cols=feature_cols,
            )
        )
        importance_frames.append(_feature_importance_frame(model, prediction_date, feature_cols))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    importance = _aggregate_feature_importance(pd.concat(importance_frames, ignore_index=True))
    return predictions, metrics, importance


def _prepare_dataset(dataset: pd.DataFrame, target: str) -> pd.DataFrame:
    required = {"date", "symbol", "label_end_date", "future_20d_return", target}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"XGBoost dataset missing columns: {sorted(missing)}")
    frame = dataset.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["label_end_date"] = pd.to_datetime(frame["label_end_date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["future_20d_return"] = pd.to_numeric(frame["future_20d_return"], errors="coerce")
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame.dropna(subset=[target]).sort_values(["date", "symbol"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("XGBoost dataset is empty after dropping missing targets")
    if frame[["date", "symbol"]].duplicated().any():
        raise ValueError("XGBoost dataset contains duplicate (date, symbol) rows")
    return frame


def _prediction_dates(frame: pd.DataFrame, config: dict[str, Any]) -> list[pd.Timestamp]:
    frequency = str(config.get("rebalance_frequency", "monthly")).lower()
    if frequency != "monthly":
        raise ValueError(f"Unsupported XGBoost rebalance_frequency: {frequency}")
    prediction_start = pd.Timestamp(config["prediction_start"]).normalize()
    dates = pd.Series(sorted(date for date in frame["date"].unique() if date >= prediction_start))
    if dates.empty:
        return []
    return dates.groupby(dates.dt.to_period("M")).max().tolist()


def _training_window(
    frame: pd.DataFrame,
    prediction_date: pd.Timestamp,
    train_window_years: int,
) -> pd.DataFrame:
    window_start = prediction_date - pd.DateOffset(years=train_window_years)
    mask = (
        (frame["date"] < prediction_date)
        & (frame["date"] >= window_start)
        & (frame["label_end_date"] <= prediction_date)
    )
    return frame.loc[mask].copy()


def _monthly_metrics(
    prediction: pd.DataFrame,
    *,
    train_count: int,
    target: str,
    feature_cols: list[str],
) -> dict[str, object]:
    factor = prediction.loc[:, ["date", "symbol", "score"]]
    labels = prediction.loc[:, ["date", "symbol", target]]
    return {
        "date": prediction["date"].iloc[0],
        "prediction_count": int(len(prediction)),
        "train_count": int(train_count),
        "ic": mean_ic(factor, labels, factor_col="score", label_col=target),
        "rank_ic": mean_rank_ic(factor, labels, factor_col="score", label_col=target),
        "feature_coverage": _feature_coverage(prediction, feature_cols),
    }


def _feature_coverage(frame: pd.DataFrame, feature_cols: list[str]) -> float:
    total = len(frame) * len(feature_cols)
    if total == 0:
        return float("nan")
    return float(frame.loc[:, feature_cols].notna().to_numpy().sum() / total)


def _feature_importance_frame(
    model: RankModel,
    prediction_date: pd.Timestamp,
    feature_cols: list[str],
) -> pd.DataFrame:
    rows = [
        {"date": prediction_date, "feature": feature, "gain": 0.0, "weight": 0.0, "cover": 0.0}
        for feature in feature_cols
    ]
    if not hasattr(model, "get_booster"):
        return pd.DataFrame(rows)
    booster = model.get_booster()
    for importance_type in ("gain", "weight", "cover"):
        for key, value in booster.get_score(importance_type=importance_type).items():
            feature = _feature_name(key, feature_cols)
            for row in rows:
                if row["feature"] == feature:
                    row[importance_type] = float(value)
                    break
    return pd.DataFrame(rows)


def _aggregate_feature_importance(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["feature", "gain", "weight", "cover", "model_count"])
    grouped = frame.groupby("feature", as_index=False).agg(
        gain=("gain", "mean"),
        weight=("weight", "sum"),
        cover=("cover", "mean"),
        model_count=("date", "nunique"),
    )
    return grouped.sort_values(
        ["gain", "weight", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _feature_name(key: str, feature_cols: list[str]) -> str:
    if key in feature_cols:
        return key
    if key.startswith("f") and key[1:].isdigit():
        index = int(key[1:])
        if 0 <= index < len(feature_cols):
            return feature_cols[index]
    return key


def _make_model(model_config: dict[str, Any]) -> RankModel:
    try:
        from xgboost import XGBRegressor
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "XGBoost is required for stage 6. Install requirements.txt dependencies."
        ) from error
    return XGBRegressor(**model_config)


def _write_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    output = _project_path(path)
    ensure_dir(output.parent)
    frame.to_parquet(output, index=False)
    return output


def _project_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else project_root() / resolved
