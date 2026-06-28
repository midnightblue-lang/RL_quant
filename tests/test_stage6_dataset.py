import numpy as np
import pandas as pd
import pytest

from quant_rl_alpha.model.dataset import make_xgboost_dataset, parse_alpha_pool


def _daily_panel() -> pd.DataFrame:
    rows = []
    for date in pd.to_datetime(["2024-01-31", "2024-02-29"]):
        for symbol, close, volume in [
            ("000001", 1.0, 100.0),
            ("000002", 2.0, 200.0),
            ("000003", 3.0, 300.0),
        ]:
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": volume,
                    "vwap": close,
                }
            )
    return pd.DataFrame(rows)


def _labels() -> pd.DataFrame:
    rows = []
    for date in pd.to_datetime(["2024-01-31", "2024-02-29"]):
        for symbol, future_return in [("000001", 0.1), ("000002", 0.2), ("000003", 0.3)]:
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "label_end_date": date + pd.Timedelta(days=30),
                    "future_20d_return": future_return,
                }
            )
    labels = pd.DataFrame(rows)
    labels["future_20d_rank"] = labels.groupby("date")["future_20d_return"].rank(pct=True)
    return labels


def test_make_xgboost_dataset_builds_wide_alpha_features() -> None:
    alpha_pool = pd.DataFrame(
        {
            "name": ["close_alpha", "volume_alpha"],
            "formula": ["close", "volume"],
            "tokens": ["BEG close SEP", "BEG volume SEP"],
        }
    )
    config = {
        "target": "future_20d_rank",
        "features": {
            "expected_alpha_count": 2,
            "winsorize_lower": 0.0,
            "winsorize_upper": 1.0,
            "standardize": True,
        },
    }

    dataset, features = make_xgboost_dataset(_daily_panel(), _labels(), alpha_pool, config)

    assert features == ["alpha_00", "alpha_01"]
    assert list(dataset.columns) == [
        "date",
        "symbol",
        "label_end_date",
        "future_20d_return",
        "future_20d_rank",
        "alpha_00",
        "alpha_01",
    ]
    assert len(dataset) == 6
    daily_mean = dataset.groupby("date")["alpha_00"].mean()
    daily_std = dataset.groupby("date")["alpha_00"].std(ddof=0)
    assert np.allclose(daily_mean.to_numpy(), 0.0)
    assert np.allclose(daily_std.to_numpy(), 1.0)


def test_make_xgboost_dataset_keeps_missing_alpha_as_nan_without_dropping_rows() -> None:
    alpha_pool = pd.DataFrame({"tokens": ["BEG close SEP"]})
    labels = pd.concat(
        [
            _labels(),
            pd.DataFrame(
                {
                    "date": [pd.Timestamp("2024-01-31")],
                    "symbol": ["000004"],
                    "label_end_date": [pd.Timestamp("2024-03-01")],
                    "future_20d_return": [0.4],
                    "future_20d_rank": [1.0],
                }
            ),
        ],
        ignore_index=True,
    )
    config = {
        "target": "future_20d_rank",
        "features": {"expected_alpha_count": 1, "standardize": False},
    }

    dataset, features = make_xgboost_dataset(_daily_panel(), labels, alpha_pool, config)

    assert features == ["alpha_00"]
    assert len(dataset) == len(labels)
    mask = (dataset["date"] == pd.Timestamp("2024-01-31")) & (dataset["symbol"] == "000004")
    missing = dataset[mask]
    assert missing["alpha_00"].isna().all()


def test_parse_alpha_pool_fails_fast_on_dimensionally_invalid_tokens() -> None:
    alpha_pool = pd.DataFrame(
        {
            "name": ["bad_alpha"],
            "tokens": ["BEG close volume Sub SEP"],
        }
    )

    with pytest.raises(ValueError, match="Invalid alpha pool tokens.*Dimensionally invalid"):
        parse_alpha_pool(alpha_pool, expected_count=1)


def test_parse_alpha_pool_requires_expected_30_when_configured() -> None:
    alpha_pool = pd.DataFrame({"tokens": ["BEG close SEP"]})

    with pytest.raises(ValueError, match="must contain 30 rows"):
        parse_alpha_pool(alpha_pool, expected_count=30)
