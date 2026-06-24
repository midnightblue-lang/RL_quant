import pandas as pd
import pytest

from quant_rl_alpha.data.labels import build_forward_return_labels


def _bars() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=25)
    rows = []
    for symbol, start in [("000001", 10.0), ("000002", 20.0)]:
        for index, date in enumerate(dates):
            rows.append({"date": date, "symbol": symbol, "close": start + index})
    return pd.DataFrame(rows)


def test_build_forward_return_labels() -> None:
    labels = build_forward_return_labels(_bars(), horizon=20)

    first = labels[(labels["date"] == pd.Timestamp("2024-01-01"))]
    assert set(first["symbol"]) == {"000001", "000002"}
    one = first[first["symbol"] == "000001"].iloc[0]
    assert one["label_end_date"] == pd.Timestamp("2024-01-29")
    assert one["future_20d_return"] == 2.0
    assert first["future_20d_rank"].between(0, 1).all()


def test_labels_can_be_filtered_to_universe() -> None:
    universe = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "symbol": ["000001", "000002"],
        }
    )

    labels = build_forward_return_labels(_bars(), horizon=20, universe=universe)

    assert labels["symbol"].tolist() == ["000001", "000002"]
    assert labels.loc[0, "future_20d_return"] == 2.0
    assert labels["future_20d_rank"].tolist() == [1.0, 0.5]


def test_rank_is_calculated_after_universe_filter() -> None:
    bars = _bars()
    extra = bars[bars["symbol"] == "000001"].copy()
    extra["symbol"] = "000003"
    extra["close"] = extra["close"] * 100
    bars = pd.concat([bars, extra], ignore_index=True)
    universe = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "symbol": ["000001", "000002"],
        }
    )

    labels = build_forward_return_labels(bars, horizon=20, universe=universe)

    assert labels["symbol"].tolist() == ["000001", "000002"]
    assert labels["future_20d_rank"].tolist() == [1.0, 0.5]


def test_missing_target_date_close_does_not_jump_to_next_stock_row() -> None:
    bars = _bars()
    target_date = pd.bdate_range("2024-01-01", periods=25)[20]
    bars = bars[~((bars["symbol"] == "000001") & (bars["date"] == target_date))]

    labels = build_forward_return_labels(bars, horizon=20)
    first_date_symbols = labels[labels["date"] == pd.Timestamp("2024-01-01")]["symbol"].tolist()

    assert first_date_symbols == ["000002"]


def test_duplicate_date_symbol_rejected() -> None:
    bars = pd.concat([_bars(), _bars().head(1)], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        build_forward_return_labels(bars, horizon=20)
