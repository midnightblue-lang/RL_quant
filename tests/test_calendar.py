import pandas as pd
import pytest

from quant_rl_alpha.utils.calendar import (
    month_end_trading_days,
    next_trading_day,
    previous_trading_days,
)


def test_month_end_trading_days() -> None:
    dates = pd.to_datetime(["2024-01-29", "2024-01-31", "2024-02-01", "2024-02-29"])

    result = month_end_trading_days(dates)

    assert result.tolist() == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]


def test_next_trading_day() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])

    assert next_trading_day(dates, "2024-01-02") == pd.Timestamp("2024-01-03")
    assert next_trading_day(dates, "2024-01-03") is None


def test_previous_trading_days_requires_positive_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        previous_trading_days(["2024-01-02"], "2024-01-02", 0)


def test_previous_trading_days() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])

    result = previous_trading_days(dates, "2024-01-04", 2)

    assert result.tolist() == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]
