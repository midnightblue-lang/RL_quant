import pandas as pd
import pytest

from quant_rl_alpha.data.schema import standardize_daily_bars, standardize_symbol_list


def test_standardize_symbol_list_excludes_bj_symbols() -> None:
    raw = pd.DataFrame(
        {
            "code": ["000001", "600000", "830001"],
            "name": ["平安银行", "浦发银行", "北交所样例"],
        }
    )

    result = standardize_symbol_list(raw, exclude_bj=True)

    assert result["symbol"].tolist() == ["000001", "600000"]


def test_standardize_daily_bars_renames_and_converts_units() -> None:
    raw = pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03"],
            "开盘": [10.0, 10.5],
            "收盘": [10.5, 10.2],
            "最高": [10.8, 10.7],
            "最低": [9.9, 10.1],
            "成交量": [1000, 2000],
            "成交额": [1_030_000, 2_060_000],
            "换手率": [0.5, 0.6],
        }
    )

    result = standardize_daily_bars(raw, symbol="1", name="平安银行")

    assert result["symbol"].unique().tolist() == ["000001"]
    assert result.loc[0, "volume"] == 100_000
    assert result.loc[0, "vwap"] == 10.3
    assert result.loc[0, "name"] == "平安银行"


def test_standardize_daily_bars_rejects_duplicate_dates() -> None:
    raw = pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-02"],
            "开盘": [10.0, 10.5],
            "收盘": [10.5, 10.2],
            "最高": [10.8, 10.7],
            "最低": [9.9, 10.1],
            "成交量": [1000, 2000],
            "成交额": [1_030_000, 2_060_000],
        }
    )

    with pytest.raises(ValueError, match="duplicate dates"):
        standardize_daily_bars(raw, symbol="000001")
