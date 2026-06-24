import pandas as pd

from quant_rl_alpha.data.quality import QualityConfig, inspect_daily_bars, summarize_quality
from quant_rl_alpha.data.schema import DAILY_COLUMNS


def _frame() -> pd.DataFrame:
    data = [
        [
            "2024-01-02",
            "000001",
            "平安银行",
            10.0,
            10.5,
            9.8,
            10.2,
            100_000,
            1_020_000,
            10.2,
            0.5,
            "akshare",
            "qfq",
        ],
        [
            "2024-01-03",
            "000001",
            "平安银行",
            10.2,
            10.4,
            10.0,
            10.3,
            0,
            0,
            float("nan"),
            0.0,
            "akshare",
            "qfq",
        ],
        [
            "2024-01-04",
            "000001",
            "平安银行",
            10.3,
            9.0,
            10.4,
            15.0,
            120_000,
            1_800_000,
            15.0,
            0.7,
            "akshare",
            "qfq",
        ],
    ]
    frame = pd.DataFrame(data, columns=DAILY_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def test_inspect_daily_bars_flags_quality_issues() -> None:
    summary = inspect_daily_bars(_frame(), QualityConfig(min_rows=5, large_return_threshold=0.30))

    assert summary.symbol == "000001"
    assert summary.rows == 3
    assert summary.zero_volume_count == 1
    assert summary.ohlc_inconsistent_count == 1
    assert summary.large_return_count == 1
    assert summary.too_few_rows is True
    assert summary.has_issue is True


def test_summarize_quality_returns_dataframe() -> None:
    result = summarize_quality([_frame()], QualityConfig(min_rows=1))

    assert result.loc[0, "symbol"] == "000001"
    assert "has_issue" in result.columns
