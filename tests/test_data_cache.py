import pandas as pd

from quant_rl_alpha.data.cache import (
    read_daily_bars,
    read_raw_frame,
    write_daily_bars,
    write_raw_frame,
)
from quant_rl_alpha.data.schema import DAILY_COLUMNS


def test_write_and_read_daily_bars(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            [
                "2024-01-02",
                "000001",
                "平安银行",
                10,
                11,
                9,
                10.5,
                100_000,
                1_050_000,
                10.5,
                0.5,
                "akshare",
                "qfq",
            ]
        ],
        columns=DAILY_COLUMNS,
    )
    frame["date"] = pd.to_datetime(frame["date"])
    path = tmp_path / "000001.parquet"

    write_daily_bars(frame, path)
    loaded = read_daily_bars(path)

    assert loaded.loc[0, "symbol"] == "000001"
    assert loaded.loc[0, "close"] == 10.5


def test_write_and_read_raw_frame(tmp_path) -> None:
    raw = pd.DataFrame({"日期": ["2024-01-02"], "收盘": [10.5]})
    path = tmp_path / "raw.parquet"

    write_raw_frame(raw, path)
    loaded = read_raw_frame(path)

    assert loaded.loc[0, "收盘"] == 10.5
