import pandas as pd

from quant_rl_alpha.data.akshare_client import download_daily_bars
from quant_rl_alpha.data.schema import DAILY_COLUMNS


class FakeClient:
    def fetch_daily_raw(
        self,
        *,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str,
        endpoint: str,
    ) -> pd.DataFrame:
        assert start_date == "2024-01-01"
        assert end_date == "2024-01-31"
        assert adjust == "qfq"
        assert endpoint == "hist_em"
        return pd.DataFrame(
            {
                "日期": ["2024-01-02"],
                "开盘": [10.0],
                "收盘": [10.5],
                "最高": [10.8],
                "最低": [9.9],
                "成交量": [1000],
                "成交额": [1_030_000],
            }
        )


class CountingDailySinaClient:
    def __init__(self) -> None:
        self.calls = 0

    def fetch_daily_raw(
        self,
        *,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str,
        endpoint: str,
    ) -> pd.DataFrame:
        self.calls += 1
        assert endpoint == "daily_sina"
        return pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "open": [10.0],
                "high": [10.8],
                "low": [9.9],
                "close": [10.5],
                "volume": [100_000],
                "amount": [1_050_000],
            }
        )


def test_download_daily_bars_with_fake_client(tmp_path, monkeypatch) -> None:
    import quant_rl_alpha.data.akshare_client as client_module

    monkeypatch.setattr(
        client_module,
        "raw_hist_path",
        lambda symbol: tmp_path / "raw" / f"{symbol}.parquet",
    )
    monkeypatch.setattr(
        client_module,
        "standard_daily_path",
        lambda symbol: tmp_path / "standard" / f"{symbol}.parquet",
    )

    results = download_daily_bars(
        FakeClient(),
        ["000001"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        sleep_seconds=0,
    )

    assert len(results) == 1
    assert results[0].error is None
    assert results[0].endpoint == "hist_em"
    assert results[0].rows == 1
    assert results[0].raw_path is not None
    assert results[0].standard_path is not None
    assert results[0].raw_path.exists()
    assert results[0].standard_path.exists()


def test_download_daily_bars_redownloads_cache_with_wrong_source(tmp_path, monkeypatch) -> None:
    import quant_rl_alpha.data.akshare_client as client_module

    raw_path = tmp_path / "raw" / "000001.parquet"
    standard_path = tmp_path / "standard" / "000001.parquet"
    raw_path.parent.mkdir(parents=True)
    standard_path.parent.mkdir(parents=True)
    pd.DataFrame({"x": [1]}).to_parquet(raw_path, index=False)
    cached = pd.DataFrame(
        [
            [
                "2024-01-02",
                "000001",
                "",
                10,
                11,
                9,
                10,
                100,
                1000,
                10,
                0.1,
                "akshare:hist_em",
                "qfq",
            ]
        ],
        columns=DAILY_COLUMNS,
    )
    cached["date"] = pd.to_datetime(cached["date"])
    cached.to_parquet(standard_path, index=False)

    monkeypatch.setattr(client_module, "raw_hist_path", lambda symbol: raw_path)
    monkeypatch.setattr(client_module, "standard_daily_path", lambda symbol: standard_path)
    client = CountingDailySinaClient()

    results = download_daily_bars(
        client,
        ["000001"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        endpoint="daily_sina",
        sleep_seconds=0,
    )

    assert client.calls == 1
    assert results[0].skipped is False
    assert results[0].attempts == 1
