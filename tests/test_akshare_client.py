import pandas as pd

from quant_rl_alpha.data.akshare_client import download_daily_bars


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
