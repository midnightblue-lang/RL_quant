from pathlib import Path

import pandas as pd

from quant_rl_alpha.data.full_ingest import build_download_manifest, download_full_market_data


class FakeFullClient:
    def list_a_stocks(self, *, exclude_bj: bool = True, source: str = "code_name") -> pd.DataFrame:
        assert exclude_bj is True
        assert source == "spot_sina"
        return pd.DataFrame({"symbol": ["000001", "600000"], "name": ["Ping An Bank", "SPD Bank"]})

    def fetch_daily_raw(
        self,
        *,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str,
        endpoint: str,
    ) -> pd.DataFrame:
        assert symbol == "000001"
        assert start_date == "2024-01-01"
        assert end_date == "2024-01-31"
        assert adjust == "qfq"
        assert endpoint == "daily_sina"
        return pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"],
                "open": [10.0, 10.2],
                "high": [10.5, 10.6],
                "low": [9.9, 10.1],
                "close": [10.3, 10.4],
                "volume": [100_000, 120_000],
                "amount": [1_030_000, 1_248_000],
            }
        )


def test_build_download_manifest_marks_pending() -> None:
    symbols = pd.DataFrame({"symbol": ["000001"], "name": ["Ping An Bank"]})

    manifest = build_download_manifest(symbols, [])

    assert manifest.loc[0, "symbol"] == "000001"
    assert manifest.loc[0, "status"] == "pending"


def test_download_full_market_data_writes_reports(tmp_path, monkeypatch) -> None:
    import quant_rl_alpha.data.akshare_client as client_module
    import quant_rl_alpha.data.full_ingest as full_ingest

    raw_dir = tmp_path / "raw"
    standard_dir = tmp_path / "standard"
    report_dir = tmp_path / "reports"

    config = {
        "endpoint": "daily_sina",
        "symbol_list_endpoint": "spot_sina",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "adjust": "qfq",
        "exclude_bj": True,
        "retry_times": 1,
        "sleep_seconds": 0,
        "skip_existing": False,
        "full_market": {
            "symbol_limit": 1,
            "manifest_flush_every": 1,
            "progress_every": 0,
            "paths": {
                "symbol_list": str(report_dir / "symbols.csv"),
                "manifest": str(report_dir / "manifest.csv"),
                "failures": str(report_dir / "failures.csv"),
                "quality_report": str(report_dir / "quality.md"),
                "summary_report": str(report_dir / "summary.md"),
            },
        },
    }

    monkeypatch.setattr(full_ingest, "load_config", lambda name: config)
    monkeypatch.setattr(full_ingest, "standard_daily_dir", lambda: standard_dir)
    monkeypatch.setattr(
        client_module, "raw_hist_path", lambda symbol: raw_dir / f"{symbol}.parquet"
    )
    monkeypatch.setattr(
        client_module, "standard_daily_path", lambda symbol: standard_dir / f"{symbol}.parquet"
    )

    result = download_full_market_data(FakeFullClient())

    assert result.symbols["symbol"].tolist() == ["000001"]
    assert result.manifest["status"].tolist() == ["downloaded"]
    assert result.quality["symbol"].tolist() == ["000001"]
    assert Path(config["full_market"]["paths"]["symbol_list"]).exists()
    assert Path(config["full_market"]["paths"]["manifest"]).exists()
    assert Path(config["full_market"]["paths"]["failures"]).exists()
    assert Path(config["full_market"]["paths"]["quality_report"]).exists()
    assert Path(config["full_market"]["paths"]["summary_report"]).exists()
