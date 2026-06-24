import pandas as pd

from quant_rl_alpha.data.stage2_pipeline import build_stage2_outputs


def test_build_stage2_outputs_writes_files(tmp_path) -> None:
    dates = pd.bdate_range("2024-01-01", periods=330)
    rows = []
    for symbol, amount in [("000001", 1000), ("000002", 2000)]:
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "name": symbol,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.0 + index,
                    "volume": 100,
                    "amount": amount + index,
                    "vwap": 10.0,
                    "turnover": 0.1,
                    "source": "test",
                    "adjust": "qfq",
                }
            )
    panel = pd.DataFrame(rows)

    _, universe, labels = build_stage2_outputs(
        daily_panel=panel,
        output_daily_panel=tmp_path / "daily_panel.parquet",
        output_universe=tmp_path / "universe.parquet",
        output_labels=tmp_path / "labels.parquet",
    )

    assert (tmp_path / "daily_panel.parquet").exists()
    assert (tmp_path / "universe.parquet").exists()
    assert (tmp_path / "labels.parquet").exists()
    assert not universe.empty
    assert not labels.empty
