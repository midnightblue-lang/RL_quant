import pandas as pd

from quant_rl_alpha.data.akshare_client import DownloadResult
from quant_rl_alpha.data.pipeline import build_quality_report, write_download_report
from quant_rl_alpha.data.schema import DAILY_COLUMNS


def test_build_quality_report_from_cached_files(tmp_path) -> None:
    input_dir = tmp_path / "daily"
    input_dir.mkdir()
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
    frame.to_parquet(input_dir / "000001.parquet", index=False)
    output_path = tmp_path / "report.md"

    summary = build_quality_report(input_dir=input_dir, output_path=output_path)

    assert summary.loc[0, "symbol"] == "000001"
    assert output_path.exists()


def test_write_download_report(tmp_path) -> None:
    path = tmp_path / "download_results.csv"
    write_download_report(
        [
            DownloadResult(
                symbol="000001",
                endpoint="daily_sina",
                standard_path=None,
                raw_path=None,
                rows=0,
                skipped=False,
                error="network error",
            )
        ],
        path,
    )

    content = path.read_text(encoding="utf-8")
    assert "daily_sina" in content
    assert "network error" in content
