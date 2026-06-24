from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_rl_alpha.data.akshare_client import AkshareClient, DownloadResult, download_daily_bars
from quant_rl_alpha.data.cache import read_daily_bars, standard_daily_dir
from quant_rl_alpha.data.quality import QualityConfig, summarize_quality, write_quality_report
from quant_rl_alpha.utils.config import load_config
from quant_rl_alpha.utils.paths import data_dir


def download_configured_sample(client: AkshareClient | None = None) -> list[DownloadResult]:
    config = load_config("data")
    client = client or AkshareClient()
    symbols = [str(symbol) for symbol in config["sample_symbols"]]
    results = download_daily_bars(
        client,
        symbols,
        start_date=config["start_date"],
        end_date=config["end_date"] or pd.Timestamp.today().strftime("%Y-%m-%d"),
        adjust=config["adjust"],
        endpoint=config["endpoint"],
        skip_existing=bool(config["skip_existing"]),
        retry_times=int(config["retry_times"]),
        sleep_seconds=float(config["sleep_seconds"]),
    )
    write_download_report(results, data_dir().parent / config["paths"]["download_report"])
    return results


def write_download_report(results: list[DownloadResult], path: str | Path) -> Path:
    rows = [
        {
            "symbol": result.symbol,
            "endpoint": result.endpoint,
            "standard_path": _text_or_empty(result.standard_path),
            "raw_path": _text_or_empty(result.raw_path),
            "rows": result.rows,
            "skipped": result.skipped,
            "error": _text_or_empty(result.error),
            "attempts": result.attempts,
            "error_type": _text_or_empty(result.error_type),
        }
        for result in results
    ]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8")
    return output


def quality_config_from_project() -> QualityConfig:
    config = load_config("data")["quality"]
    return QualityConfig(
        min_rows=int(config["min_rows"]),
        large_return_threshold=float(config["large_return_threshold"]),
        vwap_bar_tolerance=float(config["vwap_bar_tolerance"]),
    )


def build_quality_report(
    *,
    input_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    config = load_config("data")
    source_dir = Path(input_dir) if input_dir is not None else standard_daily_dir()
    report_path = (
        Path(output_path)
        if output_path is not None
        else data_dir().parent / config["paths"]["quality_report"]
    )
    paths = sorted(source_dir.glob("*.parquet"))
    frames = [read_daily_bars(path) for path in paths]
    summary = summarize_quality(frames, quality_config_from_project())
    write_quality_report(summary, report_path)
    return summary


def _text_or_empty(value: object | None) -> str:
    return "" if value is None else str(value)
