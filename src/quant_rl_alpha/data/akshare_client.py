from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_rl_alpha.data.cache import (
    raw_hist_path,
    read_daily_bars,
    standard_daily_path,
    write_daily_bars,
    write_raw_frame,
)
from quant_rl_alpha.data.schema import (
    NormalizationMeta,
    standardize_daily_bars,
    standardize_symbol_list,
)
from quant_rl_alpha.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class DownloadResult:
    symbol: str
    endpoint: str
    standard_path: Path | None
    raw_path: Path | None
    rows: int
    skipped: bool
    error: str | None = None


class AkshareClient:
    def __init__(self) -> None:
        import akshare as ak

        self._ak = ak

    def list_a_stocks(self, *, exclude_bj: bool = True) -> pd.DataFrame:
        raw = self._ak.stock_info_a_code_name()
        return standardize_symbol_list(raw, exclude_bj=exclude_bj)

    def fetch_daily_bars(
        self,
        symbol: str,
        *,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
        name: str | None = None,
        endpoint: str = "hist_em",
    ) -> pd.DataFrame:
        raw = self.fetch_daily_raw(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
            endpoint=endpoint,
        )
        return _standardize_daily_raw(raw, symbol, name=name, adjust=adjust, endpoint=endpoint)

    def fetch_daily_raw(
        self,
        *,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
        endpoint: str = "hist_em",
    ) -> pd.DataFrame:
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")
        if endpoint == "hist_em":
            return self._ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start,
                end_date=end,
                adjust=adjust,
            )
        if endpoint == "daily_sina":
            return self._ak.stock_zh_a_daily(
                symbol=akshare_exchange_symbol(symbol),
                start_date=start,
                end_date=end,
                adjust=adjust,
            )
        raise ValueError(f"Unsupported AKShare endpoint: {endpoint}")


def download_daily_bars(
    client: AkshareClient,
    symbols: Iterable[str],
    *,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
    endpoint: str = "hist_em",
    names: dict[str, str] | None = None,
    skip_existing: bool = True,
    retry_times: int = 3,
    sleep_seconds: float = 0.2,
) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    names = names or {}

    for symbol in symbols:
        raw_path = raw_hist_path(symbol)
        standard_path = standard_daily_path(symbol)
        if skip_existing and raw_path.exists() and standard_path.exists():
            cached_rows = len(read_daily_bars(standard_path))
            results.append(
                DownloadResult(
                    symbol=symbol,
                    endpoint=endpoint,
                    standard_path=standard_path,
                    raw_path=raw_path,
                    rows=cached_rows,
                    skipped=True,
                )
            )
            continue

        last_error: Exception | None = None
        for attempt in range(1, retry_times + 1):
            try:
                raw = client.fetch_daily_raw(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                    endpoint=endpoint,
                )
                write_raw_frame(raw, raw_path)
                frame = _standardize_daily_raw(
                    raw,
                    symbol,
                    name=names.get(symbol),
                    adjust=adjust,
                    endpoint=endpoint,
                )
                write_daily_bars(frame, standard_path)
                results.append(
                    DownloadResult(
                        symbol=symbol,
                        endpoint=endpoint,
                        standard_path=standard_path,
                        raw_path=raw_path,
                        rows=len(frame),
                        skipped=False,
                    )
                )
                break
            except (ValueError, KeyError, OSError, RuntimeError) as error:
                last_error = error
                if attempt == retry_times:
                    LOGGER.warning(
                        "Failed to download %s after %s attempts: %s",
                        symbol,
                        attempt,
                        error,
                    )
                    results.append(
                        DownloadResult(
                            symbol=symbol,
                            endpoint=endpoint,
                            standard_path=None,
                            raw_path=None,
                            rows=0,
                            skipped=False,
                            error=str(error),
                        )
                    )
                else:
                    time.sleep(sleep_seconds)
        if last_error is None and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return results


def _standardize_daily_raw(
    raw: pd.DataFrame,
    symbol: str,
    *,
    name: str | None,
    adjust: str,
    endpoint: str,
) -> pd.DataFrame:
    return standardize_daily_bars(
        raw,
        symbol=symbol,
        name=name,
        meta=NormalizationMeta(
            source=f"akshare:{endpoint}",
            adjust=adjust,
            volume_unit=volume_unit_for_endpoint(endpoint),
        ),
    )


def volume_unit_for_endpoint(endpoint: str) -> str:
    if endpoint == "hist_em":
        return "lots"
    if endpoint == "daily_sina":
        return "shares"
    raise ValueError(f"Unsupported AKShare endpoint: {endpoint}")


def akshare_exchange_symbol(symbol: str) -> str:
    normalized = str(symbol).strip().zfill(6)
    if normalized.startswith(("6", "5", "9")):
        return f"sh{normalized}"
    if normalized.startswith(("0", "1", "2", "3")):
        return f"sz{normalized}"
    if normalized.startswith(("4", "8")):
        return f"bj{normalized}"
    raise ValueError(f"Cannot infer exchange prefix for symbol: {symbol}")
