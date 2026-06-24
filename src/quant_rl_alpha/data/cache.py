from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from quant_rl_alpha.data.schema import assert_daily_schema, normalize_symbol
from quant_rl_alpha.utils.paths import data_dir, ensure_dir


def raw_hist_dir() -> Path:
    return ensure_dir(data_dir() / "raw" / "akshare" / "hist")


def standard_daily_dir() -> Path:
    return ensure_dir(data_dir() / "interim" / "akshare" / "daily")


def raw_hist_path(symbol: str | int) -> Path:
    return raw_hist_dir() / f"{normalize_symbol(symbol)}.parquet"


def standard_daily_path(symbol: str | int) -> Path:
    return standard_daily_dir() / f"{normalize_symbol(symbol)}.parquet"


def write_raw_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    return _write_parquet(frame, path)


def read_raw_frame(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_daily_bars(frame: pd.DataFrame, path: str | Path) -> Path:
    assert_daily_schema(frame)
    return _write_parquet(frame, path)


def read_daily_bars(path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    assert_daily_schema(frame)
    return frame


def cached_symbols() -> list[str]:
    directory = standard_daily_dir()
    return sorted(path.stem for path in directory.glob("*.parquet"))


def build_daily_panel(paths: Iterable[str | Path], output_path: str | Path) -> Path:
    frames = [read_daily_bars(path) for path in paths]
    if not frames:
        raise ValueError("No daily bar files were provided")
    panel = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    return write_daily_bars(panel, output_path)


def _write_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    resolved = Path(path)
    ensure_dir(resolved.parent)
    frame.to_parquet(resolved, index=False)
    return resolved
