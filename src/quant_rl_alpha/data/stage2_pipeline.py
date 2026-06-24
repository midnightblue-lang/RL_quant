from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_rl_alpha.data.cache import read_daily_bars, standard_daily_dir
from quant_rl_alpha.data.labels import build_forward_return_labels, write_labels
from quant_rl_alpha.data.universe import UniverseConfig, build_monthly_universe, write_universe
from quant_rl_alpha.utils.config import load_config
from quant_rl_alpha.utils.paths import data_dir, ensure_dir


def load_standard_daily_panel(input_dir: str | Path | None = None) -> pd.DataFrame:
    source_dir = Path(input_dir) if input_dir is not None else standard_daily_dir()
    paths = sorted(source_dir.glob("*.parquet"))
    if not paths:
        raise ValueError(f"No standard daily parquet files found in {source_dir}")
    frames = [read_daily_bars(path) for path in paths]
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )


def build_stage2_outputs(
    *,
    daily_panel: pd.DataFrame | None = None,
    output_daily_panel: str | Path | None = None,
    output_universe: str | Path | None = None,
    output_labels: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = load_config("universe")
    panel = daily_panel if daily_panel is not None else load_standard_daily_panel()
    paths = config["paths"]

    daily_panel_path = Path(output_daily_panel or data_dir().parent / paths["daily_panel"])
    universe_path = Path(output_universe or data_dir().parent / paths["universe_monthly"])
    labels_path = Path(output_labels or data_dir().parent / paths["labels"])

    ensure_dir(daily_panel_path.parent)
    panel.to_parquet(daily_panel_path, index=False)

    universe = build_monthly_universe(panel, universe_config_from_project())
    write_universe(universe, universe_path)

    labels = build_forward_return_labels(
        panel,
        horizon=int(config["label_horizon"]),
        universe=universe,
    )
    write_labels(labels, labels_path)
    return panel, universe, labels


def universe_config_from_project() -> UniverseConfig:
    config = load_config("universe")
    return UniverseConfig(
        min_listed_days=int(config["min_listed_days"]),
        liquidity_window=int(config["liquidity_window"]),
        top_n=int(config["top_n"]),
        exclude_st=bool(config["exclude_st"]),
        exclude_zero_volume=bool(config["exclude_zero_volume"]),
    )
