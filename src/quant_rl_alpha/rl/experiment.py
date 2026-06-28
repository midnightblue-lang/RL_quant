from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quant_rl_alpha.alpha import AlphaCalculator, AlphaPool
from quant_rl_alpha.expression import evaluate, parse_rpn
from quant_rl_alpha.reporting import write_pool_ic_live_report
from quant_rl_alpha.rl.env import AlphaMiningEnv
from quant_rl_alpha.rl.trainer import PPOTrainer
from quant_rl_alpha.utils.config import load_config
from quant_rl_alpha.utils.paths import ensure_dir, project_root
from quant_rl_alpha.utils.seed import set_seed


@dataclass(frozen=True)
class RLTrainingResult:
    pool_path: Path
    metrics_path: Path
    validation_path: Path
    config_path: Path
    pool_rows: int
    metric_rows: int
    validation_rows: int


def run_rl_alpha_mining(config_name: str = "rl") -> RLTrainingResult:
    config = load_config(config_name)
    seed = int(config["seed"])
    set_seed(seed)
    _set_torch_seed(seed)

    train_panel, train_labels = load_training_data(config)
    env = AlphaMiningEnv(train_panel, train_labels, invalid_reward=float(config["invalid_reward"]))
    trainer = PPOTrainer(env, config=config)
    iterations = int(config["train_iterations"])
    live_report_path = config["outputs"].get("pool_ic_live_report")
    if live_report_path:
        write_pool_ic_live_report(
            env.pool,
            live_report_path,
            iteration=0,
            total_iterations=iterations,
        )

    history = trainer.train_iterations(
        iterations,
        episodes_per_iteration=int(config["episodes_per_iteration"]),
        on_iteration=(
            lambda iteration, _: write_pool_ic_live_report(
                env.pool,
                live_report_path,
                iteration=iteration,
                total_iterations=iterations,
            )
            if live_report_path
            else None
        ),
    )

    pool_path = _write_frame(env.pool.summary_frame(), config["outputs"]["pool"])
    metrics_path = _write_frame(pd.DataFrame(history), config["outputs"]["metrics"])
    validation_panel, validation_labels = load_validation_data(config)
    validation = evaluate_pool_validation(env.pool, validation_panel, validation_labels)
    validation_path = _write_frame(validation, config["outputs"]["validation"])
    config_path = _write_config_snapshot(config, config["outputs"]["config"])
    return RLTrainingResult(
        pool_path=pool_path,
        metrics_path=metrics_path,
        validation_path=validation_path,
        config_path=config_path,
        pool_rows=len(env.pool.entries),
        metric_rows=len(history),
        validation_rows=len(validation),
    )


def load_training_data(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_config = config["data"]
    train_start = pd.Timestamp(data_config["train_start"]).normalize()
    train_end = pd.Timestamp(data_config["train_end"]).normalize()

    daily_panel, labels = _read_panel_and_labels(data_config)
    daily_panel = daily_panel[daily_panel["date"] <= train_end].copy()
    labels = labels[
        (labels["date"] >= train_start)
        & (labels["date"] <= train_end)
        & (labels["label_end_date"] <= train_end)
    ].copy()
    if daily_panel.empty:
        raise ValueError("Training daily panel is empty after date filtering")
    if labels.empty:
        raise ValueError("Training labels are empty after date filtering")
    return daily_panel, labels


def load_validation_data(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_config = config["data"]
    validation_start = pd.Timestamp(data_config["validation_start"]).normalize()
    validation_end = pd.Timestamp(data_config["validation_end"]).normalize()

    daily_panel, labels = _read_panel_and_labels(data_config)
    daily_panel = daily_panel[daily_panel["date"] <= validation_end].copy()
    labels = labels[
        (labels["date"] >= validation_start)
        & (labels["date"] <= validation_end)
        & (labels["label_end_date"] <= validation_end)
    ].copy()
    if daily_panel.empty:
        raise ValueError("Validation daily panel is empty after date filtering")
    if labels.empty:
        raise ValueError("Validation labels are empty after date filtering")
    return daily_panel, labels


def evaluate_pool_validation(
    pool: AlphaPool,
    daily_panel: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["name", "formula", "tokens", "validation_ic", "validation_rank_ic", "weight"]
    rows = []
    factor_values = []
    calculator = AlphaCalculator(labels)
    for entry, weight in zip(pool.entries, pool.weights, strict=True):
        if not entry.tokens:
            continue
        values = evaluate(parse_rpn(entry.tokens), daily_panel)
        factor_values.append((entry.name, values, float(weight)))
        validation_ic, validation_rank_ic = calculator.calc_single_all_ret(values)
        rows.append(
            {
                "name": entry.name,
                "formula": entry.formula,
                "tokens": " ".join(entry.tokens),
                "validation_ic": validation_ic,
                "validation_rank_ic": validation_rank_ic,
                "weight": float(weight),
            }
        )
    if factor_values:
        pool_ic, pool_rank_ic = calculator.calc_pool_all_ret(
            [values for _, values, _ in factor_values],
            [weight for _, _, weight in factor_values],
        )
        rows.append(
            {
                "name": "__pool__",
                "formula": "__weighted_pool__",
                "tokens": "",
                "validation_ic": pool_ic,
                "validation_rank_ic": pool_rank_ic,
                "weight": float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _read_panel_and_labels(data_config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_panel = pd.read_parquet(_project_path(data_config["daily_panel"]))
    labels = pd.read_parquet(_project_path(data_config["labels"]))
    daily_panel["date"] = pd.to_datetime(daily_panel["date"]).dt.normalize()
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()
    labels["label_end_date"] = pd.to_datetime(labels["label_end_date"]).dt.normalize()
    return daily_panel, labels


def _write_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    output = _project_path(path)
    ensure_dir(output.parent)
    frame.to_parquet(output, index=False)
    return output


def _write_config_snapshot(config: dict[str, Any], path: str | Path) -> Path:
    output = _project_path(path)
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    return output


def _project_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else project_root() / resolved


def _set_torch_seed(seed: int) -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
