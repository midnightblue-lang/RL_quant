import json
from pathlib import Path

import pandas as pd
import yaml

from quant_rl_alpha.reporting import build_rl_factor_report, write_pool_ic_live_report


def test_build_rl_factor_report_preflight_when_artifacts_are_missing(tmp_path: Path) -> None:
    config_path = _write_rl_report_fixture(tmp_path, pool_size=0, write_artifacts=False)

    result = build_rl_factor_report(config_path)
    html = result.report_path.read_text(encoding="utf-8")

    assert result.mode == "preflight"
    assert result.missing_artifacts == ("pool", "metrics", "validation")
    assert "Pre-flight report" in html
    assert "Missing Artifacts" in html
    assert "target_pool_size" in html


def test_build_rl_factor_report_warns_when_pool_is_too_small(tmp_path: Path) -> None:
    config_path = _write_rl_report_fixture(tmp_path, pool_size=2, write_artifacts=True)

    result = build_rl_factor_report(config_path)
    html = result.report_path.read_text(encoding="utf-8")

    assert result.mode == "final"
    assert result.pool_size == 2
    assert "Pool size warning: 2/30" in html
    assert "Mutual IC Heatmap" in html


def test_build_rl_factor_report_marks_30_factor_pool_ok(tmp_path: Path) -> None:
    config_path = _write_rl_report_fixture(tmp_path, pool_size=30, write_artifacts=True)

    result = build_rl_factor_report(config_path)
    html = result.report_path.read_text(encoding="utf-8")

    assert result.mode == "final"
    assert result.pool_size == 30
    assert "Pool size OK: 30/30" in html
    assert "Training Diagnostics" in html


def test_write_pool_ic_live_report(tmp_path: Path) -> None:
    pool = pd.DataFrame(
        {
            "name": ["alpha_1", "alpha_2"],
            "formula": ["close", "open"],
            "ic": [0.02, -0.01],
            "rank_ic": [0.03, -0.02],
            "weight": [0.1, -0.1],
        }
    )
    report_path = write_pool_ic_live_report(
        pool,
        tmp_path / "rl_pool_ic_live.html",
        iteration=3,
        total_iterations=10,
    )
    html = report_path.read_text(encoding="utf-8")

    assert "RL Pool IC Live" in html
    assert 'http-equiv="refresh"' in html
    assert "3/10" in html
    assert "Current Factor IC" in html
    assert "alpha_1" in html
    assert "0.02" in html


def test_rl_alpha_mining_notebook_is_valid_json() -> None:
    with Path("notebooks/03_rl_alpha_mining.ipynb").open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert data["nbformat"] == 4
    notebook_text = json.dumps(data, ensure_ascii=False)
    assert "build_rl_factor_report" in notebook_text
    assert "最终因子池" in notebook_text
    assert "验证期表现" in notebook_text


def _write_rl_report_fixture(
    tmp_path: Path,
    *,
    pool_size: int,
    write_artifacts: bool,
) -> Path:
    daily_panel = _daily_panel()
    labels = _labels(daily_panel)
    panel_path = tmp_path / "daily_panel.parquet"
    labels_path = tmp_path / "labels.parquet"
    daily_panel.to_parquet(panel_path, index=False)
    labels.to_parquet(labels_path, index=False)

    pool_path = tmp_path / "rl_alpha_pool.parquet"
    metrics_path = tmp_path / "rl_training_metrics.parquet"
    validation_path = tmp_path / "rl_validation_metrics.parquet"
    report_path = tmp_path / "rl_factor_report.html"
    if write_artifacts:
        _pool(pool_size).to_parquet(pool_path, index=False)
        _metrics(pool_size).to_parquet(metrics_path, index=False)
        _validation(pool_size).to_parquet(validation_path, index=False)

    config = {
        "algorithm": "ppo",
        "seed": 2026,
        "gamma": 1.0,
        "clip_epsilon": 0.2,
        "learning_rate": 0.0003,
        "ppo_epochs": 4,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "lstm_layers": 2,
        "lstm_hidden_size": 128,
        "dropout": 0.1,
        "head_hidden_size": 64,
        "pool_size": 30,
        "invalid_reward": -1.0,
        "device": "cpu",
        "train_iterations": 2,
        "episodes_per_iteration": 2,
        "data": {
            "daily_panel": str(panel_path),
            "labels": str(labels_path),
            "train_start": "2020-01-01",
            "train_end": "2020-12-31",
            "validation_start": "2021-01-01",
            "validation_end": "2021-12-31",
        },
        "outputs": {
            "pool": str(pool_path),
            "metrics": str(metrics_path),
            "validation": str(validation_path),
            "config": str(tmp_path / "rl_training_config.yml"),
            "report": str(report_path),
        },
    }
    config_path = tmp_path / "rl.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _daily_panel() -> pd.DataFrame:
    rows = []
    dates = pd.to_datetime(["2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05"])
    for date in dates:
        for symbol, close in [("000001", 1.0), ("000002", 2.0), ("000003", 3.0)]:
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close,
                    "high": close + 0.1,
                    "low": close - 0.1,
                    "close": close,
                    "volume": close * 1000,
                    "vwap": close,
                }
            )
    return pd.DataFrame(rows)


def _labels(daily_panel: pd.DataFrame) -> pd.DataFrame:
    labels = daily_panel[daily_panel["date"].isin(pd.to_datetime(["2020-12-30", "2021-01-04"]))]
    labels = labels.loc[:, ["date", "symbol", "close"]].copy()
    labels["label_end_date"] = labels["date"] + pd.offsets.BDay(1)
    labels["future_20d_return"] = labels["close"]
    return labels.loc[:, ["date", "symbol", "label_end_date", "future_20d_return"]]


def _pool(size: int) -> pd.DataFrame:
    rows = []
    for index in range(size):
        rows.append(
            {
                "name": f"alpha_{index + 1}",
                "formula": "close",
                "tokens": "BEG close SEP",
                "ic": 0.01 * (index + 1),
                "rank_ic": 0.02 * (index + 1),
                "weight": 0.1,
            }
        )
    return pd.DataFrame(rows)


def _metrics(pool_size: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mean_reward": [0.1, 0.2],
            "valid_ratio": [0.5, 0.8],
            "pool_size": [min(pool_size, 15), pool_size],
            "pool_ic": [0.01, 0.03],
            "pool_loss": [0.9, 0.7],
            "invalid_count": [2, 1],
        }
    )


def _validation(pool_size: int) -> pd.DataFrame:
    rows = [
        {
            "name": f"alpha_{index + 1}",
            "formula": "close",
            "tokens": "BEG close SEP",
            "validation_ic": 0.005 * (index + 1),
            "validation_rank_ic": 0.01 * (index + 1),
            "weight": 0.1,
        }
        for index in range(pool_size)
    ]
    rows.append(
        {
            "name": "__pool__",
            "formula": "__weighted_pool__",
            "tokens": "",
            "validation_ic": 0.05,
            "validation_rank_ic": 0.06,
            "weight": float("nan"),
        }
    )
    return pd.DataFrame(rows)
