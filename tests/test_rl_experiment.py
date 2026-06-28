from pathlib import Path

import pandas as pd
import yaml

from quant_rl_alpha.alpha import AlphaPool
from quant_rl_alpha.rl import experiment
from quant_rl_alpha.rl.experiment import (
    evaluate_pool_validation,
    load_training_data,
    load_validation_data,
)


def test_load_training_data_excludes_labels_after_training_boundary(tmp_path: Path) -> None:
    daily_panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-12-30", "2020-12-31", "2021-01-04"]),
            "symbol": ["000001", "000001", "000001"],
            "open": [1.0, 1.1, 1.2],
            "high": [1.0, 1.1, 1.2],
            "low": [1.0, 1.1, 1.2],
            "close": [1.0, 1.1, 1.2],
            "volume": [100.0, 100.0, 100.0],
            "vwap": [1.0, 1.1, 1.2],
        }
    )
    labels = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-12-30", "2020-12-31"]),
            "symbol": ["000001", "000001"],
            "label_end_date": pd.to_datetime(["2020-12-31", "2021-01-04"]),
            "future_20d_return": [0.1, 0.2],
        }
    )
    panel_path = tmp_path / "daily_panel.parquet"
    labels_path = tmp_path / "labels.parquet"
    daily_panel.to_parquet(panel_path, index=False)
    labels.to_parquet(labels_path, index=False)

    panel, train_labels = load_training_data(
        {
            "data": {
                "daily_panel": panel_path,
                "labels": labels_path,
                "train_start": "2020-01-01",
                "train_end": "2020-12-31",
            }
        }
    )

    assert panel["date"].max() == pd.Timestamp("2020-12-31")
    assert train_labels["date"].tolist() == [pd.Timestamp("2020-12-30")]
    assert train_labels["label_end_date"].max() == pd.Timestamp("2020-12-31")


def test_load_validation_data_excludes_labels_after_validation_boundary(tmp_path: Path) -> None:
    daily_panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-12-30", "2021-12-31", "2022-01-04"]),
            "symbol": ["000001", "000001", "000001"],
            "close": [1.0, 1.1, 1.2],
        }
    )
    labels = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-12-30", "2021-12-31"]),
            "symbol": ["000001", "000001"],
            "label_end_date": pd.to_datetime(["2021-12-31", "2022-01-04"]),
            "future_20d_return": [0.1, 0.2],
        }
    )
    panel_path = tmp_path / "daily_panel.parquet"
    labels_path = tmp_path / "labels.parquet"
    daily_panel.to_parquet(panel_path, index=False)
    labels.to_parquet(labels_path, index=False)

    panel, validation_labels = load_validation_data(
        {
            "data": {
                "daily_panel": panel_path,
                "labels": labels_path,
                "validation_start": "2021-01-01",
                "validation_end": "2021-12-31",
            }
        }
    )

    assert panel["date"].max() == pd.Timestamp("2021-12-31")
    assert validation_labels["date"].tolist() == [pd.Timestamp("2021-12-30")]
    assert validation_labels["label_end_date"].max() == pd.Timestamp("2021-12-31")


def test_evaluate_pool_validation_reports_factor_and_pool_rows() -> None:
    rows = []
    for date in pd.to_datetime(["2021-01-04", "2021-01-05"]):
        for symbol, close in [("000001", 1.0), ("000002", 2.0), ("000003", 3.0)]:
            rows.append({"date": date, "symbol": symbol, "close": close})
    daily_panel = pd.DataFrame(rows)
    labels = daily_panel.rename(columns={"close": "future_20d_return"}).assign(
        label_end_date=pd.Timestamp("2021-01-06")
    )
    labels = labels.loc[:, ["date", "symbol", "label_end_date", "future_20d_return"]]

    pool = AlphaPool(labels, capacity=3, learning_rate=0.1, gradient_steps=20)
    result = pool.add(
        "close_alpha",
        "close",
        daily_panel.rename(columns={"close": "value"}),
        tokens=("BEG", "close", "SEP"),
    )

    assert result.added
    validation = evaluate_pool_validation(pool, daily_panel, labels)

    assert validation["name"].tolist() == ["close_alpha", "__pool__"]
    assert validation["validation_ic"].tolist() == [1.0, 1.0]


def test_run_rl_alpha_mining_passes_selected_config_to_trainer(
    tmp_path: Path, monkeypatch
) -> None:
    daily_panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2021-01-04"]),
            "symbol": ["000001", "000001"],
            "open": [1.0, 1.1],
            "high": [1.0, 1.1],
            "low": [1.0, 1.1],
            "close": [1.0, 1.1],
            "volume": [100.0, 100.0],
            "vwap": [1.0, 1.1],
        }
    )
    labels = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2021-01-04"]),
            "symbol": ["000001", "000001"],
            "label_end_date": pd.to_datetime(["2020-01-03", "2021-01-05"]),
            "future_20d_return": [0.1, 0.2],
        }
    )
    panel_path = tmp_path / "daily_panel.parquet"
    labels_path = tmp_path / "labels.parquet"
    daily_panel.to_parquet(panel_path, index=False)
    labels.to_parquet(labels_path, index=False)
    config_path = tmp_path / "rl_custom.yml"
    config = {
        "seed": 7,
        "gamma": 0.5,
        "clip_epsilon": 0.1,
        "learning_rate": 0.001,
        "ppo_epochs": 1,
        "value_coef": 0.2,
        "entropy_coef": 0.03,
        "lstm_layers": 1,
        "lstm_hidden_size": 8,
        "dropout": 0.0,
        "head_hidden_size": 8,
        "invalid_reward": -2.0,
        "device": "cpu",
        "train_iterations": 2,
        "episodes_per_iteration": 3,
        "data": {
            "daily_panel": str(panel_path),
            "labels": str(labels_path),
            "train_start": "2020-01-01",
            "train_end": "2020-12-31",
            "validation_start": "2021-01-01",
            "validation_end": "2021-12-31",
        },
        "outputs": {
            "pool": str(tmp_path / "pool.parquet"),
            "metrics": str(tmp_path / "metrics.parquet"),
            "validation": str(tmp_path / "validation.parquet"),
            "config": str(tmp_path / "snapshot.yml"),
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    seen: list[dict] = []

    class FakeTrainer:
        def __init__(self, env, *, config=None):
            seen.append({"config": config, "invalid_reward": env.invalid_reward})

        def train_iterations(self, iterations, episodes_per_iteration=1, on_iteration=None):
            seen.append({"iterations": iterations, "episodes": episodes_per_iteration})
            return [{"pool_size": 0.0, "pool_ic": 0.0, "pool_loss": 1.0}]

    monkeypatch.setattr(experiment, "PPOTrainer", FakeTrainer)

    result = experiment.run_rl_alpha_mining(config_path)

    assert seen[0]["config"]["gamma"] == 0.5
    assert seen[0]["invalid_reward"] == -2.0
    assert seen[1] == {"iterations": 2, "episodes": 3}
    assert result.metric_rows == 1
