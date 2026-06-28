import numpy as np
import pandas as pd

from quant_rl_alpha.model.xgboost_ranker import walk_forward_predict


class FakeBooster:
    def get_score(self, importance_type: str) -> dict[str, float]:
        values = {"gain": 0.5, "weight": 2.0, "cover": 3.0}
        return {"alpha_00": values[importance_type]}


class FakeModel:
    def __init__(self, seen: list[int]) -> None:
        self.seen = seen

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self.seen.append(len(y))

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return x["alpha_00"].to_numpy(dtype=float)

    def get_booster(self) -> FakeBooster:
        return FakeBooster()


def _dataset() -> pd.DataFrame:
    rows = []
    specs = [
        ("2018-01-31", "2018-02-28", [0.4, 0.6], [10.0, 20.0]),
        ("2021-11-30", "2021-12-30", [0.2, 0.8], [1.0, 2.0]),
        ("2021-12-31", "2022-02-15", [0.9, 0.1], [2.0, 1.0]),
        ("2022-01-14", "2022-02-15", [0.1, 0.9], [9.0, 8.0]),
        ("2022-01-31", "2022-02-28", [0.25, 0.75], [1.0, 2.0]),
    ]
    for date, label_end_date, targets, features in specs:
        for symbol, target, feature in zip(["000001", "000002"], targets, features, strict=True):
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "symbol": symbol,
                    "label_end_date": pd.Timestamp(label_end_date),
                    "future_20d_return": target,
                    "future_20d_rank": target,
                    "alpha_00": feature,
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_excludes_training_labels_after_prediction_date() -> None:
    seen_train_counts: list[int] = []

    def factory(model_config):
        assert model_config == {"objective": "reg:squarederror"}
        return FakeModel(seen_train_counts)

    config = {
        "target": "future_20d_rank",
        "train_window_years": 3,
        "prediction_start": "2022-01-01",
        "rebalance_frequency": "monthly",
        "model": {"objective": "reg:squarederror"},
    }

    predictions, metrics, importance = walk_forward_predict(
        _dataset(),
        config,
        model_factory=factory,
    )

    assert seen_train_counts == [2]
    assert predictions["date"].unique().tolist() == [pd.Timestamp("2022-01-31")]
    assert metrics.loc[0, "train_count"] == 2
    assert metrics.loc[0, "prediction_count"] == 2
    assert np.isclose(metrics.loc[0, "ic"], 1.0)
    assert np.isclose(metrics.loc[0, "rank_ic"], 1.0)
    assert metrics.loc[0, "feature_coverage"] == 1.0
    assert importance.loc[0, "feature"] == "alpha_00"
    assert importance.loc[0, "gain"] == 0.5
    assert importance.loc[0, "weight"] == 2.0
    assert importance.loc[0, "cover"] == 3.0
