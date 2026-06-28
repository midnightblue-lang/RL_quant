import numpy as np
import pandas as pd
import pytest

from quant_rl_alpha.alpha.pool import AlphaPool, optimize_weights, pool_loss


def _frame(values_by_date: list[list[float]], value_col: str = "value") -> pd.DataFrame:
    rows = []
    symbols = ["000001", "000002", "000003"]
    for day, values in enumerate(values_by_date, start=1):
        date = pd.Timestamp(f"2024-01-0{day}")
        for symbol, value in zip(symbols, values, strict=True):
            rows.append({"date": date, "symbol": symbol, value_col: value})
    return pd.DataFrame(rows)


def _labels() -> pd.DataFrame:
    return _frame([[1.0, 2.0, 3.0], [1.0, 3.0, 5.0]], "future_20d_return")


def test_pool_loss_and_optimize_weights_allow_negative_weights() -> None:
    ic = np.array([-1.0])
    mutual = np.array([[1.0]])
    weights = optimize_weights(ic, mutual, l1_alpha=0.0)

    assert weights[0] < 0
    assert abs(weights[0] + 1) < 1e-6
    assert pool_loss(np.array([0.0]), ic, mutual) > pool_loss(weights, ic, mutual)


def test_alpha_pool_adds_valid_alpha_and_updates_metrics() -> None:
    pool = AlphaPool(_labels(), capacity=3, l1_alpha=0.0)
    result = pool.add(
        "a",
        "close",
        _frame([[1.0, 2.0, 3.0], [1.0, 3.0, 5.0]]),
        tokens=("BEG", "close", "SEP"),
    )

    assert result.added
    assert result.removed is None
    assert abs(result.weights["a"] - 1) < 1e-6
    assert abs(result.pool_loss) < 1e-6
    assert result.pool_ic == 1.0
    summary = pool.summary_frame()
    assert summary.loc[0, "tokens"] == "BEG close SEP"
    assert summary.loc[0, "formula"] == "close"


def test_invalid_alpha_does_not_enter_pool() -> None:
    pool = AlphaPool(_labels(), capacity=3)
    result = pool.add("constant", "1", _frame([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))

    assert not result.added
    assert result.reason == "nonfinite_ic"
    assert pool.entries == []
    assert result.weights == {}


def test_alpha_pool_aligns_values_to_label_sample() -> None:
    pool = AlphaPool(_labels(), capacity=3, l1_alpha=0.0)
    values = pd.concat(
        [
            _frame([[1.0, 2.0, 3.0], [1.0, 3.0, 5.0]]),
            pd.DataFrame([{"date": pd.Timestamp("2024-01-03"), "symbol": "999999", "value": 9.0}]),
        ],
        ignore_index=True,
    )

    result = pool.add("a", "a", values)

    assert result.added
    assert len(pool.entries[0].values) == len(_labels())
    assert "999999" not in set(pool.entries[0].values["symbol"])


def test_alpha_pool_rejects_sparse_coverage_and_duplicate_formula() -> None:
    pool = AlphaPool(
        _labels(),
        capacity=3,
        min_valid_dates=2,
        min_valid_ratio=0.5,
        min_stocks_per_date=2,
    )
    sparse = _frame([[1.0, np.nan, np.nan], [np.nan, np.nan, np.nan]])
    result = pool.add("sparse", "sparse", sparse)

    assert not result.added
    assert result.reason == "insufficient_coverage"

    valid = _frame([[1.0, 2.0, 3.0], [1.0, 3.0, 5.0]])
    assert pool.add("a", "same", valid).added
    duplicate = pool.add("b", "same", valid)
    assert not duplicate.added
    assert duplicate.reason == "duplicate_formula"


def test_pool_cap_removes_smallest_absolute_weight() -> None:
    pool = AlphaPool(_labels(), capacity=2, l1_alpha=0.0, max_steps=0)
    pool.add("keep_big", "a", _frame([[1.0, 2.0, 3.0], [1.0, 3.0, 5.0]]))
    pool.add("remove_small", "b", _frame([[3.0, 2.0, 1.0], [5.0, 3.0, 1.0]]))
    pool.weights = np.array([0.9, 0.001])

    result = pool.add("keep_negative", "c", _frame([[1.0, 3.0, 2.0], [1.0, 5.0, 3.0]]))
    names = [entry.name for entry in pool.entries]

    assert result.removed == "remove_small"
    assert names == ["keep_big", "keep_negative"]


def test_mutual_ic_cache_is_symmetric_and_names_are_unique() -> None:
    pool = AlphaPool(_labels(), capacity=3, l1_alpha=0.0)
    pool.add("a", "a", _frame([[1.0, 2.0, 3.0], [1.0, 3.0, 5.0]]))
    pool.add("b", "b", _frame([[3.0, 2.0, 1.0], [5.0, 3.0, 1.0]]))

    assert ("a", "b") in pool.mutual_cache
    assert ("b", "a") not in pool.mutual_cache
    with pytest.raises(ValueError, match="already exists"):
        pool.add("a", "duplicate", _frame([[1.0, 2.0, 3.0], [1.0, 3.0, 5.0]]))


def test_alpha_pool_accepts_high_mutual_ic_and_records_cache() -> None:
    pool = AlphaPool(_labels(), capacity=3, l1_alpha=0.0)
    first = _frame([[1.0, 2.0, 3.0], [1.0, 3.0, 5.0]])
    assert pool.add("a", "a", first).added

    duplicate_values = _frame([[2.0, 3.0, 4.0], [2.0, 4.0, 6.0]])
    result = pool.add("b", "b", duplicate_values)

    assert result.added
    assert [entry.name for entry in pool.entries] == ["a", "b"]
    assert pool.mutual_cache[("a", "b")] > 0.99
    assert np.isfinite(pool.weights).all()


def test_alpha_pool_rolls_back_when_new_alpha_is_pruned() -> None:
    pool = AlphaPool(_labels(), capacity=1, l1_alpha=0.0)
    strong = _frame([[1.0, 2.0, 3.0], [1.0, 3.0, 5.0]])
    weak = _frame([[1.0, 3.0, 2.0], [3.0, 1.0, 2.0]])
    assert pool.add("strong", "strong", strong).added

    result = pool.add("weak", "weak", weak)

    assert not result.added
    assert result.reason == "new_alpha_pruned"
    assert [entry.name for entry in pool.entries] == ["strong"]
    assert np.isfinite(pool.weights).all()


def test_alpha_pool_rejects_nonfinite_optimization_without_mutating_pool(monkeypatch) -> None:
    pool = AlphaPool(_labels(), capacity=3, l1_alpha=0.0)
    assert pool.add("a", "a", _frame([[1.0, 2.0, 3.0], [1.0, 3.0, 5.0]])).added

    def bad_optimizer(*args, **kwargs) -> np.ndarray:
        return np.array([np.nan, np.nan])

    monkeypatch.setattr("quant_rl_alpha.alpha.pool.optimize_weights", bad_optimizer)

    result = pool.add("b", "b", _frame([[3.0, 1.0, 2.0], [5.0, 1.0, 3.0]]))

    assert not result.added
    assert result.reason == "nonfinite_weights"
    assert [entry.name for entry in pool.entries] == ["a"]
    assert np.isfinite(pool.weights).all()
