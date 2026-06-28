import numpy as np
import pandas as pd

from quant_rl_alpha.alpha.metrics import (
    cross_section_l2_normalize,
    daily_ic,
    mean_ic,
    mean_rank_ic,
    mutual_ic,
)


def _frame(values_by_date: list[list[float]], value_col: str = "value") -> pd.DataFrame:
    rows = []
    symbols = ["000001", "000002", "000003"]
    for day, values in enumerate(values_by_date, start=1):
        date = pd.Timestamp(f"2024-01-0{day}")
        for symbol, value in zip(symbols, values, strict=True):
            rows.append({"date": date, "symbol": symbol, value_col: value})
    return pd.DataFrame(rows)


def test_cross_section_l2_normalize_uses_paper_equation() -> None:
    values = _frame([[1.0, 2.0, 3.0]])
    normalized = cross_section_l2_normalize(values)

    expected = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2)
    assert np.allclose(normalized["normalized"].to_numpy(), expected)
    assert abs(normalized["normalized"].mean()) < 1e-12
    assert np.isclose(np.linalg.norm(normalized["normalized"]), 1.0)


def test_cross_section_l2_normalize_rejects_constant_cross_section() -> None:
    values = _frame([[1.0, 1.0, 1.0]])
    normalized = cross_section_l2_normalize(values)

    assert normalized["normalized"].isna().all()


def test_ic_and_rank_ic_match_toy_cases() -> None:
    factor = _frame([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    label = _frame([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]], "future_20d_return")

    assert daily_ic(factor, label).tolist() == [1.0, -1.0]
    assert mean_ic(factor, label) == 0.0

    ranked_factor = _frame([[1.0, 2.0, 100.0]])
    ranked_label = _frame([[10.0, 20.0, 30.0]], "future_20d_return")
    assert mean_rank_ic(ranked_factor, ranked_label) == 1.0


def test_mutual_ic_is_signed_and_symmetric() -> None:
    factor = _frame([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    negative = factor.copy()
    negative["value"] = -negative["value"]

    assert np.isclose(mutual_ic(factor, factor), 1.0)
    assert np.isclose(mutual_ic(factor, negative), -1.0)
    assert mutual_ic(factor, negative) == mutual_ic(negative, factor)


def test_ic_handles_large_values_without_overflow() -> None:
    factor = _frame([[1e200, 2e200, 3e200], [3e200, 2e200, 1e200]])
    label = _frame([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], "future_20d_return")

    assert daily_ic(factor, label).tolist() == [1.0, -1.0]
    assert mean_ic(factor, label) == 0.0
