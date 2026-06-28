import numpy as np
import pandas as pd
import pytest

from quant_rl_alpha.expression.evaluator import evaluate, is_semantically_valid
from quant_rl_alpha.expression.rpn import parse_rpn
from quant_rl_alpha.expression.tokens import ExpressionTokens

TEST_TOKENS = ExpressionTokens(
    features=("open", "close", "high", "low", "volume", "vwap"),
    constants=("-1", "0.01", "1", "2"),
    time_deltas=("1d", "2d", "3d"),
    max_tokens=20,
)


def _bars() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=6)
    rows = []
    for symbol, base, volume_base in [("000001", 10.0, 1000.0), ("000002", 20.0, 2000.0)]:
        for index, date in enumerate(dates):
            close = base + index
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close - 0.5,
                    "high": close + 1 + index * 0.1,
                    "low": close - 1 - (0.2 if symbol == "000002" else 0.1),
                    "close": close,
                    "volume": volume_base + index * 100,
                    "amount": (volume_base + index * 100) * close,
                    "vwap": close + (0.1 if symbol == "000002" else 0.0),
                }
            )
    return pd.DataFrame(rows)


def _value(values: pd.DataFrame, date: str, symbol: str) -> float:
    row = values[(values["date"] == pd.Timestamp(date)) & (values["symbol"] == symbol)]
    return float(row.iloc[0]["value"])


def _expr(tokens: list[str]):
    return parse_rpn(tokens, TEST_TOKENS)


def test_ref_and_rolling_mean_use_only_current_and_past_rows() -> None:
    bars = _bars()
    ref = evaluate(_expr(["BEG", "close", "2d", "Ref", "SEP"]), bars)
    mean = evaluate(_expr(["BEG", "close", "2d", "Mean", "SEP"]), bars)

    assert _value(ref, "2024-01-03", "000001") == 10.0
    assert _value(mean, "2024-01-02", "000001") == 10.5

    changed = bars.copy()
    changed.loc[changed["date"] > pd.Timestamp("2024-01-03"), "close"] = 999.0
    changed_ref = evaluate(_expr(["BEG", "close", "2d", "Ref", "SEP"]), changed)
    changed_mean = evaluate(_expr(["BEG", "close", "2d", "Mean", "SEP"]), changed)

    assert _value(changed_ref, "2024-01-03", "000001") == _value(ref, "2024-01-03", "000001")
    assert _value(changed_mean, "2024-01-03", "000001") == _value(mean, "2024-01-03", "000001")


def test_cross_section_and_time_series_operators_evaluate() -> None:
    formulas = [
        ["BEG", "high", "low", "Sub", "SEP"],
        ["BEG", "close", "2", "Mul", "SEP"],
        ["BEG", "close", "2d", "Std", "SEP"],
        ["BEG", "close", "2d", "Var", "SEP"],
        ["BEG", "close", "2d", "Max", "close", "2d", "Min", "Sub", "SEP"],
        ["BEG", "close", "2d", "Mad", "SEP"],
        ["BEG", "close", "2d", "WMA", "SEP"],
        ["BEG", "close", "2d", "EMA", "SEP"],
        ["BEG", "close", "vwap", "3d", "Cov", "SEP"],
        ["BEG", "close", "volume", "3d", "Corr", "SEP"],
    ]

    for tokens in formulas:
        values = evaluate(_expr(tokens), _bars())
        assert len(values) == len(_bars())
        assert {"date", "symbol", "value"} <= set(values.columns)


def test_divide_by_zero_and_log_non_positive_are_semantically_invalid() -> None:
    bars = _bars()
    zero_div = evaluate(_expr(["BEG", "close", "close", "close", "Sub", "Div", "SEP"]), bars)
    negative = bars.copy()
    negative["close"] = -1.0
    logged = evaluate(_expr(["BEG", "close", "Log", "SEP"]), negative)

    assert zero_div["value"].isna().all()
    assert logged["value"].isna().all()
    assert not is_semantically_valid(zero_div)
    assert not is_semantically_valid(logged)


def test_semantic_validity_rejects_cross_section_constant_values() -> None:
    constant_like = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "symbol": ["000001", "000002"],
            "value": [1.0, 1.0],
        }
    )
    valid = evaluate(_expr(["BEG", "close", "SEP"]), _bars())

    assert not is_semantically_valid(constant_like)
    assert is_semantically_valid(valid)
    assert np.isfinite(valid["value"].dropna()).all()


def test_evaluator_rejects_missing_feature_and_duplicate_keys() -> None:
    bars = _bars()
    with pytest.raises(ValueError, match="feature column"):
        evaluate(_expr(["BEG", "close", "SEP"]), bars.drop(columns=["close"]))

    duplicate = pd.concat([bars, bars.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(_expr(["BEG", "close", "SEP"]), duplicate)
