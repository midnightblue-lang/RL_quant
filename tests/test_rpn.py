import pytest

from quant_rl_alpha.expression.rpn import RPNError, parse_rpn


def test_parse_rpn_and_render_formula() -> None:
    expr = parse_rpn(["BEG", "close", "20d", "Mean", "close", "Div", "1", "Sub", "SEP"])

    assert expr.to_formula() == "((Mean(close, 20) / close) - 1)"


def test_unknown_token_is_rejected() -> None:
    with pytest.raises(RPNError, match="Unknown"):
        parse_rpn(["BEG", "close", "NotAnOperator", "SEP"])


@pytest.mark.parametrize(
    "tokens",
    [
        ["BEG", "close", "10d", "Mean", "SEP"],
        ["BEG", "close", "10d", "Ref", "SEP"],
        ["BEG", "close", "10d", "Delta", "SEP"],
        ["BEG", "close", "10d", "Std", "SEP"],
        ["BEG", "close", "10d", "Var", "SEP"],
        ["BEG", "close", "10d", "Max", "SEP"],
        ["BEG", "close", "10d", "Min", "SEP"],
        ["BEG", "close", "10d", "Mad", "SEP"],
        ["BEG", "close", "10d", "WMA", "SEP"],
        ["BEG", "close", "10d", "EMA", "SEP"],
        ["BEG", "close", "volume", "10d", "Corr", "SEP"],
        ["BEG", "close", "vwap", "10d", "Cov", "SEP"],
        ["BEG", "high", "low", "Sub", "Abs", "SEP"],
        ["BEG", "vwap", "close", "Div", "Log", "SEP"],
    ],
)
def test_handwritten_rpn_formulas_parse(tokens: list[str]) -> None:
    expr = parse_rpn(tokens)

    assert expr.to_formula()
    assert not expr.is_pure_constant


@pytest.mark.parametrize(
    "tokens, message",
    [
        (["BEG", "close", "Mean", "SEP"], "time-delta"),
        (["BEG", "close", "10d", "Add", "SEP"], "expression operand"),
        (["BEG", "1", "2", "Add", "SEP"], "pure constants"),
        (["BEG", "close", "SEP", "1"], "boundaries"),
        (["BEG", "close", "10d", "SEP"], "exactly one expression"),
    ],
)
def test_invalid_rpn_is_rejected(tokens: list[str], message: str) -> None:
    with pytest.raises(RPNError, match=message):
        parse_rpn(tokens)


@pytest.mark.parametrize(
    "tokens",
    [
        ["BEG", "vwap", "volume", "Sub", "SEP"],
        ["BEG", "vwap", "volume", "Div", "SEP"],
        ["BEG", "vwap", "vwap", "Mul", "SEP"],
        ["BEG", "vwap", "0.5", "Less", "SEP"],
        ["BEG", "close", "volume", "10d", "Cov", "SEP"],
    ],
)
def test_dimensionally_invalid_rpn_is_rejected(tokens: list[str]) -> None:
    with pytest.raises(RPNError, match="Dimensionally invalid"):
        parse_rpn(tokens)


def test_dimensionally_valid_ratio_can_subtract_constant() -> None:
    expr = parse_rpn(["BEG", "close", "vwap", "Div", "1", "Sub", "SEP"])

    assert expr.to_formula() == "((close / vwap) - 1)"
