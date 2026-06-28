from __future__ import annotations

from typing import Any, Final

from quant_rl_alpha.expression.tokens import (
    CS_UNARY_OPERATORS,
    TS_BINARY_OPERATORS,
    TS_UNARY_OPERATORS,
    is_constant,
    is_feature,
)

PRICE: Final = "price"
VOLUME: Final = "volume"
DIMENSIONLESS: Final = "dimensionless"
CONSTANT: Final = "constant"

FEATURE_DIMENSIONS: Final = {
    "open": PRICE,
    "close": PRICE,
    "high": PRICE,
    "low": PRICE,
    "vwap": PRICE,
    "volume": VOLUME,
}


def expression_dimension(expr: Any) -> str | None:
    token = str(expr.token)
    if is_constant(token):
        return CONSTANT
    if is_feature(token):
        return FEATURE_DIMENSIONS.get(token, DIMENSIONLESS)
    child_dims = tuple(expression_dimension(child) for child in expr.children)
    if any(dim is None for dim in child_dims):
        return None
    return operator_dimension(token, child_dims)


def is_dimensionally_valid(expr: Any) -> bool:
    return expression_dimension(expr) is not None


def operator_dimension(token: str, child_dims: tuple[str | None, ...]) -> str | None:
    if any(dim is None for dim in child_dims):
        return None
    dims = tuple(str(dim) for dim in child_dims)
    if token in CS_UNARY_OPERATORS:
        return DIMENSIONLESS if token == "Log" else dims[0]
    if token in {"Add", "Sub"}:
        return _additive_dimension(dims[0], dims[1])
    if token == "Mul":
        return _multiplicative_dimension(dims[0], dims[1])
    if token == "Div":
        return _division_dimension(dims[0], dims[1])
    if token in {"Greater", "Less"}:
        return _comparable_dimension(dims[0], dims[1])
    if token in TS_UNARY_OPERATORS:
        return dims[0]
    if token == "Corr":
        return DIMENSIONLESS if CONSTANT not in dims else None
    if token == "Cov":
        return DIMENSIONLESS if CONSTANT not in dims and dims[0] == dims[1] else None
    if token in TS_BINARY_OPERATORS:
        return DIMENSIONLESS
    return None


def _additive_dimension(left: str, right: str) -> str | None:
    if left == right:
        return left
    if {left, right} <= {DIMENSIONLESS, CONSTANT}:
        return DIMENSIONLESS
    return None


def _multiplicative_dimension(left: str, right: str) -> str | None:
    if left == CONSTANT:
        return right
    if right == CONSTANT:
        return left
    if left == DIMENSIONLESS:
        return right
    if right == DIMENSIONLESS:
        return left
    return None


def _division_dimension(left: str, right: str) -> str | None:
    if right == CONSTANT:
        return left
    if left == right:
        return DIMENSIONLESS
    if right == DIMENSIONLESS:
        return left
    if left == CONSTANT and right == DIMENSIONLESS:
        return DIMENSIONLESS
    return None


def _comparable_dimension(left: str, right: str) -> str | None:
    if left == right:
        return left
    if {left, right} <= {DIMENSIONLESS, CONSTANT}:
        return DIMENSIONLESS
    return None
