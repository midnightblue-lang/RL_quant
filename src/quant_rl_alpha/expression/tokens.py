from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from quant_rl_alpha.utils.config import load_config

BEG: Final = "BEG"
SEP: Final = "SEP"

CS_UNARY_OPERATORS: Final[frozenset[str]] = frozenset({"Abs", "Log"})
CS_BINARY_OPERATORS: Final[frozenset[str]] = frozenset(
    {"Add", "Sub", "Mul", "Div", "Greater", "Less"}
)
TS_UNARY_OPERATORS: Final[frozenset[str]] = frozenset(
    {"Ref", "Mean", "Med", "Sum", "Std", "Var", "Max", "Min", "Mad", "Delta", "WMA", "EMA"}
)
TS_BINARY_OPERATORS: Final[frozenset[str]] = frozenset({"Cov", "Corr"})
OPERATORS: Final[frozenset[str]] = (
    CS_UNARY_OPERATORS | CS_BINARY_OPERATORS | TS_UNARY_OPERATORS | TS_BINARY_OPERATORS
)


@dataclass(frozen=True)
class ExpressionTokens:
    features: tuple[str, ...]
    constants: tuple[str, ...]
    time_deltas: tuple[str, ...]
    max_tokens: int

    @property
    def operands(self) -> tuple[str, ...]:
        return self.features + self.constants

    @property
    def all_tokens(self) -> tuple[str, ...]:
        return (
            (BEG, SEP)
            + self.features
            + self.constants
            + self.time_deltas
            + tuple(sorted(OPERATORS))
        )


def expression_tokens() -> ExpressionTokens:
    config = load_config("expression")
    return ExpressionTokens(
        features=tuple(str(item) for item in config["features"]),
        constants=tuple(_format_number(item) for item in config["constants"]),
        time_deltas=tuple(f"{int(item)}d" for item in config["time_deltas"]),
        max_tokens=int(config["max_tokens"]),
    )


def normalize_token(token: object) -> str:
    if isinstance(token, int | float):
        return _format_number(token)
    return str(token)


def is_constant(token: str, registry: ExpressionTokens | None = None) -> bool:
    registry = registry or expression_tokens()
    return normalize_token(token) in registry.constants


def is_feature(token: str, registry: ExpressionTokens | None = None) -> bool:
    registry = registry or expression_tokens()
    return normalize_token(token) in registry.features


def is_time_delta(token: str, registry: ExpressionTokens | None = None) -> bool:
    registry = registry or expression_tokens()
    return normalize_token(token) in registry.time_deltas


def parse_time_delta(token: str, registry: ExpressionTokens | None = None) -> int:
    text = normalize_token(token)
    if not is_time_delta(text, registry):
        raise ValueError(f"Not a time-delta token: {token}")
    return int(text[:-1])


def parse_constant(token: str, registry: ExpressionTokens | None = None) -> float:
    text = normalize_token(token)
    if not is_constant(text, registry):
        raise ValueError(f"Not a constant token: {token}")
    return float(text)


def _format_number(value: object) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"
