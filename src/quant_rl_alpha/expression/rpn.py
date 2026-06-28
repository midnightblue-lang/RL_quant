from __future__ import annotations

from dataclasses import dataclass

from quant_rl_alpha.expression.dimensions import expression_dimension, operator_dimension
from quant_rl_alpha.expression.tokens import (
    BEG,
    CS_BINARY_OPERATORS,
    CS_UNARY_OPERATORS,
    SEP,
    TS_BINARY_OPERATORS,
    TS_UNARY_OPERATORS,
    ExpressionTokens,
    expression_tokens,
    is_constant,
    is_feature,
    is_time_delta,
    normalize_token,
    parse_time_delta,
)


class RPNError(ValueError):
    """Raised when an RPN token sequence is not a valid formula."""


@dataclass(frozen=True)
class Expression:
    token: str
    children: tuple[Expression, ...] = ()
    window: int | None = None

    @property
    def is_pure_constant(self) -> bool:
        if is_constant(self.token):
            return True
        if is_feature(self.token):
            return False
        return bool(self.children) and all(child.is_pure_constant for child in self.children)

    def to_formula(self) -> str:
        if not self.children:
            return self.token
        if self.token in CS_UNARY_OPERATORS:
            return f"{self.token}({self.children[0].to_formula()})"
        if self.token in {"Add", "Sub", "Mul", "Div"}:
            symbols = {"Add": "+", "Sub": "-", "Mul": "*", "Div": "/"}
            left, right = self.children
            return f"({left.to_formula()} {symbols[self.token]} {right.to_formula()})"
        if self.token in {"Greater", "Less"}:
            left, right = self.children
            return f"{self.token}({left.to_formula()}, {right.to_formula()})"
        if self.token in TS_UNARY_OPERATORS:
            return f"{self.token}({self.children[0].to_formula()}, {self.window})"
        if self.token in TS_BINARY_OPERATORS:
            left, right = self.children
            return f"{self.token}({left.to_formula()}, {right.to_formula()}, {self.window})"
        raise RPNError(f"Unknown expression token: {self.token}")


StackItem = Expression | int


def parse_rpn(
    tokens: list[object] | tuple[object, ...],
    registry: ExpressionTokens | None = None,
) -> Expression:
    registry = registry or expression_tokens()
    sequence = [normalize_token(token) for token in tokens]
    if not sequence:
        raise RPNError("RPN token sequence is empty")
    if sequence[0] == BEG:
        sequence = sequence[1:]
    if sequence and sequence[-1] == SEP:
        sequence = sequence[:-1]
    if BEG in sequence or SEP in sequence:
        raise RPNError("BEG and SEP are only allowed at sequence boundaries")

    stack = stack_from_tokens(sequence, registry)
    if len(stack) != 1 or not isinstance(stack[0], Expression):
        raise RPNError("RPN sequence must leave exactly one expression on the stack")
    expr = stack[0]
    if expr.is_pure_constant:
        raise RPNError("Pure constant expressions are not valid alphas")
    return expr


def stack_from_tokens(
    tokens: list[str],
    registry: ExpressionTokens | None = None,
    *,
    partial: bool = False,
) -> list[StackItem]:
    registry = registry or expression_tokens()
    stack: list[StackItem] = []
    for token in tokens:
        _apply_token(stack, token, registry, partial=partial)
    return stack


def can_add_time_delta(stack: list[StackItem]) -> bool:
    return bool(stack) and isinstance(stack[-1], Expression) and not stack[-1].is_pure_constant


def can_apply_operator(stack: list[StackItem], token: str) -> bool:
    if token in CS_UNARY_OPERATORS:
        return (
            _has_expressions(stack, 1)
            and not stack[-1].is_pure_constant
            and _operator_dimension(token, (stack[-1],)) is not None
        )
    if token in CS_BINARY_OPERATORS:
        if not _has_expressions(stack, 2):
            return False
        return not (
            stack[-1].is_pure_constant and stack[-2].is_pure_constant
        ) and _operator_dimension(token, (stack[-2], stack[-1])) is not None
    if token in TS_UNARY_OPERATORS:
        has_operand = _has_expressions(stack[:-1], 1)
        return (
            _has_window(stack)
            and has_operand
            and not stack[-2].is_pure_constant
            and _operator_dimension(token, (stack[-2],)) is not None
        )
    if token in TS_BINARY_OPERATORS:
        if not _has_window(stack) or not _has_expressions(stack[:-1], 2):
            return False
        return not (
            stack[-2].is_pure_constant and stack[-3].is_pure_constant
        ) and _operator_dimension(token, (stack[-3], stack[-2])) is not None
    return False


def is_complete_stack(stack: list[StackItem]) -> bool:
    return len(stack) == 1 and isinstance(stack[0], Expression) and not stack[0].is_pure_constant


def _apply_token(
    stack: list[StackItem],
    token: str,
    registry: ExpressionTokens,
    *,
    partial: bool,
) -> None:
    if is_feature(token, registry) or is_constant(token, registry):
        if partial and _has_window(stack):
            raise RPNError("Operands may not follow a pending time-delta")
        stack.append(Expression(token))
        return
    if is_time_delta(token, registry):
        if partial and not can_add_time_delta(stack):
            raise RPNError("Time-delta requires a non-constant expression before it")
        stack.append(parse_time_delta(token, registry))
        return
    if partial and not can_apply_operator(stack, token):
        raise RPNError(f"Operator is not valid for current stack: {token}")
    if token in CS_UNARY_OPERATORS:
        child = _pop_expression(stack, token)
        _push_operator(stack, Expression(token, (child,)))
        return
    if token in CS_BINARY_OPERATORS:
        right = _pop_expression(stack, token)
        left = _pop_expression(stack, token)
        _push_operator(stack, Expression(token, (left, right)))
        return
    if token in TS_UNARY_OPERATORS:
        window = _pop_window(stack, token)
        child = _pop_expression(stack, token)
        _push_operator(stack, Expression(token, (child,), window))
        return
    if token in TS_BINARY_OPERATORS:
        window = _pop_window(stack, token)
        right = _pop_expression(stack, token)
        left = _pop_expression(stack, token)
        _push_operator(stack, Expression(token, (left, right), window))
        return
    raise RPNError(f"Unknown RPN token: {token}")


def _has_expressions(stack: list[StackItem], count: int) -> bool:
    return len(stack) >= count and all(isinstance(item, Expression) for item in stack[-count:])


def _has_window(stack: list[StackItem]) -> bool:
    return bool(stack) and isinstance(stack[-1], int)


def _push_operator(stack: list[StackItem], expr: Expression) -> None:
    if expr.is_pure_constant:
        raise RPNError("Multi-token expressions may not be pure constants")
    if expression_dimension(expr) is None:
        raise RPNError("Dimensionally invalid expression")
    stack.append(expr)


def _operator_dimension(token: str, items: tuple[StackItem, ...]) -> str | None:
    dims = tuple(expression_dimension(item) for item in items if isinstance(item, Expression))
    return operator_dimension(token, dims)


def _pop_expression(stack: list[StackItem], operator: str) -> Expression:
    if not stack or not isinstance(stack[-1], Expression):
        raise RPNError(f"{operator} requires an expression operand")
    return stack.pop()


def _pop_window(stack: list[StackItem], operator: str) -> int:
    if not stack or not isinstance(stack[-1], int):
        raise RPNError(f"{operator} requires a trailing time-delta operand")
    return stack.pop()
