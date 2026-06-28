from __future__ import annotations

from quant_rl_alpha.expression.rpn import (
    RPNError,
    can_add_time_delta,
    can_apply_operator,
    is_complete_stack,
    stack_from_tokens,
)
from quant_rl_alpha.expression.tokens import (
    BEG,
    OPERATORS,
    SEP,
    ExpressionTokens,
    expression_tokens,
    normalize_token,
)


def all_action_tokens(registry: ExpressionTokens | None = None) -> tuple[str, ...]:
    registry = registry or expression_tokens()
    return (
        (BEG,)
        + registry.features
        + registry.constants
        + registry.time_deltas
        + tuple(sorted(OPERATORS))
        + (SEP,)
    )


def valid_actions(
    tokens: list[object] | tuple[object, ...],
    registry: ExpressionTokens | None = None,
) -> list[str]:
    registry = registry or expression_tokens()
    sequence = [normalize_token(token) for token in tokens]
    if not sequence:
        return [BEG]
    if sequence[0] != BEG or SEP in sequence:
        return []

    try:
        stack = stack_from_tokens(sequence[1:], registry, partial=True)
    except (RPNError, ValueError):
        return []
    if len(sequence) >= registry.max_tokens:
        return []

    actions: list[str] = []
    if not stack or not isinstance(stack[-1], int):
        actions.extend(registry.operands)
    if can_add_time_delta(stack):
        actions.extend(registry.time_deltas)
    actions.extend(token for token in sorted(OPERATORS) if can_apply_operator(stack, token))
    if is_complete_stack(stack):
        actions.append(SEP)
    return actions


def valid_action_mask(
    tokens: list[object] | tuple[object, ...],
    registry: ExpressionTokens | None = None,
) -> dict[str, bool]:
    registry = registry or expression_tokens()
    valid = set(valid_actions(tokens, registry))
    return {token: token in valid for token in all_action_tokens(registry)}
