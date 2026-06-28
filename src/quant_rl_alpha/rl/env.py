from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_rl_alpha.alpha import AlphaPool, alpha_pool_from_config
from quant_rl_alpha.expression import evaluate, is_semantically_valid, parse_rpn, valid_actions
from quant_rl_alpha.expression.rpn import RPNError
from quant_rl_alpha.expression.tokens import (
    BEG,
    SEP,
    ExpressionTokens,
    expression_tokens,
    normalize_token,
)
from quant_rl_alpha.utils.config import load_config


@dataclass
class AlphaMiningEnv:
    daily_panel: pd.DataFrame
    labels: pd.DataFrame
    pool: AlphaPool | None = None
    registry: ExpressionTokens = field(default_factory=expression_tokens)
    invalid_reward: float | None = None
    state: list[str] = field(default_factory=lambda: [BEG])
    episode_index: int = 0

    def __post_init__(self) -> None:
        if self.invalid_reward is None:
            self.invalid_reward = float(load_config("rl")["invalid_reward"])
        if self.pool is None:
            self.pool = alpha_pool_from_config(self.labels)

    def reset(self) -> tuple[str, ...]:
        self.episode_index += 1
        self.state = [BEG]
        return tuple(self.state)

    def valid_actions(self) -> list[str]:
        return valid_actions(self.state, self.registry)

    def step(self, action: object) -> tuple[tuple[str, ...], float, bool, dict[str, object]]:
        token = normalize_token(action)
        allowed = self.valid_actions()
        if token not in allowed:
            raise ValueError(f"Invalid action {token!r}; valid actions are {allowed}")

        self.state.append(token)
        done = token == SEP or len(self.state) >= self.registry.max_tokens
        if not done:
            return tuple(self.state), 0.0, False, {"valid_actions": self.valid_actions()}
        return self._finish_episode()

    def _finish_episode(self) -> tuple[tuple[str, ...], float, bool, dict[str, object]]:
        info: dict[str, object] = {"tokens": tuple(self.state)}
        try:
            expr = parse_rpn(self.state, self.registry)
        except (RPNError, ValueError) as error:
            info.update({"valid": False, "reason": str(error)})
            return tuple(self.state), float(self.invalid_reward), True, info
        values = evaluate(expr, self.daily_panel)

        if not is_semantically_valid(values):
            info.update(
                {"valid": False, "formula": expr.to_formula(), "reason": "semantic_invalid"}
            )
            return tuple(self.state), float(self.invalid_reward), True, info

        name = f"alpha_{self.episode_index}"
        result = self.pool.add(name, expr.to_formula(), values, tokens=tuple(self.state))
        if not result.added or not np.isfinite(result.pool_ic):
            reason = result.reason or "pool_rejected"
            info.update({"valid": False, "formula": expr.to_formula(), "reason": reason})
            return tuple(self.state), float(self.invalid_reward), True, info

        info.update(
            {
                "valid": True,
                "name": name,
                "formula": expr.to_formula(),
                "pool_ic": result.pool_ic,
                "pool_loss": result.pool_loss,
                "weights": result.weights,
                "removed": result.removed,
            }
        )
        return tuple(self.state), float(result.pool_ic), True, info
