import importlib.util

import numpy as np
import pandas as pd
import pytest

from quant_rl_alpha.alpha import AlphaPool
from quant_rl_alpha.expression.tokens import ExpressionTokens
from quant_rl_alpha.rl import AlphaMiningEnv, PolicyValueNet, discounted_returns

TEST_TOKENS = ExpressionTokens(
    features=("open", "close", "high", "low", "volume", "vwap"),
    constants=("0.01", "1"),
    time_deltas=("1d",),
    max_tokens=8,
)


def _daily_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=4)
    rows = []
    for symbol, base in [("000001", 1.0), ("000002", 2.0), ("000003", 4.0)]:
        for index, date in enumerate(dates):
            close = base + index
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close,
                    "high": close + 0.1,
                    "low": close - 0.1,
                    "close": close,
                    "volume": 100.0 * int(symbol[-1]) + index,
                    "amount": (100.0 * int(symbol[-1]) + index) * close,
                    "vwap": close,
                }
            )
    return pd.DataFrame(rows)


def _labels() -> pd.DataFrame:
    panel = _daily_panel()
    labels = panel[panel["date"] == pd.Timestamp("2024-01-01")].copy()
    labels["future_20d_return"] = labels["close"]
    return labels.loc[:, ["date", "symbol", "future_20d_return"]]


def _env() -> AlphaMiningEnv:
    labels = _labels()
    pool = AlphaPool(labels, capacity=3, learning_rate=0.1, gradient_steps=50)
    return AlphaMiningEnv(
        _daily_panel(),
        labels,
        pool=pool,
        registry=TEST_TOKENS,
        invalid_reward=-1.0,
    )


def test_alpha_mining_env_keeps_pool_across_episodes() -> None:
    env = _env()
    assert env.reset() == ("BEG",)

    _, _, first_done, _ = env.step("close")
    assert not first_done
    _, reward, done, info = env.step("SEP")

    assert done
    assert reward > 0
    assert info["valid"]
    assert len(env.pool.entries) == 1

    env.reset()
    env.step("volume")
    _, _, second_done, _ = env.step("SEP")
    assert second_done
    assert len(env.pool.entries) == 2


def test_alpha_mining_env_rejects_invalid_actions_and_semantic_invalid_formula() -> None:
    env = _env()
    env.reset()

    with pytest.raises(ValueError, match="Invalid action"):
        env.step("SEP")

    env.reset()
    env.step("close")
    env.step("close")
    env.step("Sub")
    _, reward, done, info = env.step("SEP")

    assert done
    assert reward == -1.0
    assert info["reason"] == "semantic_invalid"
    assert len(env.pool.entries) == 0


def test_discounted_returns_uses_paper_gamma_one_by_default() -> None:
    assert np.allclose(discounted_returns([0.0, 0.0, 1.5]), [1.5, 1.5, 1.5])
    assert np.allclose(discounted_returns([1.0, 1.0], gamma=0.5), [1.5, 1.0])


def test_policy_value_net_requires_torch_when_dependency_is_missing() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed in this environment")
    with pytest.raises(ModuleNotFoundError, match="PyTorch"):
        PolicyValueNet(4)
