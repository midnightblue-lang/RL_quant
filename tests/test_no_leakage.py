import pandas as pd

from quant_rl_alpha.data.labels import build_forward_return_labels
from quant_rl_alpha.data.universe import UniverseConfig, build_monthly_universe


def _universe_bars() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=60)
    rows = []
    for symbol, amount in [("000001", 1000), ("000002", 900)]:
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "name": symbol,
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.0,
                    "close": 10.0 + index,
                    "volume": 100,
                    "amount": amount,
                }
            )
    return pd.DataFrame(rows)


def test_future_liquidity_does_not_change_current_month_universe() -> None:
    config = UniverseConfig(min_listed_days=20, liquidity_window=5, top_n=1)
    base = _universe_bars()
    changed = base.copy()
    current_month_end = pd.Timestamp("2024-01-31")
    changed.loc[
        (changed["symbol"] == "000002") & (changed["date"] > current_month_end),
        "amount",
    ] = 999_999

    base_universe = build_monthly_universe(base, config)
    changed_universe = build_monthly_universe(changed, config)
    base_current = base_universe[base_universe["date"] == current_month_end]
    changed_current = changed_universe[changed_universe["date"] == current_month_end]

    assert base_current["symbol"].tolist() == ["000001"]
    assert changed_current["symbol"].tolist() == ["000001"]


def test_price_after_label_horizon_does_not_change_label() -> None:
    dates = pd.bdate_range("2024-01-01", periods=25)
    rows = [
        {"date": date, "symbol": "000001", "close": 10.0 + index}
        for index, date in enumerate(dates)
    ]
    base = pd.DataFrame(rows)
    changed = base.copy()
    changed.loc[changed["date"] == dates[21], "close"] = 999.0

    base_label = build_forward_return_labels(base, horizon=20).iloc[0]
    changed_label = build_forward_return_labels(changed, horizon=20).iloc[0]

    assert changed_label["label_end_date"] == dates[20]
    assert changed_label["future_20d_return"] == base_label["future_20d_return"]
