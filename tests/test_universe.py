import pandas as pd

from quant_rl_alpha.data.universe import UniverseConfig, build_monthly_universe


def _bars() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=25)
    rows = []
    for symbol, name, amount_base in [
        ("000001", "平安银行", 1000),
        ("000002", "万科A", 3000),
        ("000003", "ST样例", 5000),
    ]:
        for index, date in enumerate(dates):
            volume = 100
            amount = amount_base + index
            if symbol == "000002" and date == dates[-1]:
                volume = 0
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "name": name,
                    "open": 10.0,
                    "high": 10.5,
                    "low": 9.5,
                    "close": 10.1,
                    "volume": volume,
                    "amount": amount,
                }
            )
    return pd.DataFrame(rows)


def test_build_monthly_universe_filters_st_and_zero_volume() -> None:
    universe = build_monthly_universe(
        _bars(),
        UniverseConfig(
            min_listed_days=20,
            liquidity_window=5,
            top_n=10,
            exclude_st=True,
            exclude_zero_volume=True,
        ),
    )

    last_date = pd.Timestamp("2024-02-02")
    selected = universe[universe["date"] == last_date]
    assert selected["symbol"].tolist() == ["000001"]


def test_build_monthly_universe_ranks_by_trailing_amount() -> None:
    universe = build_monthly_universe(
        _bars(),
        UniverseConfig(
            min_listed_days=20,
            liquidity_window=5,
            top_n=2,
            exclude_st=True,
            exclude_zero_volume=False,
        ),
    )

    selected = universe[universe["date"] == pd.Timestamp("2024-02-02")]
    assert selected["symbol"].tolist() == ["000002", "000001"]
    assert selected["rank"].tolist() == [1, 2]
