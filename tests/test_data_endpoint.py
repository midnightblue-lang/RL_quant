import pytest

from quant_rl_alpha.data.akshare_client import akshare_exchange_symbol, volume_unit_for_endpoint


def test_akshare_exchange_symbol() -> None:
    assert akshare_exchange_symbol("000001") == "sz000001"
    assert akshare_exchange_symbol("600000") == "sh600000"


def test_volume_unit_for_endpoint() -> None:
    assert volume_unit_for_endpoint("hist_em") == "lots"
    assert volume_unit_for_endpoint("daily_sina") == "shares"


def test_unknown_endpoint_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        volume_unit_for_endpoint("unknown")
