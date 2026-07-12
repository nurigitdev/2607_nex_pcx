from decimal import Decimal

from app.core.dashboard_metrics import _float_or_none


def test_float_or_none_handles_null_decimal_and_numeric_values() -> None:
    assert _float_or_none(None) is None
    assert _float_or_none(Decimal("1.25")) == 1.25
    assert _float_or_none(2) == 2.0
