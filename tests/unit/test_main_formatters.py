from decimal import Decimal

from app.main import _percent_label, _percent_value


def test_percent_value_formats_decimal_and_numeric_values_to_two_places() -> None:
    assert _percent_value(Decimal("0E-20")) == "0.00"
    assert _percent_value(Decimal("33.33333333333333333333")) == "33.33"
    assert _percent_value(100) == "100.00"


def test_percent_label_appends_unit_and_handles_missing_values() -> None:
    assert _percent_label(Decimal("50")) == "50.00%"
    assert _percent_label(None) == "-"
