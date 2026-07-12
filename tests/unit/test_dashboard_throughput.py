import pytest

from app.core.dashboard_throughput import (
    InvalidDashboardThroughputError,
    _optional_float,
    _rate_per_second,
    _rounded_float,
    _success_rate_percent,
    validate_lookback_hours,
)


def test_validate_lookback_hours_accepts_positive_window() -> None:
    assert validate_lookback_hours(24) == 24


@pytest.mark.parametrize("lookback_hours", [0, -1, 721])
def test_validate_lookback_hours_rejects_out_of_range_values(
    lookback_hours: int,
) -> None:
    with pytest.raises(InvalidDashboardThroughputError):
        validate_lookback_hours(lookback_hours)


def test_rate_per_second_handles_empty_or_zero_elapsed_values() -> None:
    assert _rate_per_second(0, 100) == 0
    assert _rate_per_second(10, 0) == 0
    assert _rate_per_second(3, 1500) == 2


def test_success_rate_percent_handles_empty_and_processed_counts() -> None:
    assert _success_rate_percent(0, 0) == 0
    assert _success_rate_percent(3, 4) == 75


def test_float_helpers_handle_numeric_and_null_values() -> None:
    assert _optional_float(2) == 2.0
    assert _rounded_float(None) is None
    assert _rounded_float(2.345, digits=2) == 2.35
