import pytest

from app.core.dashboard_failures import (
    InvalidDashboardFailureError,
    get_dashboard_failure_detail,
    get_dashboard_recent_failures,
)


def test_get_dashboard_recent_failures_validates_limit_before_db() -> None:
    with pytest.raises(InvalidDashboardFailureError, match="at least 1"):
        get_dashboard_recent_failures("postgresql://example/db", limit=0)

    with pytest.raises(InvalidDashboardFailureError, match="less than or equal"):
        get_dashboard_recent_failures("postgresql://example/db", limit=51)


def test_get_dashboard_failure_detail_validates_source_and_reference_before_db() -> None:
    with pytest.raises(InvalidDashboardFailureError, match="Unsupported failure source"):
        get_dashboard_failure_detail(
            "postgresql://example/db",
            source="unknown",
            reference_id="1",
        )

    with pytest.raises(InvalidDashboardFailureError, match="positive integer"):
        get_dashboard_failure_detail(
            "postgresql://example/db",
            source="pipeline",
            reference_id="abc",
        )

    with pytest.raises(InvalidDashboardFailureError, match="positive integer"):
        get_dashboard_failure_detail(
            "postgresql://example/db",
            source="pipeline",
            reference_id="0",
        )
