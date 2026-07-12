import pytest

from app.core.dashboard_failures import (
    InvalidDashboardFailureError,
    get_dashboard_recent_failures,
)


def test_get_dashboard_recent_failures_validates_limit_before_db() -> None:
    with pytest.raises(InvalidDashboardFailureError, match="at least 1"):
        get_dashboard_recent_failures("postgresql://example/db", limit=0)

    with pytest.raises(InvalidDashboardFailureError, match="less than or equal"):
        get_dashboard_recent_failures("postgresql://example/db", limit=51)
