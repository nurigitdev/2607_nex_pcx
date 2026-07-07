import pytest

from app.core.evaluation_dashboard import (
    InvalidEvaluationDashboardError,
    _status_counts_by_name,
    get_evaluation_dashboard_summary,
)


def test_evaluation_dashboard_validates_recent_limit() -> None:
    with pytest.raises(InvalidEvaluationDashboardError, match="recent_limit"):
        get_evaluation_dashboard_summary("postgresql://unused", recent_limit=0)

    with pytest.raises(InvalidEvaluationDashboardError, match="less than or equal"):
        get_evaluation_dashboard_summary("postgresql://unused", recent_limit=21)


def test_status_counts_fill_missing_statuses_and_ignore_unknown_rows() -> None:
    status_counts = _status_counts_by_name(
        [
            {"status": "succeeded", "run_count": 3},
            {"status": "failed", "run_count": 1},
            {"status": "archived", "run_count": 99},
        ],
    )

    assert {item.status: item.count for item in status_counts} == {
        "pending": 0,
        "running": 0,
        "succeeded": 3,
        "failed": 1,
    }
