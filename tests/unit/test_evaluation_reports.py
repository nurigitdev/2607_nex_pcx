import pytest

from app.core.evaluation_reports import (
    InvalidEvaluationReportError,
    _require_positive_id,
    _validate_limit,
)


def test_validate_limit_accepts_boundary_values() -> None:
    assert _validate_limit(1) == 1
    assert _validate_limit(100) == 100


@pytest.mark.parametrize("value", [None, 0, -1])
def test_require_positive_id_rejects_missing_or_non_positive_values(value: int | None) -> None:
    with pytest.raises(InvalidEvaluationReportError, match="question_set_id"):
        _require_positive_id(value, "question_set_id")


@pytest.mark.parametrize(
    ("limit", "message"),
    [
        (0, "greater than 0"),
        (101, "less than or equal to 100"),
    ],
)
def test_validate_limit_rejects_out_of_range_values(limit: int, message: str) -> None:
    with pytest.raises(InvalidEvaluationReportError, match=message):
        _validate_limit(limit)
