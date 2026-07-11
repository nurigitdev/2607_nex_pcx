from datetime import UTC, datetime

import pytest

from app.core.embedding_provider_preflight_schedules import (
    EmbeddingProviderPreflightScheduleInput,
    InvalidEmbeddingProviderPreflightScheduleError,
    validate_embedding_provider_preflight_schedule_input,
)


def test_validate_preflight_schedule_input_normalizes_values() -> None:
    next_run_at = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)

    validated = validate_embedding_provider_preflight_schedule_input(
        EmbeddingProviderPreflightScheduleInput(
            schedule_name=" provider-preflight ",
            description=" Hourly route check ",
            profile_name=" kure_v1_1024 ",
            active_only=False,
            interval_minutes=30,
            is_enabled=True,
            next_run_at=next_run_at,
        )
    )

    assert validated.schedule_name == "provider-preflight"
    assert validated.description == "Hourly route check"
    assert validated.profile_name == "kure_v1_1024"
    assert validated.active_only is False
    assert validated.interval_minutes == 30
    assert validated.is_enabled is True
    assert validated.next_run_at == next_run_at


@pytest.mark.parametrize(
    ("schedule_input", "message"),
    [
        (
            EmbeddingProviderPreflightScheduleInput(schedule_name=" "),
            "schedule_name",
        ),
        (
            EmbeddingProviderPreflightScheduleInput(
                schedule_name="bad-profile",
                profile_name=" ",
            ),
            "profile_name",
        ),
        (
            EmbeddingProviderPreflightScheduleInput(
                schedule_name="bad-interval",
                interval_minutes=0,
            ),
            "interval_minutes must be greater than 0",
        ),
        (
            EmbeddingProviderPreflightScheduleInput(
                schedule_name="too-wide",
                interval_minutes=10081,
            ),
            "interval_minutes must be less than or equal to 10080",
        ),
    ],
)
def test_validate_preflight_schedule_input_rejects_invalid_values(
    schedule_input: EmbeddingProviderPreflightScheduleInput,
    message: str,
) -> None:
    with pytest.raises(InvalidEmbeddingProviderPreflightScheduleError, match=message):
        validate_embedding_provider_preflight_schedule_input(schedule_input)
