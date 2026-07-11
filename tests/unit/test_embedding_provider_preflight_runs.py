from datetime import UTC, datetime

import pytest

from app.core.embedding_provider_preflight_runs import (
    EmbeddingProviderPreflightRunInput,
    InvalidEmbeddingProviderPreflightRunError,
    list_embedding_provider_preflight_runs,
    validate_embedding_provider_preflight_run_input,
)


def test_validate_preflight_run_input_normalizes_values() -> None:
    started_at = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)

    validated = validate_embedding_provider_preflight_run_input(
        EmbeddingProviderPreflightRunInput(
            schedule_name=" hourly-kure ",
            trigger_source=" MANUAL_API ",
            profile_name=" kure_v1_1024 ",
            active_only=False,
            status=" SUCCEEDED ",
            result={"route_count": 1},
            elapsed_ms=25,
            started_at=started_at,
            completed_at=started_at,
        )
    )

    assert validated.schedule_name == "hourly-kure"
    assert validated.trigger_source == "manual_api"
    assert validated.profile_name == "kure_v1_1024"
    assert validated.status == "succeeded"
    assert validated.active_only is False
    assert validated.elapsed_ms == 25
    assert validated.started_at == started_at


@pytest.mark.parametrize(
    ("run_input", "message"),
    [
        (
            EmbeddingProviderPreflightRunInput(
                trigger_source="unknown",
                status="succeeded",
                result={},
            ),
            "Unsupported trigger_source",
        ),
        (
            EmbeddingProviderPreflightRunInput(
                trigger_source="manual_api",
                status="never_run",
                result={},
            ),
            "Unsupported status",
        ),
        (
            EmbeddingProviderPreflightRunInput(
                trigger_source="manual_api",
                status="succeeded",
                schedule_name=" ",
                result={},
            ),
            "schedule_name is required",
        ),
        (
            EmbeddingProviderPreflightRunInput(
                trigger_source="manual_api",
                status="succeeded",
                result={},
                elapsed_ms=-1,
            ),
            "elapsed_ms",
        ),
    ],
)
def test_validate_preflight_run_input_rejects_invalid_values(
    run_input: EmbeddingProviderPreflightRunInput,
    message: str,
) -> None:
    with pytest.raises(InvalidEmbeddingProviderPreflightRunError, match=message):
        validate_embedding_provider_preflight_run_input(run_input)


@pytest.mark.parametrize("limit", [0, 201])
def test_list_preflight_runs_rejects_invalid_limit(limit: int) -> None:
    with pytest.raises(InvalidEmbeddingProviderPreflightRunError, match="limit"):
        list_embedding_provider_preflight_runs("postgresql://example/db", limit=limit)
