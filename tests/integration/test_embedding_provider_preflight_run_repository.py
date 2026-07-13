from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from app.core.database import connect, fetch_one
from app.core.embedding_provider_preflight_runs import (
    EmbeddingProviderPreflightRunInput,
    list_embedding_provider_preflight_runs,
    record_embedding_provider_preflight_run,
)
from app.core.embedding_provider_preflight_schedules import (
    EmbeddingProviderPreflightScheduleInput,
    upsert_embedding_provider_preflight_schedule,
)

pytestmark = pytest.mark.integration


def _cleanup_preflight_runs(database_url: str, run_ids: list[int]) -> None:
    if not run_ids:
        return
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM embedding_provider_preflight_runs WHERE run_id = ANY(%s)",
                (run_ids,),
            )


def _cleanup_schedules(database_url: str, schedule_names: list[str]) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM embedding_provider_preflight_schedules
                WHERE schedule_name = ANY(%s)
                """,
                (schedule_names,),
            )


def test_embedding_provider_preflight_runs_table_exists(
    migrated_database_url: str,
) -> None:
    table_name = fetch_one(
        migrated_database_url,
        """
        SELECT to_regclass(
            'public.embedding_provider_preflight_runs'
        ) AS table_name
        """,
    )

    assert table_name["table_name"] == "embedding_provider_preflight_runs"


def test_preflight_run_repository_records_and_filters_history(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    schedule_name = f"hourly-preflight-{suffix}"
    completed_at = datetime(2026, 7, 11, 13, 0, tzinfo=UTC)
    run_ids: list[int] = []

    try:
        upsert_embedding_provider_preflight_schedule(
            migrated_database_url,
            EmbeddingProviderPreflightScheduleInput(
                schedule_name=schedule_name,
                profile_name="kure_v1_1024",
                interval_minutes=60,
                is_enabled=True,
            ),
        )
        scheduled_run = record_embedding_provider_preflight_run(
            migrated_database_url,
            EmbeddingProviderPreflightRunInput(
                schedule_name=schedule_name,
                trigger_source="scheduled_cli",
                profile_name="kure_v1_1024",
                active_only=True,
                status="failed",
                result={
                    "route_count": 2,
                    "passed_count": 1,
                    "failed_count": 1,
                    "sample_set": {
                        "sample_set_name": "default_route_contract",
                        "input_type": "document",
                        "sample_text_count": 1,
                    },
                },
                started_at=completed_at - timedelta(seconds=2),
                completed_at=completed_at,
                elapsed_ms=2000,
            ),
        )
        manual_run = record_embedding_provider_preflight_run(
            migrated_database_url,
            EmbeddingProviderPreflightRunInput(
                trigger_source="manual_api",
                profile_name=None,
                active_only=False,
                status="succeeded",
                result={"route_count": 0, "passed_count": 0, "failed_count": 0},
                completed_at=completed_at + timedelta(minutes=5),
            ),
        )
        run_ids.extend([scheduled_run.run_id, manual_run.run_id])

        latest_runs = list_embedding_provider_preflight_runs(
            migrated_database_url,
            limit=200,
        )
        failed_runs = list_embedding_provider_preflight_runs(
            migrated_database_url,
            status="failed",
            limit=10,
        )
        scheduled_runs = list_embedding_provider_preflight_runs(
            migrated_database_url,
            schedule_name=schedule_name,
            limit=10,
        )
        profile_runs = list_embedding_provider_preflight_runs(
            migrated_database_url,
            profile_name="kure_v1_1024",
            limit=10,
        )

        assert manual_run.run_id in [run.run_id for run in latest_runs]
        assert scheduled_run.run_id in [run.run_id for run in latest_runs]
        assert scheduled_run.route_count == 2
        assert scheduled_run.passed_count == 1
        assert scheduled_run.failed_count == 1
        assert scheduled_run.sample_set_name == "default_route_contract"
        assert scheduled_run.input_type == "document"
        assert scheduled_run.elapsed_ms == 2000
        assert [run.run_id for run in scheduled_runs] == [scheduled_run.run_id]
        profile_run_ids = [run.run_id for run in profile_runs]
        assert scheduled_run.run_id in profile_run_ids
        assert manual_run.run_id not in profile_run_ids
        assert scheduled_run.run_id in [run.run_id for run in failed_runs]
    finally:
        _cleanup_preflight_runs(migrated_database_url, run_ids)
        _cleanup_schedules(migrated_database_url, [schedule_name])
