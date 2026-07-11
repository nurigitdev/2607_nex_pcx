from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from app.core.database import connect, fetch_one
from app.core.embedding_provider_preflight_schedules import (
    DEFAULT_PROVIDER_PREFLIGHT_SCHEDULE_NAME,
    EmbeddingProviderPreflightScheduleInput,
    InvalidEmbeddingProviderPreflightScheduleError,
    get_embedding_provider_preflight_schedule,
    list_due_embedding_provider_preflight_schedules,
    list_embedding_provider_preflight_schedules,
    record_embedding_provider_preflight_schedule_run,
    run_due_embedding_provider_preflight_schedules,
    upsert_embedding_provider_preflight_schedule,
)

pytestmark = pytest.mark.integration


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


def test_embedding_provider_preflight_schedule_table_seeded(
    migrated_database_url: str,
) -> None:
    table_name = fetch_one(
        migrated_database_url,
        """
        SELECT to_regclass(
            'public.embedding_provider_preflight_schedules'
        ) AS table_name
        """,
    )
    default_schedule = get_embedding_provider_preflight_schedule(
        migrated_database_url,
        DEFAULT_PROVIDER_PREFLIGHT_SCHEDULE_NAME,
    )

    assert table_name["table_name"] == "embedding_provider_preflight_schedules"
    assert default_schedule is not None
    assert default_schedule.interval_minutes == 60
    assert default_schedule.is_enabled is False
    assert default_schedule.last_status == "never_run"


def test_embedding_provider_preflight_schedule_repository_and_runner(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    due_name = f"due-preflight-{suffix}"
    future_name = f"future-preflight-{suffix}"
    run_at = datetime(2026, 7, 11, 13, 0, tzinfo=UTC)
    calls = []

    def fake_preflight_runner(database_url: str, **kwargs):
        calls.append((database_url, kwargs))
        return {
            "route_count": 1,
            "passed_count": 1,
            "failed_count": 0,
            "profile_name": kwargs["profile_name"],
            "active_only": kwargs["active_only"],
            "results": [{"route_id": 7, "contract_passed": True}],
        }

    try:
        due_schedule = upsert_embedding_provider_preflight_schedule(
            migrated_database_url,
            EmbeddingProviderPreflightScheduleInput(
                schedule_name=due_name,
                description="Due integration schedule",
                profile_name="kure_v1_1024",
                active_only=True,
                interval_minutes=15,
                is_enabled=True,
                next_run_at=run_at - timedelta(minutes=1),
            ),
        )
        future_schedule = upsert_embedding_provider_preflight_schedule(
            migrated_database_url,
            EmbeddingProviderPreflightScheduleInput(
                schedule_name=future_name,
                description="Future integration schedule",
                profile_name=None,
                active_only=False,
                interval_minutes=30,
                is_enabled=True,
                next_run_at=run_at + timedelta(minutes=30),
            ),
        )

        assert due_schedule.schedule_name == due_name
        assert due_schedule.profile_name == "kure_v1_1024"
        assert future_schedule.active_only is False
        assert any(
            schedule.schedule_name == due_name
            for schedule in list_embedding_provider_preflight_schedules(
                migrated_database_url,
                enabled_only=True,
            )
        )
        assert [
            schedule.schedule_name
            for schedule in list_due_embedding_provider_preflight_schedules(
                migrated_database_url,
                now=run_at,
            )
        ] == [due_name]

        runs = run_due_embedding_provider_preflight_schedules(
            migrated_database_url,
            now=run_at,
            preflight_runner=fake_preflight_runner,
        )

        assert calls == [
            (
                migrated_database_url,
                {"profile_name": "kure_v1_1024", "active_only": True},
            )
        ]
        assert len(runs) == 1
        assert runs[0].status == "succeeded"
        assert runs[0].updated_schedule.last_status == "succeeded"
        assert runs[0].updated_schedule.run_count == 1
        assert runs[0].updated_schedule.failure_count == 0
        assert runs[0].updated_schedule.next_run_at == run_at + timedelta(minutes=15)
        assert runs[0].updated_schedule.last_result["route_count"] == 1
        assert runs[0].run_record.schedule_name == due_name
        assert runs[0].run_record.trigger_source == "scheduled_cli"
        assert runs[0].run_record.status == "succeeded"
        assert runs[0].run_record.route_count == 1

        assert (
            list_due_embedding_provider_preflight_schedules(
                migrated_database_url,
                now=run_at,
            )
            == []
        )
    finally:
        _cleanup_schedules(migrated_database_url, [due_name, future_name])


def test_embedding_provider_preflight_schedule_records_errors(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    schedule_name = f"error-preflight-{suffix}"
    run_at = datetime(2026, 7, 11, 13, 30, tzinfo=UTC)

    def failing_preflight_runner(*args, **kwargs):
        raise RuntimeError("provider route preflight failed")

    try:
        upsert_embedding_provider_preflight_schedule(
            migrated_database_url,
            EmbeddingProviderPreflightScheduleInput(
                schedule_name=schedule_name,
                interval_minutes=10,
                is_enabled=True,
                next_run_at=run_at,
            ),
        )

        runs = run_due_embedding_provider_preflight_schedules(
            migrated_database_url,
            now=run_at,
            preflight_runner=failing_preflight_runner,
        )

        assert len(runs) == 1
        assert runs[0].status == "error"
        assert runs[0].updated_schedule.failure_count == 1
        assert (
            runs[0].updated_schedule.last_result["error_message"]
            == "provider route preflight failed"
        )
        assert runs[0].run_record.schedule_name == schedule_name
        assert runs[0].run_record.status == "error"
        assert runs[0].run_record.error_message == "provider route preflight failed"
    finally:
        _cleanup_schedules(migrated_database_url, [schedule_name])


def test_embedding_provider_preflight_schedule_validates_filters(
    migrated_database_url: str,
) -> None:
    with pytest.raises(InvalidEmbeddingProviderPreflightScheduleError, match="limit"):
        list_due_embedding_provider_preflight_schedules(migrated_database_url, limit=0)

    default_schedule = get_embedding_provider_preflight_schedule(
        migrated_database_url,
        DEFAULT_PROVIDER_PREFLIGHT_SCHEDULE_NAME,
    )
    assert default_schedule is not None

    with pytest.raises(
        InvalidEmbeddingProviderPreflightScheduleError,
        match="Unsupported status",
    ):
        record_embedding_provider_preflight_schedule_run(
            migrated_database_url,
            default_schedule,
            status="never_run",
            result={},
        )
