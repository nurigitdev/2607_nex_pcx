from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect, fetch_one
from app.core.embedding_provider_preflight_schedules import (
    DEFAULT_PROVIDER_PREFLIGHT_SCHEDULE_NAME,
    EmbeddingProviderPreflightScheduleInput,
    InvalidEmbeddingProviderPreflightScheduleError,
    claim_due_embedding_provider_preflight_schedules,
    get_embedding_provider_preflight_schedule,
    list_due_embedding_provider_preflight_schedules,
    list_embedding_provider_preflight_schedules,
    record_embedding_provider_preflight_schedule_run,
    run_due_embedding_provider_preflight_schedules,
    upsert_embedding_provider_preflight_schedule,
)
from app.main import create_app

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


def _cleanup_preflight_runs(database_url: str, run_ids: list[int]) -> None:
    if not run_ids:
        return
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM embedding_provider_preflight_runs WHERE run_id = ANY(%s)",
                (run_ids,),
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


def test_embedding_provider_preflight_schedule_claim_skips_locked_rows(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    schedule_name = f"locked-preflight-{suffix}"
    run_at = datetime(2026, 7, 11, 13, 20, tzinfo=UTC)

    try:
        upsert_embedding_provider_preflight_schedule(
            migrated_database_url,
            EmbeddingProviderPreflightScheduleInput(
                schedule_name=schedule_name,
                interval_minutes=20,
                is_enabled=True,
                next_run_at=run_at,
            ),
        )

        with connect(migrated_database_url) as locked_connection:
            with locked_connection.cursor() as locked_cursor:
                locked_cursor.execute(
                    """
                    SELECT *
                    FROM embedding_provider_preflight_schedules
                    WHERE schedule_name = %s
                    FOR UPDATE
                    """,
                    (schedule_name,),
                )
                assert locked_cursor.fetchone() is not None

                skipped_claims = claim_due_embedding_provider_preflight_schedules(
                    migrated_database_url,
                    now=run_at,
                    schedule_name=schedule_name,
                )

        claimed = claim_due_embedding_provider_preflight_schedules(
            migrated_database_url,
            now=run_at,
            schedule_name=schedule_name,
        )

        assert skipped_claims == []
        assert len(claimed) == 1
        assert claimed[0].schedule_name == schedule_name
        assert claimed[0].next_run_at == run_at + timedelta(minutes=20)
        assert (
            list_due_embedding_provider_preflight_schedules(
                migrated_database_url,
                now=run_at,
                schedule_name=schedule_name,
            )
            == []
        )
    finally:
        _cleanup_schedules(migrated_database_url, [schedule_name])


def test_embedding_provider_preflight_schedule_runner_claim_blocks_duplicate_reentry(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    schedule_name = f"duplicate-preflight-{suffix}"
    run_at = datetime(2026, 7, 11, 13, 25, tzinfo=UTC)
    nested_runs = []
    run_ids: list[int] = []

    def nested_runner(database_url: str, **kwargs):
        nested_runs.extend(
            run_due_embedding_provider_preflight_schedules(
                database_url,
                now=run_at,
                schedule_name=schedule_name,
                preflight_runner=lambda *_args, **_kwargs: {
                    "route_count": 99,
                    "passed_count": 99,
                    "failed_count": 0,
                },
            )
        )
        return {
            "route_count": 1,
            "passed_count": 1,
            "failed_count": 0,
            "profile_name": kwargs["profile_name"],
            "active_only": kwargs["active_only"],
        }

    try:
        upsert_embedding_provider_preflight_schedule(
            migrated_database_url,
            EmbeddingProviderPreflightScheduleInput(
                schedule_name=schedule_name,
                profile_name="kure_v1_1024",
                interval_minutes=20,
                is_enabled=True,
                next_run_at=run_at,
            ),
        )

        runs = run_due_embedding_provider_preflight_schedules(
            migrated_database_url,
            now=run_at,
            schedule_name=schedule_name,
            preflight_runner=nested_runner,
        )

        assert len(runs) == 1
        run_ids.extend(run.run_record.run_id for run in runs)
        assert nested_runs == []
        assert runs[0].run_record.route_count == 1
        assert runs[0].updated_schedule.run_count == 1
        assert runs[0].updated_schedule.next_run_at == run_at + timedelta(minutes=20)
    finally:
        _cleanup_preflight_runs(migrated_database_url, run_ids)
        _cleanup_schedules(migrated_database_url, [schedule_name])


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


def test_embedding_provider_preflight_schedule_admin_api_round_trips(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    schedule_name = f"api-preflight-{suffix}"
    next_run_at = "2026-07-11T15:00:00+00:00"
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            upsert_response = client.put(
                f"/api/admin/embedding-provider-routes/preflight-schedules/{schedule_name}",
                json={
                    "description": " API preflight schedule ",
                    "profile_name": "kure_v1_1024",
                    "active_only": True,
                    "interval_minutes": 45,
                    "is_enabled": True,
                    "next_run_at": next_run_at,
                },
            )
            list_response = client.get(
                "/api/admin/embedding-provider-routes/preflight-schedules",
                params={"enabled_only": "true"},
            )
            get_response = client.get(
                f"/api/admin/embedding-provider-routes/preflight-schedules/{schedule_name}",
            )
            invalid_response = client.put(
                f"/api/admin/embedding-provider-routes/preflight-schedules/{schedule_name}",
                json={
                    "profile_name": " ",
                    "interval_minutes": 45,
                    "is_enabled": True,
                },
            )
            missing_response = client.get(
                "/api/admin/embedding-provider-routes/preflight-schedules/"
                f"missing-schedule-{suffix}",
            )

        assert upsert_response.status_code == 200
        schedule = upsert_response.json()["schedule"]
        assert schedule["schedule_name"] == schedule_name
        assert schedule["description"] == "API preflight schedule"
        assert schedule["profile_name"] == "kure_v1_1024"
        assert schedule["interval_minutes"] == 45
        assert schedule["is_enabled"] is True
        assert datetime.fromisoformat(schedule["next_run_at"]) == datetime.fromisoformat(
            next_run_at
        )
        assert list_response.status_code == 200
        assert any(
            item["schedule_name"] == schedule_name for item in list_response.json()["schedules"]
        )
        assert get_response.status_code == 200
        assert get_response.json()["schedule"]["schedule_name"] == schedule_name
        assert invalid_response.status_code == 400
        assert "profile_name is required" in invalid_response.json()["detail"]
        assert missing_response.status_code == 404
    finally:
        _cleanup_schedules(migrated_database_url, [schedule_name])


def test_embedding_provider_preflight_schedule_due_api_previews_and_runs(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    due_name = f"api-due-preflight-{suffix}"
    future_name = f"api-future-preflight-{suffix}"
    profile_name = f"empty-preflight-profile-{suffix}"
    app = create_app(Settings(database_url=migrated_database_url))
    run_ids: list[int] = []

    try:
        upsert_embedding_provider_preflight_schedule(
            migrated_database_url,
            EmbeddingProviderPreflightScheduleInput(
                schedule_name=due_name,
                profile_name=profile_name,
                interval_minutes=30,
                is_enabled=True,
                next_run_at=datetime(2020, 1, 1, tzinfo=UTC),
            ),
        )
        upsert_embedding_provider_preflight_schedule(
            migrated_database_url,
            EmbeddingProviderPreflightScheduleInput(
                schedule_name=future_name,
                profile_name=profile_name,
                interval_minutes=30,
                is_enabled=True,
                next_run_at=datetime(2099, 1, 1, tzinfo=UTC),
            ),
        )

        with TestClient(app) as client:
            due_response = client.get(
                "/api/admin/embedding-provider-routes/preflight-schedules/due",
                params={"limit": "10"},
            )
            run_response = client.post(
                "/api/admin/embedding-provider-routes/preflight-schedules/run-due",
                json={"schedule_name": due_name, "limit": 5},
            )
            due_after_response = client.get(
                "/api/admin/embedding-provider-routes/preflight-schedules/due",
                params={"schedule_name": due_name},
            )

        assert due_response.status_code == 200
        due_schedule_names = [
            schedule["schedule_name"] for schedule in due_response.json()["schedules"]
        ]
        assert due_name in due_schedule_names
        assert future_name not in due_schedule_names
        assert run_response.status_code == 200
        body = run_response.json()
        assert body["run_count"] == 1
        assert body["failed_count"] == 0
        run = body["runs"][0]
        run_ids.append(run["run_record"]["run_id"])
        assert run["status"] == "succeeded"
        assert run["schedule"]["schedule_name"] == due_name
        assert run["updated_schedule"]["run_count"] == 1
        assert run["updated_schedule"]["last_status"] == "succeeded"
        assert run["run_record"]["schedule_name"] == due_name
        assert run["run_record"]["trigger_source"] == "scheduled_cli"
        assert run["run_record"]["route_count"] == 0
        assert due_after_response.status_code == 200
        assert due_after_response.json()["schedule_count"] == 0
    finally:
        _cleanup_preflight_runs(migrated_database_url, run_ids)
        _cleanup_schedules(migrated_database_url, [due_name, future_name])
