"""Schedule settings and runner for provider route preflight checks."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from psycopg.types.json import Json

from app.core.database import connect
from app.core.embedding_provider_route_preflight import run_embedding_provider_route_preflight

PREFLIGHT_SCHEDULE_STATUSES = ("never_run", "succeeded", "failed", "error")
DEFAULT_PROVIDER_PREFLIGHT_SCHEDULE_NAME = "default_provider_route_preflight"
MAX_PREFLIGHT_SCHEDULE_LIMIT = 100


@dataclass(frozen=True)
class EmbeddingProviderPreflightScheduleInput:
    schedule_name: str
    description: str | None = None
    profile_name: str | None = None
    active_only: bool = True
    interval_minutes: int = 60
    is_enabled: bool = False
    next_run_at: datetime | None = None


@dataclass(frozen=True)
class EmbeddingProviderPreflightScheduleRecord:
    schedule_name: str
    description: str | None
    profile_name: str | None
    active_only: bool
    interval_minutes: int
    is_enabled: bool
    next_run_at: datetime | None
    last_run_at: datetime | None
    last_status: str
    last_result: dict[str, Any]
    run_count: int
    failure_count: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ScheduledProviderRoutePreflightRun:
    schedule: EmbeddingProviderPreflightScheduleRecord
    status: str
    result: dict[str, Any]
    updated_schedule: EmbeddingProviderPreflightScheduleRecord


class InvalidEmbeddingProviderPreflightScheduleError(ValueError):
    """Raised when provider route preflight schedule data is invalid."""


PreflightRunner = Callable[..., dict[str, object]]


def validate_embedding_provider_preflight_schedule_input(
    schedule_input: EmbeddingProviderPreflightScheduleInput,
) -> EmbeddingProviderPreflightScheduleInput:
    schedule_name = _validate_nonblank(schedule_input.schedule_name, "schedule_name")
    profile_name = (
        _validate_nonblank(schedule_input.profile_name, "profile_name")
        if schedule_input.profile_name is not None
        else None
    )
    if schedule_input.interval_minutes <= 0:
        raise InvalidEmbeddingProviderPreflightScheduleError(
            "interval_minutes must be greater than 0"
        )
    if schedule_input.interval_minutes > 10080:
        raise InvalidEmbeddingProviderPreflightScheduleError(
            "interval_minutes must be less than or equal to 10080"
        )

    return EmbeddingProviderPreflightScheduleInput(
        schedule_name=schedule_name,
        description=schedule_input.description.strip() if schedule_input.description else None,
        profile_name=profile_name,
        active_only=schedule_input.active_only,
        interval_minutes=schedule_input.interval_minutes,
        is_enabled=schedule_input.is_enabled,
        next_run_at=schedule_input.next_run_at,
    )


def upsert_embedding_provider_preflight_schedule(
    database_url: str,
    schedule_input: EmbeddingProviderPreflightScheduleInput,
) -> EmbeddingProviderPreflightScheduleRecord:
    validated = validate_embedding_provider_preflight_schedule_input(schedule_input)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO embedding_provider_preflight_schedules (
                    schedule_name,
                    description,
                    profile_name,
                    active_only,
                    interval_minutes,
                    is_enabled,
                    next_run_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (schedule_name)
                DO UPDATE SET
                    description = EXCLUDED.description,
                    profile_name = EXCLUDED.profile_name,
                    active_only = EXCLUDED.active_only,
                    interval_minutes = EXCLUDED.interval_minutes,
                    is_enabled = EXCLUDED.is_enabled,
                    next_run_at = EXCLUDED.next_run_at,
                    updated_at = now()
                RETURNING *
                """,
                (
                    validated.schedule_name,
                    validated.description,
                    validated.profile_name,
                    validated.active_only,
                    validated.interval_minutes,
                    validated.is_enabled,
                    validated.next_run_at,
                ),
            )
            return _row_to_schedule_record(dict(cursor.fetchone()))


def get_embedding_provider_preflight_schedule(
    database_url: str,
    schedule_name: str,
) -> EmbeddingProviderPreflightScheduleRecord | None:
    normalized_schedule_name = _validate_nonblank(schedule_name, "schedule_name")
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM embedding_provider_preflight_schedules
                WHERE schedule_name = %s
                """,
                (normalized_schedule_name,),
            )
            row = cursor.fetchone()
    return _row_to_schedule_record(dict(row)) if row is not None else None


def list_embedding_provider_preflight_schedules(
    database_url: str,
    *,
    enabled_only: bool = False,
) -> list[EmbeddingProviderPreflightScheduleRecord]:
    where_sql = "WHERE is_enabled" if enabled_only else ""
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"""
                SELECT *
                FROM embedding_provider_preflight_schedules
                {where_sql}
                ORDER BY schedule_name ASC
                """)
            rows = cursor.fetchall()
    return [_row_to_schedule_record(dict(row)) for row in rows]


def list_due_embedding_provider_preflight_schedules(
    database_url: str,
    *,
    now: datetime | None = None,
    limit: int = 20,
    schedule_name: str | None = None,
) -> list[EmbeddingProviderPreflightScheduleRecord]:
    _validate_limit(limit)
    run_at = now or datetime.now(UTC)
    where_clauses = ["is_enabled", "(next_run_at IS NULL OR next_run_at <= %s)"]
    params: list[object] = [run_at]
    if schedule_name is not None:
        where_clauses.append("schedule_name = %s")
        params.append(_validate_nonblank(schedule_name, "schedule_name"))

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM embedding_provider_preflight_schedules
                WHERE {' AND '.join(where_clauses)}
                ORDER BY next_run_at ASC NULLS FIRST, schedule_name ASC
                LIMIT %s
                """,
                (*params, limit),
            )
            rows = cursor.fetchall()
    return [_row_to_schedule_record(dict(row)) for row in rows]


def record_embedding_provider_preflight_schedule_run(
    database_url: str,
    schedule: EmbeddingProviderPreflightScheduleRecord,
    *,
    status: str,
    result: dict[str, Any],
    ran_at: datetime | None = None,
) -> EmbeddingProviderPreflightScheduleRecord:
    if status not in PREFLIGHT_SCHEDULE_STATUSES or status == "never_run":
        raise InvalidEmbeddingProviderPreflightScheduleError(f"Unsupported status: {status}")
    completed_at = ran_at or datetime.now(UTC)
    next_run_at = completed_at + timedelta(minutes=schedule.interval_minutes)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE embedding_provider_preflight_schedules
                SET last_run_at = %s,
                    next_run_at = %s,
                    last_status = %s,
                    last_result = %s,
                    run_count = run_count + 1,
                    failure_count = failure_count + %s,
                    updated_at = now()
                WHERE schedule_name = %s
                RETURNING *
                """,
                (
                    completed_at,
                    next_run_at,
                    status,
                    Json(result),
                    0 if status == "succeeded" else 1,
                    schedule.schedule_name,
                ),
            )
            row = cursor.fetchone()
    if row is None:
        raise InvalidEmbeddingProviderPreflightScheduleError(
            f"Schedule not found: {schedule.schedule_name}"
        )
    return _row_to_schedule_record(dict(row))


def run_due_embedding_provider_preflight_schedules(
    database_url: str,
    *,
    now: datetime | None = None,
    limit: int = 20,
    schedule_name: str | None = None,
    preflight_runner: PreflightRunner = run_embedding_provider_route_preflight,
) -> list[ScheduledProviderRoutePreflightRun]:
    run_at = now or datetime.now(UTC)
    schedules = list_due_embedding_provider_preflight_schedules(
        database_url,
        now=run_at,
        limit=limit,
        schedule_name=schedule_name,
    )
    runs = []
    for schedule in schedules:
        try:
            result = dict(
                preflight_runner(
                    database_url,
                    profile_name=schedule.profile_name,
                    active_only=schedule.active_only,
                )
            )
            status = "succeeded" if int(result.get("failed_count", 0)) == 0 else "failed"
        except Exception as exc:
            result = {
                "schedule_name": schedule.schedule_name,
                "profile_name": schedule.profile_name,
                "active_only": schedule.active_only,
                "error_message": str(exc),
            }
            status = "error"
        updated_schedule = record_embedding_provider_preflight_schedule_run(
            database_url,
            schedule,
            status=status,
            result=result,
            ran_at=run_at,
        )
        runs.append(
            ScheduledProviderRoutePreflightRun(
                schedule=schedule,
                status=status,
                result=result,
                updated_schedule=updated_schedule,
            )
        )
    return runs


def _validate_limit(limit: int) -> None:
    if limit <= 0:
        raise InvalidEmbeddingProviderPreflightScheduleError("limit must be greater than 0")
    if limit > MAX_PREFLIGHT_SCHEDULE_LIMIT:
        raise InvalidEmbeddingProviderPreflightScheduleError(
            f"limit must be less than or equal to {MAX_PREFLIGHT_SCHEDULE_LIMIT}"
        )


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidEmbeddingProviderPreflightScheduleError(f"{field_name} is required")
    return normalized


def _row_to_schedule_record(
    row: dict[str, Any],
) -> EmbeddingProviderPreflightScheduleRecord:
    return EmbeddingProviderPreflightScheduleRecord(
        schedule_name=str(row["schedule_name"]),
        description=row["description"],
        profile_name=row["profile_name"],
        active_only=bool(row["active_only"]),
        interval_minutes=int(row["interval_minutes"]),
        is_enabled=bool(row["is_enabled"]),
        next_run_at=row["next_run_at"],
        last_run_at=row["last_run_at"],
        last_status=str(row["last_status"]),
        last_result=dict(row["last_result"] or {}),
        run_count=int(row["run_count"]),
        failure_count=int(row["failure_count"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
