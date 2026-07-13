"""Persistence helpers for provider route preflight run history."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from psycopg.types.json import Json

from app.core.database import connect

PREFLIGHT_RUN_STATUSES = ("succeeded", "failed", "error")
PREFLIGHT_RUN_TRIGGER_SOURCES = ("manual_api", "scheduled_cli")
MAX_PREFLIGHT_RUN_LIMIT = 200


@dataclass(frozen=True)
class EmbeddingProviderPreflightRunInput:
    trigger_source: str
    status: str
    result: dict[str, Any]
    schedule_name: str | None = None
    profile_name: str | None = None
    active_only: bool = True
    elapsed_ms: int | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass(frozen=True)
class EmbeddingProviderPreflightRunRecord:
    run_id: int
    schedule_name: str | None
    trigger_source: str
    profile_name: str | None
    active_only: bool
    status: str
    route_count: int
    passed_count: int
    failed_count: int
    sample_set_name: str | None
    input_type: str | None
    sample_text_count: int
    elapsed_ms: int | None
    result: dict[str, Any]
    error_message: str | None
    started_at: datetime
    completed_at: datetime


class InvalidEmbeddingProviderPreflightRunError(ValueError):
    """Raised when provider route preflight run history data is invalid."""


def record_embedding_provider_preflight_run(
    database_url: str,
    run_input: EmbeddingProviderPreflightRunInput,
) -> EmbeddingProviderPreflightRunRecord:
    validated = validate_embedding_provider_preflight_run_input(run_input)
    route_count = int(validated.result.get("route_count", 0) or 0)
    passed_count = int(validated.result.get("passed_count", 0) or 0)
    failed_count = int(validated.result.get("failed_count", 0) or 0)
    sample_set = dict(validated.result.get("sample_set") or {})
    completed_at = validated.completed_at or datetime.now(UTC)
    started_at = validated.started_at or completed_at
    error_message = (
        str(validated.result["error_message"])
        if validated.result.get("error_message") is not None
        else None
    )

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO embedding_provider_preflight_runs (
                    schedule_name,
                    trigger_source,
                    profile_name,
                    active_only,
                    status,
                    route_count,
                    passed_count,
                    failed_count,
                    sample_set_name,
                    input_type,
                    sample_text_count,
                    elapsed_ms,
                    result,
                    error_message,
                    started_at,
                    completed_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    validated.schedule_name,
                    validated.trigger_source,
                    validated.profile_name,
                    validated.active_only,
                    validated.status,
                    route_count,
                    passed_count,
                    failed_count,
                    sample_set.get("sample_set_name"),
                    sample_set.get("input_type"),
                    int(sample_set.get("sample_text_count") or 0),
                    validated.elapsed_ms,
                    Json(validated.result),
                    error_message,
                    started_at,
                    completed_at,
                ),
            )
            return _row_to_preflight_run_record(dict(cursor.fetchone()))


def list_embedding_provider_preflight_runs(
    database_url: str,
    *,
    schedule_name: str | None = None,
    profile_name: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[EmbeddingProviderPreflightRunRecord]:
    _validate_limit(limit)
    where_clauses = []
    params: list[object] = []
    if schedule_name is not None:
        where_clauses.append("schedule_name = %s")
        params.append(_validate_nonblank(schedule_name, "schedule_name"))
    if profile_name is not None:
        where_clauses.append("profile_name = %s")
        params.append(_validate_nonblank(profile_name, "profile_name"))
    if status is not None:
        where_clauses.append("status = %s")
        params.append(_validate_status(status))

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM embedding_provider_preflight_runs
                {where_sql}
                ORDER BY completed_at DESC, run_id DESC
                LIMIT %s
                """,
                (*params, limit),
            )
            rows = cursor.fetchall()
    return [_row_to_preflight_run_record(dict(row)) for row in rows]


def get_embedding_provider_preflight_run(
    database_url: str,
    run_id: int,
) -> EmbeddingProviderPreflightRunRecord | None:
    if run_id <= 0:
        raise InvalidEmbeddingProviderPreflightRunError("run_id must be greater than 0")
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM embedding_provider_preflight_runs
                WHERE run_id = %s
                """,
                (run_id,),
            )
            row = cursor.fetchone()
    return _row_to_preflight_run_record(dict(row)) if row is not None else None


def validate_embedding_provider_preflight_run_input(
    run_input: EmbeddingProviderPreflightRunInput,
) -> EmbeddingProviderPreflightRunInput:
    trigger_source = _validate_trigger_source(run_input.trigger_source)
    status = _validate_status(run_input.status)
    schedule_name = (
        _validate_nonblank(run_input.schedule_name, "schedule_name")
        if run_input.schedule_name is not None
        else None
    )
    profile_name = (
        _validate_nonblank(run_input.profile_name, "profile_name")
        if run_input.profile_name is not None
        else None
    )
    if run_input.elapsed_ms is not None and run_input.elapsed_ms < 0:
        raise InvalidEmbeddingProviderPreflightRunError(
            "elapsed_ms must be greater than or equal to 0"
        )
    return EmbeddingProviderPreflightRunInput(
        trigger_source=trigger_source,
        status=status,
        result=dict(run_input.result),
        schedule_name=schedule_name,
        profile_name=profile_name,
        active_only=run_input.active_only,
        elapsed_ms=run_input.elapsed_ms,
        started_at=run_input.started_at,
        completed_at=run_input.completed_at,
    )


def _validate_limit(limit: int) -> None:
    if limit <= 0:
        raise InvalidEmbeddingProviderPreflightRunError("limit must be greater than 0")
    if limit > MAX_PREFLIGHT_RUN_LIMIT:
        raise InvalidEmbeddingProviderPreflightRunError(
            f"limit must be less than or equal to {MAX_PREFLIGHT_RUN_LIMIT}"
        )


def _validate_status(status: str) -> str:
    normalized = _validate_nonblank(status, "status").lower()
    if normalized not in PREFLIGHT_RUN_STATUSES:
        raise InvalidEmbeddingProviderPreflightRunError(f"Unsupported status: {normalized}")
    return normalized


def _validate_trigger_source(trigger_source: str) -> str:
    normalized = _validate_nonblank(trigger_source, "trigger_source").lower()
    if normalized not in PREFLIGHT_RUN_TRIGGER_SOURCES:
        raise InvalidEmbeddingProviderPreflightRunError(f"Unsupported trigger_source: {normalized}")
    return normalized


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidEmbeddingProviderPreflightRunError(f"{field_name} is required")
    return normalized


def _row_to_preflight_run_record(
    row: dict[str, Any],
) -> EmbeddingProviderPreflightRunRecord:
    return EmbeddingProviderPreflightRunRecord(
        run_id=int(row["run_id"]),
        schedule_name=row["schedule_name"],
        trigger_source=str(row["trigger_source"]),
        profile_name=row["profile_name"],
        active_only=bool(row["active_only"]),
        status=str(row["status"]),
        route_count=int(row["route_count"]),
        passed_count=int(row["passed_count"]),
        failed_count=int(row["failed_count"]),
        sample_set_name=row["sample_set_name"],
        input_type=row["input_type"],
        sample_text_count=int(row["sample_text_count"]),
        elapsed_ms=int(row["elapsed_ms"]) if row["elapsed_ms"] is not None else None,
        result=dict(row["result"] or {}),
        error_message=row["error_message"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )
