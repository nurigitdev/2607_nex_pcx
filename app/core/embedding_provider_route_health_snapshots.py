"""Persistence helpers for embedding provider route health snapshots."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from psycopg import Connection
from psycopg.types.json import Json

from app.core.database import connect
from app.core.embedding_provider_route_health import (
    EmbeddingProviderRouteHealthResult,
    EmbeddingProviderRouteHealthSummary,
)


@dataclass(frozen=True)
class EmbeddingProviderRouteHealthSnapshotRecord:
    snapshot_id: int
    route_id: int
    profile_name: str
    provider_name: str
    provider_mode: str
    checked: bool
    ready: bool
    status: str
    elapsed_ms: int | None
    provider_type: str | None
    provider_model_id: str | None
    model_key: str | None
    profile_names: tuple[str, ...]
    dimension: int | None
    device: str | None
    runtime_metadata: dict[str, Any]
    validation_errors: tuple[str, ...]
    error_message: str | None
    checked_at: datetime


class InvalidEmbeddingProviderRouteHealthSnapshotError(ValueError):
    """Raised when route health snapshot filters are invalid."""


def record_embedding_provider_route_health_snapshot(
    database_url: str,
    route_health: EmbeddingProviderRouteHealthResult,
) -> EmbeddingProviderRouteHealthSnapshotRecord:
    with connect(database_url) as connection:
        return record_embedding_provider_route_health_snapshot_in_connection(
            connection,
            route_health,
        )


def record_embedding_provider_route_health_summary(
    database_url: str,
    summary: EmbeddingProviderRouteHealthSummary,
) -> list[EmbeddingProviderRouteHealthSnapshotRecord]:
    with connect(database_url) as connection:
        return [
            record_embedding_provider_route_health_snapshot_in_connection(
                connection,
                route_health,
            )
            for route_health in summary.routes
        ]


def record_embedding_provider_route_health_snapshot_in_connection(
    connection: Connection,
    route_health: EmbeddingProviderRouteHealthResult,
) -> EmbeddingProviderRouteHealthSnapshotRecord:
    route = route_health.route
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO embedding_provider_route_health_snapshots (
                route_id,
                profile_name,
                provider_name,
                provider_mode,
                checked,
                ready,
                status,
                elapsed_ms,
                provider_type,
                provider_model_id,
                model_key,
                profile_names,
                dimension,
                device,
                runtime_metadata,
                validation_errors,
                error_message
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                route.route_id,
                route.profile_name,
                route.provider_name,
                route.provider_mode,
                route_health.checked,
                route_health.ready,
                route_health.status,
                route_health.elapsed_ms,
                route_health.provider_type,
                route_health.provider_model_id,
                route_health.model_key,
                Json(list(route_health.profile_names)),
                route_health.dimension,
                route_health.device,
                Json(route_health.runtime_metadata),
                Json(list(route_health.validation_errors)),
                route_health.error_message,
            ),
        )
        return _row_to_snapshot_record(dict(cursor.fetchone()))


def list_embedding_provider_route_health_snapshots(
    database_url: str,
    *,
    profile_name: str | None = None,
    route_id: int | None = None,
    limit: int = 50,
) -> list[EmbeddingProviderRouteHealthSnapshotRecord]:
    _validate_limit(limit)
    where_clauses = []
    params: list[object] = []
    if profile_name is not None:
        where_clauses.append("profile_name = %s")
        params.append(_validate_nonblank(profile_name, "profile_name"))
    if route_id is not None:
        if route_id <= 0:
            raise InvalidEmbeddingProviderRouteHealthSnapshotError(
                "route_id must be greater than 0"
            )
        where_clauses.append("route_id = %s")
        params.append(route_id)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM embedding_provider_route_health_snapshots
                {where_sql}
                ORDER BY checked_at DESC, snapshot_id DESC
                LIMIT %s
                """,
                (*params, limit),
            )
            rows = cursor.fetchall()
    return [_row_to_snapshot_record(dict(row)) for row in rows]


def list_latest_embedding_provider_route_health_snapshots(
    database_url: str,
    route_ids: Sequence[int],
) -> dict[int, EmbeddingProviderRouteHealthSnapshotRecord]:
    validated_route_ids = _validate_route_ids(route_ids)
    if not validated_route_ids:
        return {}

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT DISTINCT ON (route_id) *
                FROM embedding_provider_route_health_snapshots
                WHERE route_id = ANY(%s)
                ORDER BY route_id, checked_at DESC, snapshot_id DESC
                """,
                (list(validated_route_ids),),
            )
            rows = cursor.fetchall()

    snapshots = [_row_to_snapshot_record(dict(row)) for row in rows]
    return {snapshot.route_id: snapshot for snapshot in snapshots}


def _validate_limit(limit: int) -> None:
    if limit <= 0:
        raise InvalidEmbeddingProviderRouteHealthSnapshotError("limit must be greater than 0")
    if limit > 500:
        raise InvalidEmbeddingProviderRouteHealthSnapshotError(
            "limit must be less than or equal to 500"
        )


def _validate_route_ids(route_ids: Sequence[int]) -> tuple[int, ...]:
    validated = []
    for route_id in dict.fromkeys(route_ids):
        if route_id <= 0:
            raise InvalidEmbeddingProviderRouteHealthSnapshotError(
                "route_id must be greater than 0"
            )
        validated.append(route_id)
    return tuple(validated)


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidEmbeddingProviderRouteHealthSnapshotError(f"{field_name} is required")
    return normalized


def _row_to_snapshot_record(
    row: dict[str, Any],
) -> EmbeddingProviderRouteHealthSnapshotRecord:
    return EmbeddingProviderRouteHealthSnapshotRecord(
        snapshot_id=int(row["snapshot_id"]),
        route_id=int(row["route_id"]),
        profile_name=str(row["profile_name"]),
        provider_name=str(row["provider_name"]),
        provider_mode=str(row["provider_mode"]),
        checked=bool(row["checked"]),
        ready=bool(row["ready"]),
        status=str(row["status"]),
        elapsed_ms=int(row["elapsed_ms"]) if row["elapsed_ms"] is not None else None,
        provider_type=row["provider_type"],
        provider_model_id=row["provider_model_id"],
        model_key=row["model_key"],
        profile_names=tuple(str(profile) for profile in row["profile_names"]),
        dimension=int(row["dimension"]) if row["dimension"] is not None else None,
        device=row["device"],
        runtime_metadata=dict(row["runtime_metadata"] or {}),
        validation_errors=tuple(str(error) for error in row["validation_errors"]),
        error_message=row["error_message"],
        checked_at=row["checked_at"],
    )
