"""Embedding provider route repository helpers."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from psycopg.types.json import Json

from app.core.database import connect
from app.core.embedding_provider_route_auth import (
    InvalidEmbeddingProviderRouteAuthError,
    normalize_embedding_provider_route_metadata,
)
from app.core.embedding_providers import EMBEDDING_PROVIDER_MODES


@dataclass(frozen=True)
class EmbeddingProviderRouteInput:
    profile_name: str
    provider_name: str
    provider_mode: str = "remote"
    provider_base_url: str | None = None
    timeout_seconds: float = 30.0
    priority: int = 100
    is_active: bool = True
    health_check_enabled: bool = True
    runtime_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class EmbeddingProviderRouteRecord:
    route_id: int
    profile_name: str
    provider_name: str
    provider_mode: str
    provider_base_url: str | None
    timeout_seconds: float
    priority: int
    is_active: bool
    health_check_enabled: bool
    runtime_metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class InvalidEmbeddingProviderRouteError(ValueError):
    """Raised when an embedding provider route is invalid."""


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidEmbeddingProviderRouteError(f"{field_name} is required")
    return normalized


def validate_embedding_provider_route_input(
    route_input: EmbeddingProviderRouteInput,
) -> EmbeddingProviderRouteInput:
    profile_name = _validate_nonblank(route_input.profile_name, "profile_name")
    provider_name = _validate_nonblank(route_input.provider_name, "provider_name")
    provider_mode = _validate_nonblank(route_input.provider_mode, "provider_mode").lower()
    if provider_mode not in EMBEDDING_PROVIDER_MODES:
        raise InvalidEmbeddingProviderRouteError(f"Unsupported provider_mode: {provider_mode}")
    provider_base_url = (
        route_input.provider_base_url.strip().rstrip("/") if route_input.provider_base_url else None
    )
    if provider_mode == "remote" and not provider_base_url:
        raise InvalidEmbeddingProviderRouteError(
            "provider_base_url is required for remote provider routes"
        )
    if route_input.timeout_seconds <= 0:
        raise InvalidEmbeddingProviderRouteError("timeout_seconds must be greater than 0")
    if route_input.priority < 0:
        raise InvalidEmbeddingProviderRouteError("priority must be greater than or equal to 0")

    try:
        runtime_metadata = normalize_embedding_provider_route_metadata(
            route_input.runtime_metadata or {}
        )
    except InvalidEmbeddingProviderRouteAuthError as exc:
        raise InvalidEmbeddingProviderRouteError(str(exc)) from exc

    return EmbeddingProviderRouteInput(
        profile_name=profile_name,
        provider_name=provider_name,
        provider_mode=provider_mode,
        provider_base_url=provider_base_url,
        timeout_seconds=route_input.timeout_seconds,
        priority=route_input.priority,
        is_active=route_input.is_active,
        health_check_enabled=route_input.health_check_enabled,
        runtime_metadata=runtime_metadata,
    )


def upsert_embedding_provider_route(
    database_url: str,
    route_input: EmbeddingProviderRouteInput,
) -> EmbeddingProviderRouteRecord:
    validated = validate_embedding_provider_route_input(route_input)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO embedding_provider_routes (
                    profile_name,
                    provider_name,
                    provider_mode,
                    provider_base_url,
                    timeout_seconds,
                    priority,
                    is_active,
                    health_check_enabled,
                    runtime_metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (profile_name, provider_name)
                DO UPDATE SET
                    provider_mode = EXCLUDED.provider_mode,
                    provider_base_url = EXCLUDED.provider_base_url,
                    timeout_seconds = EXCLUDED.timeout_seconds,
                    priority = EXCLUDED.priority,
                    is_active = EXCLUDED.is_active,
                    health_check_enabled = EXCLUDED.health_check_enabled,
                    runtime_metadata = EXCLUDED.runtime_metadata,
                    updated_at = now()
                RETURNING {_select_route_columns()}
                """,
                (
                    validated.profile_name,
                    validated.provider_name,
                    validated.provider_mode,
                    validated.provider_base_url,
                    validated.timeout_seconds,
                    validated.priority,
                    validated.is_active,
                    validated.health_check_enabled,
                    Json(validated.runtime_metadata or {}),
                ),
            )
            row = cursor.fetchone()
    return _row_to_route_record(dict(row))


def list_embedding_provider_routes(
    database_url: str,
    *,
    profile_name: str | None = None,
    active_only: bool = False,
) -> list[EmbeddingProviderRouteRecord]:
    where_clauses = []
    params: list[object] = []
    if profile_name is not None:
        where_clauses.append("profile_name = %s")
        params.append(_validate_nonblank(profile_name, "profile_name"))
    if active_only:
        where_clauses.append("is_active")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {_select_route_columns()}
                FROM embedding_provider_routes
                {where_sql}
                ORDER BY profile_name ASC, priority ASC, route_id ASC
                """,
                tuple(params),
            )
            rows = cursor.fetchall()
    return [_row_to_route_record(dict(row)) for row in rows]


def get_embedding_provider_route(
    database_url: str,
    route_id: int,
) -> EmbeddingProviderRouteRecord | None:
    if route_id <= 0:
        raise InvalidEmbeddingProviderRouteError("route_id must be greater than 0")
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {_select_route_columns()}
                FROM embedding_provider_routes
                WHERE route_id = %s
                """,
                (route_id,),
            )
            row = cursor.fetchone()
    return _row_to_route_record(dict(row)) if row else None


def select_embedding_provider_route(
    database_url: str,
    profile_name: str,
) -> EmbeddingProviderRouteRecord | None:
    profile = _validate_nonblank(profile_name, "profile_name")
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {_select_route_columns()}
                FROM embedding_provider_routes
                WHERE profile_name = %s
                  AND is_active
                ORDER BY priority ASC, route_id ASC
                LIMIT 1
                """,
                (profile,),
            )
            row = cursor.fetchone()
    return _row_to_route_record(dict(row)) if row else None


def _select_route_columns() -> str:
    return """
        route_id,
        profile_name,
        provider_name,
        provider_mode,
        provider_base_url,
        timeout_seconds,
        priority,
        is_active,
        health_check_enabled,
        runtime_metadata,
        created_at,
        updated_at
    """


def _row_to_route_record(row: dict[str, Any]) -> EmbeddingProviderRouteRecord:
    return EmbeddingProviderRouteRecord(
        route_id=int(row["route_id"]),
        profile_name=str(row["profile_name"]),
        provider_name=str(row["provider_name"]),
        provider_mode=str(row["provider_mode"]),
        provider_base_url=row["provider_base_url"],
        timeout_seconds=float(row["timeout_seconds"]),
        priority=int(row["priority"]),
        is_active=bool(row["is_active"]),
        health_check_enabled=bool(row["health_check_enabled"]),
        runtime_metadata=dict(row["runtime_metadata"] or {}),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
