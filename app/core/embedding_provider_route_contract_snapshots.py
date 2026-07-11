"""Persistence helpers for embedding provider route contract snapshots."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from psycopg import Connection
from psycopg.types.json import Json

from app.core.database import connect
from app.core.embedding_provider_route_contracts import EmbeddingProviderRouteContractResult


@dataclass(frozen=True)
class EmbeddingProviderRouteContractSnapshotRecord:
    snapshot_id: int
    route_id: int
    profile_name: str
    provider_name: str
    provider_mode: str
    passed: bool
    status: str
    elapsed_ms: int
    input_type: str
    sample_text_count: int
    expected_dimension: int | None
    provider_type: str | None
    provider_model_id: str | None
    model_key: str | None
    dimension: int | None
    input_count: int | None
    runtime_metadata: dict[str, Any]
    validation_errors: tuple[str, ...]
    error_message: str | None
    checked_at: datetime


class InvalidEmbeddingProviderRouteContractSnapshotError(ValueError):
    """Raised when route contract snapshot filters are invalid."""


def record_embedding_provider_route_contract_snapshot(
    database_url: str,
    contract: EmbeddingProviderRouteContractResult,
) -> EmbeddingProviderRouteContractSnapshotRecord:
    with connect(database_url) as connection:
        return record_embedding_provider_route_contract_snapshot_in_connection(
            connection,
            contract,
        )


def record_embedding_provider_route_contract_snapshot_in_connection(
    connection: Connection,
    contract: EmbeddingProviderRouteContractResult,
) -> EmbeddingProviderRouteContractSnapshotRecord:
    route = contract.route
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO embedding_provider_route_contract_snapshots (
                route_id,
                profile_name,
                provider_name,
                provider_mode,
                passed,
                status,
                elapsed_ms,
                input_type,
                sample_text_count,
                expected_dimension,
                provider_type,
                provider_model_id,
                model_key,
                dimension,
                input_count,
                runtime_metadata,
                validation_errors,
                error_message
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                route.route_id,
                route.profile_name,
                route.provider_name,
                route.provider_mode,
                contract.passed,
                contract.status,
                contract.elapsed_ms,
                contract.input_type,
                contract.sample_text_count,
                contract.expected_dimension,
                contract.provider_type,
                contract.provider_model_id,
                contract.model_key,
                contract.dimension,
                contract.input_count,
                Json(contract.runtime_metadata),
                Json(list(contract.validation_errors)),
                contract.error_message,
            ),
        )
        return _row_to_snapshot_record(dict(cursor.fetchone()))


def list_embedding_provider_route_contract_snapshots(
    database_url: str,
    *,
    profile_name: str | None = None,
    route_id: int | None = None,
    limit: int = 50,
) -> list[EmbeddingProviderRouteContractSnapshotRecord]:
    _validate_limit(limit)
    where_clauses = []
    params: list[object] = []
    if profile_name is not None:
        where_clauses.append("profile_name = %s")
        params.append(_validate_nonblank(profile_name, "profile_name"))
    if route_id is not None:
        if route_id <= 0:
            raise InvalidEmbeddingProviderRouteContractSnapshotError(
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
                FROM embedding_provider_route_contract_snapshots
                {where_sql}
                ORDER BY checked_at DESC, snapshot_id DESC
                LIMIT %s
                """,
                (*params, limit),
            )
            rows = cursor.fetchall()
    return [_row_to_snapshot_record(dict(row)) for row in rows]


def _validate_limit(limit: int) -> None:
    if limit <= 0:
        raise InvalidEmbeddingProviderRouteContractSnapshotError("limit must be greater than 0")
    if limit > 500:
        raise InvalidEmbeddingProviderRouteContractSnapshotError(
            "limit must be less than or equal to 500"
        )


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidEmbeddingProviderRouteContractSnapshotError(f"{field_name} is required")
    return normalized


def _row_to_snapshot_record(
    row: dict[str, Any],
) -> EmbeddingProviderRouteContractSnapshotRecord:
    return EmbeddingProviderRouteContractSnapshotRecord(
        snapshot_id=int(row["snapshot_id"]),
        route_id=int(row["route_id"]),
        profile_name=str(row["profile_name"]),
        provider_name=str(row["provider_name"]),
        provider_mode=str(row["provider_mode"]),
        passed=bool(row["passed"]),
        status=str(row["status"]),
        elapsed_ms=int(row["elapsed_ms"]),
        input_type=str(row["input_type"]),
        sample_text_count=int(row["sample_text_count"]),
        expected_dimension=(
            int(row["expected_dimension"]) if row["expected_dimension"] is not None else None
        ),
        provider_type=row["provider_type"],
        provider_model_id=row["provider_model_id"],
        model_key=row["model_key"],
        dimension=int(row["dimension"]) if row["dimension"] is not None else None,
        input_count=int(row["input_count"]) if row["input_count"] is not None else None,
        runtime_metadata=dict(row["runtime_metadata"] or {}),
        validation_errors=tuple(str(error) for error in row["validation_errors"]),
        error_message=row["error_message"],
        checked_at=row["checked_at"],
    )
