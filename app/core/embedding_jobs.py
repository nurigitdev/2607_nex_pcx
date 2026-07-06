"""Embedding job repository helpers backed by PostgreSQL."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from psycopg import Connection
from psycopg.types.json import Json

from app.core.database import connect
from app.core.pipeline_jobs import DEFAULT_LEASE_SECONDS

EMBEDDING_JOB_STATUSES = {"pending", "running", "succeeded", "failed", "skipped"}


@dataclass(frozen=True)
class EmbeddingProfileRecord:
    profile_name: str
    model_name: str
    dimension: int
    storage_type: str
    max_sequence_length: int | None
    mvp_max_input_tokens: int | None
    normalize_embeddings: bool
    pooling_strategy: str | None
    query_instruction: str | None
    document_instruction: str | None
    dtype: str | None
    adapter_name: str | None
    is_active: bool


@dataclass(frozen=True)
class EmbeddingJobInput:
    chunk_id: int
    profile_name: str
    max_attempts: int = 3
    runtime_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class EmbeddingJobRecord:
    job_id: int
    chunk_id: int
    profile_name: str
    status: str
    attempts: int
    max_attempts: int
    lease_owner: str | None
    lease_expires_at: datetime | None
    error_code: str | None
    error_message: str | None
    last_error_at: datetime | None
    runtime_metadata: dict[str, Any]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    updated_at: datetime


@dataclass(frozen=True)
class CreateEmbeddingJobResult:
    job: EmbeddingJobRecord
    created: bool


class InvalidEmbeddingJobError(ValueError):
    """Raised when an embedding job operation is invalid before reaching the DB."""


def _require_positive_id(value: int, field_name: str) -> None:
    if value <= 0:
        raise InvalidEmbeddingJobError(f"{field_name} must be greater than 0")


def _validate_profile_name(profile_name: str | None) -> str | None:
    if profile_name is None:
        return None
    profile = profile_name.strip()
    if not profile:
        raise InvalidEmbeddingJobError("profile_name is required")
    return profile


def _validate_status(status: str | None) -> None:
    if status is not None and status not in EMBEDDING_JOB_STATUSES:
        raise InvalidEmbeddingJobError(f"Unsupported embedding job status: {status}")


def _validate_worker(worker_name: str) -> str:
    worker = worker_name.strip()
    if not worker:
        raise InvalidEmbeddingJobError("worker_name is required")
    return worker


def _validate_lease_seconds(lease_seconds: int) -> None:
    if lease_seconds <= 0:
        raise InvalidEmbeddingJobError("lease_seconds must be greater than 0")


def _validate_limit(limit: int) -> None:
    if limit <= 0:
        raise InvalidEmbeddingJobError("limit must be greater than 0")
    if limit > 500:
        raise InvalidEmbeddingJobError("limit must be less than or equal to 500")


def validate_embedding_job_input(job_input: EmbeddingJobInput) -> str:
    _require_positive_id(job_input.chunk_id, "chunk_id")
    profile_name = _validate_profile_name(job_input.profile_name)
    if profile_name is None:
        raise InvalidEmbeddingJobError("profile_name is required")
    if job_input.max_attempts <= 0:
        raise InvalidEmbeddingJobError("max_attempts must be greater than 0")
    return profile_name


def _row_to_embedding_profile_record(row: dict[str, Any]) -> EmbeddingProfileRecord:
    return EmbeddingProfileRecord(
        profile_name=str(row["profile_name"]),
        model_name=str(row["model_name"]),
        dimension=int(row["dimension"]),
        storage_type=str(row["storage_type"]),
        max_sequence_length=(
            int(row["max_sequence_length"]) if row.get("max_sequence_length") is not None else None
        ),
        mvp_max_input_tokens=(
            int(row["mvp_max_input_tokens"])
            if row.get("mvp_max_input_tokens") is not None
            else None
        ),
        normalize_embeddings=bool(row["normalize_embeddings"]),
        pooling_strategy=row["pooling_strategy"],
        query_instruction=row["query_instruction"],
        document_instruction=row["document_instruction"],
        dtype=row["dtype"],
        adapter_name=row["adapter_name"],
        is_active=bool(row["is_active"]),
    )


def _row_to_embedding_job_record(row: dict[str, Any]) -> EmbeddingJobRecord:
    return EmbeddingJobRecord(
        job_id=int(row["job_id"]),
        chunk_id=int(row["chunk_id"]),
        profile_name=str(row["profile_name"]),
        status=str(row["status"]),
        attempts=int(row["attempts"]),
        max_attempts=int(row["max_attempts"]),
        lease_owner=row["lease_owner"],
        lease_expires_at=row["lease_expires_at"],
        error_code=row["error_code"],
        error_message=row["error_message"],
        last_error_at=row["last_error_at"],
        runtime_metadata=dict(row["runtime_metadata"] or {}),
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        updated_at=row["updated_at"],
    )


def _select_embedding_job_columns(alias: str = "embedding_jobs") -> str:
    return f"""
        {alias}.job_id,
        {alias}.chunk_id,
        {alias}.profile_name,
        {alias}.status,
        {alias}.attempts,
        {alias}.max_attempts,
        {alias}.lease_owner,
        {alias}.lease_expires_at,
        {alias}.error_code,
        {alias}.error_message,
        {alias}.last_error_at,
        {alias}.runtime_metadata,
        {alias}.created_at,
        {alias}.started_at,
        {alias}.finished_at,
        {alias}.updated_at
    """


def list_active_embedding_profiles_in_connection(
    connection: Connection,
) -> list[EmbeddingProfileRecord]:
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT
                profile_name,
                model_name,
                dimension,
                storage_type,
                max_sequence_length,
                mvp_max_input_tokens,
                normalize_embeddings,
                pooling_strategy,
                query_instruction,
                document_instruction,
                dtype,
                adapter_name,
                is_active
            FROM embedding_profiles
            WHERE is_active
            ORDER BY profile_name ASC
            """)
        rows = cursor.fetchall()
    return [_row_to_embedding_profile_record(dict(row)) for row in rows]


def list_active_embedding_profiles(database_url: str) -> list[EmbeddingProfileRecord]:
    with connect(database_url) as connection:
        return list_active_embedding_profiles_in_connection(connection)


def get_embedding_job_in_connection(
    connection: Connection,
    job_id: int,
) -> EmbeddingJobRecord | None:
    _require_positive_id(job_id, "job_id")
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT {_select_embedding_job_columns()}
            FROM embedding_jobs
            WHERE job_id = %s
            """,
            (job_id,),
        )
        row = cursor.fetchone()
    return _row_to_embedding_job_record(dict(row)) if row else None


def get_embedding_job(database_url: str, job_id: int) -> EmbeddingJobRecord | None:
    with connect(database_url) as connection:
        return get_embedding_job_in_connection(connection, job_id)


def list_embedding_jobs(
    database_url: str,
    *,
    chunk_id: int | None = None,
    profile_name: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> list[EmbeddingJobRecord]:
    if chunk_id is not None:
        _require_positive_id(chunk_id, "chunk_id")
    profile = _validate_profile_name(profile_name)
    _validate_status(status)
    _validate_limit(limit)

    filters: list[str] = []
    params: list[object] = []
    if chunk_id is not None:
        filters.append("chunk_id = %s")
        params.append(chunk_id)
    if profile is not None:
        filters.append("profile_name = %s")
        params.append(profile)
    if status is not None:
        filters.append("status = %s")
        params.append(status)
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    params.append(limit)

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {_select_embedding_job_columns()}
                FROM embedding_jobs
                {where_clause}
                ORDER BY created_at DESC, job_id DESC
                LIMIT %s
                """,
                tuple(params),
            )
            rows = cursor.fetchall()
    return [_row_to_embedding_job_record(dict(row)) for row in rows]


def create_embedding_job_in_connection(
    connection: Connection,
    job_input: EmbeddingJobInput,
) -> CreateEmbeddingJobResult:
    profile_name = validate_embedding_job_input(job_input)
    runtime_metadata = job_input.runtime_metadata or {}
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO embedding_jobs (
                chunk_id,
                profile_name,
                max_attempts,
                runtime_metadata
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (chunk_id, profile_name) DO NOTHING
            RETURNING {_select_embedding_job_columns()}
            """,
            (
                job_input.chunk_id,
                profile_name,
                job_input.max_attempts,
                Json(runtime_metadata),
            ),
        )
        row = cursor.fetchone()
        if row is not None:
            return CreateEmbeddingJobResult(
                job=_row_to_embedding_job_record(dict(row)),
                created=True,
            )

        cursor.execute(
            f"""
            SELECT {_select_embedding_job_columns()}
            FROM embedding_jobs
            WHERE chunk_id = %s
              AND profile_name = %s
            """,
            (job_input.chunk_id, profile_name),
        )
        existing = cursor.fetchone()
        if existing is None:
            msg = "Duplicate embedding job detected, but existing job was not found"
            raise RuntimeError(msg)
        return CreateEmbeddingJobResult(
            job=_row_to_embedding_job_record(dict(existing)),
            created=False,
        )


def create_embedding_job(
    database_url: str,
    job_input: EmbeddingJobInput,
) -> CreateEmbeddingJobResult:
    with connect(database_url) as connection:
        return create_embedding_job_in_connection(connection, job_input)


def create_embedding_jobs_for_chunk_in_connection(
    connection: Connection,
    chunk_id: int,
    *,
    profile_names: list[str] | None = None,
) -> list[CreateEmbeddingJobResult]:
    _require_positive_id(chunk_id, "chunk_id")
    if profile_names is None:
        profiles = list_active_embedding_profiles_in_connection(connection)
        effective_profile_names = [profile.profile_name for profile in profiles]
    else:
        effective_profile_names = []
        for profile_name in profile_names:
            profile = _validate_profile_name(profile_name)
            if profile is None:
                raise InvalidEmbeddingJobError("profile_name is required")
            effective_profile_names.append(profile)

    return [
        create_embedding_job_in_connection(
            connection,
            EmbeddingJobInput(chunk_id=chunk_id, profile_name=profile_name),
        )
        for profile_name in effective_profile_names
    ]


def create_embedding_jobs_for_chunk(
    database_url: str,
    chunk_id: int,
    *,
    profile_names: list[str] | None = None,
) -> list[CreateEmbeddingJobResult]:
    with connect(database_url) as connection:
        return create_embedding_jobs_for_chunk_in_connection(
            connection,
            chunk_id,
            profile_names=profile_names,
        )


def claim_next_embedding_job_in_connection(
    connection: Connection,
    worker_name: str,
    *,
    profile_name: str | None = None,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> EmbeddingJobRecord | None:
    worker = _validate_worker(worker_name)
    profile = _validate_profile_name(profile_name)
    _validate_lease_seconds(lease_seconds)
    profile_filter = "" if profile is None else "AND profile_name = %s"
    params: list[object] = []
    if profile is not None:
        params.append(profile)

    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT job_id
            FROM embedding_jobs
            WHERE (
                    status = 'pending'
                    OR (
                        status = 'running'
                        AND lease_expires_at < now()
                        AND attempts < max_attempts
                    )
                )
                {profile_filter}
            ORDER BY
                CASE WHEN status = 'pending' THEN 0 ELSE 1 END,
                created_at ASC,
                job_id ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1
            """,
            tuple(params),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        cursor.execute(
            f"""
            UPDATE embedding_jobs
            SET status = 'running',
                lease_owner = %s,
                lease_expires_at = now() + (%s * interval '1 second'),
                attempts = attempts + 1,
                started_at = COALESCE(started_at, now()),
                updated_at = now()
            WHERE job_id = %s
            RETURNING {_select_embedding_job_columns()}
            """,
            (worker, lease_seconds, row["job_id"]),
        )
        return _row_to_embedding_job_record(dict(cursor.fetchone()))


def claim_next_embedding_job(
    database_url: str,
    worker_name: str,
    *,
    profile_name: str | None = None,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> EmbeddingJobRecord | None:
    with connect(database_url) as connection:
        return claim_next_embedding_job_in_connection(
            connection,
            worker_name,
            profile_name=profile_name,
            lease_seconds=lease_seconds,
        )


def heartbeat_embedding_job_in_connection(
    connection: Connection,
    job_id: int,
    worker_name: str,
    *,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> EmbeddingJobRecord | None:
    _require_positive_id(job_id, "job_id")
    worker = _validate_worker(worker_name)
    _validate_lease_seconds(lease_seconds)
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            UPDATE embedding_jobs
            SET lease_expires_at = now() + (%s * interval '1 second'),
                updated_at = now()
            WHERE job_id = %s
              AND status = 'running'
              AND lease_owner = %s
            RETURNING {_select_embedding_job_columns()}
            """,
            (lease_seconds, job_id, worker),
        )
        row = cursor.fetchone()
    return _row_to_embedding_job_record(dict(row)) if row else None


def heartbeat_embedding_job(
    database_url: str,
    job_id: int,
    worker_name: str,
    *,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> EmbeddingJobRecord | None:
    with connect(database_url) as connection:
        return heartbeat_embedding_job_in_connection(
            connection,
            job_id,
            worker_name,
            lease_seconds=lease_seconds,
        )


def mark_embedding_job_succeeded_in_connection(
    connection: Connection,
    job_id: int,
    *,
    runtime_metadata: dict[str, Any] | None = None,
) -> EmbeddingJobRecord | None:
    _require_positive_id(job_id, "job_id")
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            UPDATE embedding_jobs
            SET status = 'succeeded',
                lease_owner = NULL,
                lease_expires_at = NULL,
                error_code = NULL,
                error_message = NULL,
                last_error_at = NULL,
                runtime_metadata = runtime_metadata || %s::jsonb,
                finished_at = COALESCE(finished_at, now()),
                updated_at = now()
            WHERE job_id = %s
            RETURNING {_select_embedding_job_columns()}
            """,
            (Json(runtime_metadata or {}), job_id),
        )
        row = cursor.fetchone()
    return _row_to_embedding_job_record(dict(row)) if row else None


def mark_embedding_job_succeeded(
    database_url: str,
    job_id: int,
    *,
    runtime_metadata: dict[str, Any] | None = None,
) -> EmbeddingJobRecord | None:
    with connect(database_url) as connection:
        return mark_embedding_job_succeeded_in_connection(
            connection,
            job_id,
            runtime_metadata=runtime_metadata,
        )


def mark_embedding_job_failed_in_connection(
    connection: Connection,
    job_id: int,
    *,
    error_code: str,
    error_message: str,
) -> EmbeddingJobRecord | None:
    _require_positive_id(job_id, "job_id")
    if not error_code.strip():
        raise InvalidEmbeddingJobError("error_code is required")
    if not error_message.strip():
        raise InvalidEmbeddingJobError("error_message is required")

    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            UPDATE embedding_jobs
            SET status = 'failed',
                lease_owner = NULL,
                lease_expires_at = NULL,
                error_code = %s,
                error_message = %s,
                last_error_at = now(),
                finished_at = now(),
                updated_at = now()
            WHERE job_id = %s
            RETURNING {_select_embedding_job_columns()}
            """,
            (error_code.strip(), error_message.strip(), job_id),
        )
        row = cursor.fetchone()
    return _row_to_embedding_job_record(dict(row)) if row else None


def mark_embedding_job_failed(
    database_url: str,
    job_id: int,
    *,
    error_code: str,
    error_message: str,
) -> EmbeddingJobRecord | None:
    with connect(database_url) as connection:
        return mark_embedding_job_failed_in_connection(
            connection,
            job_id,
            error_code=error_code,
            error_message=error_message,
        )


def mark_embedding_job_skipped_in_connection(
    connection: Connection,
    job_id: int,
    *,
    reason: str | None = None,
) -> EmbeddingJobRecord | None:
    _require_positive_id(job_id, "job_id")
    if reason is not None and not reason.strip():
        raise InvalidEmbeddingJobError("reason must not be blank")

    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            UPDATE embedding_jobs
            SET status = 'skipped',
                lease_owner = NULL,
                lease_expires_at = NULL,
                error_code = NULL,
                error_message = %s,
                last_error_at = NULL,
                finished_at = COALESCE(finished_at, now()),
                updated_at = now()
            WHERE job_id = %s
            RETURNING {_select_embedding_job_columns()}
            """,
            (reason.strip() if reason is not None else None, job_id),
        )
        row = cursor.fetchone()
    return _row_to_embedding_job_record(dict(row)) if row else None


def mark_embedding_job_skipped(
    database_url: str,
    job_id: int,
    *,
    reason: str | None = None,
) -> EmbeddingJobRecord | None:
    with connect(database_url) as connection:
        return mark_embedding_job_skipped_in_connection(connection, job_id, reason=reason)


def retry_embedding_job_in_connection(
    connection: Connection,
    job_id: int,
) -> EmbeddingJobRecord | None:
    _require_positive_id(job_id, "job_id")
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            UPDATE embedding_jobs
            SET status = 'pending',
                lease_owner = NULL,
                lease_expires_at = NULL,
                error_code = NULL,
                error_message = NULL,
                last_error_at = NULL,
                finished_at = NULL,
                updated_at = now()
            WHERE job_id = %s
              AND status = 'failed'
              AND attempts < max_attempts
            RETURNING {_select_embedding_job_columns()}
            """,
            (job_id,),
        )
        row = cursor.fetchone()
    return _row_to_embedding_job_record(dict(row)) if row else None


def retry_embedding_job(database_url: str, job_id: int) -> EmbeddingJobRecord | None:
    with connect(database_url) as connection:
        return retry_embedding_job_in_connection(connection, job_id)
