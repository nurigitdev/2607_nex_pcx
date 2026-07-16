"""Embedding job repository helpers backed by PostgreSQL."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from psycopg import Connection
from psycopg.types.json import Json

from app.core.chunk_policies import _validate_policy_name
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


@dataclass(frozen=True)
class EmbeddingJobBacklogProfileSummary:
    profile_name: str
    total_count: int
    pending_count: int
    running_count: int
    stale_running_count: int
    reclaimable_stale_running_count: int
    failed_count: int
    retryable_failed_count: int
    exhausted_failed_count: int
    succeeded_count: int
    skipped_count: int
    oldest_pending_at: datetime | None
    oldest_stale_lease_expires_at: datetime | None

    @property
    def claimable_count(self) -> int:
        return self.pending_count + self.reclaimable_stale_running_count

    @property
    def attention_count(self) -> int:
        return (
            self.retryable_failed_count
            + self.exhausted_failed_count
            + self.stale_running_count
        )


@dataclass(frozen=True)
class EmbeddingJobBacklogSummary:
    profile_summaries: tuple[EmbeddingJobBacklogProfileSummary, ...]
    total_count: int
    pending_count: int
    running_count: int
    stale_running_count: int
    reclaimable_stale_running_count: int
    failed_count: int
    retryable_failed_count: int
    exhausted_failed_count: int
    succeeded_count: int
    skipped_count: int

    @property
    def claimable_count(self) -> int:
        return self.pending_count + self.reclaimable_stale_running_count

    @property
    def attention_count(self) -> int:
        return (
            self.retryable_failed_count
            + self.exhausted_failed_count
            + self.stale_running_count
        )


@dataclass(frozen=True)
class MissingEmbeddingJobReconcileResult:
    document_id: int
    chunk_policy_name: str
    profile_name: str
    chunk_count: int
    existing_job_count: int
    missing_job_count: int
    created_job_count: int
    created_jobs: tuple[EmbeddingJobRecord, ...]


@dataclass(frozen=True)
class FailedEmbeddingJobRetryResult:
    document_id: int
    chunk_policy_name: str
    profile_name: str
    failed_job_count: int
    retryable_failed_job_count: int
    retried_job_count: int
    retried_jobs: tuple[EmbeddingJobRecord, ...]


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


def _row_to_embedding_job_backlog_profile_summary(
    row: dict[str, Any],
) -> EmbeddingJobBacklogProfileSummary:
    return EmbeddingJobBacklogProfileSummary(
        profile_name=str(row["profile_name"]),
        total_count=int(row["total_count"]),
        pending_count=int(row["pending_count"]),
        running_count=int(row["running_count"]),
        stale_running_count=int(row["stale_running_count"]),
        reclaimable_stale_running_count=int(row["reclaimable_stale_running_count"]),
        failed_count=int(row["failed_count"]),
        retryable_failed_count=int(row["retryable_failed_count"]),
        exhausted_failed_count=int(row["exhausted_failed_count"]),
        succeeded_count=int(row["succeeded_count"]),
        skipped_count=int(row["skipped_count"]),
        oldest_pending_at=row["oldest_pending_at"],
        oldest_stale_lease_expires_at=row["oldest_stale_lease_expires_at"],
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


def list_stale_embedding_jobs(
    database_url: str,
    *,
    profile_name: str | None = None,
    reclaimable_only: bool = False,
    limit: int = 100,
) -> list[EmbeddingJobRecord]:
    profile = _validate_profile_name(profile_name)
    _validate_limit(limit)

    filters = [
        "status = 'running'",
        "lease_expires_at IS NOT NULL",
        "lease_expires_at < now()",
    ]
    params: list[object] = []
    if profile is not None:
        filters.append("profile_name = %s")
        params.append(profile)
    if reclaimable_only:
        filters.append("attempts < max_attempts")
    params.append(limit)

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT {_select_embedding_job_columns()}
                FROM embedding_jobs
                WHERE {' AND '.join(filters)}
                ORDER BY lease_expires_at ASC, created_at ASC, job_id ASC
                LIMIT %s
                """,
                tuple(params),
            )
            rows = cursor.fetchall()
    return [_row_to_embedding_job_record(dict(row)) for row in rows]


def get_embedding_job_backlog_summary(database_url: str) -> EmbeddingJobBacklogSummary:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    profile_name,
                    COUNT(*)::int AS total_count,
                    COUNT(*) FILTER (WHERE status = 'pending')::int AS pending_count,
                    COUNT(*) FILTER (WHERE status = 'running')::int AS running_count,
                    COUNT(*) FILTER (
                        WHERE status = 'running'
                          AND lease_expires_at < now()
                    )::int AS stale_running_count,
                    COUNT(*) FILTER (
                        WHERE status = 'running'
                          AND lease_expires_at < now()
                          AND attempts < max_attempts
                    )::int AS reclaimable_stale_running_count,
                    COUNT(*) FILTER (WHERE status = 'failed')::int AS failed_count,
                    COUNT(*) FILTER (
                        WHERE status = 'failed'
                          AND attempts < max_attempts
                    )::int AS retryable_failed_count,
                    COUNT(*) FILTER (
                        WHERE status = 'failed'
                          AND attempts >= max_attempts
                    )::int AS exhausted_failed_count,
                    COUNT(*) FILTER (WHERE status = 'succeeded')::int AS succeeded_count,
                    COUNT(*) FILTER (WHERE status = 'skipped')::int AS skipped_count,
                    MIN(created_at) FILTER (WHERE status = 'pending') AS oldest_pending_at,
                    MIN(lease_expires_at) FILTER (
                        WHERE status = 'running'
                          AND lease_expires_at < now()
                    ) AS oldest_stale_lease_expires_at
                FROM embedding_jobs
                GROUP BY profile_name
                ORDER BY profile_name ASC
                """
            )
            rows = cursor.fetchall()

    profile_summaries = tuple(
        _row_to_embedding_job_backlog_profile_summary(dict(row)) for row in rows
    )
    return EmbeddingJobBacklogSummary(
        profile_summaries=profile_summaries,
        total_count=sum(summary.total_count for summary in profile_summaries),
        pending_count=sum(summary.pending_count for summary in profile_summaries),
        running_count=sum(summary.running_count for summary in profile_summaries),
        stale_running_count=sum(summary.stale_running_count for summary in profile_summaries),
        reclaimable_stale_running_count=sum(
            summary.reclaimable_stale_running_count for summary in profile_summaries
        ),
        failed_count=sum(summary.failed_count for summary in profile_summaries),
        retryable_failed_count=sum(
            summary.retryable_failed_count for summary in profile_summaries
        ),
        exhausted_failed_count=sum(
            summary.exhausted_failed_count for summary in profile_summaries
        ),
        succeeded_count=sum(summary.succeeded_count for summary in profile_summaries),
        skipped_count=sum(summary.skipped_count for summary in profile_summaries),
    )


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


def reconcile_missing_embedding_jobs_for_document_policy_profile_in_connection(
    connection: Connection,
    *,
    document_id: int,
    chunk_policy_name: str,
    profile_name: str,
    max_jobs: int = 500,
) -> MissingEmbeddingJobReconcileResult:
    _require_positive_id(document_id, "document_id")
    validated_policy_name = _validate_policy_name(chunk_policy_name)
    validated_profile_name = _validate_profile_name(profile_name)
    if validated_profile_name is None:
        raise InvalidEmbeddingJobError("profile_name is required")
    _validate_limit(max_jobs)

    with connection.cursor() as cursor:
        cursor.execute("SELECT 1 FROM documents WHERE document_id = %s", (document_id,))
        if cursor.fetchone() is None:
            raise InvalidEmbeddingJobError("document_id was not found")

        cursor.execute(
            "SELECT 1 FROM chunk_policies WHERE chunk_policy_name = %s",
            (validated_policy_name,),
        )
        if cursor.fetchone() is None:
            raise InvalidEmbeddingJobError("chunk_policy_name was not found")

        cursor.execute(
            """
            SELECT 1
            FROM embedding_profiles
            WHERE profile_name = %s
              AND is_active
            """,
            (validated_profile_name,),
        )
        if cursor.fetchone() is None:
            raise InvalidEmbeddingJobError("active profile_name was not found")

        cursor.execute(
            """
            SELECT
                count(c.chunk_id)::int AS chunk_count,
                count(ej.job_id)::int AS existing_job_count,
                count(c.chunk_id) FILTER (WHERE ej.job_id IS NULL)::int AS missing_job_count
            FROM chunks c
            LEFT JOIN embedding_jobs ej
              ON ej.chunk_id = c.chunk_id
             AND ej.profile_name = %s
            WHERE c.document_id = %s
              AND c.chunk_policy_name = %s
            """,
            (validated_profile_name, document_id, validated_policy_name),
        )
        summary = dict(cursor.fetchone())

        cursor.execute(
            """
            SELECT c.chunk_id
            FROM chunks c
            LEFT JOIN embedding_jobs ej
              ON ej.chunk_id = c.chunk_id
             AND ej.profile_name = %s
            WHERE c.document_id = %s
              AND c.chunk_policy_name = %s
              AND ej.job_id IS NULL
            ORDER BY c.chunk_seq ASC, c.chunk_id ASC
            LIMIT %s
            """,
            (validated_profile_name, document_id, validated_policy_name, max_jobs),
        )
        missing_chunk_ids = [int(row["chunk_id"]) for row in cursor.fetchall()]

    created_results = [
        create_embedding_job_in_connection(
            connection,
            EmbeddingJobInput(
                chunk_id=chunk_id,
                profile_name=validated_profile_name,
                runtime_metadata={
                    "reconcile_source": "multi_policy_ingestion_coverage",
                    "chunk_policy_name": validated_policy_name,
                },
            ),
        )
        for chunk_id in missing_chunk_ids
    ]
    created_jobs = tuple(result.job for result in created_results if result.created)

    return MissingEmbeddingJobReconcileResult(
        document_id=document_id,
        chunk_policy_name=validated_policy_name,
        profile_name=validated_profile_name,
        chunk_count=int(summary["chunk_count"] or 0),
        existing_job_count=int(summary["existing_job_count"] or 0),
        missing_job_count=int(summary["missing_job_count"] or 0),
        created_job_count=len(created_jobs),
        created_jobs=created_jobs,
    )


def reconcile_missing_embedding_jobs_for_document_policy_profile(
    database_url: str,
    *,
    document_id: int,
    chunk_policy_name: str,
    profile_name: str,
    max_jobs: int = 500,
) -> MissingEmbeddingJobReconcileResult:
    with connect(database_url) as connection:
        return reconcile_missing_embedding_jobs_for_document_policy_profile_in_connection(
            connection,
            document_id=document_id,
            chunk_policy_name=chunk_policy_name,
            profile_name=profile_name,
            max_jobs=max_jobs,
        )


def retry_failed_embedding_jobs_for_document_policy_profile_in_connection(
    connection: Connection,
    *,
    document_id: int,
    chunk_policy_name: str,
    profile_name: str,
    max_jobs: int = 500,
) -> FailedEmbeddingJobRetryResult:
    _require_positive_id(document_id, "document_id")
    validated_policy_name = _validate_policy_name(chunk_policy_name)
    validated_profile_name = _validate_profile_name(profile_name)
    if validated_profile_name is None:
        raise InvalidEmbeddingJobError("profile_name is required")
    _validate_limit(max_jobs)

    with connection.cursor() as cursor:
        cursor.execute("SELECT 1 FROM documents WHERE document_id = %s", (document_id,))
        if cursor.fetchone() is None:
            raise InvalidEmbeddingJobError("document_id was not found")

        cursor.execute(
            "SELECT 1 FROM chunk_policies WHERE chunk_policy_name = %s",
            (validated_policy_name,),
        )
        if cursor.fetchone() is None:
            raise InvalidEmbeddingJobError("chunk_policy_name was not found")

        cursor.execute(
            """
            SELECT 1
            FROM embedding_profiles
            WHERE profile_name = %s
              AND is_active
            """,
            (validated_profile_name,),
        )
        if cursor.fetchone() is None:
            raise InvalidEmbeddingJobError("active profile_name was not found")

        cursor.execute(
            """
            SELECT
                count(ej.job_id) FILTER (WHERE ej.status = 'failed')::int
                    AS failed_job_count,
                count(ej.job_id) FILTER (
                    WHERE ej.status = 'failed' AND ej.attempts < ej.max_attempts
                )::int AS retryable_failed_job_count
            FROM chunks c
            JOIN embedding_jobs ej
              ON ej.chunk_id = c.chunk_id
             AND ej.profile_name = %s
            WHERE c.document_id = %s
              AND c.chunk_policy_name = %s
            """,
            (validated_profile_name, document_id, validated_policy_name),
        )
        summary = dict(cursor.fetchone())

        cursor.execute(
            """
            SELECT ej.job_id
            FROM chunks c
            JOIN embedding_jobs ej
              ON ej.chunk_id = c.chunk_id
             AND ej.profile_name = %s
            WHERE c.document_id = %s
              AND c.chunk_policy_name = %s
              AND ej.status = 'failed'
              AND ej.attempts < ej.max_attempts
            ORDER BY c.chunk_seq ASC, c.chunk_id ASC, ej.job_id ASC
            LIMIT %s
            """,
            (validated_profile_name, document_id, validated_policy_name, max_jobs),
        )
        retryable_job_ids = [int(row["job_id"]) for row in cursor.fetchall()]

    retried_jobs = tuple(
        job
        for job_id in retryable_job_ids
        if (job := retry_embedding_job_in_connection(connection, job_id)) is not None
    )

    return FailedEmbeddingJobRetryResult(
        document_id=document_id,
        chunk_policy_name=validated_policy_name,
        profile_name=validated_profile_name,
        failed_job_count=int(summary["failed_job_count"] or 0),
        retryable_failed_job_count=int(summary["retryable_failed_job_count"] or 0),
        retried_job_count=len(retried_jobs),
        retried_jobs=retried_jobs,
    )


def retry_failed_embedding_jobs_for_document_policy_profile(
    database_url: str,
    *,
    document_id: int,
    chunk_policy_name: str,
    profile_name: str,
    max_jobs: int = 500,
) -> FailedEmbeddingJobRetryResult:
    with connect(database_url) as connection:
        return retry_failed_embedding_jobs_for_document_policy_profile_in_connection(
            connection,
            document_id=document_id,
            chunk_policy_name=chunk_policy_name,
            profile_name=profile_name,
            max_jobs=max_jobs,
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
    runtime_metadata: dict[str, Any] | None = None,
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
                runtime_metadata = runtime_metadata || %s::jsonb,
                last_error_at = now(),
                finished_at = now(),
                updated_at = now()
            WHERE job_id = %s
            RETURNING {_select_embedding_job_columns()}
            """,
            (error_code.strip(), error_message.strip(), Json(runtime_metadata or {}), job_id),
        )
        row = cursor.fetchone()
    return _row_to_embedding_job_record(dict(row)) if row else None


def mark_embedding_job_failed(
    database_url: str,
    job_id: int,
    *,
    error_code: str,
    error_message: str,
    runtime_metadata: dict[str, Any] | None = None,
) -> EmbeddingJobRecord | None:
    with connect(database_url) as connection:
        return mark_embedding_job_failed_in_connection(
            connection,
            job_id,
            error_code=error_code,
            error_message=error_message,
            runtime_metadata=runtime_metadata,
        )


def defer_embedding_job_in_connection(
    connection: Connection,
    job_id: int,
    *,
    lease_owner: str,
    defer_seconds: int,
    error_code: str,
    error_message: str,
    runtime_metadata: dict[str, Any] | None = None,
) -> EmbeddingJobRecord | None:
    _require_positive_id(job_id, "job_id")
    worker = _validate_worker(lease_owner)
    _validate_lease_seconds(defer_seconds)
    if not error_code.strip():
        raise InvalidEmbeddingJobError("error_code is required")
    if not error_message.strip():
        raise InvalidEmbeddingJobError("error_message is required")

    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            UPDATE embedding_jobs
            SET status = 'running',
                lease_owner = %s,
                lease_expires_at = now() + (%s * interval '1 second'),
                error_code = %s,
                error_message = %s,
                last_error_at = now(),
                runtime_metadata = runtime_metadata || %s::jsonb,
                started_at = COALESCE(started_at, now()),
                finished_at = NULL,
                updated_at = now()
            WHERE job_id = %s
            RETURNING {_select_embedding_job_columns()}
            """,
            (
                worker,
                defer_seconds,
                error_code.strip(),
                error_message.strip(),
                Json(runtime_metadata or {}),
                job_id,
            ),
        )
        row = cursor.fetchone()
    return _row_to_embedding_job_record(dict(row)) if row else None


def defer_embedding_job(
    database_url: str,
    job_id: int,
    *,
    lease_owner: str,
    defer_seconds: int,
    error_code: str,
    error_message: str,
    runtime_metadata: dict[str, Any] | None = None,
) -> EmbeddingJobRecord | None:
    with connect(database_url) as connection:
        return defer_embedding_job_in_connection(
            connection,
            job_id,
            lease_owner=lease_owner,
            defer_seconds=defer_seconds,
            error_code=error_code,
            error_message=error_message,
            runtime_metadata=runtime_metadata,
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


def release_stale_embedding_job_lease_in_connection(
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
              AND status = 'running'
              AND lease_expires_at IS NOT NULL
              AND lease_expires_at < now()
              AND attempts < max_attempts
            RETURNING {_select_embedding_job_columns()}
            """,
            (job_id,),
        )
        row = cursor.fetchone()
    return _row_to_embedding_job_record(dict(row)) if row else None


def release_stale_embedding_job_lease(
    database_url: str,
    job_id: int,
) -> EmbeddingJobRecord | None:
    with connect(database_url) as connection:
        return release_stale_embedding_job_lease_in_connection(connection, job_id)
