"""Persistence helpers for DGX ingestion benchmark results."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from psycopg import Connection
from psycopg.types.json import Json

from app.core.database import connect

MAX_DGX_INGESTION_BENCHMARK_RUN_LIMIT = 200


@dataclass(frozen=True)
class DgxIngestionBenchmarkJobInput:
    provider: str
    profile_name: str
    source_job_id: int | None
    source_chunk_id: int | None
    processed: bool
    job_status: str | None
    vector_table_name: str | None
    vector_dimension: int | None
    vector_storage_type: str | None
    provider_route_id: int | None
    provider_route_name: str | None
    provider_runtime_base_url: str | None
    provider_model_id: str | None
    provider_type: str | None
    provider_elapsed_ms: int | None
    worker_elapsed_ms: int | None
    readiness_status: str | None
    readiness_health_snapshot_id: int | None
    readiness_contract_snapshot_id: int | None
    message: str | None
    error: str | None
    passed: bool


@dataclass(frozen=True)
class DgxIngestionBenchmarkProfileInput:
    provider: str
    profile_name: str
    expected_job_count: int
    processed_count: int
    succeeded_count: int
    failed_count: int
    vector_count: int
    passed: bool
    vector_table_name: str | None = None
    vector_dimension: int | None = None
    vector_storage_type: str | None = None
    provider_route_id: int | None = None
    provider_route_name: str | None = None
    provider_runtime_base_url: str | None = None
    provider_model_id: str | None = None
    provider_type: str | None = None
    readiness_status: str | None = None
    readiness_health_snapshot_id: int | None = None
    readiness_contract_snapshot_id: int | None = None
    total_provider_elapsed_ms: int | None = None
    avg_provider_elapsed_ms: float | None = None
    max_provider_elapsed_ms: int | None = None
    total_worker_elapsed_ms: int | None = None
    avg_worker_elapsed_ms: float | None = None
    max_worker_elapsed_ms: int | None = None
    errors: tuple[str, ...] = ()
    jobs: tuple[DgxIngestionBenchmarkJobInput, ...] = ()


@dataclass(frozen=True)
class DgxIngestionBenchmarkInput:
    benchmark_run_key: str
    script_name: str
    provider_names: tuple[str, ...]
    profile_names: tuple[str, ...]
    chunk_count: int
    expected_job_count: int
    processed_count: int
    succeeded_count: int
    failed_count: int
    vector_count: int
    passed: bool
    preflight_before_worker: bool
    active_only_preflight: bool
    cleanup_attempted: bool
    cleanup_confirmed: bool
    total_elapsed_seconds: float
    total_provider_elapsed_ms: int | None = None
    total_worker_elapsed_ms: int | None = None
    fixture_file_id: int | None = None
    fixture_document_id: int | None = None
    fixture_chunk_ids: tuple[int, ...] = ()
    plan_payload: dict[str, Any] = field(default_factory=dict)
    fixture_payload: dict[str, Any] = field(default_factory=dict)
    report_payload: dict[str, Any] = field(default_factory=dict)
    created_by: str | None = None
    created_by_user_id: int | None = None
    profiles: tuple[DgxIngestionBenchmarkProfileInput, ...] = ()


@dataclass(frozen=True)
class DgxIngestionBenchmarkRunRecord:
    benchmark_run_id: int
    benchmark_run_key: str
    script_name: str
    provider_names: tuple[str, ...]
    profile_names: tuple[str, ...]
    chunk_count: int
    expected_job_count: int
    processed_count: int
    succeeded_count: int
    failed_count: int
    vector_count: int
    passed: bool
    preflight_before_worker: bool
    active_only_preflight: bool
    cleanup_attempted: bool
    cleanup_confirmed: bool
    total_elapsed_seconds: float
    total_provider_elapsed_ms: int | None
    total_worker_elapsed_ms: int | None
    fixture_file_id: int | None
    fixture_document_id: int | None
    fixture_chunk_ids: tuple[int, ...]
    plan_payload: dict[str, Any]
    fixture_payload: dict[str, Any]
    report_payload: dict[str, Any]
    created_by: str | None
    created_by_user_id: int | None
    created_at: datetime


@dataclass(frozen=True)
class DgxIngestionBenchmarkProfileRecord:
    benchmark_profile_id: int
    benchmark_run_id: int
    provider: str
    profile_name: str
    expected_job_count: int
    processed_count: int
    succeeded_count: int
    failed_count: int
    vector_count: int
    passed: bool
    vector_table_name: str | None
    vector_dimension: int | None
    vector_storage_type: str | None
    provider_route_id: int | None
    provider_route_name: str | None
    provider_runtime_base_url: str | None
    provider_model_id: str | None
    provider_type: str | None
    readiness_status: str | None
    readiness_health_snapshot_id: int | None
    readiness_contract_snapshot_id: int | None
    total_provider_elapsed_ms: int | None
    avg_provider_elapsed_ms: float | None
    max_provider_elapsed_ms: int | None
    total_worker_elapsed_ms: int | None
    avg_worker_elapsed_ms: float | None
    max_worker_elapsed_ms: int | None
    errors: tuple[str, ...]
    created_at: datetime


@dataclass(frozen=True)
class DgxIngestionBenchmarkJobRecord:
    benchmark_job_result_id: int
    benchmark_run_id: int
    benchmark_profile_id: int
    provider: str
    profile_name: str
    source_job_id: int | None
    source_chunk_id: int | None
    processed: bool
    job_status: str | None
    vector_table_name: str | None
    vector_dimension: int | None
    vector_storage_type: str | None
    provider_route_id: int | None
    provider_route_name: str | None
    provider_runtime_base_url: str | None
    provider_model_id: str | None
    provider_type: str | None
    provider_elapsed_ms: int | None
    worker_elapsed_ms: int | None
    readiness_status: str | None
    readiness_health_snapshot_id: int | None
    readiness_contract_snapshot_id: int | None
    message: str | None
    error: str | None
    passed: bool
    created_at: datetime


@dataclass(frozen=True)
class DgxIngestionBenchmarkDetail:
    run: DgxIngestionBenchmarkRunRecord
    profiles: tuple[DgxIngestionBenchmarkProfileRecord, ...]
    jobs: tuple[DgxIngestionBenchmarkJobRecord, ...]


class InvalidDgxIngestionBenchmarkError(ValueError):
    """Raised when DGX ingestion benchmark result data is invalid."""


def record_dgx_ingestion_benchmark(
    database_url: str,
    benchmark_input: DgxIngestionBenchmarkInput,
) -> DgxIngestionBenchmarkDetail:
    validated = validate_dgx_ingestion_benchmark_input(benchmark_input)
    with connect(database_url) as connection:
        return record_dgx_ingestion_benchmark_in_connection(connection, validated)


def record_dgx_ingestion_benchmark_in_connection(
    connection: Connection,
    benchmark_input: DgxIngestionBenchmarkInput,
) -> DgxIngestionBenchmarkDetail:
    validated = validate_dgx_ingestion_benchmark_input(benchmark_input)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO dgx_ingestion_benchmark_runs (
                benchmark_run_key,
                script_name,
                provider_names,
                profile_names,
                chunk_count,
                expected_job_count,
                processed_count,
                succeeded_count,
                failed_count,
                vector_count,
                passed,
                preflight_before_worker,
                active_only_preflight,
                cleanup_attempted,
                cleanup_confirmed,
                total_elapsed_seconds,
                total_provider_elapsed_ms,
                total_worker_elapsed_ms,
                fixture_file_id,
                fixture_document_id,
                fixture_chunk_ids,
                plan_payload,
                fixture_payload,
                report_payload,
                created_by,
                created_by_user_id
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING *
            """,
            (
                validated.benchmark_run_key,
                validated.script_name,
                Json(list(validated.provider_names)),
                Json(list(validated.profile_names)),
                validated.chunk_count,
                validated.expected_job_count,
                validated.processed_count,
                validated.succeeded_count,
                validated.failed_count,
                validated.vector_count,
                validated.passed,
                validated.preflight_before_worker,
                validated.active_only_preflight,
                validated.cleanup_attempted,
                validated.cleanup_confirmed,
                validated.total_elapsed_seconds,
                validated.total_provider_elapsed_ms,
                validated.total_worker_elapsed_ms,
                validated.fixture_file_id,
                validated.fixture_document_id,
                Json(list(validated.fixture_chunk_ids)),
                Json(validated.plan_payload),
                Json(validated.fixture_payload),
                Json(validated.report_payload),
                validated.created_by,
                validated.created_by_user_id,
            ),
        )
        run = _row_to_run_record(dict(cursor.fetchone()))
        profile_records: list[DgxIngestionBenchmarkProfileRecord] = []
        job_records: list[DgxIngestionBenchmarkJobRecord] = []
        for profile in validated.profiles:
            cursor.execute(
                """
                INSERT INTO dgx_ingestion_benchmark_profile_results (
                    benchmark_run_id,
                    provider,
                    profile_name,
                    expected_job_count,
                    processed_count,
                    succeeded_count,
                    failed_count,
                    vector_count,
                    passed,
                    vector_table_name,
                    vector_dimension,
                    vector_storage_type,
                    provider_route_id,
                    provider_route_name,
                    provider_runtime_base_url,
                    provider_model_id,
                    provider_type,
                    readiness_status,
                    readiness_health_snapshot_id,
                    readiness_contract_snapshot_id,
                    total_provider_elapsed_ms,
                    avg_provider_elapsed_ms,
                    max_provider_elapsed_ms,
                    total_worker_elapsed_ms,
                    avg_worker_elapsed_ms,
                    max_worker_elapsed_ms,
                    errors
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING *
                """,
                (
                    run.benchmark_run_id,
                    profile.provider,
                    profile.profile_name,
                    profile.expected_job_count,
                    profile.processed_count,
                    profile.succeeded_count,
                    profile.failed_count,
                    profile.vector_count,
                    profile.passed,
                    profile.vector_table_name,
                    profile.vector_dimension,
                    profile.vector_storage_type,
                    profile.provider_route_id,
                    profile.provider_route_name,
                    profile.provider_runtime_base_url,
                    profile.provider_model_id,
                    profile.provider_type,
                    profile.readiness_status,
                    profile.readiness_health_snapshot_id,
                    profile.readiness_contract_snapshot_id,
                    profile.total_provider_elapsed_ms,
                    profile.avg_provider_elapsed_ms,
                    profile.max_provider_elapsed_ms,
                    profile.total_worker_elapsed_ms,
                    profile.avg_worker_elapsed_ms,
                    profile.max_worker_elapsed_ms,
                    Json(list(profile.errors)),
                ),
            )
            profile_record = _row_to_profile_record(dict(cursor.fetchone()))
            profile_records.append(profile_record)
            for job in profile.jobs:
                cursor.execute(
                    """
                    INSERT INTO dgx_ingestion_benchmark_job_results (
                        benchmark_run_id,
                        benchmark_profile_id,
                        provider,
                        profile_name,
                        source_job_id,
                        source_chunk_id,
                        processed,
                        job_status,
                        vector_table_name,
                        vector_dimension,
                        vector_storage_type,
                        provider_route_id,
                        provider_route_name,
                        provider_runtime_base_url,
                        provider_model_id,
                        provider_type,
                        provider_elapsed_ms,
                        worker_elapsed_ms,
                        readiness_status,
                        readiness_health_snapshot_id,
                        readiness_contract_snapshot_id,
                        message,
                        error,
                        passed
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    RETURNING *
                    """,
                    (
                        run.benchmark_run_id,
                        profile_record.benchmark_profile_id,
                        job.provider,
                        job.profile_name,
                        job.source_job_id,
                        job.source_chunk_id,
                        job.processed,
                        job.job_status,
                        job.vector_table_name,
                        job.vector_dimension,
                        job.vector_storage_type,
                        job.provider_route_id,
                        job.provider_route_name,
                        job.provider_runtime_base_url,
                        job.provider_model_id,
                        job.provider_type,
                        job.provider_elapsed_ms,
                        job.worker_elapsed_ms,
                        job.readiness_status,
                        job.readiness_health_snapshot_id,
                        job.readiness_contract_snapshot_id,
                        job.message,
                        job.error,
                        job.passed,
                    ),
                )
                job_records.append(_row_to_job_record(dict(cursor.fetchone())))

    return DgxIngestionBenchmarkDetail(
        run=run,
        profiles=tuple(profile_records),
        jobs=tuple(job_records),
    )


def list_dgx_ingestion_benchmark_runs(
    database_url: str,
    *,
    provider: str | None = None,
    profile_name: str | None = None,
    passed: bool | None = None,
    limit: int = 20,
) -> list[DgxIngestionBenchmarkRunRecord]:
    validated_limit = _validate_limit(limit)
    where_clauses: list[str] = []
    params: list[object] = []
    if provider is not None:
        where_clauses.append("""
            EXISTS (
                SELECT 1
                FROM dgx_ingestion_benchmark_profile_results profile
                WHERE profile.benchmark_run_id = run.benchmark_run_id
                  AND profile.provider = %s
            )
            """)
        params.append(_validate_nonblank(provider, "provider"))
    if profile_name is not None:
        where_clauses.append("""
            EXISTS (
                SELECT 1
                FROM dgx_ingestion_benchmark_profile_results profile
                WHERE profile.benchmark_run_id = run.benchmark_run_id
                  AND profile.profile_name = %s
            )
            """)
        params.append(_validate_nonblank(profile_name, "profile_name"))
    if passed is not None:
        where_clauses.append("run.passed = %s")
        params.append(bool(passed))
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT run.*
                FROM dgx_ingestion_benchmark_runs run
                {where_sql}
                ORDER BY run.created_at DESC, run.benchmark_run_id DESC
                LIMIT %s
                """,
                (*params, validated_limit),
            )
            rows = cursor.fetchall()
    return [_row_to_run_record(dict(row)) for row in rows]


def get_dgx_ingestion_benchmark_detail(
    database_url: str,
    benchmark_run_id: int,
) -> DgxIngestionBenchmarkDetail | None:
    _validate_positive_int(benchmark_run_id, "benchmark_run_id")
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM dgx_ingestion_benchmark_runs
                WHERE benchmark_run_id = %s
                """,
                (benchmark_run_id,),
            )
            run_row = cursor.fetchone()
            if run_row is None:
                return None
            cursor.execute(
                """
                SELECT *
                FROM dgx_ingestion_benchmark_profile_results
                WHERE benchmark_run_id = %s
                ORDER BY benchmark_profile_id ASC
                """,
                (benchmark_run_id,),
            )
            profile_rows = cursor.fetchall()
            cursor.execute(
                """
                SELECT *
                FROM dgx_ingestion_benchmark_job_results
                WHERE benchmark_run_id = %s
                ORDER BY benchmark_job_result_id ASC
                """,
                (benchmark_run_id,),
            )
            job_rows = cursor.fetchall()

    return DgxIngestionBenchmarkDetail(
        run=_row_to_run_record(dict(run_row)),
        profiles=tuple(_row_to_profile_record(dict(row)) for row in profile_rows),
        jobs=tuple(_row_to_job_record(dict(row)) for row in job_rows),
    )


def validate_dgx_ingestion_benchmark_input(
    benchmark_input: DgxIngestionBenchmarkInput,
) -> DgxIngestionBenchmarkInput:
    benchmark_run_key = _validate_nonblank(
        benchmark_input.benchmark_run_key,
        "benchmark_run_key",
    )
    script_name = _validate_nonblank(benchmark_input.script_name, "script_name")
    provider_names = _validate_nonblank_tuple(
        benchmark_input.provider_names,
        "provider_names",
    )
    profile_names = _validate_nonblank_tuple(
        benchmark_input.profile_names,
        "profile_names",
    )
    chunk_count = _validate_positive_int(benchmark_input.chunk_count, "chunk_count")
    expected_job_count = _validate_nonnegative_int(
        benchmark_input.expected_job_count,
        "expected_job_count",
    )
    processed_count = _validate_nonnegative_int(
        benchmark_input.processed_count,
        "processed_count",
    )
    succeeded_count = _validate_nonnegative_int(
        benchmark_input.succeeded_count,
        "succeeded_count",
    )
    failed_count = _validate_nonnegative_int(
        benchmark_input.failed_count,
        "failed_count",
    )
    vector_count = _validate_nonnegative_int(
        benchmark_input.vector_count,
        "vector_count",
    )
    _validate_count_relationships(
        expected_job_count=expected_job_count,
        processed_count=processed_count,
        succeeded_count=succeeded_count,
        failed_count=failed_count,
        vector_count=vector_count,
        label="benchmark",
    )
    if benchmark_input.total_elapsed_seconds < 0:
        raise InvalidDgxIngestionBenchmarkError(
            "total_elapsed_seconds must be greater than or equal to 0"
        )
    total_provider_elapsed_ms = _validate_optional_nonnegative_int(
        benchmark_input.total_provider_elapsed_ms,
        "total_provider_elapsed_ms",
    )
    total_worker_elapsed_ms = _validate_optional_nonnegative_int(
        benchmark_input.total_worker_elapsed_ms,
        "total_worker_elapsed_ms",
    )
    fixture_file_id = _validate_optional_positive_int(
        benchmark_input.fixture_file_id,
        "fixture_file_id",
    )
    fixture_document_id = _validate_optional_positive_int(
        benchmark_input.fixture_document_id,
        "fixture_document_id",
    )
    fixture_chunk_ids = _validate_positive_int_tuple(
        benchmark_input.fixture_chunk_ids,
        "fixture_chunk_ids",
    )
    created_by = _validate_optional_nonblank(benchmark_input.created_by, "created_by")
    created_by_user_id = _validate_optional_positive_int(
        benchmark_input.created_by_user_id,
        "created_by_user_id",
    )
    profiles = tuple(_validate_profile_input(profile) for profile in benchmark_input.profiles)
    if sum(profile.expected_job_count for profile in profiles) != expected_job_count:
        raise InvalidDgxIngestionBenchmarkError(
            "profile expected_job_count sum must equal benchmark expected_job_count"
        )
    if sum(profile.processed_count for profile in profiles) != processed_count:
        raise InvalidDgxIngestionBenchmarkError(
            "profile processed_count sum must equal benchmark processed_count"
        )
    if sum(profile.succeeded_count for profile in profiles) != succeeded_count:
        raise InvalidDgxIngestionBenchmarkError(
            "profile succeeded_count sum must equal benchmark succeeded_count"
        )
    if sum(profile.failed_count for profile in profiles) != failed_count:
        raise InvalidDgxIngestionBenchmarkError(
            "profile failed_count sum must equal benchmark failed_count"
        )
    if sum(profile.vector_count for profile in profiles) != vector_count:
        raise InvalidDgxIngestionBenchmarkError(
            "profile vector_count sum must equal benchmark vector_count"
        )

    return DgxIngestionBenchmarkInput(
        benchmark_run_key=benchmark_run_key,
        script_name=script_name,
        provider_names=provider_names,
        profile_names=profile_names,
        chunk_count=chunk_count,
        expected_job_count=expected_job_count,
        processed_count=processed_count,
        succeeded_count=succeeded_count,
        failed_count=failed_count,
        vector_count=vector_count,
        passed=bool(benchmark_input.passed),
        preflight_before_worker=bool(benchmark_input.preflight_before_worker),
        active_only_preflight=bool(benchmark_input.active_only_preflight),
        cleanup_attempted=bool(benchmark_input.cleanup_attempted),
        cleanup_confirmed=bool(benchmark_input.cleanup_confirmed),
        total_elapsed_seconds=float(benchmark_input.total_elapsed_seconds),
        total_provider_elapsed_ms=total_provider_elapsed_ms,
        total_worker_elapsed_ms=total_worker_elapsed_ms,
        fixture_file_id=fixture_file_id,
        fixture_document_id=fixture_document_id,
        fixture_chunk_ids=fixture_chunk_ids,
        plan_payload=dict(benchmark_input.plan_payload),
        fixture_payload=dict(benchmark_input.fixture_payload),
        report_payload=dict(benchmark_input.report_payload),
        created_by=created_by,
        created_by_user_id=created_by_user_id,
        profiles=profiles,
    )


def _validate_profile_input(
    profile: DgxIngestionBenchmarkProfileInput,
) -> DgxIngestionBenchmarkProfileInput:
    provider = _validate_nonblank(profile.provider, "provider")
    profile_name = _validate_nonblank(profile.profile_name, "profile_name")
    expected_job_count = _validate_nonnegative_int(
        profile.expected_job_count,
        "profile.expected_job_count",
    )
    processed_count = _validate_nonnegative_int(
        profile.processed_count,
        "profile.processed_count",
    )
    succeeded_count = _validate_nonnegative_int(
        profile.succeeded_count,
        "profile.succeeded_count",
    )
    failed_count = _validate_nonnegative_int(
        profile.failed_count,
        "profile.failed_count",
    )
    vector_count = _validate_nonnegative_int(profile.vector_count, "profile.vector_count")
    _validate_count_relationships(
        expected_job_count=expected_job_count,
        processed_count=processed_count,
        succeeded_count=succeeded_count,
        failed_count=failed_count,
        vector_count=vector_count,
        label=f"profile {profile_name}",
    )
    jobs = tuple(_validate_job_input(job) for job in profile.jobs)
    if len(jobs) != processed_count:
        raise InvalidDgxIngestionBenchmarkError(
            f"profile {profile_name} job count must equal processed_count"
        )
    if any(job.provider != provider or job.profile_name != profile_name for job in jobs):
        raise InvalidDgxIngestionBenchmarkError(
            f"profile {profile_name} contains mismatched job provider/profile"
        )
    return DgxIngestionBenchmarkProfileInput(
        provider=provider,
        profile_name=profile_name,
        expected_job_count=expected_job_count,
        processed_count=processed_count,
        succeeded_count=succeeded_count,
        failed_count=failed_count,
        vector_count=vector_count,
        passed=bool(profile.passed),
        vector_table_name=_validate_optional_nonblank(
            profile.vector_table_name,
            "vector_table_name",
        ),
        vector_dimension=_validate_optional_positive_int(
            profile.vector_dimension,
            "vector_dimension",
        ),
        vector_storage_type=_validate_optional_nonblank(
            profile.vector_storage_type,
            "vector_storage_type",
        ),
        provider_route_id=_validate_optional_positive_int(
            profile.provider_route_id,
            "provider_route_id",
        ),
        provider_route_name=_validate_optional_nonblank(
            profile.provider_route_name,
            "provider_route_name",
        ),
        provider_runtime_base_url=_validate_optional_nonblank(
            profile.provider_runtime_base_url,
            "provider_runtime_base_url",
        ),
        provider_model_id=_validate_optional_nonblank(
            profile.provider_model_id,
            "provider_model_id",
        ),
        provider_type=_validate_optional_nonblank(profile.provider_type, "provider_type"),
        readiness_status=_validate_optional_nonblank(
            profile.readiness_status,
            "readiness_status",
        ),
        readiness_health_snapshot_id=_validate_optional_positive_int(
            profile.readiness_health_snapshot_id,
            "readiness_health_snapshot_id",
        ),
        readiness_contract_snapshot_id=_validate_optional_positive_int(
            profile.readiness_contract_snapshot_id,
            "readiness_contract_snapshot_id",
        ),
        total_provider_elapsed_ms=_validate_optional_nonnegative_int(
            profile.total_provider_elapsed_ms,
            "total_provider_elapsed_ms",
        ),
        avg_provider_elapsed_ms=_validate_optional_nonnegative_float(
            profile.avg_provider_elapsed_ms,
            "avg_provider_elapsed_ms",
        ),
        max_provider_elapsed_ms=_validate_optional_nonnegative_int(
            profile.max_provider_elapsed_ms,
            "max_provider_elapsed_ms",
        ),
        total_worker_elapsed_ms=_validate_optional_nonnegative_int(
            profile.total_worker_elapsed_ms,
            "total_worker_elapsed_ms",
        ),
        avg_worker_elapsed_ms=_validate_optional_nonnegative_float(
            profile.avg_worker_elapsed_ms,
            "avg_worker_elapsed_ms",
        ),
        max_worker_elapsed_ms=_validate_optional_nonnegative_int(
            profile.max_worker_elapsed_ms,
            "max_worker_elapsed_ms",
        ),
        errors=tuple(str(error) for error in profile.errors),
        jobs=jobs,
    )


def _validate_job_input(job: DgxIngestionBenchmarkJobInput) -> DgxIngestionBenchmarkJobInput:
    return DgxIngestionBenchmarkJobInput(
        provider=_validate_nonblank(job.provider, "job.provider"),
        profile_name=_validate_nonblank(job.profile_name, "job.profile_name"),
        source_job_id=_validate_optional_positive_int(job.source_job_id, "source_job_id"),
        source_chunk_id=_validate_optional_positive_int(
            job.source_chunk_id,
            "source_chunk_id",
        ),
        processed=bool(job.processed),
        job_status=_validate_optional_nonblank(job.job_status, "job_status"),
        vector_table_name=_validate_optional_nonblank(
            job.vector_table_name,
            "vector_table_name",
        ),
        vector_dimension=_validate_optional_positive_int(
            job.vector_dimension,
            "vector_dimension",
        ),
        vector_storage_type=_validate_optional_nonblank(
            job.vector_storage_type,
            "vector_storage_type",
        ),
        provider_route_id=_validate_optional_positive_int(
            job.provider_route_id,
            "provider_route_id",
        ),
        provider_route_name=_validate_optional_nonblank(
            job.provider_route_name,
            "provider_route_name",
        ),
        provider_runtime_base_url=_validate_optional_nonblank(
            job.provider_runtime_base_url,
            "provider_runtime_base_url",
        ),
        provider_model_id=_validate_optional_nonblank(
            job.provider_model_id,
            "provider_model_id",
        ),
        provider_type=_validate_optional_nonblank(job.provider_type, "provider_type"),
        provider_elapsed_ms=_validate_optional_nonnegative_int(
            job.provider_elapsed_ms,
            "provider_elapsed_ms",
        ),
        worker_elapsed_ms=_validate_optional_nonnegative_int(
            job.worker_elapsed_ms,
            "worker_elapsed_ms",
        ),
        readiness_status=_validate_optional_nonblank(
            job.readiness_status,
            "readiness_status",
        ),
        readiness_health_snapshot_id=_validate_optional_positive_int(
            job.readiness_health_snapshot_id,
            "readiness_health_snapshot_id",
        ),
        readiness_contract_snapshot_id=_validate_optional_positive_int(
            job.readiness_contract_snapshot_id,
            "readiness_contract_snapshot_id",
        ),
        message=_validate_optional_nonblank(job.message, "message"),
        error=_validate_optional_nonblank(job.error, "error"),
        passed=bool(job.passed),
    )


def _validate_count_relationships(
    *,
    expected_job_count: int,
    processed_count: int,
    succeeded_count: int,
    failed_count: int,
    vector_count: int,
    label: str,
) -> None:
    if processed_count > expected_job_count:
        raise InvalidDgxIngestionBenchmarkError(
            f"{label} processed_count must be less than or equal to expected_job_count"
        )
    if succeeded_count + failed_count > processed_count:
        raise InvalidDgxIngestionBenchmarkError(
            f"{label} terminal counts must not exceed processed_count"
        )
    if vector_count > succeeded_count:
        raise InvalidDgxIngestionBenchmarkError(
            f"{label} vector_count must be less than or equal to succeeded_count"
        )


def _validate_limit(limit: int) -> int:
    if limit <= 0:
        raise InvalidDgxIngestionBenchmarkError("limit must be greater than 0")
    if limit > MAX_DGX_INGESTION_BENCHMARK_RUN_LIMIT:
        raise InvalidDgxIngestionBenchmarkError(
            "limit must be less than or equal to " f"{MAX_DGX_INGESTION_BENCHMARK_RUN_LIMIT}"
        )
    return limit


def _validate_nonblank_tuple(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    normalized = tuple(_validate_nonblank(value, field_name) for value in values)
    if not normalized:
        raise InvalidDgxIngestionBenchmarkError(f"{field_name} is required")
    return normalized


def _validate_positive_int_tuple(values: tuple[int, ...], field_name: str) -> tuple[int, ...]:
    return tuple(_validate_positive_int(value, field_name) for value in values)


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidDgxIngestionBenchmarkError(f"{field_name} is required")
    return normalized


def _validate_optional_nonblank(value: str | None, field_name: str) -> str | None:
    return _validate_nonblank(value, field_name) if value is not None else None


def _validate_positive_int(value: int, field_name: str) -> int:
    normalized = int(value)
    if normalized <= 0:
        raise InvalidDgxIngestionBenchmarkError(f"{field_name} must be greater than 0")
    return normalized


def _validate_optional_positive_int(value: int | None, field_name: str) -> int | None:
    return _validate_positive_int(value, field_name) if value is not None else None


def _validate_nonnegative_int(value: int, field_name: str) -> int:
    normalized = int(value)
    if normalized < 0:
        raise InvalidDgxIngestionBenchmarkError(f"{field_name} must be greater than or equal to 0")
    return normalized


def _validate_optional_nonnegative_int(value: int | None, field_name: str) -> int | None:
    return _validate_nonnegative_int(value, field_name) if value is not None else None


def _validate_optional_nonnegative_float(
    value: float | None,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    normalized = float(value)
    if normalized < 0:
        raise InvalidDgxIngestionBenchmarkError(f"{field_name} must be greater than or equal to 0")
    return normalized


def _tuple_from_json_array(value: object) -> tuple:
    return tuple(value) if isinstance(value, list | tuple) else ()


def _string_tuple_from_json_array(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in _tuple_from_json_array(value))


def _int_tuple_from_json_array(value: object) -> tuple[int, ...]:
    return tuple(int(item) for item in _tuple_from_json_array(value))


def _row_to_run_record(row: dict[str, Any]) -> DgxIngestionBenchmarkRunRecord:
    return DgxIngestionBenchmarkRunRecord(
        benchmark_run_id=int(row["benchmark_run_id"]),
        benchmark_run_key=str(row["benchmark_run_key"]),
        script_name=str(row["script_name"]),
        provider_names=_string_tuple_from_json_array(row["provider_names"]),
        profile_names=_string_tuple_from_json_array(row["profile_names"]),
        chunk_count=int(row["chunk_count"]),
        expected_job_count=int(row["expected_job_count"]),
        processed_count=int(row["processed_count"]),
        succeeded_count=int(row["succeeded_count"]),
        failed_count=int(row["failed_count"]),
        vector_count=int(row["vector_count"]),
        passed=bool(row["passed"]),
        preflight_before_worker=bool(row["preflight_before_worker"]),
        active_only_preflight=bool(row["active_only_preflight"]),
        cleanup_attempted=bool(row["cleanup_attempted"]),
        cleanup_confirmed=bool(row["cleanup_confirmed"]),
        total_elapsed_seconds=float(row["total_elapsed_seconds"]),
        total_provider_elapsed_ms=(
            int(row["total_provider_elapsed_ms"])
            if row["total_provider_elapsed_ms"] is not None
            else None
        ),
        total_worker_elapsed_ms=(
            int(row["total_worker_elapsed_ms"])
            if row["total_worker_elapsed_ms"] is not None
            else None
        ),
        fixture_file_id=int(row["fixture_file_id"]) if row["fixture_file_id"] is not None else None,
        fixture_document_id=(
            int(row["fixture_document_id"]) if row["fixture_document_id"] is not None else None
        ),
        fixture_chunk_ids=_int_tuple_from_json_array(row["fixture_chunk_ids"]),
        plan_payload=dict(row["plan_payload"] or {}),
        fixture_payload=dict(row["fixture_payload"] or {}),
        report_payload=dict(row["report_payload"] or {}),
        created_by=row["created_by"],
        created_by_user_id=(
            int(row["created_by_user_id"]) if row["created_by_user_id"] is not None else None
        ),
        created_at=row["created_at"],
    )


def _row_to_profile_record(row: dict[str, Any]) -> DgxIngestionBenchmarkProfileRecord:
    return DgxIngestionBenchmarkProfileRecord(
        benchmark_profile_id=int(row["benchmark_profile_id"]),
        benchmark_run_id=int(row["benchmark_run_id"]),
        provider=str(row["provider"]),
        profile_name=str(row["profile_name"]),
        expected_job_count=int(row["expected_job_count"]),
        processed_count=int(row["processed_count"]),
        succeeded_count=int(row["succeeded_count"]),
        failed_count=int(row["failed_count"]),
        vector_count=int(row["vector_count"]),
        passed=bool(row["passed"]),
        vector_table_name=row["vector_table_name"],
        vector_dimension=(
            int(row["vector_dimension"]) if row["vector_dimension"] is not None else None
        ),
        vector_storage_type=row["vector_storage_type"],
        provider_route_id=(
            int(row["provider_route_id"]) if row["provider_route_id"] is not None else None
        ),
        provider_route_name=row["provider_route_name"],
        provider_runtime_base_url=row["provider_runtime_base_url"],
        provider_model_id=row["provider_model_id"],
        provider_type=row["provider_type"],
        readiness_status=row["readiness_status"],
        readiness_health_snapshot_id=(
            int(row["readiness_health_snapshot_id"])
            if row["readiness_health_snapshot_id"] is not None
            else None
        ),
        readiness_contract_snapshot_id=(
            int(row["readiness_contract_snapshot_id"])
            if row["readiness_contract_snapshot_id"] is not None
            else None
        ),
        total_provider_elapsed_ms=(
            int(row["total_provider_elapsed_ms"])
            if row["total_provider_elapsed_ms"] is not None
            else None
        ),
        avg_provider_elapsed_ms=(
            float(row["avg_provider_elapsed_ms"])
            if row["avg_provider_elapsed_ms"] is not None
            else None
        ),
        max_provider_elapsed_ms=(
            int(row["max_provider_elapsed_ms"])
            if row["max_provider_elapsed_ms"] is not None
            else None
        ),
        total_worker_elapsed_ms=(
            int(row["total_worker_elapsed_ms"])
            if row["total_worker_elapsed_ms"] is not None
            else None
        ),
        avg_worker_elapsed_ms=(
            float(row["avg_worker_elapsed_ms"])
            if row["avg_worker_elapsed_ms"] is not None
            else None
        ),
        max_worker_elapsed_ms=(
            int(row["max_worker_elapsed_ms"]) if row["max_worker_elapsed_ms"] is not None else None
        ),
        errors=_string_tuple_from_json_array(row["errors"]),
        created_at=row["created_at"],
    )


def _row_to_job_record(row: dict[str, Any]) -> DgxIngestionBenchmarkJobRecord:
    return DgxIngestionBenchmarkJobRecord(
        benchmark_job_result_id=int(row["benchmark_job_result_id"]),
        benchmark_run_id=int(row["benchmark_run_id"]),
        benchmark_profile_id=int(row["benchmark_profile_id"]),
        provider=str(row["provider"]),
        profile_name=str(row["profile_name"]),
        source_job_id=int(row["source_job_id"]) if row["source_job_id"] is not None else None,
        source_chunk_id=(
            int(row["source_chunk_id"]) if row["source_chunk_id"] is not None else None
        ),
        processed=bool(row["processed"]),
        job_status=row["job_status"],
        vector_table_name=row["vector_table_name"],
        vector_dimension=(
            int(row["vector_dimension"]) if row["vector_dimension"] is not None else None
        ),
        vector_storage_type=row["vector_storage_type"],
        provider_route_id=(
            int(row["provider_route_id"]) if row["provider_route_id"] is not None else None
        ),
        provider_route_name=row["provider_route_name"],
        provider_runtime_base_url=row["provider_runtime_base_url"],
        provider_model_id=row["provider_model_id"],
        provider_type=row["provider_type"],
        provider_elapsed_ms=(
            int(row["provider_elapsed_ms"]) if row["provider_elapsed_ms"] is not None else None
        ),
        worker_elapsed_ms=(
            int(row["worker_elapsed_ms"]) if row["worker_elapsed_ms"] is not None else None
        ),
        readiness_status=row["readiness_status"],
        readiness_health_snapshot_id=(
            int(row["readiness_health_snapshot_id"])
            if row["readiness_health_snapshot_id"] is not None
            else None
        ),
        readiness_contract_snapshot_id=(
            int(row["readiness_contract_snapshot_id"])
            if row["readiness_contract_snapshot_id"] is not None
            else None
        ),
        message=row["message"],
        error=row["error"],
        passed=bool(row["passed"]),
        created_at=row["created_at"],
    )
