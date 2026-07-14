import pytest

from app.core.dgx_ingestion_benchmarks import (
    DgxIngestionBenchmarkInput,
    DgxIngestionBenchmarkJobInput,
    DgxIngestionBenchmarkProfileInput,
    InvalidDgxIngestionBenchmarkError,
    list_dgx_ingestion_benchmark_runs,
    validate_dgx_ingestion_benchmark_input,
)


def _job(
    *,
    provider: str = "qwen",
    profile_name: str = "qwen3_4b_1000",
    source_job_id: int = 101,
) -> DgxIngestionBenchmarkJobInput:
    return DgxIngestionBenchmarkJobInput(
        provider=provider,
        profile_name=profile_name,
        source_job_id=source_job_id,
        source_chunk_id=201,
        processed=True,
        job_status="succeeded",
        vector_table_name="chunk_embeddings_qwen3_4b_1000",
        vector_dimension=1000,
        vector_storage_type="vector",
        provider_route_id=4,
        provider_route_name="qwen-primary",
        provider_runtime_base_url="http://192.168.20.243:9103",
        provider_model_id="local-qwen3-embedding-4b",
        provider_type="remote",
        provider_elapsed_ms=150,
        worker_elapsed_ms=200,
        readiness_status="ready",
        readiness_health_snapshot_id=11,
        readiness_contract_snapshot_id=11,
        message="Remote embedding stored",
        error=None,
        passed=True,
    )


def _profile(
    *,
    provider: str = "qwen",
    profile_name: str = "qwen3_4b_1000",
    jobs: tuple[DgxIngestionBenchmarkJobInput, ...] | None = None,
) -> DgxIngestionBenchmarkProfileInput:
    effective_jobs = (
        jobs if jobs is not None else (_job(provider=provider, profile_name=profile_name),)
    )
    return DgxIngestionBenchmarkProfileInput(
        provider=provider,
        profile_name=profile_name,
        expected_job_count=len(effective_jobs),
        processed_count=len(effective_jobs),
        succeeded_count=len(effective_jobs),
        failed_count=0,
        vector_count=len(effective_jobs),
        passed=True,
        vector_table_name="chunk_embeddings_qwen3_4b_1000",
        vector_dimension=1000,
        vector_storage_type="vector",
        provider_route_id=4,
        provider_route_name="qwen-primary",
        provider_runtime_base_url="http://192.168.20.243:9103",
        provider_model_id="local-qwen3-embedding-4b",
        provider_type="remote",
        readiness_status="ready",
        readiness_health_snapshot_id=11,
        readiness_contract_snapshot_id=11,
        total_provider_elapsed_ms=150 * len(effective_jobs),
        avg_provider_elapsed_ms=150.0,
        max_provider_elapsed_ms=150,
        total_worker_elapsed_ms=200 * len(effective_jobs),
        avg_worker_elapsed_ms=200.0,
        max_worker_elapsed_ms=200,
        errors=(),
        jobs=effective_jobs,
    )


def _benchmark(
    *,
    profiles: tuple[DgxIngestionBenchmarkProfileInput, ...] | None = None,
    expected_job_count: int | None = None,
    processed_count: int | None = None,
    succeeded_count: int | None = None,
    failed_count: int = 0,
    vector_count: int | None = None,
) -> DgxIngestionBenchmarkInput:
    effective_profiles = profiles if profiles is not None else (_profile(),)
    expected = (
        expected_job_count
        if expected_job_count is not None
        else sum(profile.expected_job_count for profile in effective_profiles)
    )
    processed = (
        processed_count
        if processed_count is not None
        else sum(profile.processed_count for profile in effective_profiles)
    )
    succeeded = (
        succeeded_count
        if succeeded_count is not None
        else sum(profile.succeeded_count for profile in effective_profiles)
    )
    vectors = (
        vector_count
        if vector_count is not None
        else sum(profile.vector_count for profile in effective_profiles)
    )
    return DgxIngestionBenchmarkInput(
        benchmark_run_key=" dgx-small-corpus-unit ",
        script_name=" run_dgx_small_corpus_embedding_benchmark.py ",
        provider_names=(" qwen ",),
        profile_names=(" qwen3_4b_1000 ",),
        chunk_count=1,
        expected_job_count=expected,
        processed_count=processed,
        succeeded_count=succeeded,
        failed_count=failed_count,
        vector_count=vectors,
        passed=True,
        preflight_before_worker=True,
        active_only_preflight=True,
        cleanup_attempted=True,
        cleanup_confirmed=True,
        total_elapsed_seconds=12.5,
        total_provider_elapsed_ms=150,
        total_worker_elapsed_ms=200,
        fixture_file_id=49,
        fixture_document_id=50,
        fixture_chunk_ids=(201,),
        plan_payload={"database_url": "postgresql://user:***@host/db"},
        fixture_payload={"job_count": expected},
        report_payload={"passed": True},
        created_by=" pytest ",
        profiles=effective_profiles,
    )


def test_validate_dgx_ingestion_benchmark_input_normalizes_strings() -> None:
    validated = validate_dgx_ingestion_benchmark_input(_benchmark())

    assert validated.benchmark_run_key == "dgx-small-corpus-unit"
    assert validated.script_name == "run_dgx_small_corpus_embedding_benchmark.py"
    assert validated.provider_names == ("qwen",)
    assert validated.profile_names == ("qwen3_4b_1000",)
    assert validated.created_by == "pytest"
    assert validated.profiles[0].jobs[0].provider_type == "remote"


def test_validate_dgx_ingestion_benchmark_allows_failed_run_without_profiles() -> None:
    validated = validate_dgx_ingestion_benchmark_input(
        _benchmark(
            profiles=(),
            expected_job_count=0,
            processed_count=0,
            succeeded_count=0,
            vector_count=0,
        )
    )

    assert validated.profiles == ()
    assert validated.expected_job_count == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("chunk_count", 0, "chunk_count must be greater than 0"),
        ("total_elapsed_seconds", -0.1, "total_elapsed_seconds"),
        ("fixture_file_id", 0, "fixture_file_id must be greater than 0"),
    ),
)
def test_validate_dgx_ingestion_benchmark_rejects_invalid_numeric_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _benchmark()

    with pytest.raises(InvalidDgxIngestionBenchmarkError, match=message):
        validate_dgx_ingestion_benchmark_input(
            DgxIngestionBenchmarkInput(
                **{**payload.__dict__, field: value},
            )
        )


def test_validate_dgx_ingestion_benchmark_rejects_inconsistent_run_counts() -> None:
    with pytest.raises(
        InvalidDgxIngestionBenchmarkError,
        match="processed_count must be less than or equal",
    ):
        validate_dgx_ingestion_benchmark_input(_benchmark(expected_job_count=1, processed_count=2))


def test_validate_dgx_ingestion_benchmark_rejects_terminal_count_overflow() -> None:
    with pytest.raises(InvalidDgxIngestionBenchmarkError, match="terminal counts"):
        validate_dgx_ingestion_benchmark_input(
            _benchmark(
                expected_job_count=2,
                processed_count=1,
                succeeded_count=1,
                failed_count=1,
                vector_count=1,
            )
        )


def test_validate_dgx_ingestion_benchmark_rejects_vector_count_overflow() -> None:
    with pytest.raises(InvalidDgxIngestionBenchmarkError, match="vector_count"):
        validate_dgx_ingestion_benchmark_input(
            _benchmark(
                expected_job_count=2,
                processed_count=2,
                succeeded_count=1,
                failed_count=0,
                vector_count=2,
            )
        )


def test_validate_dgx_ingestion_benchmark_rejects_profile_sum_mismatch() -> None:
    with pytest.raises(InvalidDgxIngestionBenchmarkError, match="profile vector_count sum"):
        validate_dgx_ingestion_benchmark_input(_benchmark(vector_count=0))


def test_validate_dgx_ingestion_benchmark_rejects_profile_job_count_mismatch() -> None:
    profile = DgxIngestionBenchmarkProfileInput(
        **{
            **_profile().__dict__,
            "expected_job_count": 2,
            "processed_count": 2,
        }
    )

    with pytest.raises(InvalidDgxIngestionBenchmarkError, match="job count"):
        validate_dgx_ingestion_benchmark_input(
            _benchmark(
                profiles=(profile,),
                expected_job_count=2,
                processed_count=2,
                succeeded_count=1,
                vector_count=1,
            )
        )


def test_validate_dgx_ingestion_benchmark_rejects_mismatched_job_profile() -> None:
    profile = _profile(jobs=(_job(profile_name="bge_m3_1024"),))

    with pytest.raises(InvalidDgxIngestionBenchmarkError, match="mismatched"):
        validate_dgx_ingestion_benchmark_input(_benchmark(profiles=(profile,)))


@pytest.mark.parametrize("limit", (0, 201))
def test_list_dgx_ingestion_benchmark_runs_validates_limit_before_connect(
    limit: int,
) -> None:
    with pytest.raises(InvalidDgxIngestionBenchmarkError, match="limit"):
        list_dgx_ingestion_benchmark_runs("postgresql://example/db", limit=limit)
