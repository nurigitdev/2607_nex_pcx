from uuid import uuid4

import pytest

from app.core.database import connect, fetch_one
from app.core.dgx_ingestion_benchmarks import (
    DgxIngestionBenchmarkInput,
    DgxIngestionBenchmarkJobInput,
    DgxIngestionBenchmarkProfileInput,
    get_dgx_ingestion_benchmark_detail,
    list_dgx_ingestion_benchmark_runs,
    record_dgx_ingestion_benchmark,
)

pytestmark = pytest.mark.integration


def _cleanup_benchmark_runs(database_url: str, benchmark_run_ids: list[int]) -> None:
    if not benchmark_run_ids:
        return
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM dgx_ingestion_benchmark_runs WHERE benchmark_run_id = ANY(%s)",
                (benchmark_run_ids,),
            )


def test_dgx_ingestion_benchmark_tables_exist(migrated_database_url: str) -> None:
    run_table = fetch_one(
        migrated_database_url,
        "SELECT to_regclass('public.dgx_ingestion_benchmark_runs') AS table_name",
    )
    profile_table = fetch_one(
        migrated_database_url,
        """
        SELECT to_regclass(
            'public.dgx_ingestion_benchmark_profile_results'
        ) AS table_name
        """,
    )
    job_table = fetch_one(
        migrated_database_url,
        "SELECT to_regclass('public.dgx_ingestion_benchmark_job_results') AS table_name",
    )

    assert run_table["table_name"] == "dgx_ingestion_benchmark_runs"
    assert profile_table["table_name"] == "dgx_ingestion_benchmark_profile_results"
    assert job_table["table_name"] == "dgx_ingestion_benchmark_job_results"


def test_dgx_ingestion_benchmark_repository_records_detail_and_filters(
    migrated_database_url: str,
) -> None:
    run_key = f"dgx-small-corpus-{uuid4().hex}"
    benchmark_run_ids: list[int] = []
    benchmark_input = DgxIngestionBenchmarkInput(
        benchmark_run_key=run_key,
        script_name="run_dgx_small_corpus_embedding_benchmark.py",
        provider_names=("qwen",),
        profile_names=("qwen3_4b_1000",),
        chunk_count=2,
        expected_job_count=2,
        processed_count=2,
        succeeded_count=2,
        failed_count=0,
        vector_count=2,
        passed=True,
        preflight_before_worker=True,
        active_only_preflight=True,
        cleanup_attempted=True,
        cleanup_confirmed=True,
        total_elapsed_seconds=12.5,
        total_provider_elapsed_ms=300,
        total_worker_elapsed_ms=420,
        fixture_file_id=49,
        fixture_document_id=50,
        fixture_chunk_ids=(1001, 1002),
        plan_payload={"database_url": "postgresql://user:***@host/db"},
        fixture_payload={"job_count": 2},
        report_payload={"passed": True},
        created_by="pytest",
        profiles=(
            DgxIngestionBenchmarkProfileInput(
                provider="qwen",
                profile_name="qwen3_4b_1000",
                expected_job_count=2,
                processed_count=2,
                succeeded_count=2,
                failed_count=0,
                vector_count=2,
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
                total_provider_elapsed_ms=300,
                avg_provider_elapsed_ms=150.0,
                max_provider_elapsed_ms=170,
                total_worker_elapsed_ms=420,
                avg_worker_elapsed_ms=210.0,
                max_worker_elapsed_ms=230,
                jobs=(
                    DgxIngestionBenchmarkJobInput(
                        provider="qwen",
                        profile_name="qwen3_4b_1000",
                        source_job_id=101,
                        source_chunk_id=1001,
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
                        provider_elapsed_ms=130,
                        worker_elapsed_ms=190,
                        readiness_status="ready",
                        readiness_health_snapshot_id=11,
                        readiness_contract_snapshot_id=11,
                        message="Remote embedding stored",
                        error=None,
                        passed=True,
                    ),
                    DgxIngestionBenchmarkJobInput(
                        provider="qwen",
                        profile_name="qwen3_4b_1000",
                        source_job_id=102,
                        source_chunk_id=1002,
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
                        provider_elapsed_ms=170,
                        worker_elapsed_ms=230,
                        readiness_status="ready",
                        readiness_health_snapshot_id=11,
                        readiness_contract_snapshot_id=11,
                        message="Remote embedding stored",
                        error=None,
                        passed=True,
                    ),
                ),
            ),
        ),
    )

    try:
        detail = record_dgx_ingestion_benchmark(migrated_database_url, benchmark_input)
        benchmark_run_ids.append(detail.run.benchmark_run_id)

        fetched = get_dgx_ingestion_benchmark_detail(
            migrated_database_url,
            detail.run.benchmark_run_id,
        )
        latest_runs = list_dgx_ingestion_benchmark_runs(
            migrated_database_url,
            provider="qwen",
            profile_name="qwen3_4b_1000",
            passed=True,
            limit=50,
        )

        assert detail.run.benchmark_run_key == run_key
        assert detail.run.total_provider_elapsed_ms == 300
        assert detail.run.fixture_chunk_ids == (1001, 1002)
        assert len(detail.profiles) == 1
        assert detail.profiles[0].avg_worker_elapsed_ms == 210.0
        assert len(detail.jobs) == 2
        assert detail.jobs[1].source_job_id == 102
        assert fetched == detail
        assert detail.run.benchmark_run_id in [run.benchmark_run_id for run in latest_runs]
    finally:
        _cleanup_benchmark_runs(migrated_database_url, benchmark_run_ids)
