from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect
from app.core.dgx_ingestion_benchmarks import (
    DgxIngestionBenchmarkInput,
    DgxIngestionBenchmarkJobInput,
    DgxIngestionBenchmarkProfileInput,
    record_dgx_ingestion_benchmark,
)
from app.main import create_app

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


def _benchmark_input(
    *,
    run_key: str,
    provider: str,
    profile_name: str,
    expected_job_count: int,
    avg_provider_elapsed_ms: float,
    avg_worker_elapsed_ms: float,
    created_by: str,
    source_job_base: int,
) -> DgxIngestionBenchmarkInput:
    total_provider_elapsed_ms = round(avg_provider_elapsed_ms * expected_job_count)
    total_worker_elapsed_ms = round(avg_worker_elapsed_ms * expected_job_count)
    jobs = tuple(
        DgxIngestionBenchmarkJobInput(
            provider=provider,
            profile_name=profile_name,
            source_job_id=source_job_base + index,
            source_chunk_id=source_job_base + 100 + index,
            processed=True,
            job_status="succeeded",
            vector_table_name=f"chunk_embeddings_{profile_name}",
            vector_dimension=1000,
            vector_storage_type="vector",
            provider_route_id=4,
            provider_route_name=f"{provider}-route",
            provider_runtime_base_url="http://192.168.20.243:9103",
            provider_model_id="local-qwen3-embedding-4b",
            provider_type="remote",
            provider_elapsed_ms=round(avg_provider_elapsed_ms),
            worker_elapsed_ms=round(avg_worker_elapsed_ms),
            readiness_status="ready",
            readiness_health_snapshot_id=101,
            readiness_contract_snapshot_id=102,
            message="Remote trend embedding stored",
            error=None,
            passed=True,
        )
        for index in range(expected_job_count)
    )
    return DgxIngestionBenchmarkInput(
        benchmark_run_key=run_key,
        script_name="run_dgx_small_corpus_embedding_benchmark.py",
        provider_names=(provider,),
        profile_names=(profile_name,),
        chunk_count=expected_job_count,
        expected_job_count=expected_job_count,
        processed_count=expected_job_count,
        succeeded_count=expected_job_count,
        failed_count=0,
        vector_count=expected_job_count,
        passed=True,
        preflight_before_worker=True,
        active_only_preflight=True,
        cleanup_attempted=True,
        cleanup_confirmed=True,
        total_elapsed_seconds=expected_job_count * 2.5,
        total_provider_elapsed_ms=total_provider_elapsed_ms,
        total_worker_elapsed_ms=total_worker_elapsed_ms,
        fixture_file_id=source_job_base + 200,
        fixture_document_id=source_job_base + 201,
        fixture_chunk_ids=tuple(
            source_job_base + 100 + index for index in range(expected_job_count)
        ),
        plan_payload={"provider": provider, "profile": profile_name},
        fixture_payload={"chunk_count": expected_job_count},
        report_payload={"passed": True, "vector_count": expected_job_count},
        created_by=created_by,
        profiles=(
            DgxIngestionBenchmarkProfileInput(
                provider=provider,
                profile_name=profile_name,
                expected_job_count=expected_job_count,
                processed_count=expected_job_count,
                succeeded_count=expected_job_count,
                failed_count=0,
                vector_count=expected_job_count,
                passed=True,
                vector_table_name=f"chunk_embeddings_{profile_name}",
                vector_dimension=1000,
                vector_storage_type="vector",
                provider_route_id=4,
                provider_route_name=f"{provider}-route",
                provider_runtime_base_url="http://192.168.20.243:9103",
                provider_model_id="local-qwen3-embedding-4b",
                provider_type="remote",
                readiness_status="ready",
                readiness_health_snapshot_id=101,
                readiness_contract_snapshot_id=102,
                total_provider_elapsed_ms=total_provider_elapsed_ms,
                avg_provider_elapsed_ms=avg_provider_elapsed_ms,
                max_provider_elapsed_ms=round(avg_provider_elapsed_ms),
                total_worker_elapsed_ms=total_worker_elapsed_ms,
                avg_worker_elapsed_ms=avg_worker_elapsed_ms,
                max_worker_elapsed_ms=round(avg_worker_elapsed_ms),
                jobs=jobs,
            ),
        ),
    )


def test_dgx_ingestion_benchmark_trend_api_and_ui(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex[:10]
    provider = f"qwen_trend_{suffix}"
    profile_name = f"qwen3_4b_1000_{suffix}"
    benchmark_run_ids: list[int] = []

    try:
        for index, (job_count, provider_ms, worker_ms) in enumerate(
            [(1, 120.0, 180.0), (2, 105.0, 160.0), (3, 90.0, 140.0)],
            start=1,
        ):
            detail = record_dgx_ingestion_benchmark(
                migrated_database_url,
                _benchmark_input(
                    run_key=f"dgx-trend-{suffix}-{index}",
                    provider=provider,
                    profile_name=profile_name,
                    expected_job_count=job_count,
                    avg_provider_elapsed_ms=provider_ms,
                    avg_worker_elapsed_ms=worker_ms,
                    created_by="slice-211-test",
                    source_job_base=81000 + (index * 100),
                ),
            )
            benchmark_run_ids.append(detail.run.benchmark_run_id)

        app = create_app(Settings(database_url=migrated_database_url))

        with TestClient(app) as client:
            trend_response = client.get(
                "/api/admin/dgx-ingestion-benchmarks/trend-summary",
                params={
                    "provider": provider,
                    "profile_name": profile_name,
                    "limit": 20,
                },
            )
            page_response = client.get(
                "/admin/dgx-ingestion-benchmarks/trends",
                params={
                    "provider": provider,
                    "profile_name": profile_name,
                    "limit": 20,
                },
            )
            history_response = client.get("/admin/dgx-ingestion-benchmarks")
            invalid_response = client.get(
                "/api/admin/dgx-ingestion-benchmarks/trend-summary",
                params={"limit": 0},
            )

        payload = trend_response.json()["trend"]
        profile_trend = payload["profiles"][0]

        assert trend_response.status_code == 200
        assert payload["run_count"] == 3
        assert payload["profile_count"] == 1
        assert profile_trend["provider"] == provider
        assert profile_trend["profile_name"] == profile_name
        assert profile_trend["run_count"] == 3
        assert profile_trend["total_vectors"] == 6
        assert profile_trend["deltas"]["vector_count"]["delta_label"] == "+2"
        assert profile_trend["deltas"]["vector_count"]["status"] == "better"
        assert profile_trend["deltas"]["avg_provider_elapsed_ms"]["delta_label"] == "-30.00 ms"
        assert profile_trend["deltas"]["avg_provider_elapsed_ms"]["status"] == "better"
        assert profile_trend["points"][0]["vector_count"] == 1
        assert profile_trend["points"][-1]["vector_count"] == 3
        assert page_response.status_code == 200
        assert "DGX Benchmark Trend 요약" in page_response.text
        assert "data-dgx-benchmark-trends-page" in page_response.text
        assert provider in page_response.text
        assert profile_name in page_response.text
        assert "+2" in page_response.text
        assert "-30.00 ms" in page_response.text
        assert "dgx-benchmark-json-viewer" in page_response.text
        assert history_response.status_code == 200
        assert "/admin/dgx-ingestion-benchmarks/trends" in history_response.text
        assert invalid_response.status_code == 400
    finally:
        _cleanup_benchmark_runs(migrated_database_url, benchmark_run_ids)
