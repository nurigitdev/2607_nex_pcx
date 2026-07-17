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
    expected_job_count: int,
    total_elapsed_seconds: float,
    total_provider_elapsed_ms: int,
    total_worker_elapsed_ms: int,
    provider_route_name: str,
    created_by: str,
    source_job_base: int,
) -> DgxIngestionBenchmarkInput:
    jobs = tuple(
        DgxIngestionBenchmarkJobInput(
            provider="qwen",
            profile_name="qwen3_4b_1000",
            source_job_id=source_job_base + index,
            source_chunk_id=source_job_base + 100 + index,
            processed=True,
            job_status="succeeded",
            vector_table_name="chunk_embeddings_qwen3_4b_1000",
            vector_dimension=1000,
            vector_storage_type="vector",
            provider_route_id=4,
            provider_route_name=provider_route_name,
            provider_runtime_base_url="http://192.168.20.243:9103",
            provider_model_id="local-qwen3-embedding-4b",
            provider_type="remote",
            provider_elapsed_ms=round(total_provider_elapsed_ms / expected_job_count),
            worker_elapsed_ms=round(total_worker_elapsed_ms / expected_job_count),
            readiness_status="ready",
            readiness_health_snapshot_id=71,
            readiness_contract_snapshot_id=72,
            message="Remote Qwen embedding stored",
            error=None,
            passed=True,
        )
        for index in range(expected_job_count)
    )
    return DgxIngestionBenchmarkInput(
        benchmark_run_key=run_key,
        script_name="run_dgx_small_corpus_embedding_benchmark.py",
        provider_names=("qwen",),
        profile_names=("qwen3_4b_1000",),
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
        total_elapsed_seconds=total_elapsed_seconds,
        total_provider_elapsed_ms=total_provider_elapsed_ms,
        total_worker_elapsed_ms=total_worker_elapsed_ms,
        fixture_file_id=source_job_base + 200,
        fixture_document_id=source_job_base + 201,
        fixture_chunk_ids=tuple(
            source_job_base + 100 + index for index in range(expected_job_count)
        ),
        plan_payload={"provider": "qwen", "profile": "qwen3_4b_1000"},
        fixture_payload={"chunk_count": expected_job_count},
        report_payload={"passed": True, "vector_count": expected_job_count},
        created_by=created_by,
        profiles=(
            DgxIngestionBenchmarkProfileInput(
                provider="qwen",
                profile_name="qwen3_4b_1000",
                expected_job_count=expected_job_count,
                processed_count=expected_job_count,
                succeeded_count=expected_job_count,
                failed_count=0,
                vector_count=expected_job_count,
                passed=True,
                vector_table_name="chunk_embeddings_qwen3_4b_1000",
                vector_dimension=1000,
                vector_storage_type="vector",
                provider_route_id=4,
                provider_route_name=provider_route_name,
                provider_runtime_base_url="http://192.168.20.243:9103",
                provider_model_id="local-qwen3-embedding-4b",
                provider_type="remote",
                readiness_status="ready",
                readiness_health_snapshot_id=71,
                readiness_contract_snapshot_id=72,
                total_provider_elapsed_ms=total_provider_elapsed_ms,
                avg_provider_elapsed_ms=total_provider_elapsed_ms / expected_job_count,
                max_provider_elapsed_ms=round(total_provider_elapsed_ms / expected_job_count),
                total_worker_elapsed_ms=total_worker_elapsed_ms,
                avg_worker_elapsed_ms=total_worker_elapsed_ms / expected_job_count,
                max_worker_elapsed_ms=round(total_worker_elapsed_ms / expected_job_count),
                jobs=jobs,
            ),
        ),
    )


def test_dgx_ingestion_benchmark_compare_api_and_ui(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    benchmark_run_ids: list[int] = []

    try:
        left_detail = record_dgx_ingestion_benchmark(
            migrated_database_url,
            _benchmark_input(
                run_key=f"dgx-compare-left-{suffix}",
                expected_job_count=1,
                total_elapsed_seconds=5.0,
                total_provider_elapsed_ms=120,
                total_worker_elapsed_ms=180,
                provider_route_name="qwen-dgx-baseline",
                created_by="slice-210-left",
                source_job_base=72000,
            ),
        )
        right_detail = record_dgx_ingestion_benchmark(
            migrated_database_url,
            _benchmark_input(
                run_key=f"dgx-compare-right-{suffix}",
                expected_job_count=2,
                total_elapsed_seconds=4.0,
                total_provider_elapsed_ms=180,
                total_worker_elapsed_ms=220,
                provider_route_name="qwen-dgx-candidate",
                created_by="slice-210-right",
                source_job_base=73000,
            ),
        )
        benchmark_run_ids.extend(
            [
                left_detail.run.benchmark_run_id,
                right_detail.run.benchmark_run_id,
            ]
        )
        app = create_app(Settings(database_url=migrated_database_url))

        with TestClient(app) as client:
            compare_response = client.get(
                "/api/admin/dgx-ingestion-benchmarks/compare",
                params={
                    "left_run_id": left_detail.run.benchmark_run_id,
                    "right_run_id": right_detail.run.benchmark_run_id,
                },
            )
            page_response = client.get(
                "/admin/dgx-ingestion-benchmarks/compare",
                params={
                    "left_run_id": left_detail.run.benchmark_run_id,
                    "right_run_id": right_detail.run.benchmark_run_id,
                },
            )
            history_response = client.get("/admin/dgx-ingestion-benchmarks")
            same_run_response = client.get(
                "/api/admin/dgx-ingestion-benchmarks/compare",
                params={
                    "left_run_id": left_detail.run.benchmark_run_id,
                    "right_run_id": left_detail.run.benchmark_run_id,
                },
            )
            missing_response = client.get(
                "/api/admin/dgx-ingestion-benchmarks/compare",
                params={
                    "left_run_id": left_detail.run.benchmark_run_id,
                    "right_run_id": 999999999,
                },
            )

        payload = compare_response.json()["comparison"]
        metric_map = {metric["metric_key"]: metric for metric in payload["run_metrics"]}
        profile_metrics = payload["profile_comparisons"][0]["metrics"]

        assert compare_response.status_code == 200
        assert payload["left"]["run"]["benchmark_run_key"].startswith("dgx-compare-left-")
        assert payload["right"]["run"]["benchmark_run_key"].startswith("dgx-compare-right-")
        assert metric_map["total_elapsed_seconds"]["delta_value"] == -1.0
        assert metric_map["total_elapsed_seconds"]["delta_label"] == "-1.00 s"
        assert metric_map["total_elapsed_seconds"]["status"] == "better"
        assert metric_map["vector_count"]["delta_value"] == 1
        assert metric_map["vector_count"]["delta_label"] == "+1"
        assert metric_map["vector_count"]["status"] == "better"
        assert profile_metrics["avg_provider_elapsed_ms"]["delta_label"] == "-30.00 ms"
        assert profile_metrics["avg_provider_elapsed_ms"]["status"] == "better"
        assert page_response.status_code == 200
        assert "DGX Benchmark Run 비교" in page_response.text
        assert "data-dgx-benchmark-compare-page" in page_response.text
        assert "qwen-dgx-baseline" in page_response.text
        assert "qwen-dgx-candidate" in page_response.text
        assert "-1.00 s" in page_response.text
        assert "+1" in page_response.text
        assert "dgx-benchmark-json-viewer" in page_response.text
        assert history_response.status_code == 200
        assert "/admin/dgx-ingestion-benchmarks/compare" in history_response.text
        assert same_run_response.status_code == 400
        assert missing_response.status_code == 404
    finally:
        _cleanup_benchmark_runs(migrated_database_url, benchmark_run_ids)
