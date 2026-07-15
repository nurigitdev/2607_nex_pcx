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


def test_dgx_ingestion_benchmark_history_api_and_ui(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    run_key = f"dgx-history-{suffix}"
    benchmark_run_ids: list[int] = []

    try:
        detail = record_dgx_ingestion_benchmark(
            migrated_database_url,
            DgxIngestionBenchmarkInput(
                benchmark_run_key=run_key,
                script_name="run_dgx_small_corpus_embedding_benchmark.py",
                provider_names=("bge",),
                profile_names=("bge_m3_1024",),
                chunk_count=1,
                expected_job_count=1,
                processed_count=1,
                succeeded_count=1,
                failed_count=0,
                vector_count=1,
                passed=True,
                preflight_before_worker=True,
                active_only_preflight=True,
                cleanup_attempted=True,
                cleanup_confirmed=True,
                total_elapsed_seconds=3.7567,
                total_provider_elapsed_ms=125,
                total_worker_elapsed_ms=180,
                fixture_file_id=9001,
                fixture_document_id=9002,
                fixture_chunk_ids=(9003,),
                plan_payload={"provider": "bge", "profiles": ["bge_m3_1024"]},
                fixture_payload={"chunk_count": 1},
                report_payload={"passed": True, "vector_count": 1},
                created_by="slice-209-test",
                profiles=(
                    DgxIngestionBenchmarkProfileInput(
                        provider="bge",
                        profile_name="bge_m3_1024",
                        expected_job_count=1,
                        processed_count=1,
                        succeeded_count=1,
                        failed_count=0,
                        vector_count=1,
                        passed=True,
                        vector_table_name="chunk_embeddings_bge_m3_1024",
                        vector_dimension=1024,
                        vector_storage_type="vector",
                        provider_route_id=2,
                        provider_route_name="bge-dgx-primary",
                        provider_runtime_base_url="http://192.168.20.243:9102",
                        provider_model_id="BAAI/bge-m3",
                        provider_type="remote",
                        readiness_status="ready",
                        readiness_health_snapshot_id=42,
                        readiness_contract_snapshot_id=43,
                        total_provider_elapsed_ms=125,
                        avg_provider_elapsed_ms=125.0,
                        max_provider_elapsed_ms=125,
                        total_worker_elapsed_ms=180,
                        avg_worker_elapsed_ms=180.0,
                        max_worker_elapsed_ms=180,
                        jobs=(
                            DgxIngestionBenchmarkJobInput(
                                provider="bge",
                                profile_name="bge_m3_1024",
                                source_job_id=51001,
                                source_chunk_id=61001,
                                processed=True,
                                job_status="succeeded",
                                vector_table_name="chunk_embeddings_bge_m3_1024",
                                vector_dimension=1024,
                                vector_storage_type="vector",
                                provider_route_id=2,
                                provider_route_name="bge-dgx-primary",
                                provider_runtime_base_url="http://192.168.20.243:9102",
                                provider_model_id="BAAI/bge-m3",
                                provider_type="remote",
                                provider_elapsed_ms=125,
                                worker_elapsed_ms=180,
                                readiness_status="ready",
                                readiness_health_snapshot_id=42,
                                readiness_contract_snapshot_id=43,
                                message="Remote BGE embedding stored",
                                error=None,
                                passed=True,
                            ),
                        ),
                    ),
                ),
            ),
        )
        benchmark_run_ids.append(detail.run.benchmark_run_id)
        app = create_app(Settings(database_url=migrated_database_url))

        with TestClient(app) as client:
            list_response = client.get(
                "/api/admin/dgx-ingestion-benchmarks",
                params={
                    "provider": "bge",
                    "profile_name": "bge_m3_1024",
                    "passed": "true",
                    "limit": 20,
                },
            )
            detail_response = client.get(
                f"/api/admin/dgx-ingestion-benchmarks/{detail.run.benchmark_run_id}"
            )
            page_response = client.get(
                "/admin/dgx-ingestion-benchmarks",
                params={
                    "benchmark_run_id": detail.run.benchmark_run_id,
                    "provider": "bge",
                    "profile_name": "bge_m3_1024",
                    "passed": "true",
                    "limit": 20,
                },
            )
            english_page_response = client.get(
                "/admin/dgx-ingestion-benchmarks",
                params={"benchmark_run_id": detail.run.benchmark_run_id, "lang": "en"},
            )
            missing_response = client.get("/api/admin/dgx-ingestion-benchmarks/999999999")
            invalid_response = client.get(
                "/api/admin/dgx-ingestion-benchmarks",
                params={"limit": 0},
            )

        list_payload = list_response.json()
        detail_payload = detail_response.json()

        assert list_response.status_code == 200
        assert list_payload["benchmark_run_count"] >= 1
        assert list_payload["summary"]["processed_count"] >= 1
        matching_runs = [
            run
            for run in list_payload["benchmark_runs"]
            if run["benchmark_run_id"] == detail.run.benchmark_run_id
        ]
        assert matching_runs
        assert matching_runs[0]["benchmark_run_key"] == run_key
        assert matching_runs[0]["created_at"]
        assert detail_response.status_code == 200
        assert detail_payload["benchmark"]["run"]["benchmark_run_key"] == run_key
        assert detail_payload["benchmark"]["run"]["total_elapsed_seconds"] == 3.7567
        assert detail_payload["benchmark"]["profiles"][0]["provider_route_name"] == "bge-dgx-primary"
        assert detail_payload["benchmark"]["jobs"][0]["message"] == "Remote BGE embedding stored"
        assert page_response.status_code == 200
        assert "DGX Ingestion Benchmark 이력" in page_response.text
        assert "data-dgx-ingestion-benchmarks-page" in page_response.text
        assert run_key in page_response.text
        assert "bge-dgx-primary" in page_response.text
        assert "Remote BGE embedding stored" in page_response.text
        assert ">3.76 s</td>" in page_response.text
        assert ">3.76 s</div>" in page_response.text
        assert "dgx-benchmark-json-viewer" in page_response.text
        assert english_page_response.status_code == 200
        assert "DGX Ingestion Benchmark History" in english_page_response.text
        assert missing_response.status_code == 404
        assert invalid_response.status_code == 400
    finally:
        _cleanup_benchmark_runs(migrated_database_url, benchmark_run_ids)
