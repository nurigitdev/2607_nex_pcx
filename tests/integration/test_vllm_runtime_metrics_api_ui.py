from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect
from app.core.vllm_runtime_metric_snapshots import record_vllm_runtime_metric_snapshot
from app.core.vllm_runtime_metrics import scrape_vllm_runtime_metrics_from_text
from app.main import create_app

pytestmark = pytest.mark.integration

PROVIDER_NAME = "pytest-vllm-runtime-api-ui"
NOW = datetime(2026, 7, 29, 13, 0, tzinfo=UTC)


def test_vllm_runtime_metric_snapshot_api_lists_persisted_snapshots(
    migrated_database_url: str,
) -> None:
    _cleanup_provider_snapshots(migrated_database_url, PROVIDER_NAME)
    try:
        first = record_vllm_runtime_metric_snapshot(
            migrated_database_url,
            _snapshot(
                """
                vllm:kv_cache_usage_perc 0.45
                vllm:num_requests_waiting 1
                vllm:time_to_first_token_seconds_sum 0.4
                vllm:time_to_first_token_seconds_count 1
                vllm:e2e_request_latency_seconds_sum 2
                vllm:e2e_request_latency_seconds_count 1
                """,
                sampled_at=NOW,
            ),
        )
        second = record_vllm_runtime_metric_snapshot(
            migrated_database_url,
            _snapshot(
                """
                vllm:kv_cache_usage_perc 0.83
                vllm:num_requests_running 2
                vllm:num_requests_waiting 3
                vllm:prompt_tokens_total 120
                vllm:generation_tokens_total 300
                vllm:time_to_first_token_seconds_sum 0.8
                vllm:time_to_first_token_seconds_count 2
                vllm:e2e_request_latency_seconds_sum 6
                vllm:e2e_request_latency_seconds_count 2
                """,
                sampled_at=NOW + timedelta(minutes=1),
            ),
        )
        app = create_app(Settings(database_url=migrated_database_url))

        with TestClient(app) as client:
            response = client.get(
                "/api/admin/vllm-runtime-metrics/snapshots",
                params={
                    "provider_name": PROVIDER_NAME,
                    "limit": "10",
                    "include_raw_samples": "true",
                },
            )
            invalid_limit_response = client.get(
                "/api/admin/vllm-runtime-metrics/snapshots",
                params={"limit": "0"},
            )
            invalid_provider_response = client.get(
                "/api/admin/vllm-runtime-metrics/snapshots",
                params={"provider_name": " "},
            )

        body = response.json()
        assert response.status_code == 200
        assert body["provider_name"] == PROVIDER_NAME
        assert body["limit"] == 10
        assert body["summary"]["snapshot_count"] == 2
        assert body["summary"]["latest_snapshot_id"] == second.snapshot_id
        assert body["summary"]["latest_kv_cache_usage_percent"] == 83.0
        assert body["summary"]["max_waiting_requests"] == 3
        assert body["snapshots"][0]["snapshot_id"] == second.snapshot_id
        assert body["snapshots"][1]["snapshot_id"] == first.snapshot_id
        assert body["snapshots"][0]["raw_samples"][0]["name"] == "vllm:kv_cache_usage_perc"
        assert invalid_limit_response.status_code == 400
        assert invalid_provider_response.status_code == 400
    finally:
        _cleanup_provider_snapshots(migrated_database_url, PROVIDER_NAME)


def test_vllm_runtime_metric_snapshot_ui_shows_persisted_snapshots(
    migrated_database_url: str,
) -> None:
    _cleanup_provider_snapshots(migrated_database_url, PROVIDER_NAME)
    try:
        record_vllm_runtime_metric_snapshot(
            migrated_database_url,
            _snapshot(
                """
                vllm:kv_cache_usage_perc 0.83
                vllm:cpu_cache_usage_perc 0.12
                vllm:num_requests_running 2
                vllm:num_requests_waiting 3
                vllm:num_requests_swapped 1
                vllm:prompt_tokens_total 120
                vllm:generation_tokens_total 300
                vllm:time_to_first_token_seconds_sum 0.8
                vllm:time_to_first_token_seconds_count 2
                vllm:e2e_request_latency_seconds_sum 6
                vllm:e2e_request_latency_seconds_count 2
                """,
                sampled_at=NOW,
            ),
        )
        app = create_app(Settings(database_url=migrated_database_url))

        with TestClient(app) as client:
            page_response = client.get(
                "/admin/vllm-runtime-metrics",
                params={"provider_name": PROVIDER_NAME, "limit": "10"},
            )
            invalid_limit_response = client.get(
                "/admin/vllm-runtime-metrics",
                params={"limit": "0"},
            )

        assert page_response.status_code == 200
        assert "vLLM Runtime Metrics" in page_response.text
        assert "data-vllm-runtime-metrics-filters" in page_response.text
        assert "data-vllm-runtime-metrics-summary" in page_response.text
        assert "data-vllm-runtime-metrics-table" in page_response.text
        assert "data-vllm-runtime-metrics-json" in page_response.text
        assert PROVIDER_NAME in page_response.text
        assert "83.00%" in page_response.text
        assert "2026-07-29 22:00:00" in page_response.text
        assert "/api/admin/vllm-runtime-metrics/snapshots" in page_response.text
        assert invalid_limit_response.status_code == 200
        assert "limit must be greater than 0" in invalid_limit_response.text
    finally:
        _cleanup_provider_snapshots(migrated_database_url, PROVIDER_NAME)


def test_vllm_runtime_metric_snapshot_ui_reports_missing_database() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        page_response = client.get("/admin/vllm-runtime-metrics")

    assert page_response.status_code == 200
    assert "vLLM Runtime Metrics" in page_response.text
    assert "NEX_PCX_DATABASE_URL is not configured." in page_response.text


def _snapshot(metrics_text: str, *, sampled_at: datetime):
    return scrape_vllm_runtime_metrics_from_text(
        metrics_text,
        provider_name=PROVIDER_NAME,
        provider_base_url="http://192.168.20.243:12000/",
        model_id="/models/qwen",
        sampled_at=sampled_at,
        scrape_elapsed_ms=19,
    )


def _cleanup_provider_snapshots(database_url: str, provider_name: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM vllm_runtime_metric_snapshots WHERE provider_name = %s",
                (provider_name,),
            )
