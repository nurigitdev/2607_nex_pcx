from datetime import UTC, datetime, timedelta

import pytest

from app.core.database import connect
from app.core.vllm_runtime_metric_snapshots import (
    InvalidVLLMRuntimeMetricSnapshotError,
    get_latest_vllm_runtime_metric_snapshot,
    list_vllm_runtime_metric_snapshots,
    record_vllm_runtime_metric_snapshot,
    summarize_vllm_runtime_metric_snapshots,
    vllm_runtime_metric_snapshot_record_payload,
    vllm_runtime_metric_snapshot_summary_payload,
)
from app.core.vllm_runtime_metrics import scrape_vllm_runtime_metrics_from_text

pytestmark = pytest.mark.integration

PROVIDER_NAME = "pytest-vllm-runtime-snapshot"
NOW = datetime(2026, 7, 29, 11, 0, tzinfo=UTC)


def test_vllm_runtime_metric_snapshot_repository_round_trip(
    migrated_database_url: str,
) -> None:
    _cleanup_provider_snapshots(migrated_database_url, PROVIDER_NAME)
    try:
        first = record_vllm_runtime_metric_snapshot(
            migrated_database_url,
            _snapshot(
                """
                vllm:kv_cache_usage_perc 0.40
                vllm:num_requests_waiting 1
                vllm:num_requests_waiting_by_reason{reason="capacity"} 1
                vllm:e2e_request_latency_seconds_sum 2
                vllm:e2e_request_latency_seconds_count 1
                """,
                sampled_at=NOW,
            ),
            runtime_metadata={"run": "first"},
        )
        second = record_vllm_runtime_metric_snapshot(
            migrated_database_url,
            _snapshot(
                """
                vllm:kv_cache_usage_perc 0.83
                vllm:num_requests_running 2
                vllm:num_requests_waiting 4
                vllm:num_requests_waiting_by_reason{reason="capacity"} 3
                vllm:num_requests_waiting_by_reason{reason="scheduler_backpressure"} 1
                vllm:e2e_request_latency_seconds_sum 8
                vllm:e2e_request_latency_seconds_count 2
                """,
                sampled_at=NOW + timedelta(minutes=1),
            ),
            runtime_metadata={"run": "second"},
        )

        assert first.snapshot_id < second.snapshot_id
        assert first.runtime_metadata["contract_version"] == "vllm_runtime_metrics_v1"
        assert first.runtime_metadata["run"] == "first"
        assert first.kv_cache_usage_percent == 40.0
        assert first.waiting_requests_by_reason == {"capacity": 1}
        assert first.raw_samples[0]["name"] == "vllm:kv_cache_usage_perc"

        provider_snapshots = list_vllm_runtime_metric_snapshots(
            migrated_database_url,
            provider_name=PROVIDER_NAME,
            limit=10,
        )
        latest = get_latest_vllm_runtime_metric_snapshot(
            migrated_database_url,
            provider_name=PROVIDER_NAME,
        )
        summary = summarize_vllm_runtime_metric_snapshots(provider_snapshots)
        summary_payload = vllm_runtime_metric_snapshot_summary_payload(summary)
        second_payload = vllm_runtime_metric_snapshot_record_payload(
            provider_snapshots[0],
            include_raw_samples=True,
        )

        assert [snapshot.snapshot_id for snapshot in provider_snapshots] == [
            second.snapshot_id,
            first.snapshot_id,
        ]
        assert latest is not None
        assert latest.snapshot_id == second.snapshot_id
        assert summary_payload["snapshot_count"] == 2
        assert summary_payload["latest_snapshot_id"] == second.snapshot_id
        assert summary_payload["max_kv_cache_usage_percent"] == 83.0
        assert summary_payload["max_waiting_requests"] == 4
        assert summary_payload["average_e2e_request_latency_seconds"] == 3.0
        assert second_payload["raw_samples"][0]["name"] == "vllm:kv_cache_usage_perc"
    finally:
        _cleanup_provider_snapshots(migrated_database_url, PROVIDER_NAME)


def test_vllm_runtime_metric_snapshot_repository_validates_filters(
    migrated_database_url: str,
) -> None:
    with pytest.raises(InvalidVLLMRuntimeMetricSnapshotError):
        list_vllm_runtime_metric_snapshots(migrated_database_url, provider_name=" ")
    with pytest.raises(InvalidVLLMRuntimeMetricSnapshotError):
        list_vllm_runtime_metric_snapshots(migrated_database_url, limit=0)


def _snapshot(metrics_text: str, *, sampled_at: datetime):
    return scrape_vllm_runtime_metrics_from_text(
        metrics_text,
        provider_name=PROVIDER_NAME,
        provider_base_url="http://192.168.20.243:12000/",
        model_id="/models/qwen",
        sampled_at=sampled_at,
        scrape_elapsed_ms=12,
    )


def _cleanup_provider_snapshots(database_url: str, provider_name: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM vllm_runtime_metric_snapshots WHERE provider_name = %s",
                (provider_name,),
            )
