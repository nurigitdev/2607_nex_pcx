from datetime import UTC, datetime, timedelta

import pytest

from app.core.vllm_runtime_metric_snapshots import (
    InvalidVLLMRuntimeMetricSnapshotError,
    VLLMRuntimeMetricSnapshotRecord,
    summarize_vllm_runtime_metric_snapshots,
    validate_vllm_runtime_metric_snapshot_limit,
    validate_vllm_runtime_metric_snapshot_metadata,
    vllm_runtime_metric_snapshot_record_payload,
    vllm_runtime_metric_snapshot_summary_payload,
)

NOW = datetime(2026, 7, 29, 10, 30, tzinfo=UTC)


def test_vllm_runtime_metric_snapshot_record_payload_hides_raw_samples_by_default() -> None:
    record = _snapshot_record()

    payload = vllm_runtime_metric_snapshot_record_payload(record)
    raw_payload = vllm_runtime_metric_snapshot_record_payload(record, include_raw_samples=True)

    assert payload["snapshot_id"] == 7
    assert payload["sampled_at"] == NOW.isoformat()
    assert payload["created_at"] == NOW.isoformat()
    assert payload["metric_names"] == ["vllm:kv_cache_usage_perc"]
    assert payload["waiting_requests_by_reason"] == {"capacity": 2}
    assert "raw_samples" not in payload
    assert raw_payload["raw_samples"] == [
        {"name": "vllm:kv_cache_usage_perc", "labels": {}, "value": 0.8}
    ]


def test_vllm_runtime_metric_snapshot_summary_handles_empty_and_populated_records() -> None:
    empty_summary = summarize_vllm_runtime_metric_snapshots(())
    assert vllm_runtime_metric_snapshot_summary_payload(empty_summary) == {
        "snapshot_count": 0,
        "provider_names": [],
        "latest_snapshot_id": None,
        "latest_provider_name": None,
        "latest_sampled_at": None,
        "latest_kv_cache_usage_percent": None,
        "latest_waiting_requests": None,
        "max_kv_cache_usage_percent": None,
        "max_waiting_requests": None,
        "average_time_to_first_token_seconds": None,
        "average_e2e_request_latency_seconds": None,
    }

    first = _snapshot_record(snapshot_id=1, sampled_at=NOW, kv_cache_usage_percent=None)
    second = _snapshot_record(
        snapshot_id=2,
        provider_name="dgx-b",
        sampled_at=NOW + timedelta(minutes=1),
        kv_cache_usage_percent=75.0,
        waiting_requests=3,
        average_time_to_first_token_seconds=0.3,
        average_e2e_request_latency_seconds=2.0,
    )

    summary = summarize_vllm_runtime_metric_snapshots([first, second])
    payload = vllm_runtime_metric_snapshot_summary_payload(summary)

    assert payload["snapshot_count"] == 2
    assert payload["provider_names"] == ["dgx-a", "dgx-b"]
    assert payload["latest_snapshot_id"] == 2
    assert payload["latest_provider_name"] == "dgx-b"
    assert payload["latest_kv_cache_usage_percent"] == 75.0
    assert payload["max_kv_cache_usage_percent"] == 75.0
    assert payload["max_waiting_requests"] == 3
    assert payload["average_time_to_first_token_seconds"] == 0.4
    assert payload["average_e2e_request_latency_seconds"] == 2.5


@pytest.mark.parametrize("limit", (0, -1, 501, True, 1.5))
def test_vllm_runtime_metric_snapshot_limit_validation_rejects_bad_values(
    limit: int,
) -> None:
    with pytest.raises(InvalidVLLMRuntimeMetricSnapshotError):
        validate_vllm_runtime_metric_snapshot_limit(limit)


def test_vllm_runtime_metric_snapshot_metadata_validation_normalizes_mappings() -> None:
    assert validate_vllm_runtime_metric_snapshot_metadata(None) == {}
    assert validate_vllm_runtime_metric_snapshot_metadata({"source": "pytest"}) == {
        "source": "pytest"
    }
    with pytest.raises(InvalidVLLMRuntimeMetricSnapshotError):
        validate_vllm_runtime_metric_snapshot_metadata(["not", "a", "mapping"])


def _snapshot_record(**overrides: object) -> VLLMRuntimeMetricSnapshotRecord:
    values = {
        "snapshot_id": 7,
        "provider_name": "dgx-a",
        "provider_base_url": "http://192.168.20.243:12000",
        "model_id": "/models/qwen",
        "sampled_at": NOW,
        "scrape_elapsed_ms": 17,
        "raw_text_bytes": 32,
        "metric_count": 1,
        "vllm_metric_count": 1,
        "metric_names": ("vllm:kv_cache_usage_perc",),
        "kv_cache_usage_ratio": 0.8,
        "kv_cache_usage_percent": 80.0,
        "cpu_cache_usage_ratio": 0.1,
        "cpu_cache_usage_percent": 10.0,
        "running_requests": 1,
        "waiting_requests": 2,
        "swapped_requests": 0,
        "waiting_requests_by_reason": {"capacity": 2},
        "request_success_total": 10,
        "prompt_tokens_total": 100,
        "generation_tokens_total": 200,
        "prompt_tokens_cached_total": 50,
        "prefix_cache_hits_total": 8,
        "prefix_cache_queries_total": 10,
        "prefix_cache_hit_rate": 0.8,
        "num_preemptions_total": 1,
        "average_time_to_first_token_seconds": 0.5,
        "average_inter_token_latency_seconds": 0.2,
        "average_e2e_request_latency_seconds": 3.0,
        "average_request_queue_time_seconds": 0.1,
        "average_request_prefill_time_seconds": 1.0,
        "average_request_decode_time_seconds": 1.5,
        "runtime_metadata": {"contract_version": "vllm_runtime_metrics_v1"},
        "raw_samples": ({"name": "vllm:kv_cache_usage_perc", "labels": {}, "value": 0.8},),
        "created_at": NOW,
    }
    values.update(overrides)
    return VLLMRuntimeMetricSnapshotRecord(**values)
