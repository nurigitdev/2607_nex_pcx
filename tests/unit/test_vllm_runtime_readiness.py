from datetime import UTC, datetime, timedelta

from app.core.vllm_runtime_metric_snapshots import VLLMRuntimeMetricSnapshotRecord
from app.core.vllm_runtime_readiness import (
    VLLM_RUNTIME_READINESS_CRITICAL,
    VLLM_RUNTIME_READINESS_OK,
    VLLM_RUNTIME_READINESS_UNKNOWN,
    VLLM_RUNTIME_READINESS_WARNING,
    VLLMRuntimeReadinessThresholds,
    assess_vllm_runtime_readiness,
    vllm_runtime_readiness_payload,
)

NOW = datetime(2026, 7, 29, 10, 30, tzinfo=UTC)


def test_vllm_runtime_readiness_reports_unknown_without_snapshots() -> None:
    readiness = assess_vllm_runtime_readiness((), now=NOW)
    payload = vllm_runtime_readiness_payload(readiness)

    assert readiness.status == VLLM_RUNTIME_READINESS_UNKNOWN
    assert readiness.reason_codes == ("no_runtime_snapshot",)
    assert payload["badge_class"] == "secondary"
    assert payload["assessed_at_label"]
    assert payload["latest_snapshot_id"] is None
    assert payload["signals"][0]["code"] == "no_runtime_snapshot"


def test_vllm_runtime_readiness_reports_ok_for_fresh_low_pressure_snapshot() -> None:
    older = _snapshot_record(snapshot_id=1, sampled_at=NOW - timedelta(minutes=1))
    latest = _snapshot_record(snapshot_id=2, sampled_at=NOW, provider_name="dgx-b")

    readiness = assess_vllm_runtime_readiness((older, latest), now=NOW)
    payload = vllm_runtime_readiness_payload(readiness)

    assert readiness.status == VLLM_RUNTIME_READINESS_OK
    assert readiness.latest_snapshot_id == 2
    assert readiness.latest_provider_name == "dgx-b"
    assert readiness.reason_codes == ()
    assert payload["badge_class"] == "success"
    assert payload["reason_codes"] == []
    assert payload["latest_sampled_at_label"]
    assert {signal["code"] for signal in payload["signals"]} >= {
        "snapshot_stale",
        "missing_vllm_metrics",
        "kv_cache_pressure",
        "waiting_queue_pressure",
        "ttft_latency",
        "e2e_latency",
    }


def test_vllm_runtime_readiness_warns_for_pressure_thresholds_and_skips_missing_metrics() -> None:
    snapshot = _snapshot_record(
        kv_cache_usage_percent=85.0,
        waiting_requests=2,
        average_e2e_request_latency_seconds=None,
    )

    readiness = assess_vllm_runtime_readiness((snapshot,), now=NOW)
    payload = vllm_runtime_readiness_payload(readiness)

    assert readiness.status == VLLM_RUNTIME_READINESS_WARNING
    assert readiness.reason_codes == ("kv_cache_pressure", "waiting_queue_pressure")
    assert payload["badge_class"] == "warning"
    assert "e2e_latency" not in {signal["code"] for signal in payload["signals"]}


def test_vllm_runtime_readiness_reports_critical_for_stale_and_missing_metric_snapshot() -> None:
    snapshot = _snapshot_record(
        sampled_at=NOW - timedelta(minutes=31),
        vllm_metric_count=0,
        kv_cache_usage_percent=95.0,
        average_time_to_first_token_seconds=6.0,
    )

    readiness = assess_vllm_runtime_readiness((snapshot,), now=NOW)
    payload = vllm_runtime_readiness_payload(readiness)

    assert readiness.status == VLLM_RUNTIME_READINESS_CRITICAL
    assert payload["badge_class"] == "danger"
    assert set(readiness.reason_codes) >= {
        "snapshot_stale",
        "missing_vllm_metrics",
        "kv_cache_pressure",
        "ttft_latency",
    }


def test_vllm_runtime_readiness_applies_custom_thresholds_to_payload() -> None:
    thresholds = VLLMRuntimeReadinessThresholds(
        kv_cache_warning_percent=70.0,
        kv_cache_critical_percent=95.0,
    )
    snapshot = _snapshot_record(kv_cache_usage_percent=83.0)

    readiness = assess_vllm_runtime_readiness(
        (snapshot,),
        thresholds=thresholds,
        now=NOW,
    )
    payload = vllm_runtime_readiness_payload(readiness)

    kv_signal = next(
        signal for signal in payload["signals"] if signal["code"] == "kv_cache_pressure"
    )
    assert readiness.status == VLLM_RUNTIME_READINESS_WARNING
    assert kv_signal["warning_threshold"] == 70.0
    assert kv_signal["critical_threshold"] == 95.0
    assert payload["thresholds"]["kv_cache_warning_percent"] == 70.0


def _snapshot_record(**overrides: object) -> VLLMRuntimeMetricSnapshotRecord:
    values = {
        "snapshot_id": 7,
        "provider_name": "dgx-a",
        "provider_base_url": "http://192.168.20.243:12000",
        "model_id": "/models/qwen",
        "sampled_at": NOW,
        "scrape_elapsed_ms": 17,
        "raw_text_bytes": 32,
        "metric_count": 4,
        "vllm_metric_count": 4,
        "metric_names": (
            "vllm:kv_cache_usage_perc",
            "vllm:num_requests_waiting",
            "vllm:time_to_first_token_seconds_sum",
            "vllm:e2e_request_latency_seconds_sum",
        ),
        "kv_cache_usage_ratio": 0.5,
        "kv_cache_usage_percent": 50.0,
        "cpu_cache_usage_ratio": 0.1,
        "cpu_cache_usage_percent": 10.0,
        "running_requests": 1,
        "waiting_requests": 0,
        "swapped_requests": 0,
        "waiting_requests_by_reason": {},
        "request_success_total": 10,
        "prompt_tokens_total": 100,
        "generation_tokens_total": 200,
        "prompt_tokens_cached_total": 50,
        "prefix_cache_hits_total": 8,
        "prefix_cache_queries_total": 10,
        "prefix_cache_hit_rate": 0.8,
        "num_preemptions_total": 0,
        "average_time_to_first_token_seconds": 0.5,
        "average_inter_token_latency_seconds": 0.2,
        "average_e2e_request_latency_seconds": 3.0,
        "average_request_queue_time_seconds": 0.1,
        "average_request_prefill_time_seconds": 1.0,
        "average_request_decode_time_seconds": 1.5,
        "runtime_metadata": {"contract_version": "vllm_runtime_metrics_v1"},
        "raw_samples": (),
        "created_at": NOW,
    }
    values.update(overrides)
    return VLLMRuntimeMetricSnapshotRecord(**values)
