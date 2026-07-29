from datetime import UTC, datetime

import httpx
import pytest

from app.core.vllm_runtime_metrics import (
    VLLM_RUNTIME_METRICS_CONTRACT_VERSION,
    InvalidVLLMRuntimeMetricsError,
    VLLMRuntimeMetricsClient,
    parse_prometheus_metrics_text,
    scrape_vllm_runtime_metrics_from_text,
    vllm_runtime_metrics_snapshot_payload,
)

NOW = datetime(2026, 7, 29, 9, 30, tzinfo=UTC)

SAMPLE_METRICS = """
# HELP vllm:kv_cache_usage_perc KV cache usage
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="/home/models/Qwen3.6-27B-NVFP4",worker="0"} 0.72
vllm:kv_cache_usage_perc{model_name="/home/models/Qwen3.6-27B-NVFP4",worker="1"} 0.81
vllm:cpu_cache_usage_perc 0.13
vllm:num_requests_running 2
vllm:num_requests_waiting 3
vllm:num_requests_waiting_by_reason{reason="capacity"} 2
vllm:num_requests_waiting_by_reason{reason="scheduler_backpressure"} 1
vllm:num_requests_swapped 1
vllm:request_success_total 12
vllm:prompt_tokens_total 1200
vllm:generation_tokens_total 3400
vllm:prompt_tokens_cached_total 500
vllm:prefix_cache_hits_total 8
vllm:prefix_cache_queries_total 10
vllm:num_preemptions_total 4
vllm:time_to_first_token_seconds_sum 2.4
vllm:time_to_first_token_seconds_count 4
vllm:inter_token_latency_seconds_sum 1.2
vllm:inter_token_latency_seconds_count 6
vllm:e2e_request_latency_seconds_sum 9
vllm:e2e_request_latency_seconds_count 3
vllm:request_queue_time_seconds_sum 1.5
vllm:request_queue_time_seconds_count 3
vllm:request_prefill_time_seconds_sum 4
vllm:request_prefill_time_seconds_count 2
vllm:request_decode_time_seconds_sum 5
vllm:request_decode_time_seconds_count 2
process_start_time_seconds 1777777777
"""


def test_parse_prometheus_metrics_text_handles_labels_and_escapes() -> None:
    samples = parse_prometheus_metrics_text(
        'vllm:test_metric{model_name="qwen, 27b",note="quote \\" ok",path="a\\\\b"} 1.5\n'
    )

    assert len(samples) == 1
    assert samples[0].name == "vllm:test_metric"
    assert samples[0].labels == {
        "model_name": "qwen, 27b",
        "note": 'quote " ok',
        "path": "a\\b",
    }
    assert samples[0].value == 1.5


def test_parse_prometheus_metrics_text_skips_nonfinite_values() -> None:
    samples = parse_prometheus_metrics_text("""
        vllm:bad_nan NaN
        vllm:bad_inf +Inf
        vllm:good 7
        """)

    assert [sample.name for sample in samples] == ["vllm:good"]


@pytest.mark.parametrize(
    "metrics_text",
    (
        "not a metric line",
        'vllm:test{label="unterminated} 1',
        "vllm:test{label=value} 1",
        'vllm:test{label="a",label="b"} 1',
    ),
)
def test_parse_prometheus_metrics_text_rejects_invalid_lines(metrics_text: str) -> None:
    with pytest.raises(InvalidVLLMRuntimeMetricsError):
        parse_prometheus_metrics_text(metrics_text)


def test_scrape_vllm_runtime_metrics_from_text_normalizes_runtime_snapshot() -> None:
    snapshot = scrape_vllm_runtime_metrics_from_text(
        SAMPLE_METRICS,
        provider_name=" dgx_vllm ",
        provider_base_url="http://192.168.20.243:12000/",
        model_id=" /models/qwen ",
        sampled_at=NOW,
        scrape_elapsed_ms=17,
    )
    payload = vllm_runtime_metrics_snapshot_payload(snapshot)

    assert snapshot.contract_version == VLLM_RUNTIME_METRICS_CONTRACT_VERSION
    assert payload["provider_name"] == "dgx_vllm"
    assert payload["provider_base_url"] == "http://192.168.20.243:12000"
    assert payload["model_id"] == "/models/qwen"
    assert payload["sampled_at"] == NOW.isoformat()
    assert payload["metric_count"] == 28
    assert payload["vllm_metric_count"] == 27
    assert payload["kv_cache_usage_ratio"] == 0.81
    assert payload["kv_cache_usage_percent"] == 81.0
    assert payload["cpu_cache_usage_percent"] == 13.0
    assert payload["running_requests"] == 2
    assert payload["waiting_requests"] == 3
    assert payload["waiting_requests_by_reason"] == {
        "capacity": 2,
        "scheduler_backpressure": 1,
    }
    assert payload["swapped_requests"] == 1
    assert payload["request_success_total"] == 12
    assert payload["prompt_tokens_total"] == 1200
    assert payload["generation_tokens_total"] == 3400
    assert payload["prompt_tokens_cached_total"] == 500
    assert payload["prefix_cache_hits_total"] == 8
    assert payload["prefix_cache_queries_total"] == 10
    assert payload["prefix_cache_hit_rate"] == 0.8
    assert payload["num_preemptions_total"] == 4
    assert payload["average_time_to_first_token_seconds"] == 0.6
    assert payload["average_inter_token_latency_seconds"] == 0.2
    assert payload["average_e2e_request_latency_seconds"] == 3.0
    assert payload["average_request_queue_time_seconds"] == 0.5
    assert payload["average_request_prefill_time_seconds"] == 2.0
    assert payload["average_request_decode_time_seconds"] == 2.5
    assert "vllm:kv_cache_usage_perc" in payload["metric_names"]


def test_scrape_vllm_runtime_metrics_from_text_supports_legacy_aliases() -> None:
    snapshot = scrape_vllm_runtime_metrics_from_text(
        """
        vllm:gpu_cache_usage_perc 0.5
        vllm:prompt_tokens 11
        vllm:generation_tokens 13
        vllm:prefix_cache_hits 2
        vllm:prefix_cache_queries 4
        vllm:time_per_output_token_seconds_sum 3
        vllm:time_per_output_token_seconds_count 2
        vllm:time_in_queue_requests_sum 6
        vllm:time_in_queue_requests_count 3
        """,
        provider_name="vllm",
        provider_base_url="http://vllm.local",
    )

    assert snapshot.kv_cache_usage_percent == 50.0
    assert snapshot.prompt_tokens_total == 11
    assert snapshot.generation_tokens_total == 13
    assert snapshot.prefix_cache_hit_rate == 0.5
    assert snapshot.average_inter_token_latency_seconds == 1.5
    assert snapshot.average_request_queue_time_seconds == 2.0


def test_vllm_runtime_metrics_snapshot_payload_can_include_raw_samples() -> None:
    snapshot = scrape_vllm_runtime_metrics_from_text(
        "vllm:num_requests_running 2\n",
        provider_name="vllm",
        provider_base_url="http://vllm.local",
    )
    payload = vllm_runtime_metrics_snapshot_payload(snapshot, include_raw_samples=True)

    assert payload["running_requests"] == 2
    assert payload["raw_samples"] == [
        {"name": "vllm:num_requests_running", "labels": {}, "value": 2.0}
    ]


def test_vllm_runtime_metrics_client_scrapes_metrics_endpoint() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["accept"] = request.headers.get("accept")
        return httpx.Response(200, text="vllm:num_requests_running 4\n")

    client = VLLMRuntimeMetricsClient(
        "http://vllm.local:12000/",
        provider_name="dgx",
        model_id="qwen",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    snapshot = client.scrape()

    assert captured == {
        "url": "http://vllm.local:12000/metrics",
        "accept": "text/plain",
    }
    assert snapshot.provider_name == "dgx"
    assert snapshot.model_id == "qwen"
    assert snapshot.running_requests == 4
    assert snapshot.scrape_elapsed_ms is not None


def test_vllm_runtime_metrics_client_reports_http_and_response_errors() -> None:
    error_client = VLLMRuntimeMetricsClient(
        "http://vllm.local",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(503, text="busy"))
        ),
    )
    with pytest.raises(InvalidVLLMRuntimeMetricsError, match="HTTP 503"):
        error_client.scrape()

    invalid_client = VLLMRuntimeMetricsClient(
        "http://vllm.local",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, text="bad line"))
        ),
    )
    with pytest.raises(InvalidVLLMRuntimeMetricsError, match="Invalid Prometheus"):
        invalid_client.scrape()

    with pytest.raises(InvalidVLLMRuntimeMetricsError, match="timeout"):
        VLLMRuntimeMetricsClient("http://vllm.local", timeout_seconds=0)
