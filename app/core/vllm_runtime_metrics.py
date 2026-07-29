"""vLLM runtime metrics scraping and normalization helpers."""

import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from urllib.parse import urljoin

VLLM_RUNTIME_METRICS_CONTRACT_VERSION = "vllm_runtime_metrics_v1"
VLLM_METRICS_PATH = "/metrics"
DEFAULT_VLLM_RUNTIME_METRICS_TIMEOUT_SECONDS = 5.0

VLLM_KV_CACHE_USAGE_ALIASES = (
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
)
VLLM_CPU_CACHE_USAGE_ALIASES = ("vllm:cpu_cache_usage_perc",)
VLLM_RUNNING_REQUEST_ALIASES = ("vllm:num_requests_running",)
VLLM_WAITING_REQUEST_ALIASES = ("vllm:num_requests_waiting",)
VLLM_SWAPPED_REQUEST_ALIASES = ("vllm:num_requests_swapped",)
VLLM_REQUEST_SUCCESS_ALIASES = ("vllm:request_success", "vllm:request_success_total")
VLLM_PROMPT_TOKEN_ALIASES = ("vllm:prompt_tokens", "vllm:prompt_tokens_total")
VLLM_GENERATION_TOKEN_ALIASES = (
    "vllm:generation_tokens",
    "vllm:generation_tokens_total",
)
VLLM_PROMPT_TOKEN_CACHED_ALIASES = (
    "vllm:prompt_tokens_cached",
    "vllm:prompt_tokens_cached_total",
)
VLLM_PREFIX_CACHE_HIT_ALIASES = ("vllm:prefix_cache_hits", "vllm:prefix_cache_hits_total")
VLLM_PREFIX_CACHE_QUERY_ALIASES = (
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_queries_total",
)
VLLM_PREEMPTION_ALIASES = ("vllm:num_preemptions", "vllm:num_preemptions_total")

_METRIC_LINE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?"
    r"\s+"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?Inf|NaN)"
    r"(?:\s+\d+)?$"
)
_LABEL_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(\".*\")$")


@dataclass(frozen=True)
class PrometheusMetricSample:
    name: str
    labels: dict[str, str]
    value: float


@dataclass(frozen=True)
class VLLMRuntimeMetricsSnapshot:
    contract_version: str
    provider_name: str
    provider_base_url: str
    model_id: str | None
    sampled_at: datetime
    scrape_elapsed_ms: int | None
    raw_text_bytes: int
    metric_count: int
    vllm_metric_count: int
    metric_names: tuple[str, ...]
    kv_cache_usage_ratio: float | None
    kv_cache_usage_percent: float | None
    cpu_cache_usage_ratio: float | None
    cpu_cache_usage_percent: float | None
    running_requests: int | None
    waiting_requests: int | None
    swapped_requests: int | None
    waiting_requests_by_reason: dict[str, int]
    request_success_total: int | None
    prompt_tokens_total: int | None
    generation_tokens_total: int | None
    prompt_tokens_cached_total: int | None
    prefix_cache_hits_total: int | None
    prefix_cache_queries_total: int | None
    prefix_cache_hit_rate: float | None
    num_preemptions_total: int | None
    average_time_to_first_token_seconds: float | None
    average_inter_token_latency_seconds: float | None
    average_e2e_request_latency_seconds: float | None
    average_request_queue_time_seconds: float | None
    average_request_prefill_time_seconds: float | None
    average_request_decode_time_seconds: float | None
    raw_samples: tuple[PrometheusMetricSample, ...] = field(default_factory=tuple)


class InvalidVLLMRuntimeMetricsError(ValueError):
    """Raised when vLLM runtime metrics cannot be scraped or parsed."""


def parse_prometheus_metrics_text(metrics_text: str) -> tuple[PrometheusMetricSample, ...]:
    if not isinstance(metrics_text, str):
        raise InvalidVLLMRuntimeMetricsError("metrics_text must be a string")
    samples: list[PrometheusMetricSample] = []
    for line_number, raw_line in enumerate(metrics_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE_RE.match(line)
        if match is None:
            raise InvalidVLLMRuntimeMetricsError(
                f"Invalid Prometheus metric line at {line_number}: {raw_line}"
            )
        value = _parse_metric_value(match.group("value"))
        if not math.isfinite(value):
            continue
        samples.append(
            PrometheusMetricSample(
                name=match.group("name"),
                labels=_parse_labels(match.group("labels") or ""),
                value=value,
            )
        )
    return tuple(samples)


def scrape_vllm_runtime_metrics_from_text(
    metrics_text: str,
    *,
    provider_name: str,
    provider_base_url: str,
    model_id: str | None = None,
    sampled_at: datetime | None = None,
    scrape_elapsed_ms: int | None = None,
) -> VLLMRuntimeMetricsSnapshot:
    samples = parse_prometheus_metrics_text(metrics_text)
    return vllm_runtime_metrics_snapshot_from_samples(
        samples,
        provider_name=provider_name,
        provider_base_url=provider_base_url,
        model_id=model_id,
        sampled_at=sampled_at,
        scrape_elapsed_ms=scrape_elapsed_ms,
        raw_text_bytes=len(metrics_text.encode("utf-8")),
    )


def vllm_runtime_metrics_snapshot_from_samples(
    samples: tuple[PrometheusMetricSample, ...],
    *,
    provider_name: str,
    provider_base_url: str,
    model_id: str | None = None,
    sampled_at: datetime | None = None,
    scrape_elapsed_ms: int | None = None,
    raw_text_bytes: int = 0,
) -> VLLMRuntimeMetricsSnapshot:
    normalized_provider_name = _nonblank(provider_name, "provider_name")
    normalized_base_url = _nonblank(provider_base_url, "provider_base_url").rstrip("/")
    normalized_model_id = (
        model_id.strip() if isinstance(model_id, str) and model_id.strip() else None
    )
    normalized_elapsed_ms = _optional_non_negative_int(scrape_elapsed_ms, "scrape_elapsed_ms")
    samples_by_name = _samples_by_name(samples)
    metric_names = tuple(sorted(samples_by_name))
    kv_ratio = _max_metric(samples_by_name, VLLM_KV_CACHE_USAGE_ALIASES)
    cpu_ratio = _max_metric(samples_by_name, VLLM_CPU_CACHE_USAGE_ALIASES)
    prefix_cache_hits = _sum_metric_int(samples_by_name, VLLM_PREFIX_CACHE_HIT_ALIASES)
    prefix_cache_queries = _sum_metric_int(samples_by_name, VLLM_PREFIX_CACHE_QUERY_ALIASES)
    return VLLMRuntimeMetricsSnapshot(
        contract_version=VLLM_RUNTIME_METRICS_CONTRACT_VERSION,
        provider_name=normalized_provider_name,
        provider_base_url=normalized_base_url,
        model_id=normalized_model_id,
        sampled_at=sampled_at or datetime.now(UTC),
        scrape_elapsed_ms=normalized_elapsed_ms,
        raw_text_bytes=max(0, int(raw_text_bytes)),
        metric_count=len(samples),
        vllm_metric_count=sum(1 for sample in samples if sample.name.startswith("vllm:")),
        metric_names=metric_names,
        kv_cache_usage_ratio=_round_optional(kv_ratio),
        kv_cache_usage_percent=_percent(kv_ratio),
        cpu_cache_usage_ratio=_round_optional(cpu_ratio),
        cpu_cache_usage_percent=_percent(cpu_ratio),
        running_requests=_sum_metric_int(samples_by_name, VLLM_RUNNING_REQUEST_ALIASES),
        waiting_requests=_sum_metric_int(samples_by_name, VLLM_WAITING_REQUEST_ALIASES),
        swapped_requests=_sum_metric_int(samples_by_name, VLLM_SWAPPED_REQUEST_ALIASES),
        waiting_requests_by_reason=_waiting_requests_by_reason(samples_by_name),
        request_success_total=_sum_metric_int(samples_by_name, VLLM_REQUEST_SUCCESS_ALIASES),
        prompt_tokens_total=_sum_metric_int(samples_by_name, VLLM_PROMPT_TOKEN_ALIASES),
        generation_tokens_total=_sum_metric_int(samples_by_name, VLLM_GENERATION_TOKEN_ALIASES),
        prompt_tokens_cached_total=_sum_metric_int(
            samples_by_name,
            VLLM_PROMPT_TOKEN_CACHED_ALIASES,
        ),
        prefix_cache_hits_total=prefix_cache_hits,
        prefix_cache_queries_total=prefix_cache_queries,
        prefix_cache_hit_rate=_ratio(prefix_cache_hits, prefix_cache_queries),
        num_preemptions_total=_sum_metric_int(samples_by_name, VLLM_PREEMPTION_ALIASES),
        average_time_to_first_token_seconds=_histogram_average_seconds(
            samples_by_name,
            ("vllm:time_to_first_token_seconds",),
        ),
        average_inter_token_latency_seconds=_histogram_average_seconds(
            samples_by_name,
            (
                "vllm:inter_token_latency_seconds",
                "vllm:time_per_output_token_seconds",
            ),
        ),
        average_e2e_request_latency_seconds=_histogram_average_seconds(
            samples_by_name,
            ("vllm:e2e_request_latency_seconds",),
        ),
        average_request_queue_time_seconds=_histogram_average_seconds(
            samples_by_name,
            ("vllm:request_queue_time_seconds", "vllm:time_in_queue_requests"),
        ),
        average_request_prefill_time_seconds=_histogram_average_seconds(
            samples_by_name,
            ("vllm:request_prefill_time_seconds",),
        ),
        average_request_decode_time_seconds=_histogram_average_seconds(
            samples_by_name,
            ("vllm:request_decode_time_seconds",),
        ),
        raw_samples=samples,
    )


def vllm_runtime_metrics_snapshot_payload(
    snapshot: VLLMRuntimeMetricsSnapshot,
    *,
    include_raw_samples: bool = False,
) -> dict[str, Any]:
    payload = {
        "contract_version": snapshot.contract_version,
        "provider_name": snapshot.provider_name,
        "provider_base_url": snapshot.provider_base_url,
        "model_id": snapshot.model_id,
        "sampled_at": snapshot.sampled_at.isoformat(),
        "scrape_elapsed_ms": snapshot.scrape_elapsed_ms,
        "raw_text_bytes": snapshot.raw_text_bytes,
        "metric_count": snapshot.metric_count,
        "vllm_metric_count": snapshot.vllm_metric_count,
        "metric_names": list(snapshot.metric_names),
        "kv_cache_usage_ratio": snapshot.kv_cache_usage_ratio,
        "kv_cache_usage_percent": snapshot.kv_cache_usage_percent,
        "cpu_cache_usage_ratio": snapshot.cpu_cache_usage_ratio,
        "cpu_cache_usage_percent": snapshot.cpu_cache_usage_percent,
        "running_requests": snapshot.running_requests,
        "waiting_requests": snapshot.waiting_requests,
        "swapped_requests": snapshot.swapped_requests,
        "waiting_requests_by_reason": dict(snapshot.waiting_requests_by_reason),
        "request_success_total": snapshot.request_success_total,
        "prompt_tokens_total": snapshot.prompt_tokens_total,
        "generation_tokens_total": snapshot.generation_tokens_total,
        "prompt_tokens_cached_total": snapshot.prompt_tokens_cached_total,
        "prefix_cache_hits_total": snapshot.prefix_cache_hits_total,
        "prefix_cache_queries_total": snapshot.prefix_cache_queries_total,
        "prefix_cache_hit_rate": snapshot.prefix_cache_hit_rate,
        "num_preemptions_total": snapshot.num_preemptions_total,
        "average_time_to_first_token_seconds": snapshot.average_time_to_first_token_seconds,
        "average_inter_token_latency_seconds": snapshot.average_inter_token_latency_seconds,
        "average_e2e_request_latency_seconds": snapshot.average_e2e_request_latency_seconds,
        "average_request_queue_time_seconds": snapshot.average_request_queue_time_seconds,
        "average_request_prefill_time_seconds": snapshot.average_request_prefill_time_seconds,
        "average_request_decode_time_seconds": snapshot.average_request_decode_time_seconds,
    }
    if include_raw_samples:
        payload["raw_samples"] = [
            {"name": sample.name, "labels": dict(sample.labels), "value": sample.value}
            for sample in snapshot.raw_samples
        ]
    return payload


class VLLMRuntimeMetricsClient:
    """HTTP scraper for a vLLM OpenAI-compatible server's /metrics endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        provider_name: str = "vllm",
        model_id: str | None = None,
        timeout_seconds: float = DEFAULT_VLLM_RUNTIME_METRICS_TIMEOUT_SECONDS,
        http_client: object | None = None,
    ) -> None:
        normalized_base_url = _nonblank(base_url, "base_url").rstrip("/")
        if timeout_seconds <= 0:
            raise InvalidVLLMRuntimeMetricsError("timeout_seconds must be greater than 0")
        self.base_url = normalized_base_url
        self.provider_name = _nonblank(provider_name, "provider_name")
        self.model_id = model_id.strip() if isinstance(model_id, str) and model_id.strip() else None
        self.timeout_seconds = timeout_seconds
        self._owns_client = http_client is None
        self._client = http_client or _create_httpx_client(timeout_seconds)

    @property
    def metrics_url(self) -> str:
        return urljoin(f"{self.base_url}/", VLLM_METRICS_PATH.lstrip("/"))

    def scrape(self) -> VLLMRuntimeMetricsSnapshot:
        started = perf_counter()
        try:
            response = self._client.request(  # type: ignore[attr-defined]
                "GET",
                self.metrics_url,
                timeout=self.timeout_seconds,
                headers={"Accept": "text/plain"},
            )
        except Exception as exc:
            raise InvalidVLLMRuntimeMetricsError(
                f"vLLM runtime metrics request failed: {exc}"
            ) from exc
        elapsed_ms = int((perf_counter() - started) * 1000)
        status_code = getattr(response, "status_code", None)
        if status_code is None or int(status_code) >= 400:
            raise InvalidVLLMRuntimeMetricsError(
                f"vLLM runtime metrics endpoint returned HTTP {status_code}"
            )
        metrics_text = getattr(response, "text", None)
        if not isinstance(metrics_text, str):
            raise InvalidVLLMRuntimeMetricsError("vLLM runtime metrics response must be text")
        return scrape_vllm_runtime_metrics_from_text(
            metrics_text,
            provider_name=self.provider_name,
            provider_base_url=self.base_url,
            model_id=self.model_id,
            sampled_at=datetime.now(UTC),
            scrape_elapsed_ms=elapsed_ms,
        )

    def close(self) -> None:
        if self._owns_client and hasattr(self._client, "close"):
            self._client.close()


def _create_httpx_client(timeout_seconds: float) -> object:
    import httpx

    return httpx.Client(timeout=timeout_seconds)


def _parse_metric_value(value: str) -> float:
    if value == "+Inf" or value == "Inf":
        return math.inf
    if value == "-Inf":
        return -math.inf
    if value == "NaN":
        return math.nan
    return float(value)


def _parse_labels(label_text: str) -> dict[str, str]:
    if not label_text:
        return {}
    labels: dict[str, str] = {}
    for label_part in _split_label_parts(label_text):
        match = _LABEL_RE.match(label_part.strip())
        if match is None:
            raise InvalidVLLMRuntimeMetricsError(f"Invalid Prometheus label: {label_part}")
        label_name, raw_value = match.groups()
        if label_name in labels:
            raise InvalidVLLMRuntimeMetricsError(f"Duplicate Prometheus label: {label_name}")
        labels[label_name] = _unescape_label_value(raw_value[1:-1])
    return labels


def _split_label_parts(label_text: str) -> tuple[str, ...]:
    parts: list[str] = []
    current: list[str] = []
    in_quotes = False
    escaped = False
    for character in label_text:
        if escaped:
            current.append(character)
            escaped = False
            continue
        if character == "\\" and in_quotes:
            current.append(character)
            escaped = True
            continue
        if character == '"':
            current.append(character)
            in_quotes = not in_quotes
            continue
        if character == "," and not in_quotes:
            parts.append("".join(current))
            current = []
            continue
        current.append(character)
    if in_quotes:
        raise InvalidVLLMRuntimeMetricsError("Unterminated Prometheus label value")
    parts.append("".join(current))
    return tuple(part for part in parts if part.strip())


def _unescape_label_value(value: str) -> str:
    return value.replace(r"\\", "\\").replace(r"\"", '"').replace(r"\n", "\n")


def _nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidVLLMRuntimeMetricsError(f"{field_name} must not be blank")
    return normalized


def _optional_non_negative_int(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise InvalidVLLMRuntimeMetricsError(f"{field_name} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise InvalidVLLMRuntimeMetricsError(f"{field_name} must be greater than or equal to 0")
    return normalized


def _samples_by_name(
    samples: tuple[PrometheusMetricSample, ...],
) -> dict[str, tuple[PrometheusMetricSample, ...]]:
    grouped: dict[str, list[PrometheusMetricSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.name, []).append(sample)
    return {name: tuple(group_samples) for name, group_samples in grouped.items()}


def _samples_for(
    samples_by_name: dict[str, tuple[PrometheusMetricSample, ...]],
    names: tuple[str, ...],
) -> tuple[PrometheusMetricSample, ...]:
    selected: list[PrometheusMetricSample] = []
    for name in names:
        selected.extend(samples_by_name.get(name, ()))
    return tuple(selected)


def _max_metric(
    samples_by_name: dict[str, tuple[PrometheusMetricSample, ...]],
    names: tuple[str, ...],
) -> float | None:
    values = [sample.value for sample in _samples_for(samples_by_name, names)]
    return max(values) if values else None


def _sum_metric(
    samples_by_name: dict[str, tuple[PrometheusMetricSample, ...]],
    names: tuple[str, ...],
) -> float | None:
    values = [sample.value for sample in _samples_for(samples_by_name, names)]
    return sum(values) if values else None


def _sum_metric_int(
    samples_by_name: dict[str, tuple[PrometheusMetricSample, ...]],
    names: tuple[str, ...],
) -> int | None:
    value = _sum_metric(samples_by_name, names)
    if value is None:
        return None
    return max(0, int(round(value)))


def _waiting_requests_by_reason(
    samples_by_name: dict[str, tuple[PrometheusMetricSample, ...]],
) -> dict[str, int]:
    reasons: dict[str, int] = {}
    for sample in samples_by_name.get("vllm:num_requests_waiting_by_reason", ()):
        reason = sample.labels.get("reason", "unknown")
        reasons[reason] = reasons.get(reason, 0) + max(0, int(round(sample.value)))
    return dict(sorted(reasons.items()))


def _histogram_average_seconds(
    samples_by_name: dict[str, tuple[PrometheusMetricSample, ...]],
    base_names: tuple[str, ...],
) -> float | None:
    for base_name in base_names:
        sum_value = _sum_metric(samples_by_name, (f"{base_name}_sum",))
        count_value = _sum_metric(samples_by_name, (f"{base_name}_count",))
        if sum_value is not None and count_value is not None and count_value > 0:
            return round(sum_value / count_value, 6)
    return None


def _ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _percent(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value * 100, 2)


def _round_optional(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None
