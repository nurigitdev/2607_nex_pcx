"""Persistence helpers for vLLM runtime metric snapshots."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from psycopg import Connection
from psycopg.types.json import Json

from app.core.database import connect
from app.core.vllm_runtime_metrics import (
    VLLMRuntimeMetricsSnapshot,
    vllm_runtime_metrics_snapshot_payload,
)

DEFAULT_VLLM_RUNTIME_METRIC_SNAPSHOT_LIMIT = 50
MAX_VLLM_RUNTIME_METRIC_SNAPSHOT_LIMIT = 500


@dataclass(frozen=True)
class VLLMRuntimeMetricSnapshotRecord:
    snapshot_id: int
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
    runtime_metadata: dict[str, Any]
    raw_samples: tuple[dict[str, Any], ...]
    created_at: datetime


@dataclass(frozen=True)
class VLLMRuntimeMetricSnapshotSummary:
    snapshot_count: int
    provider_names: tuple[str, ...]
    latest_snapshot_id: int | None
    latest_provider_name: str | None
    latest_sampled_at: datetime | None
    latest_kv_cache_usage_percent: float | None
    latest_waiting_requests: int | None
    max_kv_cache_usage_percent: float | None
    max_waiting_requests: int | None
    average_time_to_first_token_seconds: float | None
    average_e2e_request_latency_seconds: float | None


class InvalidVLLMRuntimeMetricSnapshotError(ValueError):
    """Raised when vLLM runtime metric snapshot persistence inputs are invalid."""


def record_vllm_runtime_metric_snapshot(
    database_url: str,
    snapshot: VLLMRuntimeMetricsSnapshot,
    *,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> VLLMRuntimeMetricSnapshotRecord:
    with connect(database_url) as connection:
        return record_vllm_runtime_metric_snapshot_in_connection(
            connection,
            snapshot,
            runtime_metadata=runtime_metadata,
        )


def record_vllm_runtime_metric_snapshot_in_connection(
    connection: Connection,
    snapshot: VLLMRuntimeMetricsSnapshot,
    *,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> VLLMRuntimeMetricSnapshotRecord:
    payload = vllm_runtime_metrics_snapshot_payload(snapshot, include_raw_samples=True)
    merged_metadata = _normalize_runtime_metadata(runtime_metadata)
    merged_metadata.setdefault("contract_version", snapshot.contract_version)
    raw_samples = payload.get("raw_samples", [])
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO vllm_runtime_metric_snapshots (
                provider_name,
                provider_base_url,
                model_id,
                sampled_at,
                scrape_elapsed_ms,
                raw_text_bytes,
                metric_count,
                vllm_metric_count,
                metric_names,
                kv_cache_usage_ratio,
                kv_cache_usage_percent,
                cpu_cache_usage_ratio,
                cpu_cache_usage_percent,
                running_requests,
                waiting_requests,
                swapped_requests,
                waiting_requests_by_reason,
                request_success_total,
                prompt_tokens_total,
                generation_tokens_total,
                prompt_tokens_cached_total,
                prefix_cache_hits_total,
                prefix_cache_queries_total,
                prefix_cache_hit_rate,
                num_preemptions_total,
                average_time_to_first_token_seconds,
                average_inter_token_latency_seconds,
                average_e2e_request_latency_seconds,
                average_request_queue_time_seconds,
                average_request_prefill_time_seconds,
                average_request_decode_time_seconds,
                runtime_metadata,
                raw_samples
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING *
            """,
            (
                snapshot.provider_name,
                snapshot.provider_base_url,
                snapshot.model_id,
                snapshot.sampled_at,
                snapshot.scrape_elapsed_ms,
                snapshot.raw_text_bytes,
                snapshot.metric_count,
                snapshot.vllm_metric_count,
                Json(list(snapshot.metric_names)),
                snapshot.kv_cache_usage_ratio,
                snapshot.kv_cache_usage_percent,
                snapshot.cpu_cache_usage_ratio,
                snapshot.cpu_cache_usage_percent,
                snapshot.running_requests,
                snapshot.waiting_requests,
                snapshot.swapped_requests,
                Json(dict(snapshot.waiting_requests_by_reason)),
                snapshot.request_success_total,
                snapshot.prompt_tokens_total,
                snapshot.generation_tokens_total,
                snapshot.prompt_tokens_cached_total,
                snapshot.prefix_cache_hits_total,
                snapshot.prefix_cache_queries_total,
                snapshot.prefix_cache_hit_rate,
                snapshot.num_preemptions_total,
                snapshot.average_time_to_first_token_seconds,
                snapshot.average_inter_token_latency_seconds,
                snapshot.average_e2e_request_latency_seconds,
                snapshot.average_request_queue_time_seconds,
                snapshot.average_request_prefill_time_seconds,
                snapshot.average_request_decode_time_seconds,
                Json(merged_metadata),
                Json(raw_samples),
            ),
        )
        return _row_to_snapshot_record(dict(cursor.fetchone()))


def list_vllm_runtime_metric_snapshots(
    database_url: str,
    *,
    provider_name: str | None = None,
    limit: int = DEFAULT_VLLM_RUNTIME_METRIC_SNAPSHOT_LIMIT,
) -> list[VLLMRuntimeMetricSnapshotRecord]:
    _validate_limit(limit)
    where_sql = ""
    params: list[object] = []
    if provider_name is not None:
        where_sql = "WHERE provider_name = %s"
        params.append(_validate_nonblank(provider_name, "provider_name"))
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM vllm_runtime_metric_snapshots
                {where_sql}
                ORDER BY sampled_at DESC, snapshot_id DESC
                LIMIT %s
                """,
                (*params, limit),
            )
            rows = cursor.fetchall()
    return [_row_to_snapshot_record(dict(row)) for row in rows]


def get_latest_vllm_runtime_metric_snapshot(
    database_url: str,
    *,
    provider_name: str | None = None,
) -> VLLMRuntimeMetricSnapshotRecord | None:
    snapshots = list_vllm_runtime_metric_snapshots(
        database_url,
        provider_name=provider_name,
        limit=1,
    )
    return snapshots[0] if snapshots else None


def summarize_vllm_runtime_metric_snapshots(
    snapshots: tuple[VLLMRuntimeMetricSnapshotRecord, ...] | list[VLLMRuntimeMetricSnapshotRecord],
) -> VLLMRuntimeMetricSnapshotSummary:
    if not snapshots:
        return VLLMRuntimeMetricSnapshotSummary(
            snapshot_count=0,
            provider_names=(),
            latest_snapshot_id=None,
            latest_provider_name=None,
            latest_sampled_at=None,
            latest_kv_cache_usage_percent=None,
            latest_waiting_requests=None,
            max_kv_cache_usage_percent=None,
            max_waiting_requests=None,
            average_time_to_first_token_seconds=None,
            average_e2e_request_latency_seconds=None,
        )
    ordered = sorted(snapshots, key=lambda item: (item.sampled_at, item.snapshot_id), reverse=True)
    latest = ordered[0]
    return VLLMRuntimeMetricSnapshotSummary(
        snapshot_count=len(snapshots),
        provider_names=tuple(sorted({snapshot.provider_name for snapshot in snapshots})),
        latest_snapshot_id=latest.snapshot_id,
        latest_provider_name=latest.provider_name,
        latest_sampled_at=latest.sampled_at,
        latest_kv_cache_usage_percent=latest.kv_cache_usage_percent,
        latest_waiting_requests=latest.waiting_requests,
        max_kv_cache_usage_percent=_max_optional(
            snapshot.kv_cache_usage_percent for snapshot in snapshots
        ),
        max_waiting_requests=_max_optional(snapshot.waiting_requests for snapshot in snapshots),
        average_time_to_first_token_seconds=_average_optional(
            snapshot.average_time_to_first_token_seconds for snapshot in snapshots
        ),
        average_e2e_request_latency_seconds=_average_optional(
            snapshot.average_e2e_request_latency_seconds for snapshot in snapshots
        ),
    )


def vllm_runtime_metric_snapshot_record_payload(
    record: VLLMRuntimeMetricSnapshotRecord,
    *,
    include_raw_samples: bool = False,
) -> dict[str, Any]:
    payload = {
        "snapshot_id": record.snapshot_id,
        "provider_name": record.provider_name,
        "provider_base_url": record.provider_base_url,
        "model_id": record.model_id,
        "sampled_at": record.sampled_at.isoformat(),
        "scrape_elapsed_ms": record.scrape_elapsed_ms,
        "raw_text_bytes": record.raw_text_bytes,
        "metric_count": record.metric_count,
        "vllm_metric_count": record.vllm_metric_count,
        "metric_names": list(record.metric_names),
        "kv_cache_usage_ratio": record.kv_cache_usage_ratio,
        "kv_cache_usage_percent": record.kv_cache_usage_percent,
        "cpu_cache_usage_ratio": record.cpu_cache_usage_ratio,
        "cpu_cache_usage_percent": record.cpu_cache_usage_percent,
        "running_requests": record.running_requests,
        "waiting_requests": record.waiting_requests,
        "swapped_requests": record.swapped_requests,
        "waiting_requests_by_reason": dict(record.waiting_requests_by_reason),
        "request_success_total": record.request_success_total,
        "prompt_tokens_total": record.prompt_tokens_total,
        "generation_tokens_total": record.generation_tokens_total,
        "prompt_tokens_cached_total": record.prompt_tokens_cached_total,
        "prefix_cache_hits_total": record.prefix_cache_hits_total,
        "prefix_cache_queries_total": record.prefix_cache_queries_total,
        "prefix_cache_hit_rate": record.prefix_cache_hit_rate,
        "num_preemptions_total": record.num_preemptions_total,
        "average_time_to_first_token_seconds": record.average_time_to_first_token_seconds,
        "average_inter_token_latency_seconds": record.average_inter_token_latency_seconds,
        "average_e2e_request_latency_seconds": record.average_e2e_request_latency_seconds,
        "average_request_queue_time_seconds": record.average_request_queue_time_seconds,
        "average_request_prefill_time_seconds": record.average_request_prefill_time_seconds,
        "average_request_decode_time_seconds": record.average_request_decode_time_seconds,
        "runtime_metadata": dict(record.runtime_metadata),
        "created_at": record.created_at.isoformat(),
    }
    if include_raw_samples:
        payload["raw_samples"] = [dict(sample) for sample in record.raw_samples]
    return payload


def vllm_runtime_metric_snapshot_summary_payload(
    summary: VLLMRuntimeMetricSnapshotSummary,
) -> dict[str, Any]:
    return {
        "snapshot_count": summary.snapshot_count,
        "provider_names": list(summary.provider_names),
        "latest_snapshot_id": summary.latest_snapshot_id,
        "latest_provider_name": summary.latest_provider_name,
        "latest_sampled_at": (
            summary.latest_sampled_at.isoformat() if summary.latest_sampled_at else None
        ),
        "latest_kv_cache_usage_percent": summary.latest_kv_cache_usage_percent,
        "latest_waiting_requests": summary.latest_waiting_requests,
        "max_kv_cache_usage_percent": summary.max_kv_cache_usage_percent,
        "max_waiting_requests": summary.max_waiting_requests,
        "average_time_to_first_token_seconds": summary.average_time_to_first_token_seconds,
        "average_e2e_request_latency_seconds": summary.average_e2e_request_latency_seconds,
    }


def validate_vllm_runtime_metric_snapshot_limit(limit: int) -> None:
    _validate_limit(limit)


def validate_vllm_runtime_metric_snapshot_metadata(
    runtime_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return _normalize_runtime_metadata(runtime_metadata)


def _validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise InvalidVLLMRuntimeMetricSnapshotError("limit must be an integer")
    if limit <= 0:
        raise InvalidVLLMRuntimeMetricSnapshotError("limit must be greater than 0")
    if limit > MAX_VLLM_RUNTIME_METRIC_SNAPSHOT_LIMIT:
        raise InvalidVLLMRuntimeMetricSnapshotError(
            f"limit must be less than or equal to {MAX_VLLM_RUNTIME_METRIC_SNAPSHOT_LIMIT}"
        )


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidVLLMRuntimeMetricSnapshotError(f"{field_name} is required")
    return normalized


def _normalize_runtime_metadata(
    runtime_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if runtime_metadata is None:
        return {}
    if not isinstance(runtime_metadata, Mapping):
        raise InvalidVLLMRuntimeMetricSnapshotError("runtime_metadata must be an object")
    return dict(runtime_metadata)


def _row_to_snapshot_record(row: dict[str, Any]) -> VLLMRuntimeMetricSnapshotRecord:
    return VLLMRuntimeMetricSnapshotRecord(
        snapshot_id=int(row["snapshot_id"]),
        provider_name=str(row["provider_name"]),
        provider_base_url=str(row["provider_base_url"]),
        model_id=row["model_id"],
        sampled_at=row["sampled_at"],
        scrape_elapsed_ms=_optional_int(row["scrape_elapsed_ms"]),
        raw_text_bytes=int(row["raw_text_bytes"]),
        metric_count=int(row["metric_count"]),
        vllm_metric_count=int(row["vllm_metric_count"]),
        metric_names=tuple(str(name) for name in row["metric_names"]),
        kv_cache_usage_ratio=_optional_float(row["kv_cache_usage_ratio"]),
        kv_cache_usage_percent=_optional_float(row["kv_cache_usage_percent"]),
        cpu_cache_usage_ratio=_optional_float(row["cpu_cache_usage_ratio"]),
        cpu_cache_usage_percent=_optional_float(row["cpu_cache_usage_percent"]),
        running_requests=_optional_int(row["running_requests"]),
        waiting_requests=_optional_int(row["waiting_requests"]),
        swapped_requests=_optional_int(row["swapped_requests"]),
        waiting_requests_by_reason={
            str(reason): int(count)
            for reason, count in dict(row["waiting_requests_by_reason"] or {}).items()
        },
        request_success_total=_optional_int(row["request_success_total"]),
        prompt_tokens_total=_optional_int(row["prompt_tokens_total"]),
        generation_tokens_total=_optional_int(row["generation_tokens_total"]),
        prompt_tokens_cached_total=_optional_int(row["prompt_tokens_cached_total"]),
        prefix_cache_hits_total=_optional_int(row["prefix_cache_hits_total"]),
        prefix_cache_queries_total=_optional_int(row["prefix_cache_queries_total"]),
        prefix_cache_hit_rate=_optional_float(row["prefix_cache_hit_rate"]),
        num_preemptions_total=_optional_int(row["num_preemptions_total"]),
        average_time_to_first_token_seconds=_optional_float(
            row["average_time_to_first_token_seconds"]
        ),
        average_inter_token_latency_seconds=_optional_float(
            row["average_inter_token_latency_seconds"]
        ),
        average_e2e_request_latency_seconds=_optional_float(
            row["average_e2e_request_latency_seconds"]
        ),
        average_request_queue_time_seconds=_optional_float(
            row["average_request_queue_time_seconds"]
        ),
        average_request_prefill_time_seconds=_optional_float(
            row["average_request_prefill_time_seconds"]
        ),
        average_request_decode_time_seconds=_optional_float(
            row["average_request_decode_time_seconds"]
        ),
        runtime_metadata=dict(row["runtime_metadata"] or {}),
        raw_samples=tuple(dict(sample) for sample in row["raw_samples"]),
        created_at=row["created_at"],
    )


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: object) -> float | None:
    return float(value) if value is not None else None


def _max_optional(values: Iterable[Any]) -> Any:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _average_optional(values: Iterable[Any]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return round(sum(present) / len(present), 6) if present else None
