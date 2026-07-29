"""Readiness assessment helpers for persisted vLLM runtime metric snapshots."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from app.core.vllm_runtime_metric_snapshots import VLLMRuntimeMetricSnapshotRecord

VLLM_RUNTIME_READINESS_OK = "ok"
VLLM_RUNTIME_READINESS_WARNING = "warning"
VLLM_RUNTIME_READINESS_CRITICAL = "critical"
VLLM_RUNTIME_READINESS_UNKNOWN = "unknown"

_STATUS_RANK = {
    VLLM_RUNTIME_READINESS_OK: 0,
    VLLM_RUNTIME_READINESS_UNKNOWN: 1,
    VLLM_RUNTIME_READINESS_WARNING: 2,
    VLLM_RUNTIME_READINESS_CRITICAL: 3,
}

_STATUS_BADGE_CLASS = {
    VLLM_RUNTIME_READINESS_OK: "success",
    VLLM_RUNTIME_READINESS_WARNING: "warning",
    VLLM_RUNTIME_READINESS_CRITICAL: "danger",
    VLLM_RUNTIME_READINESS_UNKNOWN: "secondary",
}


@dataclass(frozen=True)
class VLLMRuntimeReadinessThresholds:
    stale_snapshot_warning_minutes: float = 10.0
    stale_snapshot_critical_minutes: float = 30.0
    kv_cache_warning_percent: float = 80.0
    kv_cache_critical_percent: float = 90.0
    waiting_requests_warning: int = 1
    waiting_requests_critical: int = 5
    swapped_requests_warning: int = 1
    swapped_requests_critical: int = 3
    preemptions_warning_total: int = 1
    preemptions_critical_total: int = 10
    ttft_warning_seconds: float = 2.0
    ttft_critical_seconds: float = 5.0
    e2e_latency_warning_seconds: float = 30.0
    e2e_latency_critical_seconds: float = 60.0


DEFAULT_VLLM_RUNTIME_READINESS_THRESHOLDS = VLLMRuntimeReadinessThresholds()


@dataclass(frozen=True)
class VLLMRuntimeReadinessSignal:
    code: str
    status: str
    metric_label: str
    actual_value: float | int | None
    warning_threshold: float | int | None
    critical_threshold: float | int | None
    unit: str


@dataclass(frozen=True)
class VLLMRuntimeReadiness:
    status: str
    assessed_at: datetime
    latest_snapshot_id: int | None
    latest_provider_name: str | None
    latest_sampled_at: datetime | None
    reason_codes: tuple[str, ...]
    thresholds: VLLMRuntimeReadinessThresholds
    signals: tuple[VLLMRuntimeReadinessSignal, ...]


@dataclass(frozen=True)
class _ThresholdCheck:
    code: str
    metric_label: str
    value_getter: Callable[[VLLMRuntimeMetricSnapshotRecord], float | int | None]
    warning_threshold_getter: Callable[[VLLMRuntimeReadinessThresholds], float | int]
    critical_threshold_getter: Callable[[VLLMRuntimeReadinessThresholds], float | int]
    unit: str


_THRESHOLD_CHECKS = (
    _ThresholdCheck(
        code="kv_cache_pressure",
        metric_label="KV cache usage",
        value_getter=lambda snapshot: snapshot.kv_cache_usage_percent,
        warning_threshold_getter=lambda thresholds: thresholds.kv_cache_warning_percent,
        critical_threshold_getter=lambda thresholds: thresholds.kv_cache_critical_percent,
        unit="%",
    ),
    _ThresholdCheck(
        code="waiting_queue_pressure",
        metric_label="Waiting requests",
        value_getter=lambda snapshot: snapshot.waiting_requests,
        warning_threshold_getter=lambda thresholds: thresholds.waiting_requests_warning,
        critical_threshold_getter=lambda thresholds: thresholds.waiting_requests_critical,
        unit="requests",
    ),
    _ThresholdCheck(
        code="swapped_request_pressure",
        metric_label="Swapped requests",
        value_getter=lambda snapshot: snapshot.swapped_requests,
        warning_threshold_getter=lambda thresholds: thresholds.swapped_requests_warning,
        critical_threshold_getter=lambda thresholds: thresholds.swapped_requests_critical,
        unit="requests",
    ),
    _ThresholdCheck(
        code="preemption_pressure",
        metric_label="Preemptions total",
        value_getter=lambda snapshot: snapshot.num_preemptions_total,
        warning_threshold_getter=lambda thresholds: thresholds.preemptions_warning_total,
        critical_threshold_getter=lambda thresholds: thresholds.preemptions_critical_total,
        unit="count",
    ),
    _ThresholdCheck(
        code="ttft_latency",
        metric_label="Average TTFT",
        value_getter=lambda snapshot: snapshot.average_time_to_first_token_seconds,
        warning_threshold_getter=lambda thresholds: thresholds.ttft_warning_seconds,
        critical_threshold_getter=lambda thresholds: thresholds.ttft_critical_seconds,
        unit="seconds",
    ),
    _ThresholdCheck(
        code="e2e_latency",
        metric_label="Average E2E latency",
        value_getter=lambda snapshot: snapshot.average_e2e_request_latency_seconds,
        warning_threshold_getter=lambda thresholds: thresholds.e2e_latency_warning_seconds,
        critical_threshold_getter=lambda thresholds: thresholds.e2e_latency_critical_seconds,
        unit="seconds",
    ),
)


def assess_vllm_runtime_readiness(
    snapshots: Sequence[VLLMRuntimeMetricSnapshotRecord],
    *,
    thresholds: VLLMRuntimeReadinessThresholds = DEFAULT_VLLM_RUNTIME_READINESS_THRESHOLDS,
    now: datetime | None = None,
) -> VLLMRuntimeReadiness:
    assessed_at = _ensure_aware(now or datetime.now(UTC))
    if not snapshots:
        signal = VLLMRuntimeReadinessSignal(
            code="no_runtime_snapshot",
            status=VLLM_RUNTIME_READINESS_UNKNOWN,
            metric_label="Runtime snapshot",
            actual_value=None,
            warning_threshold=None,
            critical_threshold=None,
            unit="snapshot",
        )
        return VLLMRuntimeReadiness(
            status=VLLM_RUNTIME_READINESS_UNKNOWN,
            assessed_at=assessed_at,
            latest_snapshot_id=None,
            latest_provider_name=None,
            latest_sampled_at=None,
            reason_codes=(signal.code,),
            thresholds=thresholds,
            signals=(signal,),
        )

    latest = sorted(snapshots, key=lambda item: (item.sampled_at, item.snapshot_id), reverse=True)[
        0
    ]
    latest_sampled_at = _ensure_aware(latest.sampled_at)
    signals = [
        _snapshot_age_signal(latest_sampled_at, assessed_at, thresholds),
        _metric_presence_signal(latest),
    ]
    for check in _THRESHOLD_CHECKS:
        signal = _threshold_signal(latest, check, thresholds)
        if signal is not None:
            signals.append(signal)

    status = _worst_status(signal.status for signal in signals)
    reason_codes = tuple(
        signal.code for signal in signals if signal.status != VLLM_RUNTIME_READINESS_OK
    )
    return VLLMRuntimeReadiness(
        status=status,
        assessed_at=assessed_at,
        latest_snapshot_id=latest.snapshot_id,
        latest_provider_name=latest.provider_name,
        latest_sampled_at=latest_sampled_at,
        reason_codes=reason_codes,
        thresholds=thresholds,
        signals=tuple(signals),
    )


def vllm_runtime_readiness_payload(readiness: VLLMRuntimeReadiness) -> dict[str, Any]:
    return {
        "status": readiness.status,
        "badge_class": _STATUS_BADGE_CLASS.get(readiness.status, "secondary"),
        "badge_label": readiness.status.upper(),
        "assessed_at": readiness.assessed_at.isoformat(),
        "assessed_at_label": _datetime_label(readiness.assessed_at),
        "latest_snapshot_id": readiness.latest_snapshot_id,
        "latest_provider_name": readiness.latest_provider_name,
        "latest_sampled_at": (
            readiness.latest_sampled_at.isoformat() if readiness.latest_sampled_at else None
        ),
        "latest_sampled_at_label": (
            _datetime_label(readiness.latest_sampled_at) if readiness.latest_sampled_at else None
        ),
        "reason_codes": list(readiness.reason_codes),
        "thresholds": asdict(readiness.thresholds),
        "signals": [
            {
                "code": signal.code,
                "status": signal.status,
                "badge_class": _STATUS_BADGE_CLASS.get(signal.status, "secondary"),
                "metric_label": signal.metric_label,
                "actual_value": signal.actual_value,
                "warning_threshold": signal.warning_threshold,
                "critical_threshold": signal.critical_threshold,
                "unit": signal.unit,
            }
            for signal in readiness.signals
        ],
    }


def _snapshot_age_signal(
    sampled_at: datetime,
    assessed_at: datetime,
    thresholds: VLLMRuntimeReadinessThresholds,
) -> VLLMRuntimeReadinessSignal:
    age_minutes = max(0.0, round((assessed_at - sampled_at).total_seconds() / 60, 2))
    return VLLMRuntimeReadinessSignal(
        code="snapshot_stale",
        status=_threshold_status(
            age_minutes,
            thresholds.stale_snapshot_warning_minutes,
            thresholds.stale_snapshot_critical_minutes,
        ),
        metric_label="Snapshot age",
        actual_value=age_minutes,
        warning_threshold=thresholds.stale_snapshot_warning_minutes,
        critical_threshold=thresholds.stale_snapshot_critical_minutes,
        unit="minutes",
    )


def _metric_presence_signal(
    snapshot: VLLMRuntimeMetricSnapshotRecord,
) -> VLLMRuntimeReadinessSignal:
    status = (
        VLLM_RUNTIME_READINESS_CRITICAL
        if snapshot.vllm_metric_count <= 0
        else VLLM_RUNTIME_READINESS_OK
    )
    return VLLMRuntimeReadinessSignal(
        code="missing_vllm_metrics",
        status=status,
        metric_label="vLLM metric count",
        actual_value=snapshot.vllm_metric_count,
        warning_threshold=None,
        critical_threshold=1,
        unit="metrics",
    )


def _threshold_signal(
    snapshot: VLLMRuntimeMetricSnapshotRecord,
    check: _ThresholdCheck,
    thresholds: VLLMRuntimeReadinessThresholds,
) -> VLLMRuntimeReadinessSignal | None:
    actual_value = check.value_getter(snapshot)
    if actual_value is None:
        return None
    warning_threshold = check.warning_threshold_getter(thresholds)
    critical_threshold = check.critical_threshold_getter(thresholds)
    return VLLMRuntimeReadinessSignal(
        code=check.code,
        status=_threshold_status(actual_value, warning_threshold, critical_threshold),
        metric_label=check.metric_label,
        actual_value=actual_value,
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
        unit=check.unit,
    )


def _threshold_status(
    value: float | int,
    warning_threshold: float | int,
    critical_threshold: float | int,
) -> str:
    if value >= critical_threshold:
        return VLLM_RUNTIME_READINESS_CRITICAL
    if value >= warning_threshold:
        return VLLM_RUNTIME_READINESS_WARNING
    return VLLM_RUNTIME_READINESS_OK


def _worst_status(statuses: Iterable[str]) -> str:
    return max(statuses, key=lambda status: _STATUS_RANK.get(status, 0))


def _ensure_aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _datetime_label(value: datetime) -> str:
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S")
