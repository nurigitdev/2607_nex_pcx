"""Persistence helpers for DGX provider resource snapshots."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from psycopg import Connection
from psycopg.types.json import Json

from app.core.database import connect
from app.core.provider_resource_probe import (
    PROVIDER_RESOURCE_STATUS_CRITICAL,
    PROVIDER_RESOURCE_STATUS_OK,
    PROVIDER_RESOURCE_STATUS_UNKNOWN,
    PROVIDER_RESOURCE_STATUS_WARNING,
    ProviderResourceProbeReport,
    provider_resource_probe_report_payload,
)

DEFAULT_PROVIDER_RESOURCE_SNAPSHOT_LIMIT = 50
MAX_PROVIDER_RESOURCE_SNAPSHOT_LIMIT = 500
PROVIDER_RESOURCE_TYPES = ("embedding", "reranker", "vllm")
PROVIDER_RESOURCE_STATUSES = (
    PROVIDER_RESOURCE_STATUS_OK,
    PROVIDER_RESOURCE_STATUS_WARNING,
    PROVIDER_RESOURCE_STATUS_CRITICAL,
    PROVIDER_RESOURCE_STATUS_UNKNOWN,
)
PROVIDER_RESOURCE_MATCH_CONFIDENCES = ("port", "command", "missing", "unknown")


@dataclass(frozen=True)
class ProviderResourceSnapshotRecord:
    snapshot_id: int
    probe_run_id: UUID
    host: str
    provider_name: str
    provider_type: str
    model_id: str | None
    port: int
    status: str
    reason_codes: tuple[str, ...]
    match_confidence: str
    process_pid: int | None
    process_ppid: int | None
    process_user: str | None
    process_rss_bytes: int | None
    process_vms_bytes: int | None
    process_cpu_percent: float | None
    process_uptime_seconds: int | None
    process_command_preview: str | None
    process_command_hash: str | None
    listener_process_name: str | None
    listener_raw_line: str | None
    gpu_process_name: str | None
    gpu_memory_used_bytes: int | None
    system_total_ram_bytes: int | None
    system_available_ram_bytes: int | None
    system_memory_available_percent: float | None
    system_swap_total_bytes: int | None
    system_swap_used_bytes: int | None
    system_swap_used_percent: float | None
    collector_error: str | None
    collector_errors: tuple[str, ...]
    report_status: str
    report_target_count: int
    report_ok_count: int
    report_warning_count: int
    report_critical_count: int
    report_unknown_count: int
    runtime_metadata: dict[str, Any]
    raw_snapshot: dict[str, Any]
    collected_at: datetime
    created_at: datetime


@dataclass(frozen=True)
class ProviderResourceSnapshotSummary:
    snapshot_count: int
    provider_names: tuple[str, ...]
    latest_snapshot_id: int | None
    latest_provider_name: str | None
    latest_status: str | None
    latest_collected_at: datetime | None
    ok_count: int
    warning_count: int
    critical_count: int
    unknown_count: int
    max_process_rss_bytes: int | None
    max_gpu_memory_used_bytes: int | None
    max_system_swap_used_percent: float | None


class InvalidProviderResourceSnapshotError(ValueError):
    """Raised when provider resource snapshot persistence inputs are invalid."""


def record_provider_resource_probe_report(
    database_url: str,
    report: ProviderResourceProbeReport,
    *,
    runtime_metadata: Mapping[str, Any] | None = None,
    probe_run_id: UUID | str | None = None,
) -> list[ProviderResourceSnapshotRecord]:
    return record_provider_resource_probe_payload(
        database_url,
        provider_resource_probe_report_payload(report),
        runtime_metadata=runtime_metadata,
        probe_run_id=probe_run_id,
    )


def record_provider_resource_probe_payload(
    database_url: str,
    report_payload: Mapping[str, Any],
    *,
    runtime_metadata: Mapping[str, Any] | None = None,
    probe_run_id: UUID | str | None = None,
) -> list[ProviderResourceSnapshotRecord]:
    with connect(database_url) as connection:
        return record_provider_resource_probe_payload_in_connection(
            connection,
            report_payload,
            runtime_metadata=runtime_metadata,
            probe_run_id=probe_run_id,
        )


def record_provider_resource_probe_payload_in_connection(
    connection: Connection,
    report_payload: Mapping[str, Any],
    *,
    runtime_metadata: Mapping[str, Any] | None = None,
    probe_run_id: UUID | str | None = None,
) -> list[ProviderResourceSnapshotRecord]:
    validated = _validate_report_payload(report_payload)
    normalized_probe_run_id = _normalize_probe_run_id(probe_run_id)
    merged_metadata = _normalize_runtime_metadata(runtime_metadata)
    merged_metadata.setdefault("contract_version", "provider_resource_snapshot_v1")
    rows = [
        _snapshot_insert_values(
            validated,
            snapshot,
            probe_run_id=normalized_probe_run_id,
            runtime_metadata=merged_metadata,
        )
        for snapshot in validated["snapshots"]
    ]
    if not rows:
        return []

    records: list[ProviderResourceSnapshotRecord] = []
    with connection.cursor() as cursor:
        for row in rows:
            cursor.execute(
                """
                INSERT INTO provider_resource_snapshots (
                    probe_run_id,
                    host,
                    provider_name,
                    provider_type,
                    model_id,
                    port,
                    status,
                    reason_codes,
                    match_confidence,
                    process_pid,
                    process_ppid,
                    process_user,
                    process_rss_bytes,
                    process_vms_bytes,
                    process_cpu_percent,
                    process_uptime_seconds,
                    process_command_preview,
                    process_command_hash,
                    listener_process_name,
                    listener_raw_line,
                    gpu_process_name,
                    gpu_memory_used_bytes,
                    system_total_ram_bytes,
                    system_available_ram_bytes,
                    system_memory_available_percent,
                    system_swap_total_bytes,
                    system_swap_used_bytes,
                    system_swap_used_percent,
                    collector_error,
                    collector_errors,
                    report_status,
                    report_target_count,
                    report_ok_count,
                    report_warning_count,
                    report_critical_count,
                    report_unknown_count,
                    runtime_metadata,
                    raw_snapshot,
                    collected_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING *
                """,
                row,
            )
            records.append(_row_to_snapshot_record(dict(cursor.fetchone())))
    return records


def list_provider_resource_snapshots(
    database_url: str,
    *,
    provider_name: str | None = None,
    provider_type: str | None = None,
    host: str | None = None,
    status: str | None = None,
    limit: int = DEFAULT_PROVIDER_RESOURCE_SNAPSHOT_LIMIT,
) -> list[ProviderResourceSnapshotRecord]:
    validate_provider_resource_snapshot_limit(limit)
    where_clauses: list[str] = []
    params: list[object] = []
    if provider_name is not None:
        where_clauses.append("provider_name = %s")
        params.append(_validate_nonblank(provider_name, "provider_name"))
    if provider_type is not None:
        where_clauses.append("provider_type = %s")
        params.append(validate_provider_resource_type(provider_type))
    if host is not None:
        where_clauses.append("host = %s")
        params.append(_validate_nonblank(host, "host"))
    if status is not None:
        where_clauses.append("status = %s")
        params.append(validate_provider_resource_status(status))

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM provider_resource_snapshots
                {where_sql}
                ORDER BY collected_at DESC, snapshot_id DESC
                LIMIT %s
                """,
                (*params, limit),
            )
            rows = cursor.fetchall()
    return [_row_to_snapshot_record(dict(row)) for row in rows]


def list_latest_provider_resource_snapshots(
    database_url: str,
    *,
    provider_type: str | None = None,
    host: str | None = None,
) -> list[ProviderResourceSnapshotRecord]:
    where_clauses: list[str] = []
    params: list[object] = []
    if provider_type is not None:
        where_clauses.append("provider_type = %s")
        params.append(validate_provider_resource_type(provider_type))
    if host is not None:
        where_clauses.append("host = %s")
        params.append(_validate_nonblank(host, "host"))
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT DISTINCT ON (provider_name) *
                FROM provider_resource_snapshots
                {where_sql}
                ORDER BY provider_name, collected_at DESC, snapshot_id DESC
                """,
                tuple(params),
            )
            rows = cursor.fetchall()
    records = [_row_to_snapshot_record(dict(row)) for row in rows]
    return sorted(
        records, key=lambda record: (record.collected_at, record.snapshot_id), reverse=True
    )


def summarize_provider_resource_snapshots(
    snapshots: list[ProviderResourceSnapshotRecord] | tuple[ProviderResourceSnapshotRecord, ...],
) -> ProviderResourceSnapshotSummary:
    if not snapshots:
        return ProviderResourceSnapshotSummary(
            snapshot_count=0,
            provider_names=(),
            latest_snapshot_id=None,
            latest_provider_name=None,
            latest_status=None,
            latest_collected_at=None,
            ok_count=0,
            warning_count=0,
            critical_count=0,
            unknown_count=0,
            max_process_rss_bytes=None,
            max_gpu_memory_used_bytes=None,
            max_system_swap_used_percent=None,
        )
    ordered = sorted(
        snapshots, key=lambda item: (item.collected_at, item.snapshot_id), reverse=True
    )
    latest = ordered[0]
    return ProviderResourceSnapshotSummary(
        snapshot_count=len(snapshots),
        provider_names=tuple(sorted({snapshot.provider_name for snapshot in snapshots})),
        latest_snapshot_id=latest.snapshot_id,
        latest_provider_name=latest.provider_name,
        latest_status=latest.status,
        latest_collected_at=latest.collected_at,
        ok_count=sum(1 for snapshot in snapshots if snapshot.status == PROVIDER_RESOURCE_STATUS_OK),
        warning_count=sum(
            1 for snapshot in snapshots if snapshot.status == PROVIDER_RESOURCE_STATUS_WARNING
        ),
        critical_count=sum(
            1 for snapshot in snapshots if snapshot.status == PROVIDER_RESOURCE_STATUS_CRITICAL
        ),
        unknown_count=sum(
            1 for snapshot in snapshots if snapshot.status == PROVIDER_RESOURCE_STATUS_UNKNOWN
        ),
        max_process_rss_bytes=_max_optional(snapshot.process_rss_bytes for snapshot in snapshots),
        max_gpu_memory_used_bytes=_max_optional(
            snapshot.gpu_memory_used_bytes for snapshot in snapshots
        ),
        max_system_swap_used_percent=_max_optional(
            snapshot.system_swap_used_percent for snapshot in snapshots
        ),
    )


def provider_resource_snapshot_record_payload(
    record: ProviderResourceSnapshotRecord,
    *,
    include_raw_snapshot: bool = False,
) -> dict[str, Any]:
    payload = {
        "snapshot_id": record.snapshot_id,
        "probe_run_id": str(record.probe_run_id),
        "host": record.host,
        "provider_name": record.provider_name,
        "provider_type": record.provider_type,
        "model_id": record.model_id,
        "port": record.port,
        "status": record.status,
        "badge_class": provider_resource_status_badge_class(record.status),
        "reason_codes": list(record.reason_codes),
        "match_confidence": record.match_confidence,
        "process_pid": record.process_pid,
        "process_ppid": record.process_ppid,
        "process_user": record.process_user,
        "process_rss_bytes": record.process_rss_bytes,
        "process_rss_label": format_provider_resource_bytes(record.process_rss_bytes),
        "process_vms_bytes": record.process_vms_bytes,
        "process_vms_label": format_provider_resource_bytes(record.process_vms_bytes),
        "process_cpu_percent": record.process_cpu_percent,
        "process_uptime_seconds": record.process_uptime_seconds,
        "process_uptime_label": format_provider_resource_duration(record.process_uptime_seconds),
        "process_command_preview": record.process_command_preview,
        "process_command_hash": record.process_command_hash,
        "listener_process_name": record.listener_process_name,
        "listener_raw_line": record.listener_raw_line,
        "gpu_process_name": record.gpu_process_name,
        "gpu_memory_used_bytes": record.gpu_memory_used_bytes,
        "gpu_memory_used_label": format_provider_resource_bytes(record.gpu_memory_used_bytes),
        "system_total_ram_bytes": record.system_total_ram_bytes,
        "system_total_ram_label": format_provider_resource_bytes(record.system_total_ram_bytes),
        "system_available_ram_bytes": record.system_available_ram_bytes,
        "system_available_ram_label": format_provider_resource_bytes(
            record.system_available_ram_bytes
        ),
        "system_memory_available_percent": record.system_memory_available_percent,
        "system_swap_total_bytes": record.system_swap_total_bytes,
        "system_swap_total_label": format_provider_resource_bytes(record.system_swap_total_bytes),
        "system_swap_used_bytes": record.system_swap_used_bytes,
        "system_swap_used_label": format_provider_resource_bytes(record.system_swap_used_bytes),
        "system_swap_used_percent": record.system_swap_used_percent,
        "collector_error": record.collector_error,
        "collector_errors": list(record.collector_errors),
        "report_status": record.report_status,
        "report_target_count": record.report_target_count,
        "report_ok_count": record.report_ok_count,
        "report_warning_count": record.report_warning_count,
        "report_critical_count": record.report_critical_count,
        "report_unknown_count": record.report_unknown_count,
        "runtime_metadata": dict(record.runtime_metadata),
        "collected_at": record.collected_at.isoformat(),
        "created_at": record.created_at.isoformat(),
    }
    if include_raw_snapshot:
        payload["raw_snapshot"] = dict(record.raw_snapshot)
    return payload


def provider_resource_snapshot_summary_payload(
    summary: ProviderResourceSnapshotSummary,
) -> dict[str, Any]:
    return {
        "snapshot_count": summary.snapshot_count,
        "provider_names": list(summary.provider_names),
        "latest_snapshot_id": summary.latest_snapshot_id,
        "latest_provider_name": summary.latest_provider_name,
        "latest_status": summary.latest_status,
        "latest_status_badge_class": provider_resource_status_badge_class(summary.latest_status),
        "latest_collected_at": (
            summary.latest_collected_at.isoformat() if summary.latest_collected_at else None
        ),
        "ok_count": summary.ok_count,
        "warning_count": summary.warning_count,
        "critical_count": summary.critical_count,
        "unknown_count": summary.unknown_count,
        "max_process_rss_bytes": summary.max_process_rss_bytes,
        "max_process_rss_label": format_provider_resource_bytes(summary.max_process_rss_bytes),
        "max_gpu_memory_used_bytes": summary.max_gpu_memory_used_bytes,
        "max_gpu_memory_used_label": format_provider_resource_bytes(
            summary.max_gpu_memory_used_bytes
        ),
        "max_system_swap_used_percent": summary.max_system_swap_used_percent,
    }


def validate_provider_resource_snapshot_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise InvalidProviderResourceSnapshotError("limit must be an integer")
    if limit <= 0:
        raise InvalidProviderResourceSnapshotError("limit must be greater than 0")
    if limit > MAX_PROVIDER_RESOURCE_SNAPSHOT_LIMIT:
        raise InvalidProviderResourceSnapshotError(
            f"limit must be less than or equal to {MAX_PROVIDER_RESOURCE_SNAPSHOT_LIMIT}"
        )


def validate_provider_resource_type(provider_type: str) -> str:
    normalized = provider_type.strip().lower()
    if normalized not in PROVIDER_RESOURCE_TYPES:
        raise InvalidProviderResourceSnapshotError(
            f"provider_type must be one of: {', '.join(PROVIDER_RESOURCE_TYPES)}"
        )
    return normalized


def validate_provider_resource_status(status: str) -> str:
    normalized = status.strip().lower()
    if normalized not in PROVIDER_RESOURCE_STATUSES:
        raise InvalidProviderResourceSnapshotError(
            f"status must be one of: {', '.join(PROVIDER_RESOURCE_STATUSES)}"
        )
    return normalized


def provider_resource_status_badge_class(status: str | None) -> str:
    return {
        PROVIDER_RESOURCE_STATUS_OK: "success",
        PROVIDER_RESOURCE_STATUS_WARNING: "warning",
        PROVIDER_RESOURCE_STATUS_CRITICAL: "danger",
        PROVIDER_RESOURCE_STATUS_UNKNOWN: "secondary",
    }.get(status or PROVIDER_RESOURCE_STATUS_UNKNOWN, "secondary")


def format_provider_resource_bytes(value: int | None) -> str:
    if value is None:
        return "-"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"


def format_provider_resource_duration(seconds: int | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"
    hours, remaining_minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {remaining_minutes}m"
    days, remaining_hours = divmod(hours, 24)
    return f"{days}d {remaining_hours}h"


def _validate_report_payload(report_payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(report_payload, Mapping):
        raise InvalidProviderResourceSnapshotError("report payload must be an object")
    snapshots = report_payload.get("snapshots")
    if not isinstance(snapshots, (list, tuple)):
        raise InvalidProviderResourceSnapshotError("report payload snapshots must be an array")
    validated = dict(report_payload)
    validated["snapshots"] = list(snapshots)
    return validated


def _snapshot_insert_values(
    report_payload: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    probe_run_id: UUID,
    runtime_metadata: Mapping[str, Any],
) -> tuple[object, ...]:
    if not isinstance(snapshot, Mapping):
        raise InvalidProviderResourceSnapshotError("snapshot must be an object")
    process = _optional_mapping(snapshot.get("process"))
    listener = _optional_mapping(snapshot.get("listener"))
    gpu_process = _optional_mapping(snapshot.get("gpu_process"))
    system_memory = _optional_mapping(snapshot.get("system_memory")) or {}
    return (
        probe_run_id,
        _validate_nonblank(_string_value(snapshot.get("host")), "host"),
        _validate_nonblank(_string_value(snapshot.get("provider_name")), "provider_name"),
        validate_provider_resource_type(_string_value(snapshot.get("provider_type"))),
        _optional_string(snapshot.get("model_id")),
        _validate_port(snapshot.get("port")),
        validate_provider_resource_status(_string_value(snapshot.get("status"))),
        Json(_string_array(snapshot.get("reason_codes"))),
        _validate_match_confidence(_string_value(snapshot.get("match_confidence"))),
        _optional_int(process.get("pid") if process else None),
        _optional_int(process.get("ppid") if process else None),
        _optional_string(process.get("user") if process else None),
        _optional_int(process.get("rss_bytes") if process else None),
        _optional_int(process.get("vms_bytes") if process else None),
        _optional_float(process.get("cpu_percent") if process else None),
        _optional_int(process.get("uptime_seconds") if process else None),
        _optional_string(process.get("command_preview") if process else None),
        _optional_string(process.get("command_hash") if process else None),
        _optional_string(listener.get("process_name") if listener else None),
        _optional_string(listener.get("raw_line") if listener else None),
        _optional_string(gpu_process.get("process_name") if gpu_process else None),
        _optional_int(gpu_process.get("used_gpu_memory_bytes") if gpu_process else None),
        _optional_int(system_memory.get("total_ram_bytes")),
        _optional_int(system_memory.get("available_ram_bytes")),
        _optional_float(system_memory.get("memory_available_percent")),
        _optional_int(system_memory.get("swap_total_bytes")),
        _optional_int(system_memory.get("swap_used_bytes")),
        _optional_float(system_memory.get("swap_used_percent")),
        _optional_string(snapshot.get("collector_error")),
        Json(_string_array(report_payload.get("collector_errors"))),
        validate_provider_resource_status(_string_value(report_payload.get("status"))),
        _non_negative_int(report_payload.get("target_count"), "target_count"),
        _non_negative_int(report_payload.get("ok_count"), "ok_count"),
        _non_negative_int(report_payload.get("warning_count"), "warning_count"),
        _non_negative_int(report_payload.get("critical_count"), "critical_count"),
        _non_negative_int(report_payload.get("unknown_count"), "unknown_count"),
        Json(dict(runtime_metadata)),
        Json(dict(snapshot)),
        _parse_datetime(snapshot.get("collected_at") or report_payload.get("collected_at")),
    )


def _row_to_snapshot_record(row: dict[str, Any]) -> ProviderResourceSnapshotRecord:
    return ProviderResourceSnapshotRecord(
        snapshot_id=int(row["snapshot_id"]),
        probe_run_id=row["probe_run_id"],
        host=str(row["host"]),
        provider_name=str(row["provider_name"]),
        provider_type=str(row["provider_type"]),
        model_id=row["model_id"],
        port=int(row["port"]),
        status=str(row["status"]),
        reason_codes=tuple(str(code) for code in row["reason_codes"]),
        match_confidence=str(row["match_confidence"]),
        process_pid=_optional_int(row["process_pid"]),
        process_ppid=_optional_int(row["process_ppid"]),
        process_user=row["process_user"],
        process_rss_bytes=_optional_int(row["process_rss_bytes"]),
        process_vms_bytes=_optional_int(row["process_vms_bytes"]),
        process_cpu_percent=_optional_float(row["process_cpu_percent"]),
        process_uptime_seconds=_optional_int(row["process_uptime_seconds"]),
        process_command_preview=row["process_command_preview"],
        process_command_hash=row["process_command_hash"],
        listener_process_name=row["listener_process_name"],
        listener_raw_line=row["listener_raw_line"],
        gpu_process_name=row["gpu_process_name"],
        gpu_memory_used_bytes=_optional_int(row["gpu_memory_used_bytes"]),
        system_total_ram_bytes=_optional_int(row["system_total_ram_bytes"]),
        system_available_ram_bytes=_optional_int(row["system_available_ram_bytes"]),
        system_memory_available_percent=_optional_float(row["system_memory_available_percent"]),
        system_swap_total_bytes=_optional_int(row["system_swap_total_bytes"]),
        system_swap_used_bytes=_optional_int(row["system_swap_used_bytes"]),
        system_swap_used_percent=_optional_float(row["system_swap_used_percent"]),
        collector_error=row["collector_error"],
        collector_errors=tuple(str(error) for error in row["collector_errors"]),
        report_status=str(row["report_status"]),
        report_target_count=int(row["report_target_count"]),
        report_ok_count=int(row["report_ok_count"]),
        report_warning_count=int(row["report_warning_count"]),
        report_critical_count=int(row["report_critical_count"]),
        report_unknown_count=int(row["report_unknown_count"]),
        runtime_metadata=dict(row["runtime_metadata"] or {}),
        raw_snapshot=dict(row["raw_snapshot"] or {}),
        collected_at=row["collected_at"],
        created_at=row["created_at"],
    )


def _normalize_probe_run_id(probe_run_id: UUID | str | None) -> UUID:
    if probe_run_id is None:
        return uuid4()
    if isinstance(probe_run_id, UUID):
        return probe_run_id
    try:
        return UUID(str(probe_run_id))
    except ValueError as exc:
        raise InvalidProviderResourceSnapshotError("probe_run_id must be a UUID") from exc


def _normalize_runtime_metadata(runtime_metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if runtime_metadata is None:
        return {}
    if not isinstance(runtime_metadata, Mapping):
        raise InvalidProviderResourceSnapshotError("runtime_metadata must be an object")
    return dict(runtime_metadata)


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidProviderResourceSnapshotError(f"{field_name} is required")
    return normalized


def _validate_port(value: object) -> int:
    port = _non_negative_int(value, "port")
    if port <= 0 or port > 65535:
        raise InvalidProviderResourceSnapshotError("port must be between 1 and 65535")
    return port


def _validate_match_confidence(match_confidence: str) -> str:
    normalized = match_confidence.strip().lower()
    if normalized not in PROVIDER_RESOURCE_MATCH_CONFIDENCES:
        raise InvalidProviderResourceSnapshotError(
            "match_confidence must be one of: " f"{', '.join(PROVIDER_RESOURCE_MATCH_CONFIDENCES)}"
        )
    return normalized


def _non_negative_int(value: object, field_name: str) -> int:
    result = _optional_int(value)
    if result is None:
        raise InvalidProviderResourceSnapshotError(f"{field_name} is required")
    if result < 0:
        raise InvalidProviderResourceSnapshotError(f"{field_name} must be non-negative")
    return result


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: object) -> float | None:
    return float(value) if value is not None else None


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _string_value(value: object) -> str:
    return "" if value is None else str(value)


def _string_array(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise InvalidProviderResourceSnapshotError("JSON array value is required")
    return [str(item) for item in value]


def _optional_mapping(value: object) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise InvalidProviderResourceSnapshotError("snapshot child value must be an object")
    return value


def _parse_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        selected = value
    elif isinstance(value, str) and value.strip():
        try:
            selected = datetime.fromisoformat(value)
        except ValueError as exc:
            raise InvalidProviderResourceSnapshotError(
                "collected_at must be an ISO datetime"
            ) from exc
    else:
        raise InvalidProviderResourceSnapshotError("collected_at is required")
    if selected.tzinfo is None:
        return selected.replace(tzinfo=UTC)
    return selected.astimezone(UTC)


def _max_optional(values: Iterable[Any]) -> Any:
    present = [value for value in values if value is not None]
    return max(present) if present else None
