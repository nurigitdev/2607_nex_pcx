from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from app.core.provider_resource_snapshots import (
    InvalidProviderResourceSnapshotError,
    ProviderResourceSnapshotRecord,
    format_provider_resource_bytes,
    format_provider_resource_duration,
    provider_resource_snapshot_record_payload,
    provider_resource_snapshot_summary_payload,
    summarize_provider_resource_snapshots,
    validate_provider_resource_snapshot_limit,
    validate_provider_resource_status,
    validate_provider_resource_type,
)

NOW = datetime(2026, 7, 29, 13, 0, tzinfo=UTC)
PROBE_RUN_ID = UUID("11111111-1111-4111-8111-111111111111")


def test_provider_resource_snapshot_record_payload_formats_operational_fields() -> None:
    record = _snapshot_record()

    payload = provider_resource_snapshot_record_payload(record)
    raw_payload = provider_resource_snapshot_record_payload(record, include_raw_snapshot=True)

    assert payload["snapshot_id"] == 7
    assert payload["badge_class"] == "warning"
    assert payload["reason_codes"] == ["swap_pressure"]
    assert payload["process_rss_label"] == "2.00 GiB"
    assert payload["gpu_memory_used_label"] == "8.00 GiB"
    assert payload["process_uptime_label"] == "1h 1m"
    assert payload["collected_at"] == NOW.isoformat()
    assert "raw_snapshot" not in payload
    assert raw_payload["raw_snapshot"]["provider_name"] == "qwen-primary"


def test_provider_resource_snapshot_summary_handles_empty_and_populated_records() -> None:
    empty = summarize_provider_resource_snapshots(())
    assert provider_resource_snapshot_summary_payload(empty) == {
        "snapshot_count": 0,
        "provider_names": [],
        "latest_snapshot_id": None,
        "latest_provider_name": None,
        "latest_status": None,
        "latest_status_badge_class": "secondary",
        "latest_collected_at": None,
        "ok_count": 0,
        "warning_count": 0,
        "critical_count": 0,
        "unknown_count": 0,
        "max_process_rss_bytes": None,
        "max_process_rss_label": "-",
        "max_gpu_memory_used_bytes": None,
        "max_gpu_memory_used_label": "-",
        "max_system_swap_used_percent": None,
    }

    first = _snapshot_record(snapshot_id=1, provider_name="a", status="ok")
    second = _snapshot_record(
        snapshot_id=2,
        provider_name="b",
        status="critical",
        collected_at=NOW + timedelta(minutes=1),
        process_rss_bytes=3 * 1024**3,
        gpu_memory_used_bytes=None,
        system_swap_used_percent=31.2,
    )

    summary = summarize_provider_resource_snapshots([first, second])
    payload = provider_resource_snapshot_summary_payload(summary)

    assert payload["snapshot_count"] == 2
    assert payload["provider_names"] == ["a", "b"]
    assert payload["latest_snapshot_id"] == 2
    assert payload["latest_provider_name"] == "b"
    assert payload["latest_status"] == "critical"
    assert payload["latest_status_badge_class"] == "danger"
    assert payload["ok_count"] == 1
    assert payload["critical_count"] == 1
    assert payload["max_process_rss_label"] == "3.00 GiB"
    assert payload["max_gpu_memory_used_label"] == "8.00 GiB"
    assert payload["max_system_swap_used_percent"] == 31.2


@pytest.mark.parametrize("limit", (0, -1, 501, True, 1.2))
def test_provider_resource_snapshot_limit_validation_rejects_bad_values(
    limit: int,
) -> None:
    with pytest.raises(InvalidProviderResourceSnapshotError):
        validate_provider_resource_snapshot_limit(limit)


def test_provider_resource_snapshot_filter_validation_rejects_unknown_values() -> None:
    assert validate_provider_resource_type(" VLLM ") == "vllm"
    assert validate_provider_resource_status(" WARNING ") == "warning"

    with pytest.raises(InvalidProviderResourceSnapshotError, match="provider_type"):
        validate_provider_resource_type("database")
    with pytest.raises(InvalidProviderResourceSnapshotError, match="status"):
        validate_provider_resource_status("degraded")


def test_provider_resource_format_helpers_cover_boundaries() -> None:
    assert format_provider_resource_bytes(None) == "-"
    assert format_provider_resource_bytes(0) == "0 B"
    assert format_provider_resource_bytes(2048) == "2.00 KiB"
    assert format_provider_resource_bytes(2 * 1024**4) == "2.00 TiB"
    assert format_provider_resource_duration(None) == "-"
    assert format_provider_resource_duration(42) == "42s"
    assert format_provider_resource_duration(125) == "2m 5s"
    assert format_provider_resource_duration(7200) == "2h 0m"
    assert format_provider_resource_duration(3 * 86400 + 3600) == "3d 1h"


def _snapshot_record(**overrides: object) -> ProviderResourceSnapshotRecord:
    values = {
        "snapshot_id": 7,
        "probe_run_id": PROBE_RUN_ID,
        "host": "192.168.20.243",
        "provider_name": "qwen-primary",
        "provider_type": "embedding",
        "model_id": "qwen3_embedding_4b",
        "port": 9103,
        "status": "warning",
        "reason_codes": ("swap_pressure",),
        "match_confidence": "port",
        "process_pid": 123,
        "process_ppid": 1,
        "process_user": "nexpcx",
        "process_rss_bytes": 2 * 1024**3,
        "process_vms_bytes": 3 * 1024**3,
        "process_cpu_percent": 7.5,
        "process_uptime_seconds": 3661,
        "process_command_preview": "python -m uvicorn",
        "process_command_hash": "hash",
        "listener_process_name": "python",
        "listener_raw_line": "LISTEN 0 2048 0.0.0.0:9103",
        "gpu_process_name": "python",
        "gpu_memory_used_bytes": 8 * 1024**3,
        "system_total_ram_bytes": 128 * 1024**3,
        "system_available_ram_bytes": 64 * 1024**3,
        "system_memory_available_percent": 50.0,
        "system_swap_total_bytes": 8 * 1024**3,
        "system_swap_used_bytes": 2 * 1024**3,
        "system_swap_used_percent": 25.0,
        "collector_error": None,
        "collector_errors": (),
        "report_status": "warning",
        "report_target_count": 1,
        "report_ok_count": 0,
        "report_warning_count": 1,
        "report_critical_count": 0,
        "report_unknown_count": 0,
        "runtime_metadata": {"source": "pytest"},
        "raw_snapshot": {"provider_name": "qwen-primary"},
        "collected_at": NOW,
        "created_at": NOW,
    }
    values.update(overrides)
    return ProviderResourceSnapshotRecord(**values)
