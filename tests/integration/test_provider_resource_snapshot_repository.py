from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from app.core.database import connect
from app.core.provider_resource_probe import (
    ProviderResourceTarget,
    build_provider_resource_probe_report,
    provider_resource_probe_report_payload,
)
from app.core.provider_resource_snapshots import (
    InvalidProviderResourceSnapshotError,
    list_latest_provider_resource_snapshots,
    list_provider_resource_snapshots,
    provider_resource_snapshot_record_payload,
    record_provider_resource_probe_payload,
    record_provider_resource_probe_report,
    summarize_provider_resource_snapshots,
)

pytestmark = pytest.mark.integration

PROVIDER_NAME = "pytest-provider-resource"
PROBE_RUN_ID = UUID("22222222-2222-4222-8222-222222222222")
NOW = datetime(2026, 7, 29, 14, 0, tzinfo=UTC)


def test_provider_resource_snapshot_repository_round_trips_probe_report(
    migrated_database_url: str,
) -> None:
    _cleanup_provider_snapshots(migrated_database_url, PROVIDER_NAME)
    try:
        report = _report(
            collected_at=NOW,
            rss_kib=2 * 1024 * 1024,
            gpu_mib=8192,
            swap_free_kib=6 * 1024 * 1024,
        )
        records = record_provider_resource_probe_report(
            migrated_database_url,
            report,
            runtime_metadata={"run": "first"},
            probe_run_id=PROBE_RUN_ID,
        )

        assert len(records) == 1
        record = records[0]
        assert record.probe_run_id == PROBE_RUN_ID
        assert record.provider_name == PROVIDER_NAME
        assert record.match_confidence == "port"
        assert record.process_pid == 7001
        assert record.process_rss_bytes == 2 * 1024 * 1024 * 1024
        assert record.gpu_memory_used_bytes == 8192 * 1024 * 1024
        assert record.runtime_metadata["contract_version"] == "provider_resource_snapshot_v1"
        assert record.runtime_metadata["run"] == "first"
        assert record.raw_snapshot["provider_name"] == PROVIDER_NAME

        latest_report = _report(
            collected_at=NOW + timedelta(minutes=1),
            rss_kib=4 * 1024 * 1024,
            gpu_mib=16384,
            swap_free_kib=4 * 1024 * 1024,
        )
        second_records = record_provider_resource_probe_payload(
            migrated_database_url,
            provider_resource_probe_report_payload(latest_report),
            runtime_metadata={"run": "second"},
        )
        provider_records = list_provider_resource_snapshots(
            migrated_database_url,
            provider_name=PROVIDER_NAME,
            limit=10,
        )
        latest_records = list_latest_provider_resource_snapshots(
            migrated_database_url,
            provider_type="embedding",
            host="192.168.20.243",
        )
        payload = provider_resource_snapshot_record_payload(
            provider_records[0],
            include_raw_snapshot=True,
        )
        summary = summarize_provider_resource_snapshots(provider_records)

        assert [row.snapshot_id for row in provider_records] == [
            second_records[0].snapshot_id,
            record.snapshot_id,
        ]
        assert any(row.provider_name == PROVIDER_NAME for row in latest_records)
        assert set(payload["raw_snapshot"]["reason_codes"]) == {
            "ram_budget_pressure",
            "swap_pressure",
        }
        assert payload["process_rss_label"] == "4.00 GiB"
        assert summary.snapshot_count == 2
        assert summary.max_gpu_memory_used_bytes == 16384 * 1024 * 1024
    finally:
        _cleanup_provider_snapshots(migrated_database_url, PROVIDER_NAME)


def test_provider_resource_snapshot_repository_validates_filters_and_payload(
    migrated_database_url: str,
) -> None:
    with pytest.raises(InvalidProviderResourceSnapshotError):
        list_provider_resource_snapshots(migrated_database_url, provider_name=" ")
    with pytest.raises(InvalidProviderResourceSnapshotError):
        list_provider_resource_snapshots(migrated_database_url, provider_type="database")
    with pytest.raises(InvalidProviderResourceSnapshotError):
        list_provider_resource_snapshots(migrated_database_url, status="bad")
    with pytest.raises(InvalidProviderResourceSnapshotError):
        record_provider_resource_probe_payload(migrated_database_url, {"snapshots": {}})


def _report(
    *,
    collected_at: datetime,
    rss_kib: int,
    gpu_mib: int,
    swap_free_kib: int,
):
    return build_provider_resource_probe_report(
        targets=(
            ProviderResourceTarget(
                provider_name=PROVIDER_NAME,
                provider_type="embedding",
                host="192.168.20.243",
                port=9103,
                process_match="embedding_provider_service",
                model_id="pytest-model",
                ram_warning_bytes=3 * 1024**3,
            ),
        ),
        ps_text=(
            f"7001 1 nexpcx {rss_kib} {rss_kib * 2} 9.5 3661 "
            "/home/nexpcx/2607_nex_pcx/.venv/bin/python -m uvicorn "
            "app.embedding_provider_service:app"
        ),
        ss_text='LISTEN 0 2048 0.0.0.0:9103 0.0.0.0:* users:(("python",pid=7001,fd=3))',
        nvidia_smi_text=f"7001, python, {gpu_mib}",
        meminfo_text=f"""
        MemTotal:       134217728 kB
        MemAvailable:   67108864 kB
        SwapTotal:       8388608 kB
        SwapFree:        {swap_free_kib} kB
        """,
        collected_at=collected_at,
    )


def _cleanup_provider_snapshots(database_url: str, provider_name: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM provider_resource_snapshots WHERE provider_name = %s",
                (provider_name,),
            )
