from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect
from app.core.provider_resource_probe import (
    ProviderResourceTarget,
    build_provider_resource_probe_report,
)
from app.core.provider_resource_snapshots import record_provider_resource_probe_report
from app.main import create_app

pytestmark = pytest.mark.integration

PROVIDER_NAME = "pytest-provider-resource-api-ui"
NOW = datetime(2026, 7, 29, 15, 0, tzinfo=UTC)
DASHBOARD_NOW = datetime(2099, 1, 1, 15, 0, tzinfo=UTC)


def test_provider_resource_snapshot_api_lists_persisted_snapshots(
    migrated_database_url: str,
) -> None:
    _cleanup_provider_snapshots(migrated_database_url, PROVIDER_NAME)
    try:
        first = record_provider_resource_probe_report(
            migrated_database_url,
            _report(collected_at=NOW, rss_kib=2 * 1024 * 1024, gpu_mib=8192),
        )[0]
        second = record_provider_resource_probe_report(
            migrated_database_url,
            _report(collected_at=NOW.replace(minute=1), rss_kib=4 * 1024 * 1024, gpu_mib=16384),
        )[0]
        app = create_app(Settings(database_url=migrated_database_url))

        with TestClient(app) as client:
            response = client.get(
                "/api/admin/provider-resource-snapshots",
                params={
                    "provider_name": PROVIDER_NAME,
                    "provider_type": "embedding",
                    "host": "192.168.20.243",
                    "limit": "10",
                    "include_raw_snapshot": "true",
                },
            )
            invalid_limit_response = client.get(
                "/api/admin/provider-resource-snapshots",
                params={"limit": "0"},
            )
            invalid_status_response = client.get(
                "/api/admin/provider-resource-snapshots",
                params={"status": "bad"},
            )

        body = response.json()
        assert response.status_code == 200
        assert body["provider_name"] == PROVIDER_NAME
        assert body["provider_type"] == "embedding"
        assert body["summary"]["snapshot_count"] == 2
        assert body["summary"]["latest_snapshot_id"] == second.snapshot_id
        assert body["summary"]["max_process_rss_label"] == "4.00 GiB"
        assert body["summary"]["max_process_resident_memory_share_label"] == "3.12%"
        assert body["summary"]["max_gpu_memory_used_label"] == "16.00 GiB"
        assert body["snapshots"][0]["process_resident_memory_share_label"] == "3.12%"
        assert [snapshot["snapshot_id"] for snapshot in body["snapshots"]] == [
            second.snapshot_id,
            first.snapshot_id,
        ]
        assert body["snapshots"][0]["raw_snapshot"]["provider_name"] == PROVIDER_NAME
        assert invalid_limit_response.status_code == 400
        assert invalid_status_response.status_code == 400
    finally:
        _cleanup_provider_snapshots(migrated_database_url, PROVIDER_NAME)


def test_provider_resource_snapshot_ui_shows_persisted_snapshots(
    migrated_database_url: str,
) -> None:
    _cleanup_provider_snapshots(migrated_database_url, PROVIDER_NAME)
    try:
        record_provider_resource_probe_report(
            migrated_database_url,
            _report(
                collected_at=DASHBOARD_NOW,
                rss_kib=4 * 1024 * 1024,
                gpu_mib=16384,
            ),
        )
        app = create_app(Settings(database_url=migrated_database_url))

        with TestClient(app) as client:
            page_response = client.get(
                "/admin/provider-resources",
                params={"provider_name": PROVIDER_NAME, "limit": "10"},
            )
            invalid_limit_response = client.get(
                "/admin/provider-resources",
                params={"limit": "0"},
            )

        assert page_response.status_code == 200
        assert "Provider Resource Monitor" in page_response.text
        assert "data-provider-resource-filters" in page_response.text
        assert "data-provider-resource-summary" in page_response.text
        assert "data-provider-resource-table" in page_response.text
        assert "data-provider-resource-json" in page_response.text
        assert PROVIDER_NAME in page_response.text
        assert "4.00 GiB" in page_response.text
        assert "3.12%" in page_response.text
        assert "Resident Share" in page_response.text
        assert "16.00 GiB" in page_response.text
        assert "주의" in page_response.text
        assert "swap_pressure" in page_response.text
        assert "/api/admin/provider-resource-snapshots" in page_response.text
        assert invalid_limit_response.status_code == 200
        assert "limit must be greater than 0" in invalid_limit_response.text
    finally:
        _cleanup_provider_snapshots(migrated_database_url, PROVIDER_NAME)


def test_dashboard_provider_resource_card_summarizes_latest_snapshots(
    migrated_database_url: str,
) -> None:
    _cleanup_provider_snapshots(migrated_database_url, PROVIDER_NAME)
    try:
        record_provider_resource_probe_report(
            migrated_database_url,
            _report(
                collected_at=DASHBOARD_NOW,
                rss_kib=4 * 1024 * 1024,
                gpu_mib=16384,
            ),
        )
        app = create_app(Settings(database_url=migrated_database_url))

        with TestClient(app) as client:
            response = client.get("/")

        assert response.status_code == 200
        assert "data-dashboard-provider-resource-card" in response.text
        assert "Provider Resource 준비 상태" in response.text
        assert PROVIDER_NAME in response.text
        assert "2099-01-02 00:00:00" in response.text
        assert "4.00 GiB" in response.text
        assert "3.12%" in response.text
        assert "16.00 GiB" in response.text
        assert "50.00%" in response.text
        assert "위험" in response.text
        assert "ram_budget_pressure" in response.text
        assert "swap_pressure" in response.text
        assert "/admin/provider-resources" in response.text
        assert "/api/admin/provider-resource-snapshots?limit=10" in response.text
    finally:
        _cleanup_provider_snapshots(migrated_database_url, PROVIDER_NAME)


def test_provider_resource_snapshot_ui_reports_missing_database() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        page_response = client.get("/admin/provider-resources")
        api_response = client.get("/api/admin/provider-resource-snapshots")

    assert page_response.status_code == 200
    assert "Provider Resource Monitor" in page_response.text
    assert "NEX_PCX_DATABASE_URL is not configured." in page_response.text
    assert api_response.status_code == 503


def _report(*, collected_at: datetime, rss_kib: int, gpu_mib: int):
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
            f"7101 1 nexpcx {rss_kib} {rss_kib * 2} 9.5 3661 "
            "/home/nexpcx/2607_nex_pcx/.venv/bin/python -m uvicorn "
            "app.embedding_provider_service:app"
        ),
        ss_text='LISTEN 0 2048 0.0.0.0:9103 0.0.0.0:* users:(("python",pid=7101,fd=3))',
        nvidia_smi_text=f"7101, python, {gpu_mib}",
        meminfo_text="""
        MemTotal:       134217728 kB
        MemAvailable:   67108864 kB
        SwapTotal:       8388608 kB
        SwapFree:        4194304 kB
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
