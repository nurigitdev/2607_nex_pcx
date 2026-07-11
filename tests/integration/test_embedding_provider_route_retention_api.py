from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect, fetch_one
from app.core.embedding_provider_route_retention import (
    load_provider_route_retention_settings,
)
from app.core.embedding_provider_routes import (
    EmbeddingProviderRouteInput,
    upsert_embedding_provider_route,
)
from app.main import create_app

pytestmark = pytest.mark.integration


def _cleanup_route(database_url: str, provider_name: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM embedding_provider_routes WHERE provider_name = %s",
                (provider_name,),
            )


def _cleanup_preflight_runs(database_url: str, run_ids: list[int]) -> None:
    if not run_ids:
        return
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM embedding_provider_preflight_runs WHERE run_id = ANY(%s)",
                (run_ids,),
            )


def test_provider_route_retention_settings_seeded(
    migrated_database_url: str,
) -> None:
    setting_count = fetch_one(
        migrated_database_url,
        """
        SELECT count(*) AS count
        FROM app_log_settings
        WHERE setting_name IN (
            'provider_route_retention_enabled',
            'provider_route_retention_days',
            'provider_route_cleanup_batch_size'
        )
        """,
    )
    settings = load_provider_route_retention_settings(migrated_database_url)

    assert setting_count["count"] == 3
    assert settings.enabled is True
    assert settings.retention_days == 30
    assert settings.cleanup_batch_size == 1000


def test_provider_route_retention_api_previews_and_deletes_expired_operational_rows(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    provider_name = f"retention-provider-{suffix}"
    app = create_app(Settings(database_url=migrated_database_url))
    preflight_run_ids: list[int] = []

    try:
        route = upsert_embedding_provider_route(
            migrated_database_url,
            EmbeddingProviderRouteInput(
                profile_name="kure_v1_1024",
                provider_name=provider_name,
                provider_mode="mock",
                provider_base_url=None,
                runtime_metadata={"purpose": "retention-test"},
            ),
        )
        health_snapshot_ids, contract_snapshot_ids, preflight_run_ids = (
            _insert_provider_route_retention_rows(migrated_database_url, route.route_id)
        )

        with TestClient(app) as client:
            update_response = client.put(
                "/api/admin/embedding-provider-routes/retention-settings",
                json={
                    "enabled": True,
                    "retention_days": 3650,
                    "cleanup_batch_size": 1,
                },
            )
            get_response = client.get("/api/admin/embedding-provider-routes/retention-settings")
            invalid_response = client.put(
                "/api/admin/embedding-provider-routes/retention-settings",
                json={
                    "enabled": True,
                    "retention_days": 0,
                    "cleanup_batch_size": 1,
                },
            )
            preview_response = client.post(
                "/api/admin/embedding-provider-routes/cleanup",
                json={"dry_run": True},
            )
            cleanup_response = client.post(
                "/api/admin/embedding-provider-routes/cleanup",
                json={"dry_run": False},
            )

        remaining = _count_retention_fixture_rows(
            migrated_database_url,
            health_snapshot_ids,
            contract_snapshot_ids,
            preflight_run_ids,
        )

        assert update_response.status_code == 200
        assert update_response.json()["settings"] == {
            "enabled": True,
            "retention_days": 3650,
            "cleanup_batch_size": 1,
        }
        assert get_response.status_code == 200
        assert get_response.json()["settings"] == update_response.json()["settings"]
        assert invalid_response.status_code == 422
        assert preview_response.status_code == 200
        assert preview_response.json()["cleanup"]["dry_run"] is True
        assert preview_response.json()["cleanup"]["expired_health_snapshot_count"] >= 1
        assert preview_response.json()["cleanup"]["expired_contract_snapshot_count"] >= 1
        assert preview_response.json()["cleanup"]["expired_preflight_run_count"] >= 1
        assert preview_response.json()["cleanup"]["deleted_count"] == 0
        assert cleanup_response.status_code == 200
        assert cleanup_response.json()["cleanup"]["dry_run"] is False
        assert cleanup_response.json()["cleanup"]["deleted_health_snapshot_count"] == 1
        assert cleanup_response.json()["cleanup"]["deleted_contract_snapshot_count"] == 1
        assert cleanup_response.json()["cleanup"]["deleted_preflight_run_count"] == 1
        assert remaining == {
            "health_count": 1,
            "contract_count": 1,
            "preflight_count": 1,
        }
    finally:
        _cleanup_preflight_runs(migrated_database_url, preflight_run_ids)
        _cleanup_route(migrated_database_url, provider_name)
        with TestClient(app) as client:
            client.put(
                "/api/admin/embedding-provider-routes/retention-settings",
                json={
                    "enabled": True,
                    "retention_days": 30,
                    "cleanup_batch_size": 1000,
                },
            )


def _insert_provider_route_retention_rows(
    database_url: str,
    route_id: int,
) -> tuple[list[int], list[int], list[int]]:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO embedding_provider_route_health_snapshots (
                    route_id,
                    profile_name,
                    provider_name,
                    provider_mode,
                    checked,
                    ready,
                    status,
                    profile_names,
                    runtime_metadata,
                    validation_errors,
                    checked_at
                )
                VALUES
                    (
                        %s,
                        'kure_v1_1024',
                        'retention-provider',
                        'mock',
                        true,
                        true,
                        'ready',
                        '["kure_v1_1024"]'::jsonb,
                        '{}'::jsonb,
                        '[]'::jsonb,
                        now() - interval '4000 days'
                    ),
                    (
                        %s,
                        'kure_v1_1024',
                        'retention-provider',
                        'mock',
                        true,
                        true,
                        'ready',
                        '["kure_v1_1024"]'::jsonb,
                        '{}'::jsonb,
                        '[]'::jsonb,
                        now() - interval '1 day'
                    )
                RETURNING snapshot_id
                """,
                (route_id, route_id),
            )
            health_snapshot_ids = [int(row["snapshot_id"]) for row in cursor.fetchall()]

            cursor.execute(
                """
                INSERT INTO embedding_provider_route_contract_snapshots (
                    route_id,
                    profile_name,
                    provider_name,
                    provider_mode,
                    passed,
                    status,
                    elapsed_ms,
                    input_type,
                    sample_text_count,
                    runtime_metadata,
                    validation_errors,
                    checked_at
                )
                VALUES
                    (
                        %s,
                        'kure_v1_1024',
                        'retention-provider',
                        'mock',
                        true,
                        'passed',
                        10,
                        'document',
                        1,
                        '{}'::jsonb,
                        '[]'::jsonb,
                        now() - interval '4000 days'
                    ),
                    (
                        %s,
                        'kure_v1_1024',
                        'retention-provider',
                        'mock',
                        true,
                        'passed',
                        10,
                        'document',
                        1,
                        '{}'::jsonb,
                        '[]'::jsonb,
                        now() - interval '1 day'
                    )
                RETURNING snapshot_id
                """,
                (route_id, route_id),
            )
            contract_snapshot_ids = [int(row["snapshot_id"]) for row in cursor.fetchall()]

            cursor.execute(
                """
                INSERT INTO embedding_provider_preflight_runs (
                    trigger_source,
                    status,
                    result,
                    completed_at
                )
                VALUES
                    (
                        'manual_api',
                        'succeeded',
                        '{"route_count": 0, "passed_count": 0, "failed_count": 0}'::jsonb,
                        now() - interval '4000 days'
                    ),
                    (
                        'manual_api',
                        'succeeded',
                        '{"route_count": 0, "passed_count": 0, "failed_count": 0}'::jsonb,
                        now() - interval '1 day'
                    )
                RETURNING run_id
                """,
            )
            preflight_run_ids = [int(row["run_id"]) for row in cursor.fetchall()]

    return health_snapshot_ids, contract_snapshot_ids, preflight_run_ids


def _count_retention_fixture_rows(
    database_url: str,
    health_snapshot_ids: list[int],
    contract_snapshot_ids: list[int],
    preflight_run_ids: list[int],
) -> dict[str, int]:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT count(*) AS count
                FROM embedding_provider_route_health_snapshots
                WHERE snapshot_id = ANY(%s)
                """,
                (health_snapshot_ids,),
            )
            health_count = int(cursor.fetchone()["count"])
            cursor.execute(
                """
                SELECT count(*) AS count
                FROM embedding_provider_route_contract_snapshots
                WHERE snapshot_id = ANY(%s)
                """,
                (contract_snapshot_ids,),
            )
            contract_count = int(cursor.fetchone()["count"])
            cursor.execute(
                """
                SELECT count(*) AS count
                FROM embedding_provider_preflight_runs
                WHERE run_id = ANY(%s)
                """,
                (preflight_run_ids,),
            )
            preflight_count = int(cursor.fetchone()["count"])

    return {
        "health_count": health_count,
        "contract_count": contract_count,
        "preflight_count": preflight_count,
    }
