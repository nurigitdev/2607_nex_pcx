from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from psycopg import errors

from app.core.config import Settings
from app.core.database import connect, fetch_one
from app.core.embedding_provider_contract_sample_sets import (
    EmbeddingProviderContractSampleSetInput,
    delete_embedding_provider_contract_sample_set,
    get_default_embedding_provider_contract_sample_set,
    get_embedding_provider_contract_sample_set,
    list_embedding_provider_contract_sample_sets,
    upsert_embedding_provider_contract_sample_set,
)
from app.core.embedding_provider_preflight_runs import (
    get_embedding_provider_preflight_run,
    list_embedding_provider_preflight_runs,
)
from app.core.embedding_provider_route_contract_snapshots import (
    InvalidEmbeddingProviderRouteContractSnapshotError,
    list_embedding_provider_route_contract_snapshots,
)
from app.core.embedding_provider_route_health import EmbeddingProviderRouteHealthResult
from app.core.embedding_provider_route_health_snapshots import (
    InvalidEmbeddingProviderRouteHealthSnapshotError,
    list_embedding_provider_route_health_snapshots,
)
from app.core.embedding_provider_route_readiness import (
    get_embedding_provider_route_readiness_summary,
)
from app.core.embedding_provider_routes import (
    EmbeddingProviderRouteInput,
    list_embedding_provider_routes,
    select_embedding_provider_route,
    upsert_embedding_provider_route,
)
from app.main import create_app, log_embedding_provider_route_health_alert

pytestmark = pytest.mark.integration


def _cleanup_routes(database_url: str, provider_names: list[str]) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM embedding_provider_routes WHERE provider_name = ANY(%s)",
                (provider_names,),
            )


def _cleanup_app_logs(database_url: str, correlation_ids: list[str]) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM app_logs WHERE correlation_id = ANY(%s)",
                (correlation_ids,),
            )


def _cleanup_contract_sample_sets(database_url: str, sample_set_names: list[str]) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM embedding_provider_contract_sample_sets
                WHERE sample_set_name = ANY(%s)
                """,
                (sample_set_names,),
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


def _create_embedding_profile(database_url: str, profile_name: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO embedding_profiles (
                    profile_name,
                    model_name,
                    dimension,
                    storage_type,
                    is_active
                )
                VALUES (%s, 'example/mock-provider-health', 1024, 'vector', true)
                """,
                (profile_name,),
            )


def _cleanup_embedding_profile(database_url: str, profile_name: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM embedding_profiles WHERE profile_name = %s",
                (profile_name,),
            )


def test_embedding_provider_routes_table_exists_and_enforces_remote_url(
    migrated_database_url: str,
) -> None:
    table_name = fetch_one(
        migrated_database_url,
        "SELECT to_regclass('public.embedding_provider_routes') AS table_name",
    )

    assert table_name["table_name"] == "embedding_provider_routes"
    snapshot_table_name = fetch_one(
        migrated_database_url,
        """
        SELECT to_regclass(
            'public.embedding_provider_route_health_snapshots'
        ) AS table_name
        """,
    )
    assert snapshot_table_name["table_name"] == "embedding_provider_route_health_snapshots"
    contract_snapshot_table_name = fetch_one(
        migrated_database_url,
        """
        SELECT to_regclass(
            'public.embedding_provider_route_contract_snapshots'
        ) AS table_name
        """,
    )
    assert contract_snapshot_table_name["table_name"] == (
        "embedding_provider_route_contract_snapshots"
    )
    sample_set_table_name = fetch_one(
        migrated_database_url,
        """
        SELECT to_regclass(
            'public.embedding_provider_contract_sample_sets'
        ) AS table_name
        """,
    )
    assert sample_set_table_name["table_name"] == "embedding_provider_contract_sample_sets"
    preflight_run_table_name = fetch_one(
        migrated_database_url,
        """
        SELECT to_regclass(
            'public.embedding_provider_preflight_runs'
        ) AS table_name
        """,
    )
    assert preflight_run_table_name["table_name"] == "embedding_provider_preflight_runs"

    with connect(migrated_database_url) as connection:
        with connection.cursor() as cursor:
            with pytest.raises(errors.CheckViolation):
                cursor.execute(
                    """
                    INSERT INTO embedding_provider_routes (
                        profile_name,
                        provider_name,
                        provider_mode,
                        provider_base_url
                    )
                    VALUES ('kure_v1_1024', %s, 'remote', NULL)
                    """,
                    (f"invalid-{uuid4()}",),
                )
            connection.rollback()


def test_embedding_provider_contract_sample_set_repository_and_api(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    sample_set_name = f"contract-samples-{suffix}"
    inactive_sample_set_name = f"inactive-contract-samples-{suffix}"
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        default_sample_set = get_default_embedding_provider_contract_sample_set(
            migrated_database_url
        )
        assert default_sample_set.sample_set_name == "default_route_contract"
        assert default_sample_set.input_type == "document"
        assert default_sample_set.sample_texts == (
            "NeX-PCX embedding provider contract check sample.",
        )

        with TestClient(app) as client:
            inactive_response = client.post(
                "/api/admin/embedding-provider-routes/contract-sample-sets",
                json={
                    "sample_set_name": inactive_sample_set_name,
                    "description": "Inactive integration sample set",
                    "input_type": "document",
                    "sample_texts": ["inactive sample"],
                    "is_active": False,
                    "is_default": False,
                },
            )
            assert inactive_response.status_code == 200
            inactive_sample_set = inactive_response.json()["sample_set"]
            assert inactive_sample_set["is_active"] is False
            assert inactive_sample_set["is_default"] is False

            create_response = client.post(
                "/api/admin/embedding-provider-routes/contract-sample-sets",
                json={
                    "sample_set_name": sample_set_name,
                    "description": "Integration sample set",
                    "input_type": "query",
                    "sample_texts": ["query sample one", "query sample two"],
                    "is_active": True,
                    "is_default": True,
                },
            )
            assert create_response.status_code == 200
            custom_sample_set = create_response.json()["sample_set"]
            assert custom_sample_set["is_default"] is True
            assert custom_sample_set["sample_texts"] == [
                "query sample one",
                "query sample two",
            ]
            assert (
                get_default_embedding_provider_contract_sample_set(
                    migrated_database_url
                ).sample_set_name
                == sample_set_name
            )

            get_response = client.get(
                f"/api/admin/embedding-provider-routes/contract-sample-sets/{sample_set_name}"
            )
            assert get_response.status_code == 200
            assert get_response.json()["sample_set"]["sample_set_name"] == sample_set_name

            update_response = client.put(
                f"/api/admin/embedding-provider-routes/contract-sample-sets/{sample_set_name}",
                json={
                    "sample_set_name": "ignored-by-path",
                    "description": "Updated integration sample set",
                    "input_type": "query",
                    "sample_texts": ["updated query sample"],
                    "is_active": True,
                    "is_default": True,
                },
            )
            assert update_response.status_code == 200
            updated_sample_set = update_response.json()["sample_set"]
            assert updated_sample_set["sample_set_name"] == sample_set_name
            assert updated_sample_set["description"] == "Updated integration sample set"
            assert updated_sample_set["sample_texts"] == ["updated query sample"]

            invalid_default_response = client.post(
                "/api/admin/embedding-provider-routes/contract-sample-sets",
                json={
                    "sample_set_name": f"invalid-default-{suffix}",
                    "input_type": "document",
                    "sample_texts": ["invalid"],
                    "is_active": False,
                    "is_default": True,
                },
            )
            assert invalid_default_response.status_code == 400
            assert (
                "Default contract sample set must be active"
                in invalid_default_response.json()["detail"]
            )

            unset_default_response = client.put(
                f"/api/admin/embedding-provider-routes/contract-sample-sets/{sample_set_name}",
                json={
                    "sample_set_name": sample_set_name,
                    "description": "Unset default sample set",
                    "input_type": "query",
                    "sample_texts": ["updated query sample"],
                    "is_active": True,
                    "is_default": False,
                },
            )
            assert unset_default_response.status_code == 400
            assert "cannot be unset directly" in unset_default_response.json()["detail"]

            delete_default_response = client.delete(
                f"/api/admin/embedding-provider-routes/contract-sample-sets/{sample_set_name}"
            )
            assert delete_default_response.status_code == 400
            assert "cannot be deleted" in delete_default_response.json()["detail"]

            deleted_response = client.delete(
                "/api/admin/embedding-provider-routes/contract-sample-sets/"
                f"{inactive_sample_set_name}"
            )
            assert deleted_response.status_code == 200
            assert (
                deleted_response.json()["deleted_sample_set"]["sample_set_name"]
                == inactive_sample_set_name
            )
            assert (
                get_embedding_provider_contract_sample_set(
                    migrated_database_url,
                    inactive_sample_set_name,
                )
                is None
            )

            missing_delete_response = client.delete(
                "/api/admin/embedding-provider-routes/contract-sample-sets/missing-sample-set"
            )
            assert missing_delete_response.status_code == 404

            response = client.get(
                "/api/admin/embedding-provider-routes/contract-sample-sets",
                params={"active_only": "true"},
            )

        assert any(
            sample_set.sample_set_name == sample_set_name
            for sample_set in list_embedding_provider_contract_sample_sets(
                migrated_database_url,
                active_only=True,
            )
        )
        assert (
            delete_embedding_provider_contract_sample_set(
                migrated_database_url,
                "missing-sample-set",
            )
            is None
        )
        assert response.status_code == 200
        body = response.json()
        assert body["default_sample_set"]["sample_set_name"] == sample_set_name
        assert body["default_sample_set"]["input_type"] == "query"
        assert body["default_sample_set"]["sample_text_count"] == 1
        assert all(
            sample_set["sample_set_name"] != inactive_sample_set_name
            for sample_set in body["sample_sets"]
        )
    finally:
        _cleanup_contract_sample_sets(
            migrated_database_url,
            [sample_set_name, inactive_sample_set_name],
        )
        upsert_embedding_provider_contract_sample_set(
            migrated_database_url,
            EmbeddingProviderContractSampleSetInput(
                sample_set_name="default_route_contract",
                description="Default provider route embedding contract check sample set",
                input_type="document",
                sample_texts=("NeX-PCX embedding provider contract check sample.",),
                is_default=True,
            ),
        )


def test_embedding_provider_route_repository_upserts_and_selects_best_route(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    slow_name = f"gpu-slow-{suffix}"
    fast_name = f"gpu-fast-{suffix}"
    provider_names = [slow_name, fast_name]
    try:
        slow = upsert_embedding_provider_route(
            migrated_database_url,
            EmbeddingProviderRouteInput(
                profile_name="kure_v1_1024",
                provider_name=slow_name,
                provider_base_url="http://slow-provider.local/",
                timeout_seconds=20.0,
                priority=20,
                runtime_metadata={"device": "cuda:1"},
            ),
        )
        fast = upsert_embedding_provider_route(
            migrated_database_url,
            EmbeddingProviderRouteInput(
                profile_name="kure_v1_1024",
                provider_name=fast_name,
                provider_base_url="http://fast-provider.local/",
                timeout_seconds=5.0,
                priority=5,
                runtime_metadata={"device": "cuda:0"},
            ),
        )

        routes = list_embedding_provider_routes(
            migrated_database_url,
            profile_name="kure_v1_1024",
            active_only=True,
        )
        selected = select_embedding_provider_route(migrated_database_url, "kure_v1_1024")

        assert slow.provider_base_url == "http://slow-provider.local"
        assert fast.timeout_seconds == 5.0
        assert fast.runtime_metadata == {"device": "cuda:0"}
        ordered_provider_names = [
            route.provider_name for route in routes if route.provider_name in provider_names
        ]
        assert ordered_provider_names == [
            fast_name,
            slow_name,
        ]
        assert selected is not None
        assert selected.provider_name == fast_name

        updated = upsert_embedding_provider_route(
            migrated_database_url,
            EmbeddingProviderRouteInput(
                profile_name="kure_v1_1024",
                provider_name=fast_name,
                provider_base_url="http://fast-provider-v2.local",
                timeout_seconds=3.0,
                priority=30,
                is_active=False,
            ),
        )

        assert updated.route_id == fast.route_id
        assert updated.provider_base_url == "http://fast-provider-v2.local"
        assert updated.is_active is False
    finally:
        _cleanup_routes(migrated_database_url, provider_names)


def test_embedding_provider_route_admin_api_and_page_manage_routes(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    api_provider_name = f"gpu-admin-api-{suffix}"
    form_provider_name = f"gpu-admin-form-{suffix}"
    provider_names = [api_provider_name, form_provider_name]
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            post_response = client.post(
                "/api/admin/embedding-provider-routes",
                json={
                    "profile_name": "kure_v1_1024",
                    "provider_name": api_provider_name,
                    "provider_mode": "remote",
                    "provider_base_url": "http://gpu-admin-api.local/",
                    "timeout_seconds": 9.0,
                    "priority": 9,
                    "is_active": True,
                    "health_check_enabled": True,
                    "runtime_metadata": {"pool": "admin"},
                },
            )
            assert post_response.status_code == 200
            route = post_response.json()["route"]
            assert route["provider_name"] == api_provider_name
            assert route["provider_base_url"] == "http://gpu-admin-api.local"
            assert route["runtime_metadata"] == {"pool": "admin"}

            list_response = client.get(
                "/api/admin/embedding-provider-routes",
                params={"profile_name": "kure_v1_1024", "active_only": "true"},
            )
            assert list_response.status_code == 200
            provider_names_from_api = [
                route["provider_name"] for route in list_response.json()["routes"]
            ]
            assert api_provider_name in provider_names_from_api

            form_response = client.post(
                "/admin/embedding-provider-routes",
                data={
                    "profile_name": "kure_v1_1024",
                    "provider_name": form_provider_name,
                    "provider_mode": "remote",
                    "provider_base_url": "http://gpu-admin-form.local/",
                    "timeout_seconds": "7",
                    "priority": "7",
                    "is_active": "true",
                    "health_check_enabled": "true",
                },
            )
            assert form_response.status_code == 200
            assert "Embedding provider route saved." in form_response.text
            assert form_provider_name in form_response.text

            page_response = client.get("/admin/embedding-provider-routes")
            assert page_response.status_code == 200
            assert "임베딩 Provider 라우팅" in page_response.text
            assert "Provider Preset 등록" in page_response.text
            assert "data-provider-preset-registration-panel" in page_response.text
            assert "/api/admin/embedding-provider-routes/presets/register" in (page_response.text)
            assert "data-provider-launch-plan-panel" in page_response.text
            assert "/api/admin/embedding-provider-routes/presets/launch-plan" in (
                page_response.text
            )
            assert api_provider_name in page_response.text
            assert form_provider_name in page_response.text
            assert "/api/admin/embedding-provider-routes" in page_response.text
            assert "/api/admin/embedding-provider-routes/presets" in page_response.text
            assert "data-provider-operations-summary-panel" in page_response.text
            assert "/api/admin/embedding-provider-routes/operations-summary" in page_response.text
            assert "data-route-readiness-panel" in page_response.text
            assert "/api/admin/embedding-provider-routes/readiness?active_only=true" in (
                page_response.text
            )
            assert "data-manual-health-button" in page_response.text
            assert (
                f"/api/admin/embedding-provider-routes/{route['route_id']}/health-check"
                in page_response.text
            )
            assert "data-contract-button" in page_response.text
            assert (
                f"/api/admin/embedding-provider-routes/{route['route_id']}/contract-check"
                in page_response.text
            )
            assert "data-route-preflight-button" in page_response.text
            assert "/api/admin/embedding-provider-routes/preflight" in page_response.text
            assert "data-preflight-schedule-panel" in page_response.text
            assert "/api/admin/embedding-provider-routes/preflight-schedules" in page_response.text
            assert "data-due-schedule-controls" in page_response.text
            assert "data-due-preview" in page_response.text
            assert "data-run-due" in page_response.text
            assert (
                "/api/admin/embedding-provider-routes/preflight-schedules/due" in page_response.text
            )
            assert (
                "/api/admin/embedding-provider-routes/preflight-schedules/run-due"
                in page_response.text
            )
            assert "data-preflight-run-history-panel" in page_response.text
            assert "data-preflight-run-detail-panel" in page_response.text
            assert "data-preflight-run-detail-routes-table" in page_response.text
            assert (
                "/api/admin/embedding-provider-routes/preflight-runs?limit=10" in page_response.text
            )
            assert "data-provider-route-retention-panel" in page_response.text
            assert "/api/admin/embedding-provider-routes/retention-settings" in page_response.text
            assert "/api/admin/embedding-provider-routes/cleanup" in page_response.text
            assert "data-route-health-history-panel" in page_response.text
            assert (
                "/api/admin/embedding-provider-routes/health-snapshots?limit=10"
                in page_response.text
            )
            assert "data-route-contract-history-panel" in page_response.text
            assert (
                "/api/admin/embedding-provider-routes/contract-snapshots?limit=10"
                in page_response.text
            )
            assert "data-route-alerts-panel" in page_response.text
            assert "/api/admin/embedding-provider-routes/alerts" in page_response.text
            assert "data-contract-sample-set-panel" in page_response.text
            assert "data-sample-set-form" in page_response.text
            assert "/api/admin/embedding-provider-routes/contract-sample-sets" in page_response.text
    finally:
        _cleanup_routes(migrated_database_url, provider_names)


def test_embedding_provider_route_preset_registration_api_creates_qwen_routes(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    provider_name = f"qwen-preset-ui-{suffix}"
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            presets_response = client.get("/api/admin/embedding-provider-routes/presets")
            assert presets_response.status_code == 200
            presets = presets_response.json()["presets"]
            assert {preset["preset_name"] for preset in presets} >= {"kure", "bge", "qwen"}

            register_response = client.post(
                "/api/admin/embedding-provider-routes/presets/register",
                json={
                    "preset_name": "qwen",
                    "provider_name": provider_name,
                    "host": "gpu-qwen.local",
                    "port": 19103,
                    "timeout_seconds": 42.0,
                    "priority": 12,
                    "is_active": True,
                    "health_check_enabled": True,
                    "runtime_metadata": {"operator": "integration"},
                },
            )
            assert register_response.status_code == 200
            body = register_response.json()
            assert body["registered_count"] == 2
            assert body["preflight"] is None
            assert [route["profile_name"] for route in body["routes"]] == [
                "qwen3_4b_1000",
                "qwen3_4b_2560",
            ]
            assert {route["provider_name"] for route in body["routes"]} == {provider_name}
            assert {route["provider_base_url"] for route in body["routes"]} == {
                "http://gpu-qwen.local:19103"
            }
            assert all(
                route["runtime_metadata"]["source"] == "preset_registration_ui"
                for route in body["routes"]
            )
            assert all(
                route["runtime_metadata"]["operator"] == "integration" for route in body["routes"]
            )

            list_response = client.get(
                "/api/admin/embedding-provider-routes",
                params={"active_only": "true"},
            )
            assert list_response.status_code == 200
            registered_routes = [
                route
                for route in list_response.json()["routes"]
                if route["provider_name"] == provider_name
            ]
            assert {route["profile_name"] for route in registered_routes} == {
                "qwen3_4b_1000",
                "qwen3_4b_2560",
            }
    finally:
        _cleanup_routes(migrated_database_url, [provider_name])


def test_embedding_provider_route_preset_registration_api_can_run_immediate_preflight(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    provider_name = f"kure-preset-preflight-{suffix}"
    created_run_ids: list[int] = []
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            register_response = client.post(
                "/api/admin/embedding-provider-routes/presets/register",
                json={
                    "preset_name": "kure",
                    "provider_name": provider_name,
                    "host": "127.0.0.1",
                    "port": 1,
                    "timeout_seconds": 0.001,
                    "priority": 13,
                    "is_active": True,
                    "health_check_enabled": True,
                    "run_preflight": True,
                },
            )
            assert register_response.status_code == 200
            body = register_response.json()
            preflight = body["preflight"]
            assert body["registered_count"] == 1
            assert preflight["route_count"] == 1
            assert preflight["passed_count"] == 0
            assert preflight["failed_count"] == 1
            assert preflight["trigger_source"] == "preset_registration"
            assert preflight["preflight_run"]["status"] == "failed"
            created_run_ids.append(preflight["preflight_run"]["run_id"])
            result = preflight["results"][0]
            assert result["route"]["provider_name"] == provider_name
            assert result["health"]["status"] == "unreachable"
            assert result["contract"]["status"] == "health_unreachable"

            detail_response = client.get(
                f"/api/admin/embedding-provider-routes/preflight-runs/{created_run_ids[0]}"
            )
            assert detail_response.status_code == 200
            assert detail_response.json()["run"]["result"]["trigger_source"] == (
                "preset_registration"
            )
    finally:
        _cleanup_routes(migrated_database_url, [provider_name])
        _cleanup_preflight_runs(migrated_database_url, created_run_ids)


def test_embedding_provider_route_health_api_summarizes_mock_routes(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    profile_name = f"route_health_profile_{suffix}"
    provider_name = f"mock-route-health-{suffix}"
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        _create_embedding_profile(migrated_database_url, profile_name)
        upsert_embedding_provider_route(
            migrated_database_url,
            EmbeddingProviderRouteInput(
                profile_name=profile_name,
                provider_name=provider_name,
                provider_mode="mock",
                provider_base_url=None,
                priority=3,
                runtime_metadata={"purpose": "api-health-test"},
            ),
        )

        with TestClient(app) as client:
            response = client.get(
                "/api/admin/embedding-provider-routes/health",
                params={"profile_name": profile_name},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["route_count"] == 1
        assert body["checked_count"] == 1
        assert body["ready_count"] == 1
        assert body["routes"][0]["checked"] is True
        assert body["routes"][0]["ready"] is True
        assert body["routes"][0]["status"] == "ready"
        assert body["routes"][0]["provider_type"] == "mock"
        assert body["routes"][0]["provider_model_id"] == "mock-provider"
        assert body["routes"][0]["route"]["profile_name"] == profile_name
        assert body["routes"][0]["route"]["provider_name"] == provider_name
        assert body["snapshot_count"] == 0
        assert body["snapshots"] == []
        assert (
            list_embedding_provider_route_health_snapshots(
                migrated_database_url,
                profile_name=profile_name,
            )
            == []
        )
    finally:
        _cleanup_routes(migrated_database_url, [provider_name])
        _cleanup_embedding_profile(migrated_database_url, profile_name)


def test_embedding_provider_route_health_api_persists_snapshots_when_requested(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    profile_name = f"route_health_snapshot_profile_{suffix}"
    provider_name = f"mock-route-health-snapshot-{suffix}"
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        _create_embedding_profile(migrated_database_url, profile_name)
        route = upsert_embedding_provider_route(
            migrated_database_url,
            EmbeddingProviderRouteInput(
                profile_name=profile_name,
                provider_name=provider_name,
                provider_mode="mock",
                provider_base_url=None,
                priority=3,
                runtime_metadata={"purpose": "api-health-snapshot-test"},
            ),
        )

        with TestClient(app) as client:
            response = client.get(
                "/api/admin/embedding-provider-routes/health",
                params={"profile_name": profile_name, "persist": "true"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["route_count"] == 1
        assert body["snapshot_count"] == 1
        assert body["snapshots"][0]["route_id"] == route.route_id
        assert body["snapshots"][0]["profile_name"] == profile_name
        assert body["snapshots"][0]["provider_name"] == provider_name
        assert body["snapshots"][0]["status"] == "ready"
        assert body["snapshots"][0]["ready"] is True
        assert body["snapshots"][0]["profile_names"] == [profile_name]
        assert body["snapshots"][0]["runtime_metadata"] == {"provider": "mock"}

        snapshots = list_embedding_provider_route_health_snapshots(
            migrated_database_url,
            profile_name=profile_name,
        )
        assert len(snapshots) == 1
        assert snapshots[0].snapshot_id == body["snapshots"][0]["snapshot_id"]
        assert snapshots[0].route_id == route.route_id
        assert snapshots[0].status == "ready"
        assert snapshots[0].provider_model_id == "mock-provider"
    finally:
        _cleanup_routes(migrated_database_url, [provider_name])
        _cleanup_embedding_profile(migrated_database_url, profile_name)


def test_embedding_provider_route_manual_health_check_api_persists_snapshot(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    profile_name = f"route_manual_health_profile_{suffix}"
    provider_name = f"mock-route-manual-health-{suffix}"
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        _create_embedding_profile(migrated_database_url, profile_name)
        route = upsert_embedding_provider_route(
            migrated_database_url,
            EmbeddingProviderRouteInput(
                profile_name=profile_name,
                provider_name=provider_name,
                provider_mode="mock",
                provider_base_url=None,
                priority=3,
                runtime_metadata={"purpose": "manual-health-test"},
            ),
        )

        with TestClient(app) as client:
            response = client.post(
                f"/api/admin/embedding-provider-routes/{route.route_id}/health-check",
            )
            history_response = client.get(
                "/api/admin/embedding-provider-routes/health-snapshots",
                params={"route_id": route.route_id, "limit": "5"},
            )

        assert response.status_code == 200
        assert history_response.status_code == 200
        body = response.json()
        history_body = history_response.json()
        assert body["route_health"]["status"] == "ready"
        assert body["route_health"]["route"]["route_id"] == route.route_id
        assert body["snapshot"]["route_id"] == route.route_id
        assert body["snapshot"]["status"] == "ready"
        assert body["snapshot"]["profile_names"] == [profile_name]
        assert history_body["snapshot_count"] == 1
        assert history_body["snapshots"][0]["snapshot_id"] == body["snapshot"]["snapshot_id"]

        snapshots = list_embedding_provider_route_health_snapshots(
            migrated_database_url,
            route_id=route.route_id,
        )
        assert len(snapshots) == 1
        assert snapshots[0].snapshot_id == body["snapshot"]["snapshot_id"]
    finally:
        _cleanup_routes(migrated_database_url, [provider_name])
        _cleanup_embedding_profile(migrated_database_url, profile_name)


def test_embedding_provider_route_contract_check_api_validates_mock_route(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    provider_name = f"mock-route-contract-{suffix}"
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        route = upsert_embedding_provider_route(
            migrated_database_url,
            EmbeddingProviderRouteInput(
                profile_name="kure_v1_1024",
                provider_name=provider_name,
                provider_mode="mock",
                provider_base_url=None,
                priority=4,
                runtime_metadata={"purpose": "contract-test"},
            ),
        )

        with TestClient(app) as client:
            response = client.post(
                f"/api/admin/embedding-provider-routes/{route.route_id}/contract-check",
            )
            history_response = client.get(
                "/api/admin/embedding-provider-routes/contract-snapshots",
                params={"route_id": route.route_id, "limit": "5"},
            )

        assert response.status_code == 200
        assert history_response.status_code == 200
        body = response.json()
        history_body = history_response.json()
        contract = body["contract"]
        snapshot = body["snapshot"]
        assert body["sample_set"]["sample_set_name"] == "default_route_contract"
        assert body["sample_set"]["sample_text_count"] == 1
        assert contract["passed"] is True
        assert contract["status"] == "passed"
        assert contract["route"]["route_id"] == route.route_id
        assert contract["route"]["provider_name"] == provider_name
        assert contract["health"]["status"] == "ready"
        assert contract["provider_type"] == "mock"
        assert contract["provider_model_id"] == "mock-provider"
        assert contract["expected_dimension"] == 1024
        assert contract["dimension"] == 1024
        assert contract["input_count"] == 1
        assert contract["runtime_metadata"]["contract_sample_set_name"] == "default_route_contract"
        assert contract["validation_errors"] == []
        assert snapshot["route_id"] == route.route_id
        assert snapshot["profile_name"] == "kure_v1_1024"
        assert snapshot["provider_name"] == provider_name
        assert snapshot["passed"] is True
        assert snapshot["status"] == "passed"
        assert snapshot["provider_model_id"] == "mock-provider"
        assert snapshot["expected_dimension"] == 1024
        assert snapshot["dimension"] == 1024
        assert snapshot["input_count"] == 1
        assert history_body["snapshot_count"] == 1
        assert history_body["snapshots"][0]["snapshot_id"] == snapshot["snapshot_id"]

        snapshots = list_embedding_provider_route_contract_snapshots(
            migrated_database_url,
            route_id=route.route_id,
        )
        assert len(snapshots) == 1
        assert snapshots[0].snapshot_id == snapshot["snapshot_id"]
        assert snapshots[0].passed is True
        assert snapshots[0].status == "passed"
    finally:
        _cleanup_routes(migrated_database_url, [provider_name])


def test_embedding_provider_route_contract_check_api_logs_failed_contract(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    provider_name = f"mock-route-contract-fail-{suffix}"
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        route = upsert_embedding_provider_route(
            migrated_database_url,
            EmbeddingProviderRouteInput(
                profile_name="kure_v1_1024",
                provider_name=provider_name,
                provider_mode="remote",
                provider_base_url="http://127.0.0.1:9",
                timeout_seconds=0.1,
                priority=4,
                runtime_metadata={"purpose": "contract-alert-test"},
            ),
        )
        correlation_id = f"embedding-provider-route:{route.route_id}:contract"
        _cleanup_app_logs(migrated_database_url, [correlation_id])

        with TestClient(app) as client:
            response = client.post(
                f"/api/admin/embedding-provider-routes/{route.route_id}/contract-check",
            )

        assert response.status_code == 200
        body = response.json()
        contract = body["contract"]
        snapshot = body["snapshot"]
        assert contract["passed"] is False
        assert contract["status"] == "health_unreachable"
        assert snapshot["route_id"] == route.route_id
        assert snapshot["passed"] is False
        assert snapshot["status"] == "health_unreachable"
        assert "Remote provider request failed" in snapshot["error_message"]

        log_row = fetch_one(
            migrated_database_url,
            """
            SELECT level, event_type, message, detail
            FROM app_logs
            WHERE correlation_id = %s
            ORDER BY log_id DESC
            LIMIT 1
            """,
            (correlation_id,),
        )
        assert log_row["level"] == "ERROR"
        assert log_row["event_type"] == "embedding_provider_route_contract_alert"
        assert provider_name in log_row["message"]
        assert log_row["detail"]["route_id"] == route.route_id
        assert log_row["detail"]["status"] == "health_unreachable"
        assert "Remote provider request failed" in log_row["detail"]["error_message"]
    finally:
        if "route" in locals():
            _cleanup_app_logs(
                migrated_database_url,
                [f"embedding-provider-route:{route.route_id}:contract"],
            )
        _cleanup_routes(migrated_database_url, [provider_name])


def test_embedding_provider_route_alert_api_filters_and_acknowledges_alerts(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    provider_name = f"mock-route-alert-ack-{suffix}"
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        route = upsert_embedding_provider_route(
            migrated_database_url,
            EmbeddingProviderRouteInput(
                profile_name="kure_v1_1024",
                provider_name=provider_name,
                provider_mode="remote",
                provider_base_url="http://127.0.0.1:9",
                timeout_seconds=0.1,
                priority=4,
                runtime_metadata={"purpose": "alert-ack-test"},
            ),
        )
        correlation_id = f"embedding-provider-route:{route.route_id}:contract"
        _cleanup_app_logs(migrated_database_url, [correlation_id])

        with TestClient(app) as client:
            contract_response = client.post(
                f"/api/admin/embedding-provider-routes/{route.route_id}/contract-check",
            )
            alerts_response = client.get(
                "/api/admin/embedding-provider-routes/alerts",
                params={"level": "ERROR", "acknowledged": "false"},
            )

        assert contract_response.status_code == 200
        assert alerts_response.status_code == 200
        matching_alerts = [
            alert
            for alert in alerts_response.json()["alerts"]
            if alert["correlation_id"] == correlation_id
        ]
        assert matching_alerts
        alert = matching_alerts[0]
        assert alert["acknowledged_at"] is None
        assert alert["detail"]["provider_name"] == provider_name
        assert alert["detail"]["status"] == "health_unreachable"

        with TestClient(app) as client:
            acknowledge_response = client.post(
                f"/api/admin/embedding-provider-routes/alerts/{alert['log_id']}/acknowledge",
                json={
                    "acknowledged_by": "integration-test",
                    "acknowledgement_note": "operator reviewed",
                },
            )
            acknowledged_response = client.get(
                "/api/admin/embedding-provider-routes/alerts",
                params={"acknowledged": "true"},
            )

        assert acknowledge_response.status_code == 200
        acknowledged_alert = acknowledge_response.json()["alert"]
        assert acknowledged_alert["log_id"] == alert["log_id"]
        assert acknowledged_alert["acknowledged_at"] is not None
        assert acknowledged_alert["acknowledged_by"] == "integration-test"
        assert acknowledged_alert["acknowledgement_note"] == "operator reviewed"
        assert any(
            item["log_id"] == alert["log_id"] for item in acknowledged_response.json()["alerts"]
        )
    finally:
        if "route" in locals():
            _cleanup_app_logs(
                migrated_database_url,
                [f"embedding-provider-route:{route.route_id}:contract"],
            )
        _cleanup_routes(migrated_database_url, [provider_name])


def test_embedding_provider_route_preflight_api_persists_health_and_contract_snapshots(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    provider_name = f"mock-route-preflight-{suffix}"
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        route = upsert_embedding_provider_route(
            migrated_database_url,
            EmbeddingProviderRouteInput(
                profile_name="kure_v1_1024",
                provider_name=provider_name,
                provider_mode="mock",
                provider_base_url=None,
                priority=2,
                runtime_metadata={"purpose": "preflight-test"},
            ),
        )

        with TestClient(app) as client:
            response = client.post(
                "/api/admin/embedding-provider-routes/preflight",
                params={"profile_name": "kure_v1_1024"},
            )
            history_response = client.get(
                "/api/admin/embedding-provider-routes/preflight-runs",
                params={"limit": "10"},
            )
            operations_response = client.get(
                "/api/admin/embedding-provider-routes/operations-summary"
            )
            created_run_id = response.json()["preflight_run"]["run_id"]
            detail_response = client.get(
                f"/api/admin/embedding-provider-routes/preflight-runs/{created_run_id}"
            )
            invalid_detail_response = client.get(
                "/api/admin/embedding-provider-routes/preflight-runs/0"
            )

        assert response.status_code == 200
        assert history_response.status_code == 200
        assert operations_response.status_code == 200
        assert detail_response.status_code == 200
        assert invalid_detail_response.status_code == 400
        body = response.json()
        preflight_run = body["preflight_run"]
        operations_summary = operations_response.json()["operations_summary"]
        detail = detail_response.json()["run"]
        route_results = [
            result
            for result in body["results"]
            if result["route"]["provider_name"] == provider_name
        ]
        assert route_results
        route_result = route_results[0]
        assert body["route_count"] >= 1
        assert body["passed_count"] >= 1
        assert body["sample_set"]["sample_set_name"] == "default_route_contract"
        assert body["sample_set"]["sample_text_count"] == 1
        assert preflight_run["trigger_source"] == "manual_api"
        assert preflight_run["status"] == "succeeded"
        assert preflight_run["route_count"] == body["route_count"]
        assert preflight_run["passed_count"] == body["passed_count"]
        assert preflight_run["sample_set_name"] == "default_route_contract"
        assert route_result["health"]["status"] == "ready"
        assert route_result["health_snapshot"]["route_id"] == route.route_id
        assert route_result["contract"]["passed"] is True
        assert route_result["contract_snapshot"]["route_id"] == route.route_id
        assert route_result["contract_snapshot"]["status"] == "passed"
        assert preflight_run["run_id"] in [run["run_id"] for run in history_response.json()["runs"]]
        assert detail["run_id"] == preflight_run["run_id"]
        assert detail["result"]["sample_set"]["sample_set_name"] == "default_route_contract"
        detail_route_results = [
            result
            for result in detail["result"]["results"]
            if result["route"]["provider_name"] == provider_name
        ]
        assert detail_route_results
        assert detail_route_results[0]["contract"]["status"] == "passed"
        assert operations_summary["active_route_count"] >= 1
        assert operations_summary["ready_route_count"] >= 1
        assert operations_summary["overall_status"] in {"ready", "attention", "blocked"}
        assert operations_summary["overall_status_reason"]
        assert isinstance(operations_summary["unacknowledged_alert_count"], int)
        assert operations_summary["latest_preflight_run"] is not None
        assert operations_summary["latest_preflight_run"]["status"] in {
            "succeeded",
            "failed",
            "error",
        }

        health_snapshots = list_embedding_provider_route_health_snapshots(
            migrated_database_url,
            route_id=route.route_id,
        )
        contract_snapshots = list_embedding_provider_route_contract_snapshots(
            migrated_database_url,
            route_id=route.route_id,
        )
        assert len(health_snapshots) == 1
        assert len(contract_snapshots) == 1
        assert preflight_run["run_id"] in [
            run.run_id
            for run in list_embedding_provider_preflight_runs(
                migrated_database_url,
                limit=10,
            )
        ]
        assert (
            get_embedding_provider_preflight_run(
                migrated_database_url,
                preflight_run["run_id"],
            ).run_id
            == preflight_run["run_id"]
        )
    finally:
        _cleanup_routes(migrated_database_url, [provider_name])


def test_embedding_provider_route_readiness_api_aggregates_latest_snapshots(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    provider_name = f"mock-route-readiness-{suffix}"
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        route = upsert_embedding_provider_route(
            migrated_database_url,
            EmbeddingProviderRouteInput(
                profile_name="kure_v1_1024",
                provider_name=provider_name,
                provider_mode="mock",
                provider_base_url=None,
                priority=2,
                runtime_metadata={"purpose": "readiness-test"},
            ),
        )

        initial_summary = get_embedding_provider_route_readiness_summary(
            migrated_database_url,
            profile_name="kure_v1_1024",
            active_only=True,
        )
        initial_item = next(
            item for item in initial_summary.routes if item.route.route_id == route.route_id
        )
        assert initial_item.status == "needs_contract"
        assert initial_item.ready is False
        assert initial_item.latest_health_snapshot is None
        assert initial_item.latest_contract_snapshot is None

        with TestClient(app) as client:
            preflight_response = client.post(
                "/api/admin/embedding-provider-routes/preflight",
                params={"profile_name": "kure_v1_1024"},
            )
            readiness_response = client.get(
                "/api/admin/embedding-provider-routes/readiness",
                params={"profile_name": "kure_v1_1024", "active_only": "true"},
            )

        assert preflight_response.status_code == 200
        assert readiness_response.status_code == 200
        body = readiness_response.json()
        route_items = [
            item for item in body["routes"] if item["route"]["provider_name"] == provider_name
        ]
        assert route_items
        route_item = route_items[0]
        assert body["route_count"] >= 1
        assert body["active_count"] >= 1
        assert body["ready_count"] >= 1
        assert route_item["ready"] is True
        assert route_item["status"] == "ready"
        assert route_item["recovery_action"] == "ready_for_worker"
        assert route_item["latest_health_snapshot"]["route_id"] == route.route_id
        assert route_item["latest_health_snapshot"]["status"] == "ready"
        assert route_item["latest_contract_snapshot"]["route_id"] == route.route_id
        assert route_item["latest_contract_snapshot"]["status"] == "passed"
        assert route_item["latest_contract_snapshot"]["passed"] is True
    finally:
        _cleanup_routes(migrated_database_url, [provider_name])


def test_embedding_provider_route_health_alert_logs_mismatch_detail(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    provider_name = f"mock-route-health-alert-{suffix}"

    try:
        route = upsert_embedding_provider_route(
            migrated_database_url,
            EmbeddingProviderRouteInput(
                profile_name="kure_v1_1024",
                provider_name=provider_name,
                provider_mode="mock",
                provider_base_url=None,
                priority=5,
            ),
        )
        correlation_id = f"embedding-provider-route:{route.route_id}:health"
        _cleanup_app_logs(migrated_database_url, [correlation_id])

        log_id = log_embedding_provider_route_health_alert(
            migrated_database_url,
            EmbeddingProviderRouteHealthResult(
                route=route,
                checked=True,
                ready=False,
                status="mismatch",
                elapsed_ms=3,
                provider_type="remote",
                provider_model_id="gpu-bge",
                model_key="bge_m3",
                profile_names=("bge_m3_1024",),
                dimension=1024,
                device="cuda:1",
                runtime_metadata={},
                validation_errors=("model_key mismatch: expected kure_v1, got bge_m3",),
            ),
        )

        log_row = fetch_one(
            migrated_database_url,
            """
            SELECT log_id, level, event_type, detail
            FROM app_logs
            WHERE correlation_id = %s
            ORDER BY log_id DESC
            LIMIT 1
            """,
            (correlation_id,),
        )
        assert log_id == log_row["log_id"]
        assert log_row["level"] == "WARNING"
        assert log_row["event_type"] == "embedding_provider_route_health_alert"
        assert log_row["detail"]["route_id"] == route.route_id
        assert log_row["detail"]["status"] == "mismatch"
        assert log_row["detail"]["validation_errors"] == [
            "model_key mismatch: expected kure_v1, got bge_m3"
        ]
    finally:
        if "route" in locals():
            _cleanup_app_logs(
                migrated_database_url,
                [f"embedding-provider-route:{route.route_id}:health"],
            )
        _cleanup_routes(migrated_database_url, [provider_name])


def test_embedding_provider_route_health_snapshot_filters_validate_inputs(
    migrated_database_url: str,
) -> None:
    with pytest.raises(InvalidEmbeddingProviderRouteHealthSnapshotError):
        list_embedding_provider_route_health_snapshots(migrated_database_url, limit=0)

    with pytest.raises(InvalidEmbeddingProviderRouteHealthSnapshotError):
        list_embedding_provider_route_health_snapshots(migrated_database_url, route_id=0)


def test_embedding_provider_route_contract_snapshot_filters_validate_inputs(
    migrated_database_url: str,
) -> None:
    with pytest.raises(InvalidEmbeddingProviderRouteContractSnapshotError):
        list_embedding_provider_route_contract_snapshots(migrated_database_url, limit=0)

    with pytest.raises(InvalidEmbeddingProviderRouteContractSnapshotError):
        list_embedding_provider_route_contract_snapshots(migrated_database_url, route_id=0)
