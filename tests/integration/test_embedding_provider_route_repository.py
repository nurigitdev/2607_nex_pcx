from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from psycopg import errors

from app.core.config import Settings
from app.core.database import connect, fetch_one
from app.core.embedding_provider_route_contract_snapshots import (
    InvalidEmbeddingProviderRouteContractSnapshotError,
    list_embedding_provider_route_contract_snapshots,
)
from app.core.embedding_provider_route_health import EmbeddingProviderRouteHealthResult
from app.core.embedding_provider_route_health_snapshots import (
    InvalidEmbeddingProviderRouteHealthSnapshotError,
    list_embedding_provider_route_health_snapshots,
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
            assert api_provider_name in page_response.text
            assert form_provider_name in page_response.text
            assert "/api/admin/embedding-provider-routes" in page_response.text
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
            assert "data-route-health-history-panel" in page_response.text
            assert (
                "/api/admin/embedding-provider-routes/health-snapshots?limit=10"
                in page_response.text
            )
    finally:
        _cleanup_routes(migrated_database_url, provider_names)


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

        assert response.status_code == 200
        body = response.json()
        contract = body["contract"]
        snapshot = body["snapshot"]
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
