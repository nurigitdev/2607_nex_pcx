from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_embedding_job_monitor_shows_configuration_message_without_database_url() -> None:
    app = create_app(Settings())

    with TestClient(app) as client:
        response = client.get("/admin/embedding-jobs")

    assert response.status_code == 200
    assert "Embedding Job Monitor" in response.text
    assert "임베딩 Queue Backlog" in response.text
    assert "Failed Job Bulk Retry" in response.text
    assert "Stale Lease Recovery" in response.text
    assert "/api/admin/embedding-jobs/backlog-summary" in response.text
    assert "/api/admin/embedding-jobs/retry-failed" in response.text
    assert "/api/admin/embedding-jobs/stale-leases" in response.text
    assert "NEX_PCX_DATABASE_URL is not configured." in response.text


def test_embedding_job_admin_apis_require_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        responses = [
            client.get("/api/admin/embedding-jobs/backlog-summary"),
            client.post("/api/admin/embedding-jobs/retry-failed", json={}),
            client.get("/api/admin/embedding-jobs/stale-leases"),
            client.post("/api/admin/embedding-jobs/1/release-stale-lease"),
        ]

    assert [response.status_code for response in responses] == [503, 503, 503, 503]
    assert all(
        response.json()["detail"] == "NEX_PCX_DATABASE_URL is not configured."
        for response in responses
    )


def test_embedding_job_retry_api_requires_database_url() -> None:
    app = create_app(Settings())

    with TestClient(app) as client:
        response = client.post("/api/embedding/jobs/1/retry")

    assert response.status_code == 503
    assert response.json()["detail"] == "NEX_PCX_DATABASE_URL is not configured."


def test_embedding_batch_runs_page_shows_configuration_message_without_database_url() -> None:
    app = create_app(Settings())

    with TestClient(app) as client:
        response = client.get("/admin/embedding-batch-runs")

    assert response.status_code == 200
    assert "임베딩 Batch 실행 이력" in response.text
    assert "data-embedding-batch-runs-page" in response.text
    assert "/api/admin/embedding-batch-runs" in response.text
    assert "Embedding Throughput Trend" in response.text
    assert "/api/admin/embedding-batch-runs/throughput-summary" in response.text
    assert "Batch Run Retention" in response.text
    assert "/api/admin/embedding-batch-runs/retention-settings" in response.text
    assert "/api/admin/embedding-batch-runs/cleanup" in response.text
    assert "NEX_PCX_DATABASE_URL is not configured." in response.text


def test_embedding_batch_run_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        responses = [
            client.get("/api/admin/embedding-batch-runs"),
            client.get("/api/admin/embedding-batch-runs/throughput-summary"),
            client.get("/api/admin/embedding-batch-runs/retention-settings"),
            client.put("/api/admin/embedding-batch-runs/retention-settings", json={}),
            client.post("/api/admin/embedding-batch-runs/cleanup", json={"dry_run": True}),
            client.get("/api/admin/embedding-batch-runs/1"),
            client.post("/api/admin/embedding-batch-runs/1/retry-failed"),
        ]

    assert [response.status_code for response in responses] == [
        503,
        503,
        503,
        503,
        503,
        503,
        503,
    ]
    assert all(
        response.json()["detail"] == "NEX_PCX_DATABASE_URL is not configured."
        for response in responses
    )


def test_embedding_model_readiness_api_reports_local_bundle_state(tmp_path) -> None:
    kure_dir = tmp_path / "kure_v1"
    kure_dir.mkdir()
    (kure_dir / "config.json").write_text("{}", encoding="utf-8")
    (kure_dir / "model.safetensors").write_bytes(b"weights")
    app = create_app(Settings(embedding_models_dir=tmp_path))

    with TestClient(app) as client:
        response = client.get("/api/embedding/models/readiness")

    assert response.status_code == 200
    body = response.json()
    assert body["models_dir"] == str(tmp_path)
    assert body["model_count"] == 3
    assert body["ready_count"] == 1
    models = {model["model_key"]: model for model in body["models"]}
    assert models["kure_v1"]["ready"] is True
    assert models["bge_m3"]["ready"] is False
    assert models["qwen3_embedding_4b"]["profile_names"] == [
        "qwen3_4b_1000",
        "qwen3_4b_2560",
    ]


def test_embedding_provider_health_api_reports_mock_provider() -> None:
    app = create_app(Settings())

    with TestClient(app) as client:
        response = client.get("/api/embedding/providers/health")

    assert response.status_code == 200
    body = response.json()
    assert body["provider_mode"] == "mock"
    assert body["ready"] is True
    assert body["provider_model_id"] == "mock-provider"


def test_embedding_provider_health_page_shows_mock_provider() -> None:
    app = create_app(Settings())

    with TestClient(app) as client:
        response = client.get("/admin/embedding-provider")

    assert response.status_code == 200
    assert "임베딩 Provider 상태" in response.text
    assert "mock-provider" in response.text
    assert "/api/embedding/providers/health" in response.text


def test_embedding_provider_routes_page_shows_configuration_message_without_database_url() -> None:
    app = create_app(Settings())

    with TestClient(app) as client:
        response = client.get("/admin/embedding-provider-routes")

    assert response.status_code == 200
    assert "임베딩 Provider 라우팅" in response.text
    assert "Provider Preset 등록" in response.text
    assert 'id="provider-preset-registration"' in response.text
    assert "data-provider-preset-registration-panel" in response.text
    assert "data-provider-preset-form" in response.text
    assert "data-preset-run-preflight" in response.text
    assert "data-provider-preset-result-panel" in response.text
    assert "data-provider-preset-preview-table" in response.text
    assert "/api/admin/embedding-provider-routes/presets" in response.text
    assert "Provider Runtime Launch Plan" in response.text
    assert 'id="provider-launch-plan"' in response.text
    assert "data-provider-launch-plan-panel" in response.text
    assert "data-provider-launch-shell-command" in response.text
    assert "/api/admin/embedding-provider-routes/presets/launch-plan" in response.text
    assert "Provider Route Import/Export" in response.text
    assert "data-provider-route-import-export-panel" in response.text
    assert "/api/admin/embedding-provider-routes/export" in response.text
    assert "/api/admin/embedding-provider-routes/import" in response.text
    assert "request_headers_json" in response.text
    assert "auth_token_env" in response.text
    assert "Request Auth" in response.text
    assert "data-provider-model-availability-panel" in response.text
    assert "/api/admin/embedding-provider-routes/model-availability" in response.text
    assert "권장 조치" in response.text
    assert "data-provider-model-availability-drilldown-panel" in response.text
    assert "wireAvailabilityProfileShortcuts" in response.text
    assert "Provider 가용성 상세" in response.text
    assert "/api/admin/embedding-provider-routes/presets/register" in response.text
    assert "Provider 운영 요약" in response.text
    assert "운영 Playbook" in response.text
    assert "/admin/embedding-provider-routes/playbook" in response.text
    assert "data-provider-operations-summary-panel" in response.text
    assert "data-operations-summary-refresh" in response.text
    assert "Provider Route Health" in response.text
    assert "Provider Route Readiness" in response.text
    assert "운영 상태" in response.text
    assert "다음 조치" in response.text
    assert "data-route-readiness-panel" in response.text
    assert "data-route-readiness-table" in response.text
    assert "data-route-health-panel" in response.text
    assert "Preflight 실행" in response.text
    assert "data-route-preflight-button" in response.text
    assert "data-provider-route-management-panel" in response.text
    assert "wireProviderRouteManagementControls" in response.text
    assert "Preflight 스케줄" in response.text
    assert "data-preflight-schedule-panel" in response.text
    assert "data-due-schedule-controls" in response.text
    assert "data-due-preview" in response.text
    assert "data-run-due" in response.text
    assert "Preflight 실행 이력" in response.text
    assert "data-preflight-run-history-panel" in response.text
    assert "data-preflight-run-detail-panel" in response.text
    assert "data-preflight-run-detail-routes-table" in response.text
    assert "preflight-run-json-viewer" in response.text
    assert "운영 데이터 보존 설정" in response.text
    assert "data-provider-route-retention-panel" in response.text
    assert "Health Snapshot 이력" in response.text
    assert "data-route-health-history-panel" in response.text
    assert "Contract Snapshot 이력" in response.text
    assert "data-route-contract-history-panel" in response.text
    assert "Provider Route Alert" in response.text
    assert "data-route-alerts-panel" in response.text
    assert "routeAlertNote" in response.text
    assert "Contract Sample Sets" in response.text
    assert "data-contract-sample-set-panel" in response.text
    assert "data-sample-set-form" in response.text
    assert "wireContractButtons" in response.text
    assert "NEX_PCX_DATABASE_URL is not configured." in response.text
    assert "/api/admin/embedding-provider-routes" in response.text
    assert "/api/admin/embedding-provider-routes/operations-summary" in response.text
    assert "/api/admin/embedding-provider-routes/readiness?active_only=true" in response.text
    assert "/api/admin/embedding-provider-routes/health" in response.text
    assert "/api/admin/embedding-provider-routes/contract-sample-sets" in response.text
    assert "/api/admin/embedding-provider-routes/preflight" in response.text
    assert "/api/admin/embedding-provider-routes/preflight-schedules" in response.text
    assert "/api/admin/embedding-provider-routes/preflight-schedules/due" in response.text
    assert "/api/admin/embedding-provider-routes/preflight-schedules/run-due" in response.text
    assert "/api/admin/embedding-provider-routes/preflight-runs?limit=10" in response.text
    assert "/api/admin/embedding-provider-routes/preflight-runs" in response.text
    assert "/api/admin/embedding-provider-routes/retention-settings" in response.text
    assert "/api/admin/embedding-provider-routes/cleanup" in response.text
    assert "/api/admin/embedding-provider-routes/health-snapshots?limit=10" in response.text
    assert "/api/admin/embedding-provider-routes/contract-snapshots?limit=10" in response.text
    assert "/api/admin/embedding-provider-routes/alerts" in response.text


def test_embedding_provider_launch_plan_api_works_without_database_url() -> None:
    app = create_app(Settings())

    with TestClient(app) as client:
        response = client.post(
            "/api/admin/embedding-provider-routes/presets/launch-plan",
            json={
                "preset_name": "qwen",
                "host": "0.0.0.0",
                "port": 19103,
                "device": "cuda:0",
                "models_dir": "/srv/nex_pcx/models",
                "provider_model_id": "gpu-qwen3-4b",
                "python_bin": "/opt/nex-pcx/.venv/bin/python",
                "reload": True,
            },
        )

    assert response.status_code == 200
    plan = response.json()["plan"]
    assert plan["base_url"] == "http://0.0.0.0:19103"
    assert plan["provider_model_id"] == "gpu-qwen3-4b"
    assert plan["environment"]["NEX_PCX_PROVIDER_BACKEND"] == "qwen_embedding"
    assert "--reload" in plan["command"]


def test_embedding_provider_route_import_export_api_requires_database_url() -> None:
    app = create_app(Settings())

    with TestClient(app) as client:
        export_response = client.get("/api/admin/embedding-provider-routes/export")
        availability_response = client.get(
            "/api/admin/embedding-provider-routes/model-availability"
        )
        drilldown_response = client.get(
            "/api/admin/embedding-provider-routes/model-availability/kure_v1_1024"
        )
        import_response = client.post(
            "/api/admin/embedding-provider-routes/import",
            json={
                "routes": [
                    {
                        "profile_name": "kure_v1_1024",
                        "provider_name": "gpu-smoke",
                        "provider_mode": "remote",
                        "provider_base_url": "http://gpu-smoke.local",
                    }
                ],
                "dry_run": True,
            },
        )

    assert export_response.status_code == 503
    assert availability_response.status_code == 503
    assert drilldown_response.status_code == 503
    assert import_response.status_code == 503


def test_embedding_provider_route_playbook_page_loads_document_source() -> None:
    app = create_app(Settings())

    with TestClient(app) as client:
        response = client.get("/admin/embedding-provider-routes/playbook")

    assert response.status_code == 200
    assert "Provider 운영 Playbook" in response.text
    assert "data-provider-playbook-page" in response.text
    assert "docs/provider_operations_playbook.md" in response.text
    assert "Embedding Provider Operations Playbook" in response.text
    assert "Release Checklist" in response.text


def test_embedding_provider_contract_sample_set_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))
    payload = {
        "sample_set_name": "smoke_samples",
        "description": "Smoke sample set",
        "input_type": "document",
        "sample_texts": ["sample text"],
        "is_active": True,
        "is_default": False,
    }

    with TestClient(app) as client:
        responses = [
            client.get("/api/admin/embedding-provider-routes/contract-sample-sets"),
            client.post(
                "/api/admin/embedding-provider-routes/contract-sample-sets",
                json=payload,
            ),
            client.get("/api/admin/embedding-provider-routes/contract-sample-sets/smoke_samples"),
            client.put(
                "/api/admin/embedding-provider-routes/contract-sample-sets/smoke_samples",
                json=payload,
            ),
            client.delete(
                "/api/admin/embedding-provider-routes/contract-sample-sets/smoke_samples"
            ),
        ]

    assert [response.status_code for response in responses] == [503, 503, 503, 503, 503]
    assert all(
        response.json()["detail"] == "NEX_PCX_DATABASE_URL is not configured."
        for response in responses
    )


def test_embedding_provider_route_retention_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))
    retention_payload = {
        "enabled": True,
        "retention_days": 30,
        "cleanup_batch_size": 1000,
    }

    with TestClient(app) as client:
        responses = [
            client.get("/api/admin/embedding-provider-routes/retention-settings"),
            client.put(
                "/api/admin/embedding-provider-routes/retention-settings",
                json=retention_payload,
            ),
            client.post(
                "/api/admin/embedding-provider-routes/cleanup",
                json={"dry_run": True},
            ),
        ]

    assert [response.status_code for response in responses] == [503, 503, 503]
    assert all(
        response.json()["detail"] == "NEX_PCX_DATABASE_URL is not configured."
        for response in responses
    )


def test_embedding_provider_operations_summary_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/api/admin/embedding-provider-routes/operations-summary")

    assert response.status_code == 503
    assert response.json()["detail"] == "NEX_PCX_DATABASE_URL is not configured."


def test_embedding_provider_preflight_run_detail_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/api/admin/embedding-provider-routes/preflight-runs/1")

    assert response.status_code == 503
    assert response.json()["detail"] == "NEX_PCX_DATABASE_URL is not configured."


def test_embedding_provider_preflight_schedule_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))
    schedule_payload = {
        "description": "Smoke schedule",
        "profile_name": None,
        "active_only": True,
        "interval_minutes": 60,
        "is_enabled": False,
        "next_run_at": None,
    }

    with TestClient(app) as client:
        responses = [
            client.get("/api/admin/embedding-provider-routes/preflight-schedules"),
            client.get("/api/admin/embedding-provider-routes/preflight-schedules/due"),
            client.post(
                "/api/admin/embedding-provider-routes/preflight-schedules/run-due",
                json={"schedule_name": "smoke-schedule", "limit": 5},
            ),
            client.get("/api/admin/embedding-provider-routes/preflight-schedules/smoke-schedule"),
            client.put(
                "/api/admin/embedding-provider-routes/preflight-schedules/smoke-schedule",
                json=schedule_payload,
            ),
        ]

    assert [response.status_code for response in responses] == [503, 503, 503, 503, 503]
    assert all(
        response.json()["detail"] == "NEX_PCX_DATABASE_URL is not configured."
        for response in responses
    )


def test_embedding_model_readiness_page_shows_without_database_url(tmp_path) -> None:
    app = create_app(Settings(embedding_models_dir=tmp_path))

    with TestClient(app) as client:
        response = client.get("/admin/embedding-models")

    assert response.status_code == 200
    assert "임베딩 모델 준비도" in response.text
    assert str(tmp_path) in response.text
    assert "/api/embedding/models/readiness" in response.text
    assert "qwen3_embedding_4b" in response.text
