from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.remote_reranker_operations import RemoteRerankerOperationsStatus
from app.main import create_app

pytestmark = pytest.mark.integration


def _status_payload(*, request_smoke_checked: bool = False) -> dict[str, Any]:
    return {
        "checked_at": "2026-07-25T01:02:03+00:00",
        "passed": True,
        "operations_status": "ready",
        "app_runtime": {
            "status": "remote_selected",
            "mode": "remote",
            "remote_base_url": "http://192.168.20.243:9104",
            "timeout_seconds": 300.0,
            "configured_for_remote": True,
            "error": None,
        },
        "request_smoke_requested": request_smoke_checked,
        "status": "running",
        "pid": "2437559",
        "provider": {
            "provider_name": "qwen-reranker-primary",
            "provider_model_id": "Qwen/Qwen3-Reranker-4B",
            "reranker_profile_name": "qwen3_reranker_4b",
            "backend": "qwen_reranker",
            "device": "cuda:0",
            "base_url": "http://192.168.20.243:9104",
            "health_url": "http://192.168.20.243:9104/healthz",
            "ssh_target": "nexpcx@192.168.20.243",
            "workdir": "/home/nexpcx/2607_nex_pcx",
            "pid_file": "/home/nexpcx/2607_nex_pcx/run/remote_reranker_provider_9104.pid",
            "log_file": "/home/nexpcx/2607_nex_pcx/logs/remote_reranker_provider_9104.log",
        },
        "plan": {},
        "command_observation": {
            "ok": True,
            "exit_code": 0,
            "stdout": "status=running\npid=2437559\n",
            "stderr": "",
            "values": {"status": "running", "pid": "2437559"},
            "stdout_line_count": 2,
            "stderr_line_count": 0,
        },
        "health": {
            "checked": True,
            "ok": True,
            "status_code": 200,
            "payload": {
                "ready": True,
                "provider_type": "remote",
                "runtime_metadata": {
                    "service": "nex_pcx_reranker_provider_service",
                    "backend": "qwen_reranker",
                    "device": "cuda:0",
                },
            },
            "error": None,
            "mismatches": [],
        },
        "request_smoke": {
            "checked": request_smoke_checked,
            "passed": True if request_smoke_checked else None,
            "summary": (
                {
                    "passed": True,
                    "request_elapsed_ms": 17,
                    "provider_elapsed_ms": 11,
                    "candidate_count": 3,
                    "returned_count": 2,
                    "result_previews": [
                        {
                            "candidate_key": "candidate-1",
                            "rank": 1,
                            "score": 8.756176,
                            "source_rank": 1,
                            "score_components": {"source_rank": 1},
                        }
                    ],
                    "runtime_metadata": {"backend": "qwen_reranker"},
                    "mismatches": [],
                    "error": None,
                }
                if request_smoke_checked
                else None
            ),
        },
        "elapsed_ms": 42,
        "error": None,
    }


def test_remote_reranker_operations_status_api(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.main as main_module

    calls: list[bool] = []

    def fake_status(_settings: Settings, *, request_smoke: bool = False):
        calls.append(request_smoke)
        return RemoteRerankerOperationsStatus(
            status_code=200,
            payload=_status_payload(request_smoke_checked=request_smoke),
        )

    monkeypatch.setattr(main_module, "get_remote_reranker_operations_status", fake_status)
    app = create_app(
        Settings(
            reranker_provider_mode="remote",
            remote_reranker_provider_url="http://192.168.20.243:9104",
        )
    )

    with TestClient(app) as client:
        response = client.get("/api/admin/reranker-provider/status?request_smoke=true")

    assert response.status_code == 200
    assert calls == [True]
    assert response.json()["operations_status"] == "ready"
    assert response.json()["request_smoke"]["summary"]["returned_count"] == 2


def test_remote_reranker_operations_page_renders_korean_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.main as main_module

    calls: list[bool] = []

    def fake_status(_settings: Settings, *, request_smoke: bool = False):
        calls.append(request_smoke)
        return RemoteRerankerOperationsStatus(
            status_code=200,
            payload=_status_payload(request_smoke_checked=request_smoke),
        )

    monkeypatch.setattr(main_module, "get_remote_reranker_operations_status", fake_status)
    app = create_app(
        Settings(
            reranker_provider_mode="remote",
            remote_reranker_provider_url="http://192.168.20.243:9104",
        )
    )

    with TestClient(app) as client:
        response = client.get("/admin/reranker-provider?request_smoke=true")

    assert response.status_code == 200
    assert calls == [True]
    assert "Reranker Provider 상태" in response.text
    assert "Request Smoke 실행" in response.text
    assert "data-reranker-provider-status-panel" in response.text
    assert "data-reranker-request-smoke-panel" in response.text
    assert "qwen-reranker-primary" in response.text
    assert "/api/admin/reranker-provider/status?request_smoke=true" in response.text
