from fastapi.testclient import TestClient

from app.main import create_app


def test_foreground_worker_runtime_api_returns_runtime_summary() -> None:
    app = create_app()

    with TestClient(app) as client:
        response = client.get("/api/admin/foreground-worker-runtime")

    assert response.status_code == 200
    payload = response.json()
    runtime = payload["foreground_worker_runtime"]
    assert runtime["status"] in {"ready", "warning", "blocked"}
    assert "summary" in runtime
    assert "supervisor_evidence" in runtime
    assert "worker_runner_evidence" in runtime
