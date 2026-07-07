from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_embedding_job_monitor_shows_configuration_message_without_database_url() -> None:
    app = create_app(Settings())

    with TestClient(app) as client:
        response = client.get("/admin/embedding-jobs")

    assert response.status_code == 200
    assert "Embedding Job Monitor" in response.text
    assert "NEX_PCX_DATABASE_URL is not configured." in response.text


def test_embedding_job_retry_api_requires_database_url() -> None:
    app = create_app(Settings())

    with TestClient(app) as client:
        response = client.post("/api/embedding/jobs/1/retry")

    assert response.status_code == 503
    assert response.json()["detail"] == "NEX_PCX_DATABASE_URL is not configured."
