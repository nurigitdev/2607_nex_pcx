from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_admin_logs_shows_configuration_message_without_database_url() -> None:
    app = create_app(Settings())

    with TestClient(app) as client:
        response = client.get("/admin/logs")

    assert response.status_code == 200
    assert "Application Logs" in response.text
    assert "NEX_PCX_DATABASE_URL is not configured." in response.text
