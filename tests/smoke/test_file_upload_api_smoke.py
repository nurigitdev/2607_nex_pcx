from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_upload_file_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.post(
            "/api/files",
            files={"file": ("example.md", b"# Example", "text/markdown")},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "NEX_PCX_DATABASE_URL is not configured."
