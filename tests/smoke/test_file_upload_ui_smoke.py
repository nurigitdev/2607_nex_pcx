from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_file_upload_page_renders_configuration_state() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/files/upload")

    assert response.status_code == 200
    assert "File Upload" in response.text
    assert "Drop file here" in response.text
    assert "data-upload-progress" in response.text
    assert "NEX_PCX_DATABASE_URL is not configured." in response.text
    assert "No upload result" in response.text


def test_file_upload_form_shows_configuration_error_without_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.post(
            "/files/upload",
            data={"document_group": "docs", "security_level": "internal"},
            files={"file": ("example.md", b"# Example", "text/markdown")},
        )

    assert response.status_code == 200
    assert "NEX_PCX_DATABASE_URL is not configured." in response.text
    assert "No upload result" in response.text
