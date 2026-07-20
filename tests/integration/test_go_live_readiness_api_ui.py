from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_go_live_readiness_api_reads_migrated_database(
    migrated_database_url: str,
    tmp_path,
) -> None:
    app = create_app(
        Settings(
            database_url=migrated_database_url,
            upload_storage_dir=tmp_path / "uploads",
            embedding_models_dir=tmp_path / "models",
        )
    )

    with TestClient(app) as client:
        response = client.get("/api/admin/go-live-readiness")

    assert response.status_code == 200
    payload = response.json()["go_live_readiness"]
    assert payload["status"] in {"ready", "warning", "blocked"}
    checks = {
        check["code"]: check for section in payload["sections"] for check in section["checks"]
    }
    assert checks["database_configured"]["status"] == "passed"
    assert checks["database_connectivity"]["status"] == "passed"
    assert checks["active_embedding_profiles"]["metadata"]["active_profile_count"] >= 1


def test_go_live_readiness_page_reads_migrated_database(
    migrated_database_url: str,
    tmp_path,
) -> None:
    app = create_app(
        Settings(
            database_url=migrated_database_url,
            upload_storage_dir=tmp_path / "uploads",
            embedding_models_dir=tmp_path / "models",
        )
    )

    with TestClient(app) as client:
        response = client.get("/admin/go-live-readiness")

    assert response.status_code == 200
    assert "운영 준비도" in response.text
    assert "Database 연결" in response.text
    assert "활성 임베딩 Profile" in response.text
