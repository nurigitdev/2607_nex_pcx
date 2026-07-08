from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_search_compare_page_shows_configuration_message_without_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/search")

    assert response.status_code == 200
    assert "Search Compare" in response.text
    assert "NEX_PCX_DATABASE_URL is not configured." in response.text
    assert 'href="/search"' in response.text
    assert 'id="search-permission-summary"' in response.text
    assert "검색 권한 설명" in response.text
    assert "권한 검색 Matrix" in response.text
    assert 'id="permission-matrix-submit"' in response.text
    assert "/api/search/permission-matrix" in response.text


def test_search_history_page_shows_configuration_message_without_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/search/logs")

    assert response.status_code == 200
    assert "Search History" in response.text
    assert "NEX_PCX_DATABASE_URL is not configured." in response.text
    assert 'href="/search/logs"' in response.text


def test_golden_evaluation_page_shows_configuration_message_without_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/evaluations")

    assert response.status_code == 200
    assert "Golden Evaluation Monitor" in response.text
    assert "NEX_PCX_DATABASE_URL is not configured." in response.text
    assert 'href="/evaluations"' in response.text
    assert 'id="evaluation-execute-form"' in response.text
    assert "/api/evaluations/runs/execute" in response.text
