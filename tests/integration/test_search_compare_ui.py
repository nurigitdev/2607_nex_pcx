import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

pytestmark = pytest.mark.integration


def test_search_compare_page_renders_actor_and_profile_options(
    migrated_database_url: str,
) -> None:
    app = create_app(Settings(database_url=migrated_database_url))

    with TestClient(app) as client:
        response = client.get("/search")

    assert response.status_code == 200
    assert "Search Compare" in response.text
    assert "Alice Member / alice.member" in response.text
    assert "kure_v1_1024" in response.text
    assert "bge_m3_1024" in response.text
    assert 'id="search-compare-form"' in response.text
    assert 'id="search-results"' in response.text
    assert 'id="feedback-summary-grid"' in response.text
    assert "search-feedback-button" in response.text
    assert "/api/search/feedback" in response.text
    assert "/api/search/feedback/summary" in response.text
    assert "권한 검색 Matrix" in response.text
    assert 'id="permission-matrix-form"' in response.text
    assert "/api/search/permission-matrix" in response.text


def test_search_history_page_renders_filters_and_log_table(
    migrated_database_url: str,
) -> None:
    app = create_app(Settings(database_url=migrated_database_url))

    with TestClient(app) as client:
        response = client.get("/search/logs")

    assert response.status_code == 200
    assert "Search History" in response.text
    assert "Alice Member / alice.member" in response.text
    assert 'class="search-history-filter"' in response.text
    assert 'class="table table-sm align-middle mb-0 search-log-table"' in response.text
