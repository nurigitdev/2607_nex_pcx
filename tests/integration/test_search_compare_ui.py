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
