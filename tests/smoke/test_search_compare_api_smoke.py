from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_search_compare_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.post(
            "/api/search/compare",
            json={
                "query_text": "hello",
                "actor_user_id": 1,
                "requested_search_scope": "company",
            },
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}
