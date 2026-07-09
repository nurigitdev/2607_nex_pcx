from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_golden_question_page_shows_configuration_message_without_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/evaluations/questions")

    assert response.status_code == 200
    assert "Golden Questions" in response.text
    assert "NEX_PCX_DATABASE_URL is not configured." in response.text
    assert 'href="/evaluations/questions"' in response.text
    assert 'id="question-set-form"' in response.text
    assert 'id="question-set-exchange-json"' in response.text
    assert 'id="golden-candidate-table-body"' in response.text
