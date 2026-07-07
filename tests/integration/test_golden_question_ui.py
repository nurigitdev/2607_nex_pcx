import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

pytestmark = pytest.mark.integration


def test_golden_question_manager_page_renders_crud_shell(
    migrated_database_url: str,
) -> None:
    app = create_app(Settings(database_url=migrated_database_url))

    with TestClient(app) as client:
        response = client.get("/evaluations/questions")

    assert response.status_code == 200
    assert "Golden Questions" in response.text
    assert "Alice Member / alice.member" in response.text
    assert 'href="/evaluations/questions"' in response.text
    assert 'id="question-set-form"' in response.text
    assert 'id="question-form"' in response.text
    assert 'id="target-form"' in response.text
    assert 'id="question-set-table-body"' in response.text
    assert 'id="question-table-body"' in response.text
    assert 'id="target-table-body"' in response.text
    assert "/api/evaluations/question-sets" in response.text
    assert "/api/evaluations/questions" in response.text
    assert "/api/evaluations/expected-targets" in response.text
