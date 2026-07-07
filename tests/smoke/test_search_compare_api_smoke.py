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


def test_search_feedback_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.post(
            "/api/search/feedback",
            json={
                "search_log_result_id": 1,
                "relevance_label": "correct",
            },
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_search_feedback_summary_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/api/search/feedback/summary")

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_search_result_promotion_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.post(
            "/api/search/results/1/promote-golden-question",
            json={"question_set_id": 1},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_search_logs_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/api/search/logs")

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_search_log_detail_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/api/search/logs/1")

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_evaluation_question_sets_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/api/evaluations/question-sets")

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_evaluation_question_set_create_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.post(
            "/api/evaluations/question-sets",
            json={"set_name": "baseline"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_evaluation_question_create_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.post(
            "/api/evaluations/questions",
            json={"question_set_id": 1, "question_text": "hello"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_evaluation_expected_target_create_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.post(
            "/api/evaluations/expected-targets",
            json={"question_id": 1, "expected_heading_path": ["Policy"]},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_evaluation_runs_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/api/evaluations/runs")

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_evaluation_run_detail_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/api/evaluations/runs/1")

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}
