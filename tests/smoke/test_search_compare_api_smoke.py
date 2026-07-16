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


def test_search_log_profile_retry_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.post(
            "/api/search/logs/1/retry-profile",
            json={"profile_name": "kure_v1_1024"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_search_permission_matrix_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.post(
            "/api/search/permission-matrix",
            json={
                "query_text": "hello",
                "entries": [
                    {
                        "actor_user_id": 1,
                        "requested_search_scope": "company",
                    },
                ],
            },
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_search_experiment_run_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.post(
            "/api/search/experiments/run",
            json={
                "run_name": "smoke",
                "query_text": "hello",
                "actor_user_id": 1,
            },
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_search_experiment_list_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/api/search/experiments")

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_search_experiment_detail_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/api/search/experiments/1")

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_golden_search_experiment_batch_summary_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        list_response = client.get("/api/search/experiments/golden-question-batches")
        detail_response = client.get("/api/search/experiments/golden-question-batches/dummy")
        metric_response = client.get(
            "/api/search/experiments/golden-question-batches/dummy/metrics"
        )
        snapshot_list_response = client.get(
            "/api/search/experiments/golden-question-batches/dummy/metric-snapshots"
        )
        snapshot_trend_response = client.get(
            "/api/search/experiments/golden-question-batches/dummy/metric-snapshots/trend"
        )
        snapshot_create_response = client.post(
            "/api/search/experiments/golden-question-batches/dummy/metric-snapshots"
        )
        snapshot_detail_response = client.get(
            "/api/search/experiments/golden-question-batch-metric-snapshots/1"
        )
        snapshot_compare_response = client.get(
            "/api/search/experiments/golden-question-batch-metric-snapshots/compare",
            params={"base_snapshot_id": 1, "target_snapshot_id": 2},
        )

    assert list_response.status_code == 503
    assert list_response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}
    assert detail_response.status_code == 503
    assert detail_response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}
    assert metric_response.status_code == 503
    assert metric_response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}
    assert snapshot_list_response.status_code == 503
    assert snapshot_list_response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}
    assert snapshot_trend_response.status_code == 503
    assert snapshot_trend_response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}
    assert snapshot_create_response.status_code == 503
    assert snapshot_create_response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}
    assert snapshot_detail_response.status_code == 503
    assert snapshot_detail_response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}
    assert snapshot_compare_response.status_code == 503
    assert snapshot_compare_response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


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


def test_search_feedback_comments_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/api/search/feedback/comments")

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


def test_golden_question_candidates_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/api/evaluations/golden-question-candidates")

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_golden_question_candidate_batch_promotion_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.post(
            "/api/evaluations/golden-question-candidates/promote",
            json={"question_set_id": 1, "search_log_result_ids": [1]},
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


def test_search_log_metadata_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.patch(
            "/api/search/logs/1/metadata",
            json={"review_tags": ["baseline"]},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_search_log_export_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/api/search/logs/1/export")

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_search_experiment_report_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/api/search/logs/1/experiment-report")

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_search_log_compare_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get(
            "/api/search/logs/compare",
            params={"left_search_log_id": 1, "right_search_log_id": 2},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_search_log_retention_settings_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        get_response = client.get("/api/search/logs/retention-settings")
        update_response = client.put(
            "/api/search/logs/retention-settings",
            json={"enabled": True, "retention_days": 30, "cleanup_batch_size": 1000},
        )

    assert get_response.status_code == 503
    assert get_response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}
    assert update_response.status_code == 503
    assert update_response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_search_log_cleanup_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.post("/api/search/logs/cleanup", json={"dry_run": True})

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


def test_evaluation_question_set_export_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/api/evaluations/question-sets/1/export")

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_evaluation_question_set_import_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.post(
            "/api/evaluations/question-sets/import",
            json={"question_set": {"set_name": "baseline"}},
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


def test_evaluation_execute_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.post(
            "/api/evaluations/runs/execute",
            json={"question_set_id": 1, "profile_name": "kure_v1_1024"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_golden_search_experiment_batch_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.post(
            "/api/search/experiments/golden-question-set/run",
            json={"question_set_id": 1, "profiles": ["kure_v1_1024"]},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_evaluation_profile_comparison_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get(
            "/api/evaluations/profile-comparison",
            params={"question_set_id": 1},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_evaluation_run_detail_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/api/evaluations/runs/1")

    assert response.status_code == 503


def test_evaluation_permission_audit_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/api/evaluations/runs/1/permission-audit")

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}


def test_evaluation_run_export_api_requires_database_url() -> None:
    app = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        response = client.get("/api/evaluations/runs/1/export")

    assert response.status_code == 503
    assert response.json() == {"detail": "NEX_PCX_DATABASE_URL is not configured."}
