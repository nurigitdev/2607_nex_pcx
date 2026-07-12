from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect
from app.core.evaluation_metrics import (
    ExpectedTarget,
    QuestionEvaluationInput,
    evaluate_question,
    summarize_question_metrics,
)
from app.core.evaluation_runs import (
    EvaluationResultInput,
    EvaluationRunInput,
    complete_evaluation_run,
    create_evaluation_result,
    create_evaluation_run,
)
from app.main import create_app

pytestmark = pytest.mark.integration


def _create_evaluation_api_fixture(database_url: str) -> dict[str, object]:
    set_name = f"slice-038-{uuid4()}"
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT user_id FROM app_users WHERE login_id = 'alice.member'")
            user_id = cursor.fetchone()["user_id"]
            cursor.execute(
                """
                INSERT INTO golden_question_sets (
                    set_name,
                    description,
                    created_by_user_id
                )
                VALUES (%s, 'Slice 038 API fixture', %s)
                RETURNING question_set_id
                """,
                (set_name, user_id),
            )
            question_set_id = cursor.fetchone()["question_set_id"]
            cursor.execute(
                """
                INSERT INTO golden_questions (
                    question_set_id,
                    question_text,
                    question_type,
                    actor_user_id,
                    requested_search_scope,
                    top_k,
                    created_by_user_id
                )
                VALUES (
                    %s,
                    'Which nonexistent API policy applies?',
                    'no_answer',
                    %s,
                    'company',
                    5,
                    %s
                )
                RETURNING question_id
                """,
                (question_set_id, user_id, user_id),
            )
            question_id = cursor.fetchone()["question_id"]

    metric = evaluate_question(
        QuestionEvaluationInput(
            question_id=question_id,
            top_k=5,
            expected_targets=(
                ExpectedTarget(expected_heading_path=("No Answer",), relevance_grade=0),
            ),
            ranked_results=(),
        )
    )
    run = create_evaluation_run(
        database_url,
        EvaluationRunInput(
            question_set_id=question_set_id,
            run_name=f"slice-038-run-{uuid4()}",
            profile_name="kure_v1_1024",
            top_k=5,
        ),
    )
    result = create_evaluation_result(
        database_url,
        EvaluationResultInput(evaluation_run_id=run.evaluation_run_id, metric=metric),
    )
    completed_run = complete_evaluation_run(
        database_url,
        run.evaluation_run_id,
        summarize_question_metrics((metric,)),
    )
    return {
        "set_name": set_name,
        "question_set_id": question_set_id,
        "question_id": question_id,
        "evaluation_run_id": completed_run.evaluation_run_id,
        "evaluation_result_id": result.evaluation_result_id,
    }


def _cleanup_fixture(database_url: str, fixture: dict[str, object]) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM golden_question_sets WHERE set_name = %s",
                (fixture["set_name"],),
            )


def test_golden_evaluation_read_api_returns_question_sets_runs_and_detail(
    migrated_database_url: str,
) -> None:
    fixture = _create_evaluation_api_fixture(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))
    try:
        with TestClient(app) as client:
            question_sets_response = client.get("/api/evaluations/question-sets")
            runs_response = client.get(
                "/api/evaluations/runs",
                params={
                    "question_set_id": fixture["question_set_id"],
                    "profile_name": "kure_v1_1024",
                    "status": "succeeded",
                },
            )
            detail_response = client.get(f"/api/evaluations/runs/{fixture['evaluation_run_id']}")
            permission_audit_response = client.get(
                f"/api/evaluations/runs/{fixture['evaluation_run_id']}/permission-audit"
            )
            export_json_response = client.get(
                f"/api/evaluations/runs/{fixture['evaluation_run_id']}/export",
            )
            export_csv_response = client.get(
                f"/api/evaluations/runs/{fixture['evaluation_run_id']}/export",
                params={"format": "csv"},
            )
            page_response = client.get(
                "/evaluations",
                params={
                    "evaluation_run_id": fixture["evaluation_run_id"],
                    "question_set_id": fixture["question_set_id"],
                },
            )
            comparison_response = client.get(
                "/api/evaluations/profile-comparison",
                params={"question_set_id": fixture["question_set_id"]},
            )
            missing_comparison_response = client.get(
                "/api/evaluations/profile-comparison",
                params={"question_set_id": 999999999},
            )
            bad_comparison_response = client.get(
                "/api/evaluations/profile-comparison",
                params={"question_set_id": fixture["question_set_id"], "limit": 0},
            )
            missing_response = client.get("/api/evaluations/runs/999999999")
            missing_audit_response = client.get("/api/evaluations/runs/999999999/permission-audit")
            bad_audit_response = client.get(
                f"/api/evaluations/runs/{fixture['evaluation_run_id']}/permission-audit",
                params={"limit": 0},
            )
            missing_export_response = client.get("/api/evaluations/runs/999999999/export")
            bad_export_response = client.get(
                f"/api/evaluations/runs/{fixture['evaluation_run_id']}/export",
                params={"format": "xlsx"},
            )
            bad_request_response = client.get("/api/evaluations/runs", params={"limit": 0})

        question_sets = question_sets_response.json()["question_sets"]
        runs = runs_response.json()["runs"]
        detail = detail_response.json()
        permission_audit = permission_audit_response.json()["audit"]
        export_json = export_json_response.json()
        export_csv = export_csv_response.text
        result = detail["results"][0]
        comparison = comparison_response.json()["profiles"][0]

        assert question_sets_response.status_code == 200
        assert fixture["question_set_id"] in {item["question_set_id"] for item in question_sets}
        assert runs_response.status_code == 200
        assert len(runs) == 1
        assert runs[0]["evaluation_run_id"] == fixture["evaluation_run_id"]
        assert runs[0]["question_count"] == 1
        assert runs[0]["no_answer_success_rate"] == pytest.approx(1)
        assert detail_response.status_code == 200
        assert detail["run"]["status"] == "succeeded"
        assert detail["question_set"]["set_name"] == fixture["set_name"]
        assert result["evaluation_result_id"] == fixture["evaluation_result_id"]
        assert result["question_id"] == fixture["question_id"]
        assert result["no_answer_success"] is True
        assert permission_audit_response.status_code == 200
        assert permission_audit[0]["evaluation_result_id"] == fixture["evaluation_result_id"]
        assert permission_audit[0]["question_id"] == fixture["question_id"]
        assert permission_audit[0]["actor_login_id"] == "alice.member"
        assert permission_audit[0]["requested_search_scope"] == "company"
        assert permission_audit[0]["effective_search_scope"] == "company"
        assert permission_audit[0]["search_log_id"] is None
        assert export_json_response.status_code == 200
        assert export_json_response.headers["content-disposition"].endswith('.json"')
        assert export_json["version"] == 1
        assert export_json["run"]["evaluation_run_id"] == fixture["evaluation_run_id"]
        assert export_json["question_set"]["set_name"] == fixture["set_name"]
        assert export_json["results"][0]["evaluation_result_id"] == fixture["evaluation_result_id"]
        assert export_csv_response.status_code == 200
        assert export_csv_response.headers["content-type"].startswith("text/csv")
        assert export_csv_response.headers["content-disposition"].endswith('.csv"')
        assert "evaluation_run_id,run_name,question_set_id" in export_csv
        assert str(fixture["evaluation_run_id"]) in export_csv
        assert str(fixture["question_id"]) in export_csv
        assert comparison_response.status_code == 200
        assert comparison["evaluation_run_id"] == fixture["evaluation_run_id"]
        assert comparison["profile_name"] == "kure_v1_1024"
        assert comparison["no_answer_success_rate"] == pytest.approx(1)
        assert page_response.status_code == 200
        assert "Golden Evaluation Monitor" in page_response.text
        assert "Run Evaluation" in page_response.text
        assert "Profile Comparison" in page_response.text
        assert "Permission Audit" in page_response.text
        assert "/api/evaluations/profile-comparison" in page_response.text
        assert f"/api/evaluations/runs/{fixture['evaluation_run_id']}/permission-audit" in (
            page_response.text
        )
        assert 'id="evaluation-execute-form"' in page_response.text
        assert "/api/evaluations/runs/execute" in page_response.text
        assert "Run Search Experiment Batch" in page_response.text
        assert 'id="golden-search-experiment-form"' in page_response.text
        assert "/api/search/experiments/golden-question-set/run" in page_response.text
        assert 'href="/search/experiments"' in page_response.text
        assert f"#{fixture['evaluation_run_id']}" in page_response.text
        assert f"/api/evaluations/runs/{fixture['evaluation_run_id']}/export?format=json" in (
            page_response.text
        )
        assert f"/api/evaluations/runs/{fixture['evaluation_run_id']}/export?format=csv" in (
            page_response.text
        )
        assert fixture["set_name"] in page_response.text
        assert "kure_v1_1024" in page_response.text
        assert missing_comparison_response.status_code == 404
        assert bad_comparison_response.status_code == 400
        assert missing_response.status_code == 404
        assert missing_audit_response.status_code == 404
        assert bad_audit_response.status_code == 400
        assert missing_export_response.status_code == 404
        assert bad_export_response.status_code == 400
        assert bad_request_response.status_code == 400
    finally:
        _cleanup_fixture(migrated_database_url, fixture)
