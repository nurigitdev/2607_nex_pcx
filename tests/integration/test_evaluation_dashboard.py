from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect
from app.core.evaluation_dashboard import get_evaluation_dashboard_summary
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


def _create_dashboard_fixture(database_url: str) -> dict[str, object]:
    set_name = f"slice-048-{uuid4()}"
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
                VALUES (%s, 'Slice 048 dashboard fixture', %s)
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
                    'Which dashboard-only policy should not exist?',
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
            cursor.execute(
                """
                INSERT INTO golden_question_expected_targets (
                    question_id,
                    expected_heading_path,
                    expectation_type,
                    relevance_grade
                )
                VALUES (%s, ARRAY['No Answer']::TEXT[], 'visible', 0)
                """,
                (question_id,),
            )

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
    succeeded_run = create_evaluation_run(
        database_url,
        EvaluationRunInput(
            question_set_id=question_set_id,
            run_name=f"slice-048-succeeded-{uuid4()}",
            profile_name="kure_v1_1024",
            top_k=5,
        ),
    )
    create_evaluation_result(
        database_url,
        EvaluationResultInput(evaluation_run_id=succeeded_run.evaluation_run_id, metric=metric),
    )
    completed_run = complete_evaluation_run(
        database_url,
        succeeded_run.evaluation_run_id,
        summarize_question_metrics((metric,)),
    )
    failed_run = create_evaluation_run(
        database_url,
        EvaluationRunInput(
            question_set_id=question_set_id,
            run_name=f"slice-048-failed-{uuid4()}",
            profile_name="kure_v1_1024",
            status="failed",
            top_k=5,
        ),
    )
    return {
        "set_name": set_name,
        "question_set_id": question_set_id,
        "question_id": question_id,
        "succeeded_run_id": completed_run.evaluation_run_id,
        "failed_run_id": failed_run.evaluation_run_id,
    }


def _cleanup_dashboard_fixture(database_url: str, fixture: dict[str, object]) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM golden_question_sets WHERE set_name = %s",
                (fixture["set_name"],),
            )


def test_evaluation_dashboard_summary_api_and_page(
    migrated_database_url: str,
) -> None:
    fixture = _create_dashboard_fixture(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))
    try:
        summary = get_evaluation_dashboard_summary(migrated_database_url, recent_limit=10)
        status_counts = {item.status: item.count for item in summary.status_counts}

        with TestClient(app) as client:
            api_response = client.get("/api/dashboard/evaluations", params={"recent_limit": 10})
            bad_response = client.get("/api/dashboard/evaluations", params={"recent_limit": 0})
            page_response = client.get("/")

        api_payload = api_response.json()["evaluations"]
        recent_run_ids = {run.evaluation_run_id for run in summary.recent_runs}
        api_recent_run_ids = {run["evaluation_run_id"] for run in api_payload["recent_runs"]}

        assert summary.active_question_set_count >= 1
        assert summary.question_count >= 1
        assert summary.expected_target_count >= 1
        assert status_counts["succeeded"] >= 1
        assert status_counts["failed"] >= 1
        assert fixture["succeeded_run_id"] in recent_run_ids
        assert fixture["failed_run_id"] in recent_run_ids
        assert api_response.status_code == 200
        assert api_payload["active_question_set_count"] == summary.active_question_set_count
        assert fixture["succeeded_run_id"] in api_recent_run_ids
        assert bad_response.status_code == 400
        assert page_response.status_code == 200
        assert "Golden Evaluation Snapshot" in page_response.text
        assert "Active Question Sets" in page_response.text
        assert fixture["set_name"] in page_response.text
        assert f"#{fixture['succeeded_run_id']}" in page_response.text
        assert "/api/dashboard/evaluations" in page_response.text
    finally:
        _cleanup_dashboard_fixture(migrated_database_url, fixture)
