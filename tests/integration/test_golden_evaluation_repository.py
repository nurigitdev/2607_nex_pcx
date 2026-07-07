from uuid import uuid4

import pytest

from app.core.database import connect
from app.core.evaluation_metrics import (
    ExpectedTarget,
    QuestionEvaluationInput,
    RankedSearchResult,
    evaluate_question,
    summarize_question_metrics,
)
from app.core.evaluation_runs import (
    EvaluationResultInput,
    EvaluationRunInput,
    complete_evaluation_run,
    create_evaluation_result,
    create_evaluation_run,
    get_evaluation_run,
    list_evaluation_results,
    list_evaluation_runs,
    run_golden_evaluation,
)

pytestmark = pytest.mark.integration


def _create_evaluation_fixture(database_url: str) -> dict[str, object]:
    checksum = f"golden-evaluation-repository-{uuid4()}"
    document_group = f"slice-037-{uuid4()}"
    set_name = f"slice-037-{uuid4()}"
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT user_id FROM app_users WHERE login_id = 'alice.member'")
            user_id = cursor.fetchone()["user_id"]
            cursor.execute(
                """
                INSERT INTO files (
                    original_file_name,
                    stored_file_name,
                    file_ext,
                    file_size_bytes,
                    sha256_checksum,
                    storage_path,
                    uploaded_by_user_id,
                    document_group
                )
                VALUES (%s, %s, '.md', 1, %s, %s, %s, %s)
                RETURNING file_id
                """,
                (
                    f"{checksum}.md",
                    f"{checksum}.stored.md",
                    checksum,
                    f"/tmp/{checksum}.md",
                    user_id,
                    document_group,
                ),
            )
            file_id = cursor.fetchone()["file_id"]
            cursor.execute(
                """
                INSERT INTO documents (
                    file_id,
                    document_title,
                    document_group,
                    owner_user_id,
                    access_scope
                )
                VALUES (%s, %s, %s, %s, 'company')
                RETURNING document_id
                """,
                (file_id, f"Golden evaluation repository {checksum}", document_group, user_id),
            )
            document_id = cursor.fetchone()["document_id"]
            chunk_ids = []
            for index, chunk_text in enumerate(("Visible answer chunk", "Hidden answer chunk")):
                cursor.execute(
                    """
                    INSERT INTO chunks (
                        document_id,
                        chunk_seq,
                        chunk_text,
                        content_hash,
                        chunk_policy_name,
                        char_count
                    )
                    VALUES (%s, %s, %s, %s, 'heading_512_64', %s)
                    RETURNING chunk_id
                    """,
                    (
                        document_id,
                        index,
                        chunk_text,
                        f"chunk-{checksum}-{index}",
                        len(chunk_text),
                    ),
                )
                chunk_ids.append(cursor.fetchone()["chunk_id"])

            cursor.execute(
                """
                INSERT INTO golden_question_sets (
                    set_name,
                    description,
                    created_by_user_id
                )
                VALUES (%s, 'Slice 037 repository fixture', %s)
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
                    document_group,
                    chunk_policy_name,
                    top_k,
                    created_by_user_id
                )
                VALUES (
                    %s,
                    'Which chunk is expected?',
                    'single_fact',
                    %s,
                    'company',
                    %s,
                    'heading_512_64',
                    2,
                    %s
                )
                RETURNING question_id
                """,
                (question_set_id, user_id, document_group, user_id),
            )
            fact_question_id = cursor.fetchone()["question_id"]
            cursor.execute(
                """
                INSERT INTO golden_question_expected_targets (
                    question_id,
                    chunk_id,
                    expectation_type,
                    relevance_grade
                )
                VALUES
                    (%s, %s, 'visible', 3),
                    (%s, %s, 'hidden', 0)
                """,
                (
                    fact_question_id,
                    chunk_ids[0],
                    fact_question_id,
                    chunk_ids[1],
                ),
            )
            cursor.execute(
                """
                INSERT INTO golden_questions (
                    question_set_id,
                    question_text,
                    question_type,
                    actor_user_id,
                    requested_search_scope,
                    document_group,
                    top_k,
                    created_by_user_id
                )
                VALUES (
                    %s,
                    'Which nonexistent policy applies?',
                    'no_answer',
                    %s,
                    'company',
                    %s,
                    2,
                    %s
                )
                RETURNING question_id
                """,
                (question_set_id, user_id, document_group, user_id),
            )
            no_answer_question_id = cursor.fetchone()["question_id"]
            cursor.execute(
                """
                INSERT INTO golden_question_expected_targets (
                    question_id,
                    expected_heading_path,
                    expectation_type,
                    relevance_grade
                )
                VALUES (%s, ARRAY['No Answer'], 'visible', 0)
                """,
                (no_answer_question_id,),
            )

    return {
        "file_id": file_id,
        "set_name": set_name,
        "question_set_id": question_set_id,
        "fact_question_id": fact_question_id,
        "no_answer_question_id": no_answer_question_id,
        "visible_chunk_id": chunk_ids[0],
        "hidden_chunk_id": chunk_ids[1],
    }


def _cleanup_fixture(database_url: str, fixture: dict[str, object]) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM golden_question_sets WHERE set_name = %s",
                (fixture["set_name"],),
            )
            cursor.execute("DELETE FROM files WHERE file_id = %s", (fixture["file_id"],))


def test_run_golden_evaluation_persists_results_and_summary(
    migrated_database_url: str,
) -> None:
    fixture = _create_evaluation_fixture(migrated_database_url)
    try:
        report = run_golden_evaluation(
            migrated_database_url,
            EvaluationRunInput(
                question_set_id=fixture["question_set_id"],
                run_name=f"slice-037-runner-{uuid4()}",
                profile_name="kure_v1_1024",
                chunk_policy_name="heading_512_64",
                top_k=2,
                runtime_metadata={"runner": "unit-ranked-results"},
            ),
            ranked_results_by_question={
                fixture["fact_question_id"]: (
                    RankedSearchResult(rank=1, chunk_id=fixture["visible_chunk_id"]),
                    RankedSearchResult(rank=2, chunk_id=fixture["hidden_chunk_id"]),
                )
            },
        )
        stored_run = get_evaluation_run(migrated_database_url, report.run.evaluation_run_id)
        listed_runs = list_evaluation_runs(
            migrated_database_url,
            question_set_id=fixture["question_set_id"],
            profile_name="kure_v1_1024",
            status="succeeded",
        )
        stored_results = list_evaluation_results(
            migrated_database_url,
            report.run.evaluation_run_id,
        )
        result_by_question = {result.question_id: result for result in stored_results}
        fact_result = result_by_question[fixture["fact_question_id"]]
        no_answer_result = result_by_question[fixture["no_answer_question_id"]]
        expected_summary = summarize_question_metrics(
            (
                evaluate_question(
                    QuestionEvaluationInput(
                        question_id=fixture["fact_question_id"],
                        top_k=2,
                        expected_targets=(
                            ExpectedTarget(
                                chunk_id=fixture["visible_chunk_id"],
                                relevance_grade=3,
                            ),
                            ExpectedTarget(
                                chunk_id=fixture["hidden_chunk_id"],
                                expectation_type="hidden",
                                relevance_grade=0,
                            ),
                        ),
                        ranked_results=(
                            RankedSearchResult(rank=1, chunk_id=fixture["visible_chunk_id"]),
                            RankedSearchResult(rank=2, chunk_id=fixture["hidden_chunk_id"]),
                        ),
                    )
                ),
                evaluate_question(
                    QuestionEvaluationInput(
                        question_id=fixture["no_answer_question_id"],
                        top_k=2,
                        expected_targets=(
                            ExpectedTarget(
                                expected_heading_path=("No Answer",),
                                relevance_grade=0,
                            ),
                        ),
                        ranked_results=(),
                    )
                ),
            )
        )

        assert stored_run == report.run
        assert report.run.status == "succeeded"
        assert report.run.started_at is not None
        assert report.run.finished_at is not None
        assert report.run.question_count == 2
        assert report.run.recall_question_count == 1
        assert report.run.ndcg_question_count == 1
        assert report.run.no_answer_question_count == 1
        assert report.run.hidden_violation_count == 1
        assert report.run.mean_recall_at_k == pytest.approx(1)
        assert report.run.mean_reciprocal_rank == pytest.approx(1)
        assert report.run.mean_ndcg == pytest.approx(1)
        assert report.run.no_answer_success_rate == pytest.approx(1)
        assert report.summary == expected_summary
        assert report.run in listed_runs
        assert len(report.results) == 2
        assert stored_results == list(report.results)
        assert fact_result.matched_chunk_ids == (fixture["visible_chunk_id"],)
        assert fact_result.hidden_violation_chunk_ids == (fixture["hidden_chunk_id"],)
        assert fact_result.recall_at_k == pytest.approx(1)
        assert no_answer_result.no_answer_success is True
    finally:
        _cleanup_fixture(migrated_database_url, fixture)


def test_evaluation_repository_supports_manual_run_and_result_lifecycle(
    migrated_database_url: str,
) -> None:
    fixture = _create_evaluation_fixture(migrated_database_url)
    try:
        run = create_evaluation_run(
            migrated_database_url,
            EvaluationRunInput(
                question_set_id=fixture["question_set_id"],
                run_name=f"slice-037-manual-{uuid4()}",
                profile_name="bge_m3_1024",
                top_k=3,
            ),
        )
        metric = evaluate_question(
            QuestionEvaluationInput(
                question_id=fixture["fact_question_id"],
                top_k=3,
                expected_targets=(ExpectedTarget(chunk_id=fixture["visible_chunk_id"]),),
                ranked_results=(RankedSearchResult(rank=1, chunk_id=fixture["visible_chunk_id"]),),
            )
        )
        result = create_evaluation_result(
            migrated_database_url,
            EvaluationResultInput(
                evaluation_run_id=run.evaluation_run_id,
                metric=metric,
                metadata={"manual": True},
            ),
        )
        summary = summarize_question_metrics((metric,))
        completed_run = complete_evaluation_run(
            migrated_database_url,
            run.evaluation_run_id,
            summary,
        )

        assert run.status == "pending"
        assert run.started_at is None
        assert result.metadata == {"manual": True}
        assert result.matched_chunk_ids == (fixture["visible_chunk_id"],)
        assert completed_run.status == "succeeded"
        assert completed_run.question_count == 1
        assert list_evaluation_results(migrated_database_url, run.evaluation_run_id) == [result]
    finally:
        _cleanup_fixture(migrated_database_url, fixture)


def test_evaluation_repository_returns_none_and_empty_results(
    migrated_database_url: str,
) -> None:
    assert get_evaluation_run(migrated_database_url, 999999999) is None
    assert list_evaluation_results(migrated_database_url, 999999999) == []
