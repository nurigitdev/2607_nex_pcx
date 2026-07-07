import csv
import io
from datetime import UTC, datetime

from app.core.evaluation_runs import EvaluationResultRecord, EvaluationRunRecord
from app.core.golden_questions import GoldenQuestionSetRecord
from app.main import evaluation_results_csv, evaluation_run_export_payload


def _run_record(now: datetime) -> EvaluationRunRecord:
    return EvaluationRunRecord(
        evaluation_run_id=10,
        question_set_id=20,
        run_name="export-run",
        profile_name="kure_v1_1024",
        chunk_policy_name=None,
        similarity_metric="cosine",
        top_k=5,
        status="succeeded",
        question_count=1,
        recall_question_count=1,
        ndcg_question_count=1,
        no_answer_question_count=0,
        hidden_violation_count=0,
        mean_recall_at_k=1.0,
        mean_reciprocal_rank=1.0,
        mean_ndcg=1.0,
        no_answer_success_rate=None,
        runtime_metadata={"adapter": "mock"},
        error_message=None,
        started_at=now,
        finished_at=now,
        created_at=now,
        updated_at=now,
    )


def _question_set_record(now: datetime) -> GoldenQuestionSetRecord:
    return GoldenQuestionSetRecord(
        question_set_id=20,
        set_name="Export Set",
        description="Export fixture",
        is_active=True,
        metadata={},
        created_by_user_id=None,
        created_at=now,
        updated_at=now,
    )


def _result_record(now: datetime) -> EvaluationResultRecord:
    return EvaluationResultRecord(
        evaluation_result_id=30,
        evaluation_run_id=10,
        question_id=40,
        search_log_id=None,
        top_k=5,
        visible_expected_count=2,
        retrieved_count=5,
        matched_visible_count=2,
        hidden_violation_count=0,
        matched_chunk_ids=(101, 102),
        hidden_violation_chunk_ids=(),
        recall_at_k=1.0,
        reciprocal_rank=0.5,
        dcg=1.5,
        ideal_dcg=1.5,
        ndcg=1.0,
        no_answer_success=None,
        metadata={"note": "ok"},
        created_at=now,
    )


def test_evaluation_run_export_payload_and_csv_include_results() -> None:
    now = datetime(2026, 7, 7, 12, 0, tzinfo=UTC)
    run = _run_record(now)
    question_set = _question_set_record(now)
    result = _result_record(now)

    payload = evaluation_run_export_payload(run, question_set, [result])
    csv_text = evaluation_results_csv(run, question_set, [result])
    rows = list(csv.DictReader(io.StringIO(csv_text)))

    assert payload["version"] == 1
    assert payload["run"]["evaluation_run_id"] == 10
    assert payload["question_set"]["set_name"] == "Export Set"
    assert payload["results"][0]["matched_chunk_ids"] == [101, 102]
    assert rows[0]["evaluation_run_id"] == "10"
    assert rows[0]["question_set_name"] == "Export Set"
    assert rows[0]["matched_chunk_ids"] == "101,102"
    assert rows[0]["chunk_policy_name"] == ""


def test_evaluation_results_csv_writes_header_without_results() -> None:
    now = datetime(2026, 7, 7, 12, 0, tzinfo=UTC)

    csv_text = evaluation_results_csv(_run_record(now), None, [])

    assert csv_text.startswith("evaluation_run_id,run_name,question_set_id")
    assert list(csv.DictReader(io.StringIO(csv_text))) == []
