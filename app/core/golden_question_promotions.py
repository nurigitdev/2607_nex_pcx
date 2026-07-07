"""Promote curated search results into golden question fixtures."""

from dataclasses import dataclass, field
from typing import Any

from app.core.database import connect
from app.core.golden_questions import (
    GoldenQuestionExpectedTargetInput,
    GoldenQuestionExpectedTargetRecord,
    GoldenQuestionInput,
    GoldenQuestionRecord,
    create_expected_target_in_connection,
    create_golden_question_in_connection,
)
from app.core.search_logs import (
    SearchLogDetailRecord,
    SearchLogRecord,
    SearchLogResultDetailRecord,
    get_search_log,
    get_search_log_detail,
    get_search_log_result,
)


@dataclass(frozen=True)
class GoldenQuestionPromotionInput:
    question_set_id: int
    search_log_result_id: int
    question_type: str = "single_fact"
    expectation_type: str = "visible"
    relevance_grade: int = 3
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_by_user_id: int | None = None


@dataclass(frozen=True)
class GoldenQuestionPromotionRecord:
    question: GoldenQuestionRecord
    expected_target: GoldenQuestionExpectedTargetRecord
    source_search_log: SearchLogRecord
    source_result: SearchLogResultDetailRecord


class InvalidGoldenQuestionPromotionError(ValueError):
    """Raised when a search result cannot be promoted into a golden question."""


def _require_positive_id(value: int | None, field_name: str, *, required: bool = True) -> None:
    if value is None:
        if required:
            raise InvalidGoldenQuestionPromotionError(f"{field_name} must be greater than 0")
        return
    if value <= 0:
        raise InvalidGoldenQuestionPromotionError(f"{field_name} must be greater than 0")


def _validate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise InvalidGoldenQuestionPromotionError("metadata must be a JSON object")
    return dict(metadata)


def validate_golden_question_promotion_input(
    promotion_input: GoldenQuestionPromotionInput,
) -> GoldenQuestionPromotionInput:
    _require_positive_id(promotion_input.question_set_id, "question_set_id")
    _require_positive_id(promotion_input.search_log_result_id, "search_log_result_id")
    _require_positive_id(
        promotion_input.created_by_user_id,
        "created_by_user_id",
        required=False,
    )
    return GoldenQuestionPromotionInput(
        question_set_id=promotion_input.question_set_id,
        search_log_result_id=promotion_input.search_log_result_id,
        question_type=promotion_input.question_type,
        expectation_type=promotion_input.expectation_type,
        relevance_grade=promotion_input.relevance_grade,
        notes=promotion_input.notes,
        metadata=_validate_metadata(promotion_input.metadata),
        created_by_user_id=promotion_input.created_by_user_id,
    )


def _find_source_result(
    detail: SearchLogDetailRecord,
    search_log_result_id: int,
) -> SearchLogResultDetailRecord | None:
    return next(
        (
            result
            for result in detail.results
            if result.search_log_result.search_log_result_id == search_log_result_id
        ),
        None,
    )


def _promotion_metadata(
    source_search_log: SearchLogRecord,
    source_result: SearchLogResultDetailRecord,
) -> dict[str, object]:
    result = source_result.search_log_result
    return {
        "source": "search_result_promotion",
        "search_log_id": source_search_log.search_log_id,
        "search_log_result_id": result.search_log_result_id,
        "profile_name": result.profile_name,
        "rank": result.rank,
        "chunk_id": result.chunk_id,
        "feedback_labels": [feedback.relevance_label for feedback in source_result.feedback],
    }


def promote_search_result_to_golden_question(
    database_url: str,
    promotion_input: GoldenQuestionPromotionInput,
) -> GoldenQuestionPromotionRecord | None:
    validated = validate_golden_question_promotion_input(promotion_input)
    search_result = get_search_log_result(database_url, validated.search_log_result_id)
    if search_result is None:
        return None

    source_search_log = get_search_log(database_url, search_result.search_log_id)
    if source_search_log is None:
        return None

    detail = get_search_log_detail(database_url, source_search_log.search_log_id)
    if detail is None:
        return None
    source_result = _find_source_result(detail, validated.search_log_result_id)
    if source_result is None:
        return None

    promotion_metadata = _promotion_metadata(source_search_log, source_result)
    metadata = {**validated.metadata, "promotion": promotion_metadata}
    created_by_user_id = (
        validated.created_by_user_id
        or source_search_log.created_by_user_id
        or source_search_log.actor_user_id
    )

    with connect(database_url) as connection:
        question = create_golden_question_in_connection(
            connection,
            GoldenQuestionInput(
                question_set_id=validated.question_set_id,
                question_text=source_search_log.query_text,
                question_type=validated.question_type,
                actor_user_id=source_search_log.actor_user_id,
                requested_search_scope=source_search_log.requested_search_scope or "company",
                document_group=source_search_log.document_group or source_result.document_group,
                file_type=source_search_log.file_type or source_result.file_ext,
                chunk_policy_name=(
                    source_search_log.chunk_policy_name or source_result.chunk_policy_name
                ),
                top_k=source_search_log.top_k,
                metadata=metadata,
                created_by_user_id=created_by_user_id,
            ),
        )
        expected_target = create_expected_target_in_connection(
            connection,
            GoldenQuestionExpectedTargetInput(
                question_id=question.question_id,
                chunk_id=source_result.search_log_result.chunk_id,
                expected_heading_path=source_result.heading_path,
                expectation_type=validated.expectation_type,
                relevance_grade=validated.relevance_grade,
                notes=validated.notes,
                metadata=metadata,
            ),
        )

    return GoldenQuestionPromotionRecord(
        question=question,
        expected_target=expected_target,
        source_search_log=source_search_log,
        source_result=source_result,
    )
