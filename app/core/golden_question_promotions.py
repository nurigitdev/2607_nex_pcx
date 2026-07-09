"""Promote curated search results into golden question fixtures."""

from dataclasses import dataclass, field
from datetime import datetime
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
class GoldenQuestionBatchPromotionInput:
    question_set_id: int
    search_log_result_ids: tuple[int, ...]
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


@dataclass(frozen=True)
class GoldenQuestionCandidateRecord:
    search_log_result_id: int
    search_log_id: int
    query_text: str
    actor_user_id: int | None
    actor_login_id: str | None
    actor_display_name: str | None
    requested_search_scope: str | None
    document_group: str | None
    file_type: str | None
    chunk_policy_name: str | None
    top_k: int
    profile_name: str
    rank: int
    chunk_id: int
    score: float | None
    document_id: int
    document_title: str | None
    original_file_name: str
    heading_path: tuple[str, ...]
    chunk_preview: str
    feedback_count: int
    correct_count: int
    partial_count: int
    feedback_labels: tuple[str, ...]
    latest_feedback_comment: str | None
    latest_feedback_at: datetime | None
    already_promoted: bool


@dataclass(frozen=True)
class GoldenQuestionBatchPromotionRecord:
    promotions: tuple[GoldenQuestionPromotionRecord, ...]
    skipped_search_log_result_ids: tuple[int, ...]
    missing_search_log_result_ids: tuple[int, ...]


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


def _validate_limit(limit: int, *, max_limit: int = 100) -> int:
    if limit <= 0:
        raise InvalidGoldenQuestionPromotionError("limit must be greater than 0")
    if limit > max_limit:
        raise InvalidGoldenQuestionPromotionError(
            f"limit must be less than or equal to {max_limit}"
        )
    return limit


def _validate_search_log_result_ids(
    search_log_result_ids: tuple[int, ...],
    *,
    max_count: int = 50,
) -> tuple[int, ...]:
    if not search_log_result_ids:
        raise InvalidGoldenQuestionPromotionError("search_log_result_ids must not be empty")
    if len(search_log_result_ids) > max_count:
        raise InvalidGoldenQuestionPromotionError(
            f"search_log_result_ids must contain at most {max_count} ids"
        )

    normalized: list[int] = []
    seen: set[int] = set()
    for search_log_result_id in search_log_result_ids:
        _require_positive_id(search_log_result_id, "search_log_result_id")
        if search_log_result_id not in seen:
            normalized.append(search_log_result_id)
            seen.add(search_log_result_id)
    return tuple(normalized)


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


def validate_golden_question_batch_promotion_input(
    promotion_input: GoldenQuestionBatchPromotionInput,
) -> GoldenQuestionBatchPromotionInput:
    _require_positive_id(promotion_input.question_set_id, "question_set_id")
    _require_positive_id(
        promotion_input.created_by_user_id,
        "created_by_user_id",
        required=False,
    )
    return GoldenQuestionBatchPromotionInput(
        question_set_id=promotion_input.question_set_id,
        search_log_result_ids=_validate_search_log_result_ids(
            tuple(promotion_input.search_log_result_ids)
        ),
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


def _chunk_preview(chunk_text: str, limit: int = 220) -> str:
    normalized = " ".join(chunk_text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "..."


def _row_to_golden_question_candidate_record(row: dict[str, Any]) -> GoldenQuestionCandidateRecord:
    return GoldenQuestionCandidateRecord(
        search_log_result_id=int(row["search_log_result_id"]),
        search_log_id=int(row["search_log_id"]),
        query_text=str(row["query_text"]),
        actor_user_id=int(row["actor_user_id"]) if row["actor_user_id"] is not None else None,
        actor_login_id=row["actor_login_id"],
        actor_display_name=row["actor_display_name"],
        requested_search_scope=row["requested_search_scope"],
        document_group=row["document_group"],
        file_type=row["file_type"],
        chunk_policy_name=row["chunk_policy_name"],
        top_k=int(row["top_k"]),
        profile_name=str(row["profile_name"]),
        rank=int(row["rank"]),
        chunk_id=int(row["chunk_id"]),
        score=float(row["score"]) if row["score"] is not None else None,
        document_id=int(row["document_id"]),
        document_title=row["document_title"],
        original_file_name=str(row["original_file_name"]),
        heading_path=tuple(row["heading_path"] or ()),
        chunk_preview=_chunk_preview(str(row["chunk_text"])),
        feedback_count=int(row["feedback_count"] or 0),
        correct_count=int(row["correct_count"] or 0),
        partial_count=int(row["partial_count"] or 0),
        feedback_labels=tuple(row["feedback_labels"] or ()),
        latest_feedback_comment=row["latest_feedback_comment"],
        latest_feedback_at=row["latest_feedback_at"],
        already_promoted=bool(row["already_promoted"]),
    )


def list_golden_question_candidates(
    database_url: str,
    *,
    document_group: str | None = None,
    include_promoted: bool = False,
    limit: int = 20,
) -> list[GoldenQuestionCandidateRecord]:
    validated_limit = _validate_limit(limit)
    filters = ["srf.relevance_label IN ('correct', 'partial')"]
    params: list[object] = []
    if document_group is not None:
        normalized_document_group = document_group.strip()
        if not normalized_document_group:
            raise InvalidGoldenQuestionPromotionError("document_group must not be blank")
        filters.append("sl.document_group = %s")
        params.append(normalized_document_group)
    if not include_promoted:
        filters.append("""
            NOT EXISTS (
                SELECT 1
                FROM golden_questions gq
                WHERE gq.metadata #>> '{promotion,search_log_result_id}' =
                      slr.search_log_result_id::text
            )
            """)

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    slr.search_log_result_id,
                    sl.search_log_id,
                    sl.query_text,
                    sl.actor_user_id,
                    au.login_id AS actor_login_id,
                    au.display_name AS actor_display_name,
                    sl.requested_search_scope,
                    sl.document_group,
                    sl.file_type,
                    sl.chunk_policy_name,
                    sl.top_k,
                    slr.profile_name,
                    slr.rank,
                    slr.chunk_id,
                    slr.score,
                    c.document_id,
                    c.chunk_text,
                    c.heading_path,
                    d.document_title,
                    f.original_file_name,
                    count(srf.feedback_id) AS feedback_count,
                    count(srf.feedback_id) FILTER (
                        WHERE srf.relevance_label = 'correct'
                    ) AS correct_count,
                    count(srf.feedback_id) FILTER (
                        WHERE srf.relevance_label = 'partial'
                    ) AS partial_count,
                    array_agg(DISTINCT srf.relevance_label) AS feedback_labels,
                    (
                        array_agg(
                            srf.comment
                            ORDER BY srf.created_at DESC, srf.feedback_id DESC
                        ) FILTER (
                            WHERE srf.comment IS NOT NULL
                              AND length(btrim(srf.comment)) > 0
                        )
                    )[1] AS latest_feedback_comment,
                    max(srf.created_at) AS latest_feedback_at,
                    EXISTS (
                        SELECT 1
                        FROM golden_questions gq
                        WHERE gq.metadata #>> '{{promotion,search_log_result_id}}' =
                              slr.search_log_result_id::text
                    ) AS already_promoted
                FROM search_result_feedback srf
                JOIN search_log_results slr
                  ON slr.search_log_result_id = srf.search_log_result_id
                JOIN search_logs sl ON sl.search_log_id = slr.search_log_id
                LEFT JOIN app_users au ON au.user_id = sl.actor_user_id
                JOIN chunks c ON c.chunk_id = slr.chunk_id
                JOIN documents d ON d.document_id = c.document_id
                JOIN files f ON f.file_id = d.file_id
                WHERE {' AND '.join(filters)}
                GROUP BY
                    slr.search_log_result_id,
                    sl.search_log_id,
                    au.login_id,
                    au.display_name,
                    c.document_id,
                    c.chunk_text,
                    c.heading_path,
                    d.document_title,
                    f.original_file_name
                ORDER BY
                    count(srf.feedback_id) FILTER (
                        WHERE srf.relevance_label = 'correct'
                    ) DESC,
                    count(srf.feedback_id) DESC,
                    max(srf.created_at) DESC,
                    slr.rank ASC,
                    slr.search_log_result_id DESC
                LIMIT %s
                """,
                [*params, validated_limit],
            )
            rows = cursor.fetchall()
    return [_row_to_golden_question_candidate_record(dict(row)) for row in rows]


def _promoted_search_log_result_ids(
    database_url: str,
    search_log_result_ids: tuple[int, ...],
) -> set[int]:
    if not search_log_result_ids:
        return set()
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT metadata #>> '{promotion,search_log_result_id}' AS search_log_result_id
                FROM golden_questions
                WHERE metadata #>> '{promotion,search_log_result_id}' = ANY(%s)
                """,
                ([str(search_log_result_id) for search_log_result_id in search_log_result_ids],),
            )
            rows = cursor.fetchall()
    return {int(row["search_log_result_id"]) for row in rows if row["search_log_result_id"]}


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


def promote_search_results_to_golden_questions(
    database_url: str,
    promotion_input: GoldenQuestionBatchPromotionInput,
) -> GoldenQuestionBatchPromotionRecord:
    validated = validate_golden_question_batch_promotion_input(promotion_input)
    skipped_ids = _promoted_search_log_result_ids(database_url, validated.search_log_result_ids)
    promotions: list[GoldenQuestionPromotionRecord] = []
    missing_ids: list[int] = []

    for search_log_result_id in validated.search_log_result_ids:
        if search_log_result_id in skipped_ids:
            continue
        promotion = promote_search_result_to_golden_question(
            database_url,
            GoldenQuestionPromotionInput(
                question_set_id=validated.question_set_id,
                search_log_result_id=search_log_result_id,
                question_type=validated.question_type,
                expectation_type=validated.expectation_type,
                relevance_grade=validated.relevance_grade,
                notes=validated.notes,
                metadata=validated.metadata,
                created_by_user_id=validated.created_by_user_id,
            ),
        )
        if promotion is None:
            missing_ids.append(search_log_result_id)
        else:
            promotions.append(promotion)

    return GoldenQuestionBatchPromotionRecord(
        promotions=tuple(promotions),
        skipped_search_log_result_ids=tuple(
            search_log_result_id
            for search_log_result_id in validated.search_log_result_ids
            if search_log_result_id in skipped_ids
        ),
        missing_search_log_result_ids=tuple(missing_ids),
    )
