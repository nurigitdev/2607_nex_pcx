"""Search log, result, and feedback repository helpers."""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from psycopg import Connection
from psycopg.types.json import Json

from app.core.database import connect

SEARCH_SCOPES = {"mine", "team", "managed_org", "company"}
SIMILARITY_METRICS = {"cosine", "l2", "inner_product"}
FEEDBACK_LABELS = {"correct", "partial", "wrong", "duplicate", "insufficient_context"}


@dataclass(frozen=True)
class SearchLogInput:
    query_text: str
    top_k: int
    profiles: tuple[str, ...]
    normalized_query_text: str | None = None
    actor_user_id: int | None = None
    requested_search_scope: str | None = None
    effective_search_scope: str | None = None
    permission_filter_metadata: dict[str, Any] = field(default_factory=dict)
    document_group: str | None = None
    file_type: str | None = None
    chunk_policy_name: str | None = None
    similarity_metric: str = "cosine"
    query_runtime_metadata: dict[str, Any] = field(default_factory=dict)
    total_elapsed_ms: int | None = None
    created_by: str | None = None
    created_by_user_id: int | None = None


@dataclass(frozen=True)
class SearchLogRecord:
    search_log_id: int
    query_text: str
    normalized_query_text: str | None
    actor_user_id: int | None
    requested_search_scope: str | None
    effective_search_scope: str | None
    permission_filter_metadata: dict[str, Any]
    document_group: str | None
    file_type: str | None
    chunk_policy_name: str | None
    top_k: int
    similarity_metric: str
    profiles: tuple[str, ...]
    query_runtime_metadata: dict[str, Any]
    total_elapsed_ms: int | None
    created_by: str | None
    created_by_user_id: int | None
    created_at: datetime


@dataclass(frozen=True)
class SearchLogResultInput:
    search_log_id: int
    profile_name: str
    rank: int
    chunk_id: int
    distance: float | None = None
    score: float | None = None
    profile_elapsed_ms: int | None = None


@dataclass(frozen=True)
class SearchLogResultRecord:
    search_log_result_id: int
    search_log_id: int
    profile_name: str
    rank: int
    chunk_id: int
    distance: float | None
    score: float | None
    profile_elapsed_ms: int | None
    created_at: datetime


@dataclass(frozen=True)
class SearchResultFeedbackInput:
    search_log_result_id: int
    relevance_label: str
    comment: str | None = None
    created_by: str | None = None
    created_by_user_id: int | None = None


@dataclass(frozen=True)
class SearchResultFeedbackRecord:
    feedback_id: int
    search_log_result_id: int
    relevance_label: str
    comment: str | None
    created_by: str | None
    created_by_user_id: int | None
    created_at: datetime


@dataclass(frozen=True)
class SearchFeedbackProfileSummaryRecord:
    profile_name: str
    feedback_count: int
    search_log_count: int
    result_count: int
    correct_count: int
    partial_count: int
    wrong_count: int
    duplicate_count: int
    insufficient_context_count: int
    average_rank: float | None
    average_score: float | None
    average_profile_elapsed_ms: float | None
    latest_feedback_at: datetime | None

    @property
    def relevant_count(self) -> int:
        return self.correct_count + self.partial_count


@dataclass(frozen=True)
class SearchFeedbackSummaryRecord:
    feedback_count: int
    search_log_count: int
    result_count: int
    latest_feedback_at: datetime | None
    profiles: tuple[SearchFeedbackProfileSummaryRecord, ...]


class InvalidSearchLogError(ValueError):
    """Raised when a search log operation is invalid before reaching the DB."""


def _require_positive_id(value: int | None, field_name: str) -> None:
    if value is not None and value <= 0:
        raise InvalidSearchLogError(f"{field_name} must be greater than 0")


def _validate_nonblank(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise InvalidSearchLogError(f"{field_name} must not be blank")
    return normalized


def _validate_scope(scope: str | None, field_name: str) -> str | None:
    normalized = _validate_nonblank(scope, field_name)
    if normalized is not None and normalized not in SEARCH_SCOPES:
        raise InvalidSearchLogError(f"Unsupported {field_name}: {normalized}")
    return normalized


def _validate_similarity_metric(similarity_metric: str) -> str:
    metric = similarity_metric.strip()
    if metric not in SIMILARITY_METRICS:
        raise InvalidSearchLogError(f"Unsupported similarity metric: {similarity_metric}")
    return metric


def _validate_profiles(profiles: tuple[str, ...]) -> tuple[str, ...]:
    if not profiles:
        raise InvalidSearchLogError("profiles must not be empty")
    normalized_profiles = tuple(_validate_nonblank(profile, "profile_name") for profile in profiles)
    if len(set(normalized_profiles)) != len(normalized_profiles):
        raise InvalidSearchLogError("profiles must be unique")
    return normalized_profiles


def _validate_elapsed_ms(elapsed_ms: int | None, field_name: str) -> None:
    if elapsed_ms is not None and elapsed_ms < 0:
        raise InvalidSearchLogError(f"{field_name} must be greater than or equal to 0")


def _validate_finite_float(value: float | None, field_name: str) -> None:
    if value is not None and not math.isfinite(value):
        raise InvalidSearchLogError(f"{field_name} must be finite")


def validate_search_log_input(search_log_input: SearchLogInput) -> SearchLogInput:
    query_text = _validate_nonblank(search_log_input.query_text, "query_text")
    if search_log_input.top_k <= 0:
        raise InvalidSearchLogError("top_k must be greater than 0")
    _require_positive_id(search_log_input.actor_user_id, "actor_user_id")
    _require_positive_id(search_log_input.created_by_user_id, "created_by_user_id")
    _validate_elapsed_ms(search_log_input.total_elapsed_ms, "total_elapsed_ms")
    return SearchLogInput(
        query_text=query_text or search_log_input.query_text,
        normalized_query_text=_validate_nonblank(
            search_log_input.normalized_query_text,
            "normalized_query_text",
        ),
        actor_user_id=search_log_input.actor_user_id,
        requested_search_scope=_validate_scope(
            search_log_input.requested_search_scope,
            "requested_search_scope",
        ),
        effective_search_scope=_validate_scope(
            search_log_input.effective_search_scope,
            "effective_search_scope",
        ),
        permission_filter_metadata=dict(search_log_input.permission_filter_metadata),
        document_group=_validate_nonblank(search_log_input.document_group, "document_group"),
        file_type=_validate_nonblank(search_log_input.file_type, "file_type"),
        chunk_policy_name=_validate_nonblank(
            search_log_input.chunk_policy_name,
            "chunk_policy_name",
        ),
        top_k=search_log_input.top_k,
        similarity_metric=_validate_similarity_metric(search_log_input.similarity_metric),
        profiles=_validate_profiles(search_log_input.profiles),
        query_runtime_metadata=dict(search_log_input.query_runtime_metadata),
        total_elapsed_ms=search_log_input.total_elapsed_ms,
        created_by=_validate_nonblank(search_log_input.created_by, "created_by"),
        created_by_user_id=search_log_input.created_by_user_id,
    )


def validate_search_log_result_input(
    result_input: SearchLogResultInput,
) -> SearchLogResultInput:
    _require_positive_id(result_input.search_log_id, "search_log_id")
    _require_positive_id(result_input.chunk_id, "chunk_id")
    if result_input.rank <= 0:
        raise InvalidSearchLogError("rank must be greater than 0")
    _validate_finite_float(result_input.distance, "distance")
    _validate_finite_float(result_input.score, "score")
    _validate_elapsed_ms(result_input.profile_elapsed_ms, "profile_elapsed_ms")
    return SearchLogResultInput(
        search_log_id=result_input.search_log_id,
        profile_name=_validate_nonblank(result_input.profile_name, "profile_name")
        or result_input.profile_name,
        rank=result_input.rank,
        chunk_id=result_input.chunk_id,
        distance=result_input.distance,
        score=result_input.score,
        profile_elapsed_ms=result_input.profile_elapsed_ms,
    )


def validate_search_result_feedback_input(
    feedback_input: SearchResultFeedbackInput,
) -> SearchResultFeedbackInput:
    _require_positive_id(feedback_input.search_log_result_id, "search_log_result_id")
    label = _validate_nonblank(feedback_input.relevance_label, "relevance_label")
    if label not in FEEDBACK_LABELS:
        raise InvalidSearchLogError(f"Unsupported relevance_label: {label}")
    _require_positive_id(feedback_input.created_by_user_id, "created_by_user_id")
    return SearchResultFeedbackInput(
        search_log_result_id=feedback_input.search_log_result_id,
        relevance_label=label,
        comment=_validate_nonblank(feedback_input.comment, "comment"),
        created_by=_validate_nonblank(feedback_input.created_by, "created_by"),
        created_by_user_id=feedback_input.created_by_user_id,
    )


def _row_to_search_log_record(row: dict[str, Any]) -> SearchLogRecord:
    return SearchLogRecord(
        search_log_id=int(row["search_log_id"]),
        query_text=str(row["query_text"]),
        normalized_query_text=row["normalized_query_text"],
        actor_user_id=int(row["actor_user_id"]) if row["actor_user_id"] is not None else None,
        requested_search_scope=row["requested_search_scope"],
        effective_search_scope=row["effective_search_scope"],
        permission_filter_metadata=dict(row["permission_filter_metadata"] or {}),
        document_group=row["document_group"],
        file_type=row["file_type"],
        chunk_policy_name=row["chunk_policy_name"],
        top_k=int(row["top_k"]),
        similarity_metric=str(row["similarity_metric"]),
        profiles=tuple(row["profiles"] or ()),
        query_runtime_metadata=dict(row["query_runtime_metadata"] or {}),
        total_elapsed_ms=(
            int(row["total_elapsed_ms"]) if row["total_elapsed_ms"] is not None else None
        ),
        created_by=row["created_by"],
        created_by_user_id=(
            int(row["created_by_user_id"]) if row["created_by_user_id"] is not None else None
        ),
        created_at=row["created_at"],
    )


def _row_to_search_log_result_record(row: dict[str, Any]) -> SearchLogResultRecord:
    return SearchLogResultRecord(
        search_log_result_id=int(row["search_log_result_id"]),
        search_log_id=int(row["search_log_id"]),
        profile_name=str(row["profile_name"]),
        rank=int(row["rank"]),
        chunk_id=int(row["chunk_id"]),
        distance=float(row["distance"]) if row["distance"] is not None else None,
        score=float(row["score"]) if row["score"] is not None else None,
        profile_elapsed_ms=(
            int(row["profile_elapsed_ms"]) if row["profile_elapsed_ms"] is not None else None
        ),
        created_at=row["created_at"],
    )


def _row_to_search_result_feedback_record(row: dict[str, Any]) -> SearchResultFeedbackRecord:
    return SearchResultFeedbackRecord(
        feedback_id=int(row["feedback_id"]),
        search_log_result_id=int(row["search_log_result_id"]),
        relevance_label=str(row["relevance_label"]),
        comment=row["comment"],
        created_by=row["created_by"],
        created_by_user_id=(
            int(row["created_by_user_id"]) if row["created_by_user_id"] is not None else None
        ),
        created_at=row["created_at"],
    )


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _row_to_search_feedback_profile_summary_record(
    row: dict[str, Any],
) -> SearchFeedbackProfileSummaryRecord:
    return SearchFeedbackProfileSummaryRecord(
        profile_name=str(row["profile_name"]),
        feedback_count=int(row["feedback_count"] or 0),
        search_log_count=int(row["search_log_count"] or 0),
        result_count=int(row["result_count"] or 0),
        correct_count=int(row["correct_count"] or 0),
        partial_count=int(row["partial_count"] or 0),
        wrong_count=int(row["wrong_count"] or 0),
        duplicate_count=int(row["duplicate_count"] or 0),
        insufficient_context_count=int(row["insufficient_context_count"] or 0),
        average_rank=_optional_float(row["average_rank"]),
        average_score=_optional_float(row["average_score"]),
        average_profile_elapsed_ms=_optional_float(row["average_profile_elapsed_ms"]),
        latest_feedback_at=row["latest_feedback_at"],
    )


def create_search_log_in_connection(
    connection: Connection,
    search_log_input: SearchLogInput,
) -> SearchLogRecord:
    validated = validate_search_log_input(search_log_input)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO search_logs (
                query_text,
                normalized_query_text,
                actor_user_id,
                requested_search_scope,
                effective_search_scope,
                permission_filter_metadata,
                document_group,
                file_type,
                chunk_policy_name,
                top_k,
                similarity_metric,
                profiles,
                query_runtime_metadata,
                total_elapsed_ms,
                created_by,
                created_by_user_id
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING *
            """,
            (
                validated.query_text,
                validated.normalized_query_text,
                validated.actor_user_id,
                validated.requested_search_scope,
                validated.effective_search_scope,
                Json(validated.permission_filter_metadata),
                validated.document_group,
                validated.file_type,
                validated.chunk_policy_name,
                validated.top_k,
                validated.similarity_metric,
                Json(list(validated.profiles)),
                Json(validated.query_runtime_metadata),
                validated.total_elapsed_ms,
                validated.created_by,
                validated.created_by_user_id,
            ),
        )
        return _row_to_search_log_record(dict(cursor.fetchone()))


def create_search_log(database_url: str, search_log_input: SearchLogInput) -> SearchLogRecord:
    with connect(database_url) as connection:
        return create_search_log_in_connection(connection, search_log_input)


def get_search_log(database_url: str, search_log_id: int) -> SearchLogRecord | None:
    _require_positive_id(search_log_id, "search_log_id")
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM search_logs WHERE search_log_id = %s",
                (search_log_id,),
            )
            row = cursor.fetchone()
    return _row_to_search_log_record(dict(row)) if row else None


def get_search_log_result(
    database_url: str,
    search_log_result_id: int,
) -> SearchLogResultRecord | None:
    _require_positive_id(search_log_result_id, "search_log_result_id")
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM search_log_results WHERE search_log_result_id = %s",
                (search_log_result_id,),
            )
            row = cursor.fetchone()
    return _row_to_search_log_result_record(dict(row)) if row else None


def create_search_log_result_in_connection(
    connection: Connection,
    result_input: SearchLogResultInput,
) -> SearchLogResultRecord:
    validated = validate_search_log_result_input(result_input)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO search_log_results (
                search_log_id,
                profile_name,
                rank,
                chunk_id,
                distance,
                score,
                profile_elapsed_ms
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                validated.search_log_id,
                validated.profile_name,
                validated.rank,
                validated.chunk_id,
                validated.distance,
                validated.score,
                validated.profile_elapsed_ms,
            ),
        )
        return _row_to_search_log_result_record(dict(cursor.fetchone()))


def create_search_log_results(
    database_url: str,
    result_inputs: list[SearchLogResultInput],
) -> list[SearchLogResultRecord]:
    with connect(database_url) as connection:
        return [
            create_search_log_result_in_connection(connection, result_input)
            for result_input in result_inputs
        ]


def list_search_log_results(
    database_url: str,
    search_log_id: int,
) -> list[SearchLogResultRecord]:
    _require_positive_id(search_log_id, "search_log_id")
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM search_log_results
                WHERE search_log_id = %s
                ORDER BY profile_name ASC, rank ASC
                """,
                (search_log_id,),
            )
            rows = cursor.fetchall()
    return [_row_to_search_log_result_record(dict(row)) for row in rows]


def create_search_result_feedback(
    database_url: str,
    feedback_input: SearchResultFeedbackInput,
) -> SearchResultFeedbackRecord:
    validated = validate_search_result_feedback_input(feedback_input)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO search_result_feedback (
                    search_log_result_id,
                    relevance_label,
                    comment,
                    created_by,
                    created_by_user_id
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    validated.search_log_result_id,
                    validated.relevance_label,
                    validated.comment,
                    validated.created_by,
                    validated.created_by_user_id,
                ),
            )
            return _row_to_search_result_feedback_record(dict(cursor.fetchone()))


def _feedback_summary_filter_clause(
    *,
    document_group: str | None,
) -> tuple[str, list[str]]:
    params: list[str] = []
    clauses: list[str] = []
    normalized_document_group = _validate_nonblank(document_group, "document_group")
    if normalized_document_group is not None:
        clauses.append("sl.document_group = %s")
        params.append(normalized_document_group)
    if not clauses:
        return "", params
    return f"WHERE {' AND '.join(clauses)}", params


def summarize_search_feedback(
    database_url: str,
    *,
    document_group: str | None = None,
) -> SearchFeedbackSummaryRecord:
    where_clause, params = _feedback_summary_filter_clause(document_group=document_group)
    feedback_cte = f"""
        WITH feedback AS (
            SELECT
                srf.feedback_id,
                srf.relevance_label,
                srf.created_at AS feedback_created_at,
                slr.search_log_result_id,
                slr.search_log_id,
                slr.profile_name,
                slr.rank,
                slr.score,
                slr.profile_elapsed_ms
            FROM search_result_feedback srf
            JOIN search_log_results slr
              ON slr.search_log_result_id = srf.search_log_result_id
            JOIN search_logs sl
              ON sl.search_log_id = slr.search_log_id
            {where_clause}
        )
        """
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                {feedback_cte}
                SELECT
                    count(feedback_id) AS feedback_count,
                    count(DISTINCT search_log_id) AS search_log_count,
                    count(DISTINCT search_log_result_id) AS result_count,
                    max(feedback_created_at) AS latest_feedback_at
                FROM feedback
                """,
                params,
            )
            total_row = dict(cursor.fetchone())
            cursor.execute(
                f"""
                {feedback_cte}
                SELECT
                    ep.profile_name,
                    count(f.feedback_id) AS feedback_count,
                    count(DISTINCT f.search_log_id) AS search_log_count,
                    count(DISTINCT f.search_log_result_id) AS result_count,
                    count(f.feedback_id) FILTER (
                        WHERE f.relevance_label = 'correct'
                    ) AS correct_count,
                    count(f.feedback_id) FILTER (
                        WHERE f.relevance_label = 'partial'
                    ) AS partial_count,
                    count(f.feedback_id) FILTER (
                        WHERE f.relevance_label = 'wrong'
                    ) AS wrong_count,
                    count(f.feedback_id) FILTER (
                        WHERE f.relevance_label = 'duplicate'
                    ) AS duplicate_count,
                    count(f.feedback_id) FILTER (
                        WHERE f.relevance_label = 'insufficient_context'
                    ) AS insufficient_context_count,
                    avg(f.rank) AS average_rank,
                    avg(f.score) AS average_score,
                    avg(f.profile_elapsed_ms) AS average_profile_elapsed_ms,
                    max(f.feedback_created_at) AS latest_feedback_at
                FROM embedding_profiles ep
                LEFT JOIN feedback f ON f.profile_name = ep.profile_name
                WHERE ep.is_active
                GROUP BY ep.profile_name
                ORDER BY ep.profile_name ASC
                """,
                params,
            )
            profile_rows = cursor.fetchall()

    return SearchFeedbackSummaryRecord(
        feedback_count=int(total_row["feedback_count"] or 0),
        search_log_count=int(total_row["search_log_count"] or 0),
        result_count=int(total_row["result_count"] or 0),
        latest_feedback_at=total_row["latest_feedback_at"],
        profiles=tuple(
            _row_to_search_feedback_profile_summary_record(dict(row)) for row in profile_rows
        ),
    )
