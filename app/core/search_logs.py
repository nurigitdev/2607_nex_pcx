"""Search log, result, and feedback repository helpers."""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from psycopg import Connection
from psycopg.types.json import Json

from app.core.database import connect

SEARCH_SCOPES = {"mine", "team", "managed_org", "company"}
SIMILARITY_METRICS = {"cosine", "l2", "inner_product", "bm25"}
FEEDBACK_LABELS = {"correct", "partial", "wrong", "duplicate", "insufficient_context"}
MAX_REVIEW_TAGS = 12
MAX_REVIEW_TAG_LENGTH = 40
MAX_REVIEW_MEMO_LENGTH = 2000


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
    strategy_name: str = "vector_cosine"
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
    strategy_name: str
    top_k: int
    similarity_metric: str
    profiles: tuple[str, ...]
    query_runtime_metadata: dict[str, Any]
    total_elapsed_ms: int | None
    created_by: str | None
    created_by_user_id: int | None
    review_tags: tuple[str, ...]
    review_memo: str | None
    reviewed_by_user_id: int | None
    reviewed_at: datetime | None
    created_at: datetime


@dataclass(frozen=True)
class SearchLogResultInput:
    search_log_id: int
    profile_name: str
    rank: int
    chunk_id: int
    distance: float | None = None
    score: float | None = None
    search_profile_name: str | None = None
    retrieval_strategy: str | None = None
    score_components: dict[str, Any] = field(default_factory=dict)
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
    search_profile_name: str | None
    retrieval_strategy: str | None
    score_components: dict[str, Any]
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
class SearchLogReviewMetadataInput:
    search_log_id: int
    review_tags: tuple[str, ...] = ()
    review_memo: str | None = None
    reviewed_by_user_id: int | None = None


@dataclass(frozen=True)
class SearchLogRetentionSettings:
    enabled: bool = True
    retention_days: int = 30
    cleanup_batch_size: int = 1000


@dataclass(frozen=True)
class SearchLogRetentionSettingsInput:
    enabled: bool = True
    retention_days: int = 30
    cleanup_batch_size: int = 1000


@dataclass(frozen=True)
class SearchLogCleanupResult:
    enabled: bool
    dry_run: bool
    retention_days: int
    cleanup_batch_size: int
    expired_count: int
    deleted_count: int
    cutoff_at: datetime


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


@dataclass(frozen=True)
class SearchFeedbackCommentRecord:
    feedback_id: int
    search_log_result_id: int
    search_log_id: int
    query_text: str
    document_group: str | None
    actor_login_id: str | None
    actor_display_name: str | None
    profile_name: str
    rank: int
    chunk_id: int
    document_title: str | None
    original_file_name: str
    relevance_label: str
    comment: str
    created_by_user_id: int | None
    created_at: datetime


@dataclass(frozen=True)
class SearchLogListItem:
    search_log: SearchLogRecord
    actor_login_id: str | None
    actor_display_name: str | None
    result_count: int
    feedback_count: int
    correct_count: int
    partial_count: int
    wrong_count: int
    duplicate_count: int
    insufficient_context_count: int
    latest_feedback_at: datetime | None


@dataclass(frozen=True)
class SearchRuntimeFailureRecord:
    search_log_id: int
    query_text: str
    actor_user_id: int | None
    actor_login_id: str | None
    actor_display_name: str | None
    requested_search_scope: str | None
    effective_search_scope: str | None
    document_group: str | None
    file_type: str | None
    chunk_policy_name: str | None
    top_k: int
    profile_name: str
    error_code: str | None
    error_message: str | None
    elapsed_ms: int | None
    created_at: datetime


@dataclass(frozen=True)
class SearchLatencyOutlierRecord:
    search_log_id: int
    query_text: str
    actor_user_id: int | None
    actor_login_id: str | None
    actor_display_name: str | None
    requested_search_scope: str | None
    effective_search_scope: str | None
    document_group: str | None
    file_type: str | None
    chunk_policy_name: str | None
    top_k: int
    profiles: tuple[str, ...]
    total_elapsed_ms: int
    succeeded_profile_count: int
    failed_profile_count: int
    created_at: datetime


@dataclass(frozen=True)
class SearchNoResultRecord:
    search_log_id: int
    query_text: str
    actor_user_id: int | None
    actor_login_id: str | None
    actor_display_name: str | None
    requested_search_scope: str | None
    effective_search_scope: str | None
    document_group: str | None
    file_type: str | None
    chunk_policy_name: str | None
    top_k: int
    profiles: tuple[str, ...]
    total_elapsed_ms: int | None
    failed_profile_count: int
    created_at: datetime


@dataclass(frozen=True)
class SearchDuplicateFingerprintRecord:
    condition_fingerprint: str
    duplicate_count: int
    latest_search_log_id: int
    first_search_log_id: int
    query_text: str
    actor_user_id: int | None
    actor_login_id: str | None
    actor_display_name: str | None
    requested_search_scope: str | None
    effective_search_scope: str | None
    document_group: str | None
    file_type: str | None
    chunk_policy_name: str | None
    top_k: int
    similarity_metric: str
    profiles: tuple[str, ...]
    zero_result_count: int
    runtime_failure_count: int
    average_total_elapsed_ms: float | None
    first_created_at: datetime
    latest_created_at: datetime


@dataclass(frozen=True)
class SearchOperationsSummaryRecord:
    lookback_hours: int
    min_total_elapsed_ms: int
    search_count: int
    result_row_count: int
    no_result_count: int
    runtime_failure_count: int
    latency_outlier_count: int
    real_provider_required_count: int
    mock_fallback_allowed_count: int
    duplicate_fingerprint_count: int
    max_duplicate_count: int
    average_total_elapsed_ms: float | None
    latest_search_at: datetime | None


@dataclass(frozen=True)
class SearchLogResultDetailRecord:
    search_log_result: SearchLogResultRecord
    document_id: int
    file_id: int
    chunk_preview: str
    content_hash: str
    chunk_policy_name: str
    heading_path: tuple[str, ...]
    page_no: int | None
    slide_no: int | None
    sheet_name: str | None
    cell_range: str | None
    document_title: str | None
    document_group: str
    original_file_name: str
    file_ext: str | None
    feedback: tuple[SearchResultFeedbackRecord, ...]


@dataclass(frozen=True)
class SearchLogDetailRecord:
    search_log: SearchLogRecord
    actor_login_id: str | None
    actor_display_name: str | None
    results: tuple[SearchLogResultDetailRecord, ...]


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


def _validate_review_tags(review_tags: tuple[str, ...]) -> tuple[str, ...]:
    if len(review_tags) > MAX_REVIEW_TAGS:
        raise InvalidSearchLogError(f"review_tags must contain at most {MAX_REVIEW_TAGS} tags")
    normalized_tags = tuple(_validate_nonblank(tag, "review_tag") for tag in review_tags)
    if any(tag is not None and len(tag) > MAX_REVIEW_TAG_LENGTH for tag in normalized_tags):
        raise InvalidSearchLogError(
            f"review_tag must be less than or equal to {MAX_REVIEW_TAG_LENGTH} characters"
        )
    if len(set(normalized_tags)) != len(normalized_tags):
        raise InvalidSearchLogError("review_tags must be unique")
    return tuple(tag for tag in normalized_tags if tag is not None)


def _normalize_review_memo(review_memo: str | None) -> str | None:
    if review_memo is None:
        return None
    normalized_memo = review_memo.strip()
    return normalized_memo or None


def validate_search_log_review_metadata_input(
    metadata_input: SearchLogReviewMetadataInput,
) -> SearchLogReviewMetadataInput:
    _require_positive_id(metadata_input.search_log_id, "search_log_id")
    _require_positive_id(metadata_input.reviewed_by_user_id, "reviewed_by_user_id")
    normalized_memo = _normalize_review_memo(metadata_input.review_memo)
    if normalized_memo is not None and len(normalized_memo) > MAX_REVIEW_MEMO_LENGTH:
        raise InvalidSearchLogError(
            f"review_memo must be less than or equal to {MAX_REVIEW_MEMO_LENGTH} characters"
        )
    return SearchLogReviewMetadataInput(
        search_log_id=metadata_input.search_log_id,
        review_tags=_validate_review_tags(metadata_input.review_tags),
        review_memo=normalized_memo,
        reviewed_by_user_id=metadata_input.reviewed_by_user_id,
    )


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
        strategy_name=_validate_nonblank(search_log_input.strategy_name, "strategy_name")
        or search_log_input.strategy_name,
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
    if not isinstance(result_input.score_components, dict):
        raise InvalidSearchLogError("score_components must be a JSON object")
    profile_name = (
        _validate_nonblank(result_input.profile_name, "profile_name") or result_input.profile_name
    )
    return SearchLogResultInput(
        search_log_id=result_input.search_log_id,
        profile_name=profile_name,
        rank=result_input.rank,
        chunk_id=result_input.chunk_id,
        distance=result_input.distance,
        score=result_input.score,
        search_profile_name=_validate_nonblank(
            result_input.search_profile_name,
            "search_profile_name",
        )
        or profile_name,
        retrieval_strategy=_validate_nonblank(
            result_input.retrieval_strategy,
            "retrieval_strategy",
        )
        or "vector_cosine",
        score_components=dict(result_input.score_components),
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
        strategy_name=str(row.get("strategy_name") or "vector_cosine"),
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
        review_tags=tuple(row["review_tags"] or ()),
        review_memo=row["review_memo"],
        reviewed_by_user_id=(
            int(row["reviewed_by_user_id"]) if row["reviewed_by_user_id"] is not None else None
        ),
        reviewed_at=row["reviewed_at"],
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
        search_profile_name=row.get("search_profile_name"),
        retrieval_strategy=row.get("retrieval_strategy"),
        score_components=dict(row.get("score_components") or {}),
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


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _row_to_search_runtime_failure_record(row: dict[str, Any]) -> SearchRuntimeFailureRecord:
    return SearchRuntimeFailureRecord(
        search_log_id=int(row["search_log_id"]),
        query_text=str(row["query_text"]),
        actor_user_id=int(row["actor_user_id"]) if row["actor_user_id"] is not None else None,
        actor_login_id=row["actor_login_id"],
        actor_display_name=row["actor_display_name"],
        requested_search_scope=row["requested_search_scope"],
        effective_search_scope=row["effective_search_scope"],
        document_group=row["document_group"],
        file_type=row["file_type"],
        chunk_policy_name=row["chunk_policy_name"],
        top_k=int(row["top_k"]),
        profile_name=str(row["profile_name"]),
        error_code=row["error_code"],
        error_message=row["error_message"],
        elapsed_ms=_optional_int(row["elapsed_ms"]),
        created_at=row["created_at"],
    )


def _row_to_search_latency_outlier_record(row: dict[str, Any]) -> SearchLatencyOutlierRecord:
    return SearchLatencyOutlierRecord(
        search_log_id=int(row["search_log_id"]),
        query_text=str(row["query_text"]),
        actor_user_id=int(row["actor_user_id"]) if row["actor_user_id"] is not None else None,
        actor_login_id=row["actor_login_id"],
        actor_display_name=row["actor_display_name"],
        requested_search_scope=row["requested_search_scope"],
        effective_search_scope=row["effective_search_scope"],
        document_group=row["document_group"],
        file_type=row["file_type"],
        chunk_policy_name=row["chunk_policy_name"],
        top_k=int(row["top_k"]),
        profiles=tuple(row["profiles"] or ()),
        total_elapsed_ms=int(row["total_elapsed_ms"]),
        succeeded_profile_count=int(row["succeeded_profile_count"] or 0),
        failed_profile_count=int(row["failed_profile_count"] or 0),
        created_at=row["created_at"],
    )


def _row_to_search_no_result_record(row: dict[str, Any]) -> SearchNoResultRecord:
    return SearchNoResultRecord(
        search_log_id=int(row["search_log_id"]),
        query_text=str(row["query_text"]),
        actor_user_id=int(row["actor_user_id"]) if row["actor_user_id"] is not None else None,
        actor_login_id=row["actor_login_id"],
        actor_display_name=row["actor_display_name"],
        requested_search_scope=row["requested_search_scope"],
        effective_search_scope=row["effective_search_scope"],
        document_group=row["document_group"],
        file_type=row["file_type"],
        chunk_policy_name=row["chunk_policy_name"],
        top_k=int(row["top_k"]),
        profiles=tuple(row["profiles"] or ()),
        total_elapsed_ms=(
            int(row["total_elapsed_ms"]) if row["total_elapsed_ms"] is not None else None
        ),
        failed_profile_count=int(row["failed_profile_count"] or 0),
        created_at=row["created_at"],
    )


def _row_to_search_duplicate_fingerprint_record(
    row: dict[str, Any],
) -> SearchDuplicateFingerprintRecord:
    return SearchDuplicateFingerprintRecord(
        condition_fingerprint=str(row["condition_fingerprint"]),
        duplicate_count=int(row["duplicate_count"] or 0),
        latest_search_log_id=int(row["latest_search_log_id"]),
        first_search_log_id=int(row["first_search_log_id"]),
        query_text=str(row["query_text"]),
        actor_user_id=int(row["actor_user_id"]) if row["actor_user_id"] is not None else None,
        actor_login_id=row["actor_login_id"],
        actor_display_name=row["actor_display_name"],
        requested_search_scope=row["requested_search_scope"],
        effective_search_scope=row["effective_search_scope"],
        document_group=row["document_group"],
        file_type=row["file_type"],
        chunk_policy_name=row["chunk_policy_name"],
        top_k=int(row["top_k"]),
        similarity_metric=str(row["similarity_metric"]),
        profiles=tuple(row["profiles"] or ()),
        zero_result_count=int(row["zero_result_count"] or 0),
        runtime_failure_count=int(row["runtime_failure_count"] or 0),
        average_total_elapsed_ms=_optional_float(row["average_total_elapsed_ms"]),
        first_created_at=row["first_created_at"],
        latest_created_at=row["latest_created_at"],
    )


def _row_to_search_operations_summary_record(row: dict[str, Any]) -> SearchOperationsSummaryRecord:
    return SearchOperationsSummaryRecord(
        lookback_hours=int(row["lookback_hours"]),
        min_total_elapsed_ms=int(row["min_total_elapsed_ms"]),
        search_count=int(row["search_count"] or 0),
        result_row_count=int(row["result_row_count"] or 0),
        no_result_count=int(row["no_result_count"] or 0),
        runtime_failure_count=int(row["runtime_failure_count"] or 0),
        latency_outlier_count=int(row["latency_outlier_count"] or 0),
        real_provider_required_count=int(row["real_provider_required_count"] or 0),
        mock_fallback_allowed_count=int(row["mock_fallback_allowed_count"] or 0),
        duplicate_fingerprint_count=int(row["duplicate_fingerprint_count"] or 0),
        max_duplicate_count=int(row["max_duplicate_count"] or 0),
        average_total_elapsed_ms=_optional_float(row["average_total_elapsed_ms"]),
        latest_search_at=row["latest_search_at"],
    )


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


def _row_to_search_feedback_comment_record(row: dict[str, Any]) -> SearchFeedbackCommentRecord:
    return SearchFeedbackCommentRecord(
        feedback_id=int(row["feedback_id"]),
        search_log_result_id=int(row["search_log_result_id"]),
        search_log_id=int(row["search_log_id"]),
        query_text=str(row["query_text"]),
        document_group=row["document_group"],
        actor_login_id=row["actor_login_id"],
        actor_display_name=row["actor_display_name"],
        profile_name=str(row["profile_name"]),
        rank=int(row["rank"]),
        chunk_id=int(row["chunk_id"]),
        document_title=row["document_title"],
        original_file_name=str(row["original_file_name"]),
        relevance_label=str(row["relevance_label"]),
        comment=str(row["comment"]),
        created_by_user_id=(
            int(row["created_by_user_id"]) if row["created_by_user_id"] is not None else None
        ),
        created_at=row["created_at"],
    )


def _row_to_search_log_list_item(row: dict[str, Any]) -> SearchLogListItem:
    return SearchLogListItem(
        search_log=_row_to_search_log_record(row),
        actor_login_id=row["actor_login_id"],
        actor_display_name=row["actor_display_name"],
        result_count=int(row["result_count"] or 0),
        feedback_count=int(row["feedback_count"] or 0),
        correct_count=int(row["correct_count"] or 0),
        partial_count=int(row["partial_count"] or 0),
        wrong_count=int(row["wrong_count"] or 0),
        duplicate_count=int(row["duplicate_count"] or 0),
        insufficient_context_count=int(row["insufficient_context_count"] or 0),
        latest_feedback_at=row["latest_feedback_at"],
    )


def _chunk_preview(chunk_text: str, limit: int = 240) -> str:
    normalized = " ".join(chunk_text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "..."


def _row_to_search_log_result_detail_record(
    row: dict[str, Any],
    feedback: tuple[SearchResultFeedbackRecord, ...],
) -> SearchLogResultDetailRecord:
    return SearchLogResultDetailRecord(
        search_log_result=_row_to_search_log_result_record(row),
        document_id=int(row["document_id"]),
        file_id=int(row["file_id"]),
        chunk_preview=_chunk_preview(str(row["chunk_text"])),
        content_hash=str(row["content_hash"]),
        chunk_policy_name=str(row["chunk_policy_name"]),
        heading_path=tuple(row["heading_path"] or ()),
        page_no=int(row["page_no"]) if row["page_no"] is not None else None,
        slide_no=int(row["slide_no"]) if row["slide_no"] is not None else None,
        sheet_name=row["sheet_name"],
        cell_range=row["cell_range"],
        document_title=row["document_title"],
        document_group=str(row["document_group"]),
        original_file_name=str(row["original_file_name"]),
        file_ext=row["file_ext"],
        feedback=feedback,
    )


def _validate_limit(limit: int, *, max_limit: int = 200) -> int:
    if limit <= 0:
        raise InvalidSearchLogError("limit must be greater than 0")
    if limit > max_limit:
        raise InvalidSearchLogError(f"limit must be less than or equal to {max_limit}")
    return limit


def _validate_min_total_elapsed_ms(min_total_elapsed_ms: int) -> int:
    if min_total_elapsed_ms < 0:
        raise InvalidSearchLogError("min_total_elapsed_ms must be greater than or equal to 0")
    return min_total_elapsed_ms


def _validate_min_duplicate_count(min_count: int) -> int:
    if min_count < 2:
        raise InvalidSearchLogError("min_count must be greater than or equal to 2")
    if min_count > 1000:
        raise InvalidSearchLogError("min_count must be less than or equal to 1000")
    return min_count


def _validate_search_operations_lookback_hours(lookback_hours: int) -> int:
    if lookback_hours <= 0:
        raise InvalidSearchLogError("lookback_hours must be greater than 0")
    if lookback_hours > 720:
        raise InvalidSearchLogError("lookback_hours must be less than or equal to 720")
    return lookback_hours


def _parse_bool(value: str, default: bool) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_positive_int(value: str, default: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def search_log_retention_settings_from_rows(
    rows: list[dict[str, Any]],
) -> SearchLogRetentionSettings:
    values = {row["setting_name"]: row["setting_value"] for row in rows}
    defaults = SearchLogRetentionSettings()
    return SearchLogRetentionSettings(
        enabled=_parse_bool(
            values.get("search_log_retention_enabled", str(defaults.enabled)),
            defaults.enabled,
        ),
        retention_days=_parse_positive_int(
            values.get("search_log_retention_days", str(defaults.retention_days)),
            defaults.retention_days,
        ),
        cleanup_batch_size=_parse_positive_int(
            values.get("search_log_cleanup_batch_size", str(defaults.cleanup_batch_size)),
            defaults.cleanup_batch_size,
        ),
    )


def validate_search_log_retention_settings_input(
    settings_input: SearchLogRetentionSettingsInput,
) -> SearchLogRetentionSettingsInput:
    if settings_input.retention_days <= 0 or settings_input.retention_days > 3650:
        raise InvalidSearchLogError("retention_days must be between 1 and 3650")
    if settings_input.cleanup_batch_size <= 0 or settings_input.cleanup_batch_size > 100000:
        raise InvalidSearchLogError("cleanup_batch_size must be between 1 and 100000")
    return SearchLogRetentionSettingsInput(
        enabled=bool(settings_input.enabled),
        retention_days=settings_input.retention_days,
        cleanup_batch_size=settings_input.cleanup_batch_size,
    )


def load_search_log_retention_settings(database_url: str) -> SearchLogRetentionSettings:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT setting_name, setting_value
                FROM app_log_settings
                WHERE setting_name IN (
                    'search_log_retention_enabled',
                    'search_log_retention_days',
                    'search_log_cleanup_batch_size'
                )
                """)
            rows = cursor.fetchall()
    return search_log_retention_settings_from_rows([dict(row) for row in rows])


def update_search_log_retention_settings(
    database_url: str,
    settings_input: SearchLogRetentionSettingsInput,
) -> SearchLogRetentionSettings:
    validated = validate_search_log_retention_settings_input(settings_input)
    rows = (
        (
            "search_log_retention_enabled",
            "true" if validated.enabled else "false",
            "bool",
            "Enable search log retention cleanup actions",
        ),
        (
            "search_log_retention_days",
            str(validated.retention_days),
            "int",
            "Number of days to retain search_logs and dependent rows",
        ),
        (
            "search_log_cleanup_batch_size",
            str(validated.cleanup_batch_size),
            "int",
            "Maximum search_logs rows cleaned up in one admin action",
        ),
    )
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO app_log_settings (
                    setting_name,
                    setting_value,
                    value_type,
                    description,
                    updated_at
                )
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (setting_name) DO UPDATE
                SET setting_value = EXCLUDED.setting_value,
                    value_type = EXCLUDED.value_type,
                    description = EXCLUDED.description,
                    updated_at = now()
                """,
                rows,
            )
    return SearchLogRetentionSettings(
        enabled=validated.enabled,
        retention_days=validated.retention_days,
        cleanup_batch_size=validated.cleanup_batch_size,
    )


def cleanup_expired_search_logs(
    database_url: str, *, dry_run: bool = True
) -> SearchLogCleanupResult:
    retention_settings = load_search_log_retention_settings(database_url)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT now() - (%s::int * interval '1 day') AS cutoff_at
                """,
                (retention_settings.retention_days,),
            )
            cutoff_at = cursor.fetchone()["cutoff_at"]
            cursor.execute(
                """
                SELECT count(*) AS expired_count
                FROM search_logs
                WHERE created_at < %s
                """,
                (cutoff_at,),
            )
            expired_count = int(cursor.fetchone()["expired_count"] or 0)
            deleted_count = 0
            if retention_settings.enabled and not dry_run:
                cursor.execute(
                    """
                    WITH doomed AS (
                        SELECT search_log_id
                        FROM search_logs
                        WHERE created_at < %s
                        ORDER BY created_at ASC, search_log_id ASC
                        LIMIT %s
                    )
                    DELETE FROM search_logs
                    WHERE search_log_id IN (SELECT search_log_id FROM doomed)
                    """,
                    (cutoff_at, retention_settings.cleanup_batch_size),
                )
                deleted_count = cursor.rowcount or 0

    return SearchLogCleanupResult(
        enabled=retention_settings.enabled,
        dry_run=dry_run,
        retention_days=retention_settings.retention_days,
        cleanup_batch_size=retention_settings.cleanup_batch_size,
        expired_count=expired_count,
        deleted_count=deleted_count,
        cutoff_at=cutoff_at,
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
                strategy_name,
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
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s
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
                validated.strategy_name,
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


def update_search_log_review_metadata(
    database_url: str,
    metadata_input: SearchLogReviewMetadataInput,
) -> SearchLogRecord | None:
    validated = validate_search_log_review_metadata_input(metadata_input)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE search_logs
                SET review_tags = %s,
                    review_memo = %s,
                    reviewed_by_user_id = %s,
                    reviewed_at = now()
                WHERE search_log_id = %s
                RETURNING *
                """,
                (
                    Json(list(validated.review_tags)),
                    validated.review_memo,
                    validated.reviewed_by_user_id,
                    validated.search_log_id,
                ),
            )
            row = cursor.fetchone()
    return _row_to_search_log_record(dict(row)) if row else None


def _search_log_filter_clause(
    *,
    actor_user_id: int | None,
    requested_search_scope: str | None,
    document_group: str | None,
    provider_mode_filter: str | None,
) -> tuple[str, list[object]]:
    params: list[object] = []
    clauses: list[str] = []
    _require_positive_id(actor_user_id, "actor_user_id")
    normalized_scope = _validate_scope(requested_search_scope, "requested_search_scope")
    normalized_document_group = _validate_nonblank(document_group, "document_group")
    normalized_provider_mode = _validate_nonblank(
        provider_mode_filter,
        "provider_mode_filter",
    )
    if normalized_provider_mode not in {
        None,
        "real_provider_required",
        "mock_fallback_allowed",
    }:
        raise InvalidSearchLogError("Unsupported provider_mode_filter")
    if actor_user_id is not None:
        clauses.append("sl.actor_user_id = %s")
        params.append(actor_user_id)
    if normalized_scope is not None:
        clauses.append("sl.requested_search_scope = %s")
        params.append(normalized_scope)
    if normalized_document_group is not None:
        clauses.append("sl.document_group = %s")
        params.append(normalized_document_group)
    if normalized_provider_mode == "real_provider_required":
        clauses.append("sl.query_runtime_metadata ->> 'real_provider_required' = 'true'")
    elif normalized_provider_mode == "mock_fallback_allowed":
        clauses.append("sl.query_runtime_metadata ->> 'allow_mock_fallback' = 'true'")
    if not clauses:
        return "", params
    return f"WHERE {' AND '.join(clauses)}", params


def list_search_logs(
    database_url: str,
    *,
    actor_user_id: int | None = None,
    requested_search_scope: str | None = None,
    document_group: str | None = None,
    provider_mode_filter: str | None = None,
    limit: int = 50,
) -> list[SearchLogListItem]:
    validated_limit = _validate_limit(limit)
    where_clause, params = _search_log_filter_clause(
        actor_user_id=actor_user_id,
        requested_search_scope=requested_search_scope,
        document_group=document_group,
        provider_mode_filter=provider_mode_filter,
    )
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    sl.*,
                    au.login_id AS actor_login_id,
                    au.display_name AS actor_display_name,
                    count(DISTINCT slr.search_log_result_id) AS result_count,
                    count(srf.feedback_id) AS feedback_count,
                    count(srf.feedback_id) FILTER (
                        WHERE srf.relevance_label = 'correct'
                    ) AS correct_count,
                    count(srf.feedback_id) FILTER (
                        WHERE srf.relevance_label = 'partial'
                    ) AS partial_count,
                    count(srf.feedback_id) FILTER (
                        WHERE srf.relevance_label = 'wrong'
                    ) AS wrong_count,
                    count(srf.feedback_id) FILTER (
                        WHERE srf.relevance_label = 'duplicate'
                    ) AS duplicate_count,
                    count(srf.feedback_id) FILTER (
                        WHERE srf.relevance_label = 'insufficient_context'
                    ) AS insufficient_context_count,
                    max(srf.created_at) AS latest_feedback_at
                FROM search_logs sl
                LEFT JOIN app_users au ON au.user_id = sl.actor_user_id
                LEFT JOIN search_log_results slr ON slr.search_log_id = sl.search_log_id
                LEFT JOIN search_result_feedback srf
                  ON srf.search_log_result_id = slr.search_log_result_id
                {where_clause}
                GROUP BY sl.search_log_id, au.login_id, au.display_name
                ORDER BY sl.created_at DESC, sl.search_log_id DESC
                LIMIT %s
                """,
                [*params, validated_limit],
            )
            rows = cursor.fetchall()
    return [_row_to_search_log_list_item(dict(row)) for row in rows]


def list_search_runtime_failures(
    database_url: str,
    *,
    profile_name: str | None = None,
    limit: int = 20,
) -> list[SearchRuntimeFailureRecord]:
    validated_limit = _validate_limit(limit)
    normalized_profile = _validate_nonblank(profile_name, "profile_name")
    params: list[object] = []
    profile_filter = ""
    if normalized_profile is not None:
        profile_filter = "WHERE failure.key = %s"
        params.append(normalized_profile)

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    sl.search_log_id,
                    sl.query_text,
                    sl.actor_user_id,
                    au.login_id AS actor_login_id,
                    au.display_name AS actor_display_name,
                    sl.requested_search_scope,
                    sl.effective_search_scope,
                    sl.document_group,
                    sl.file_type,
                    sl.chunk_policy_name,
                    sl.top_k,
                    sl.created_at,
                    failure.key AS profile_name,
                    failure.value ->> 'error_code' AS error_code,
                    failure.value ->> 'error_message' AS error_message,
                    failure.value ->> 'elapsed_ms' AS elapsed_ms
                FROM search_logs sl
                LEFT JOIN app_users au ON au.user_id = sl.actor_user_id
                JOIN LATERAL jsonb_each(
                    COALESCE(sl.query_runtime_metadata -> 'profile_failures', '{{}}'::jsonb)
                ) AS failure(key, value) ON TRUE
                {profile_filter}
                ORDER BY sl.created_at DESC, sl.search_log_id DESC, failure.key ASC
                LIMIT %s
                """,
                [*params, validated_limit],
            )
            rows = cursor.fetchall()
    return [_row_to_search_runtime_failure_record(dict(row)) for row in rows]


def list_search_latency_outliers(
    database_url: str,
    *,
    min_total_elapsed_ms: int = 1000,
    limit: int = 20,
) -> list[SearchLatencyOutlierRecord]:
    validated_threshold = _validate_min_total_elapsed_ms(min_total_elapsed_ms)
    validated_limit = _validate_limit(limit)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    sl.search_log_id,
                    sl.query_text,
                    sl.actor_user_id,
                    au.login_id AS actor_login_id,
                    au.display_name AS actor_display_name,
                    sl.requested_search_scope,
                    sl.effective_search_scope,
                    sl.document_group,
                    sl.file_type,
                    sl.chunk_policy_name,
                    sl.top_k,
                    sl.profiles,
                    sl.total_elapsed_ms,
                    COALESCE(
                        NULLIF(
                            sl.query_runtime_metadata #>> '{profile_status_counts,succeeded}',
                            ''
                        )::int,
                        0
                    ) AS succeeded_profile_count,
                    COALESCE(
                        NULLIF(
                            sl.query_runtime_metadata #>> '{profile_status_counts,failed}',
                            ''
                        )::int,
                        0
                    ) AS failed_profile_count,
                    sl.created_at
                FROM search_logs sl
                LEFT JOIN app_users au ON au.user_id = sl.actor_user_id
                WHERE sl.total_elapsed_ms IS NOT NULL
                  AND sl.total_elapsed_ms >= %s
                ORDER BY sl.total_elapsed_ms DESC, sl.created_at DESC, sl.search_log_id DESC
                LIMIT %s
                """,
                (validated_threshold, validated_limit),
            )
            rows = cursor.fetchall()
    return [_row_to_search_latency_outlier_record(dict(row)) for row in rows]


def list_search_no_result_logs(
    database_url: str,
    *,
    limit: int = 20,
) -> list[SearchNoResultRecord]:
    validated_limit = _validate_limit(limit)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    sl.search_log_id,
                    sl.query_text,
                    sl.actor_user_id,
                    au.login_id AS actor_login_id,
                    au.display_name AS actor_display_name,
                    sl.requested_search_scope,
                    sl.effective_search_scope,
                    sl.document_group,
                    sl.file_type,
                    sl.chunk_policy_name,
                    sl.top_k,
                    sl.profiles,
                    sl.total_elapsed_ms,
                    COALESCE(
                        NULLIF(
                            sl.query_runtime_metadata #>> '{profile_status_counts,failed}',
                            ''
                        )::int,
                        0
                    ) AS failed_profile_count,
                    sl.created_at
                FROM search_logs sl
                LEFT JOIN app_users au ON au.user_id = sl.actor_user_id
                LEFT JOIN search_log_results slr ON slr.search_log_id = sl.search_log_id
                GROUP BY sl.search_log_id, au.login_id, au.display_name
                HAVING count(slr.search_log_result_id) = 0
                ORDER BY sl.created_at DESC, sl.search_log_id DESC
                LIMIT %s
                """,
                (validated_limit,),
            )
            rows = cursor.fetchall()
    return [_row_to_search_no_result_record(dict(row)) for row in rows]


def list_search_duplicate_fingerprints(
    database_url: str,
    *,
    min_count: int = 2,
    limit: int = 20,
) -> list[SearchDuplicateFingerprintRecord]:
    validated_min_count = _validate_min_duplicate_count(min_count)
    validated_limit = _validate_limit(limit)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH log_base AS (
                    SELECT
                        sl.*,
                        au.login_id AS actor_login_id,
                        au.display_name AS actor_display_name,
                        count(slr.search_log_result_id) AS result_count,
                        md5(
                            jsonb_build_object(
                                'normalized_query_text',
                                COALESCE(
                                    sl.normalized_query_text,
                                    lower(btrim(sl.query_text))
                                ),
                                'actor_user_id', sl.actor_user_id,
                                'requested_search_scope', sl.requested_search_scope,
                                'effective_search_scope', sl.effective_search_scope,
                                'permission_filter_metadata',
                                sl.permission_filter_metadata,
                                'document_group', sl.document_group,
                                'file_type', sl.file_type,
                                'chunk_policy_name', sl.chunk_policy_name,
                                'top_k', sl.top_k,
                                'similarity_metric', sl.similarity_metric,
                                'profiles', sl.profiles
                            )::text
                        ) AS condition_fingerprint
                    FROM search_logs sl
                    LEFT JOIN app_users au ON au.user_id = sl.actor_user_id
                    LEFT JOIN search_log_results slr ON slr.search_log_id = sl.search_log_id
                    GROUP BY sl.search_log_id, au.login_id, au.display_name
                )
                SELECT
                    condition_fingerprint,
                    count(*) AS duplicate_count,
                    (array_agg(search_log_id ORDER BY created_at DESC, search_log_id DESC))[1]
                        AS latest_search_log_id,
                    (array_agg(search_log_id ORDER BY created_at ASC, search_log_id ASC))[1]
                        AS first_search_log_id,
                    (array_agg(query_text ORDER BY created_at DESC, search_log_id DESC))[1]
                        AS query_text,
                    (array_agg(actor_user_id ORDER BY created_at DESC, search_log_id DESC))[1]
                        AS actor_user_id,
                    (array_agg(actor_login_id ORDER BY created_at DESC, search_log_id DESC))[1]
                        AS actor_login_id,
                    (array_agg(actor_display_name ORDER BY created_at DESC, search_log_id DESC))[1]
                        AS actor_display_name,
                    (
                        array_agg(
                            requested_search_scope
                            ORDER BY created_at DESC, search_log_id DESC
                        )
                    )[1] AS requested_search_scope,
                    (
                        array_agg(
                            effective_search_scope
                            ORDER BY created_at DESC, search_log_id DESC
                        )
                    )[1] AS effective_search_scope,
                    (array_agg(document_group ORDER BY created_at DESC, search_log_id DESC))[1]
                        AS document_group,
                    (array_agg(file_type ORDER BY created_at DESC, search_log_id DESC))[1]
                        AS file_type,
                    (array_agg(chunk_policy_name ORDER BY created_at DESC, search_log_id DESC))[1]
                        AS chunk_policy_name,
                    (array_agg(top_k ORDER BY created_at DESC, search_log_id DESC))[1]
                        AS top_k,
                    (array_agg(similarity_metric ORDER BY created_at DESC, search_log_id DESC))[1]
                        AS similarity_metric,
                    (array_agg(profiles ORDER BY created_at DESC, search_log_id DESC))[1]
                        AS profiles,
                    count(*) FILTER (WHERE result_count = 0) AS zero_result_count,
                    count(*) FILTER (
                        WHERE COALESCE(
                            query_runtime_metadata -> 'profile_failures',
                            '{}'::jsonb
                        ) <> '{}'::jsonb
                    ) AS runtime_failure_count,
                    avg(total_elapsed_ms) FILTER (WHERE total_elapsed_ms IS NOT NULL)
                        AS average_total_elapsed_ms,
                    min(created_at) AS first_created_at,
                    max(created_at) AS latest_created_at
                FROM log_base
                GROUP BY condition_fingerprint
                HAVING count(*) >= %s
                ORDER BY duplicate_count DESC, latest_created_at DESC
                LIMIT %s
                """,
                (validated_min_count, validated_limit),
            )
            rows = cursor.fetchall()
    return [_row_to_search_duplicate_fingerprint_record(dict(row)) for row in rows]


def get_search_operations_summary(
    database_url: str,
    *,
    lookback_hours: int = 24,
    min_total_elapsed_ms: int = 1000,
) -> SearchOperationsSummaryRecord:
    validated_lookback_hours = _validate_search_operations_lookback_hours(lookback_hours)
    validated_threshold = _validate_min_total_elapsed_ms(min_total_elapsed_ms)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH scoped_logs AS (
                    SELECT
                        sl.*,
                        count(slr.search_log_result_id) AS result_count,
                        COALESCE(
                            sl.query_runtime_metadata -> 'profile_failures',
                            '{}'::jsonb
                        ) <> '{}'::jsonb AS has_runtime_failure,
                        md5(
                            jsonb_build_object(
                                'normalized_query_text',
                                COALESCE(
                                    sl.normalized_query_text,
                                    lower(btrim(sl.query_text))
                                ),
                                'actor_user_id', sl.actor_user_id,
                                'requested_search_scope', sl.requested_search_scope,
                                'effective_search_scope', sl.effective_search_scope,
                                'permission_filter_metadata',
                                sl.permission_filter_metadata,
                                'document_group', sl.document_group,
                                'file_type', sl.file_type,
                                'chunk_policy_name', sl.chunk_policy_name,
                                'top_k', sl.top_k,
                                'similarity_metric', sl.similarity_metric,
                                'profiles', sl.profiles
                            )::text
                        ) AS condition_fingerprint
                    FROM search_logs sl
                    LEFT JOIN search_log_results slr ON slr.search_log_id = sl.search_log_id
                    WHERE sl.created_at >= now() - (%s::int * interval '1 hour')
                    GROUP BY sl.search_log_id
                ),
                duplicate_groups AS (
                    SELECT condition_fingerprint, count(*) AS duplicate_count
                    FROM scoped_logs
                    GROUP BY condition_fingerprint
                    HAVING count(*) >= 2
                )
                SELECT
                    %s::int AS lookback_hours,
                    %s::int AS min_total_elapsed_ms,
                    count(*) AS search_count,
                    COALESCE(sum(result_count), 0) AS result_row_count,
                    count(*) FILTER (WHERE result_count = 0) AS no_result_count,
                    count(*) FILTER (WHERE has_runtime_failure) AS runtime_failure_count,
                    count(*) FILTER (
                        WHERE total_elapsed_ms IS NOT NULL
                          AND total_elapsed_ms >= %s
                    ) AS latency_outlier_count,
                    count(*) FILTER (
                        WHERE query_runtime_metadata ->> 'real_provider_required' = 'true'
                    ) AS real_provider_required_count,
                    count(*) FILTER (
                        WHERE query_runtime_metadata ->> 'allow_mock_fallback' = 'true'
                    ) AS mock_fallback_allowed_count,
                    (
                        SELECT count(*)
                        FROM duplicate_groups
                    ) AS duplicate_fingerprint_count,
                    (
                        SELECT COALESCE(max(duplicate_count), 0)
                        FROM duplicate_groups
                    ) AS max_duplicate_count,
                    avg(total_elapsed_ms) FILTER (WHERE total_elapsed_ms IS NOT NULL)
                        AS average_total_elapsed_ms,
                    max(created_at) AS latest_search_at
                FROM scoped_logs
                """,
                (
                    validated_lookback_hours,
                    validated_lookback_hours,
                    validated_threshold,
                    validated_threshold,
                ),
            )
            row = cursor.fetchone()
    return _row_to_search_operations_summary_record(dict(row))


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
                search_profile_name,
                retrieval_strategy,
                score_components,
                profile_elapsed_ms
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                validated.search_log_id,
                validated.profile_name,
                validated.rank,
                validated.chunk_id,
                validated.distance,
                validated.score,
                validated.search_profile_name,
                validated.retrieval_strategy,
                Json(validated.score_components),
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


def get_search_log_detail(
    database_url: str,
    search_log_id: int,
) -> SearchLogDetailRecord | None:
    _require_positive_id(search_log_id, "search_log_id")
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    sl.*,
                    au.login_id AS actor_login_id,
                    au.display_name AS actor_display_name
                FROM search_logs sl
                LEFT JOIN app_users au ON au.user_id = sl.actor_user_id
                WHERE sl.search_log_id = %s
                """,
                (search_log_id,),
            )
            log_row = cursor.fetchone()
            if log_row is None:
                return None

            cursor.execute(
                """
                SELECT
                    slr.*,
                    c.document_id,
                    d.file_id,
                    c.chunk_text,
                    c.content_hash,
                    c.chunk_policy_name,
                    c.heading_path,
                    c.page_no,
                    c.slide_no,
                    c.sheet_name,
                    c.cell_range,
                    d.document_title,
                    d.document_group,
                    f.original_file_name,
                    f.file_ext
                FROM search_log_results slr
                JOIN chunks c ON c.chunk_id = slr.chunk_id
                JOIN documents d ON d.document_id = c.document_id
                JOIN files f ON f.file_id = d.file_id
                WHERE slr.search_log_id = %s
                ORDER BY slr.profile_name ASC, slr.rank ASC
                """,
                (search_log_id,),
            )
            result_rows = [dict(row) for row in cursor.fetchall()]

            cursor.execute(
                """
                SELECT srf.*
                FROM search_result_feedback srf
                JOIN search_log_results slr
                  ON slr.search_log_result_id = srf.search_log_result_id
                WHERE slr.search_log_id = %s
                ORDER BY srf.search_log_result_id ASC, srf.created_at ASC
                """,
                (search_log_id,),
            )
            feedback_rows = cursor.fetchall()

    feedback_by_result_id: dict[int, list[SearchResultFeedbackRecord]] = {}
    for row in feedback_rows:
        feedback = _row_to_search_result_feedback_record(dict(row))
        feedback_by_result_id.setdefault(feedback.search_log_result_id, []).append(feedback)

    return SearchLogDetailRecord(
        search_log=_row_to_search_log_record(dict(log_row)),
        actor_login_id=log_row["actor_login_id"],
        actor_display_name=log_row["actor_display_name"],
        results=tuple(
            _row_to_search_log_result_detail_record(
                row,
                tuple(feedback_by_result_id.get(int(row["search_log_result_id"]), ())),
            )
            for row in result_rows
        ),
    )


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


def list_search_feedback_comments(
    database_url: str,
    *,
    document_group: str | None = None,
    limit: int = 20,
) -> list[SearchFeedbackCommentRecord]:
    validated_limit = _validate_limit(limit, max_limit=100)
    where_clause, params = _feedback_summary_filter_clause(document_group=document_group)
    comment_filter_clause = where_clause.replace("WHERE", "AND", 1)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    srf.feedback_id,
                    srf.search_log_result_id,
                    srf.relevance_label,
                    srf.comment,
                    srf.created_by_user_id,
                    srf.created_at,
                    sl.search_log_id,
                    sl.query_text,
                    sl.document_group,
                    au.login_id AS actor_login_id,
                    au.display_name AS actor_display_name,
                    slr.profile_name,
                    slr.rank,
                    slr.chunk_id,
                    d.document_title,
                    f.original_file_name
                FROM search_result_feedback srf
                JOIN search_log_results slr
                  ON slr.search_log_result_id = srf.search_log_result_id
                JOIN search_logs sl ON sl.search_log_id = slr.search_log_id
                LEFT JOIN app_users au ON au.user_id = sl.actor_user_id
                JOIN chunks c ON c.chunk_id = slr.chunk_id
                JOIN documents d ON d.document_id = c.document_id
                JOIN files f ON f.file_id = d.file_id
                WHERE srf.comment IS NOT NULL
                  AND length(btrim(srf.comment)) > 0
                {comment_filter_clause}
                ORDER BY srf.created_at DESC, srf.feedback_id DESC
                LIMIT %s
                """,
                [*params, validated_limit],
            )
            rows = cursor.fetchall()
    return [_row_to_search_feedback_comment_record(dict(row)) for row in rows]


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
