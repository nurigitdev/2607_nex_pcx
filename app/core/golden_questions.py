"""Golden question set repository helpers for evaluation fixtures."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from psycopg import Connection
from psycopg.types.json import Json

from app.core.database import connect

GOLDEN_QUESTION_TYPES = {"single_fact", "section", "comparison", "no_answer", "table_figure"}
GOLDEN_EXPECTATION_TYPES = {"visible", "hidden"}
SEARCH_SCOPES = {"mine", "team", "managed_org", "company"}


@dataclass(frozen=True)
class GoldenQuestionSetInput:
    set_name: str
    description: str | None = None
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    created_by_user_id: int | None = None


@dataclass(frozen=True)
class GoldenQuestionSetRecord:
    question_set_id: int
    set_name: str
    description: str | None
    is_active: bool
    metadata: dict[str, Any]
    created_by_user_id: int | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class GoldenQuestionInput:
    question_set_id: int
    question_text: str
    normalized_question_text: str | None = None
    question_type: str = "single_fact"
    actor_user_id: int | None = None
    requested_search_scope: str = "company"
    document_group: str | None = None
    file_type: str | None = None
    chunk_policy_name: str | None = None
    top_k: int = 5
    metadata: dict[str, Any] = field(default_factory=dict)
    created_by_user_id: int | None = None


@dataclass(frozen=True)
class GoldenQuestionRecord:
    question_id: int
    question_set_id: int
    question_text: str
    normalized_question_text: str | None
    question_type: str
    actor_user_id: int | None
    requested_search_scope: str
    document_group: str | None
    file_type: str | None
    chunk_policy_name: str | None
    top_k: int
    metadata: dict[str, Any]
    created_by_user_id: int | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class GoldenQuestionExpectedTargetInput:
    question_id: int
    chunk_id: int | None = None
    expected_heading_path: tuple[str, ...] = ()
    expectation_type: str = "visible"
    relevance_grade: int = 3
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GoldenQuestionExpectedTargetRecord:
    expected_target_id: int
    question_id: int
    chunk_id: int | None
    expected_heading_path: tuple[str, ...]
    expectation_type: str
    relevance_grade: int
    notes: str | None
    metadata: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class GoldenQuestionDetailRecord:
    question: GoldenQuestionRecord
    expected_targets: tuple[GoldenQuestionExpectedTargetRecord, ...]


class InvalidGoldenQuestionError(ValueError):
    """Raised when a golden question operation is invalid before reaching the DB."""


def normalize_question_text(question_text: str) -> str:
    return " ".join(question_text.casefold().split())


def _require_positive_id(value: int | None, field_name: str) -> None:
    if value is not None and value <= 0:
        raise InvalidGoldenQuestionError(f"{field_name} must be greater than 0")


def _validate_nonblank(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise InvalidGoldenQuestionError(f"{field_name} must not be blank")
    return normalized


def _validate_metadata(metadata: dict[str, Any], field_name: str = "metadata") -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise InvalidGoldenQuestionError(f"{field_name} must be a JSON object")
    return dict(metadata)


def _validate_scope(scope: str, field_name: str = "requested_search_scope") -> str:
    normalized = _validate_nonblank(scope, field_name)
    if normalized is None or normalized not in SEARCH_SCOPES:
        raise InvalidGoldenQuestionError(f"Unsupported {field_name}: {normalized}")
    return normalized


def _validate_question_type(question_type: str) -> str:
    normalized = _validate_nonblank(question_type, "question_type")
    if normalized is None or normalized not in GOLDEN_QUESTION_TYPES:
        raise InvalidGoldenQuestionError(f"Unsupported question_type: {normalized}")
    return normalized


def _validate_expectation_type(expectation_type: str) -> str:
    normalized = _validate_nonblank(expectation_type, "expectation_type")
    if normalized is None or normalized not in GOLDEN_EXPECTATION_TYPES:
        raise InvalidGoldenQuestionError(f"Unsupported expectation_type: {normalized}")
    return normalized


def _validate_heading_path(expected_heading_path: tuple[str, ...]) -> tuple[str, ...]:
    heading_path = tuple(expected_heading_path)
    for heading in heading_path:
        if not isinstance(heading, str) or not heading.strip():
            raise InvalidGoldenQuestionError("expected_heading_path values must not be blank")
    return tuple(heading.strip() for heading in heading_path)


def _validate_limit(limit: int, *, max_limit: int = 500) -> int:
    if limit <= 0:
        raise InvalidGoldenQuestionError("limit must be greater than 0")
    if limit > max_limit:
        raise InvalidGoldenQuestionError(f"limit must be less than or equal to {max_limit}")
    return limit


def validate_golden_question_set_input(
    question_set_input: GoldenQuestionSetInput,
) -> GoldenQuestionSetInput:
    _require_positive_id(question_set_input.created_by_user_id, "created_by_user_id")
    set_name = _validate_nonblank(question_set_input.set_name, "set_name")
    return GoldenQuestionSetInput(
        set_name=set_name or question_set_input.set_name,
        description=_validate_nonblank(question_set_input.description, "description"),
        is_active=question_set_input.is_active,
        metadata=_validate_metadata(question_set_input.metadata),
        created_by_user_id=question_set_input.created_by_user_id,
    )


def validate_golden_question_input(question_input: GoldenQuestionInput) -> GoldenQuestionInput:
    _require_positive_id(question_input.question_set_id, "question_set_id")
    _require_positive_id(question_input.actor_user_id, "actor_user_id")
    _require_positive_id(question_input.created_by_user_id, "created_by_user_id")
    if question_input.top_k <= 0:
        raise InvalidGoldenQuestionError("top_k must be greater than 0")

    question_text = _validate_nonblank(question_input.question_text, "question_text")
    normalized_question_text = _validate_nonblank(
        question_input.normalized_question_text,
        "normalized_question_text",
    )
    return GoldenQuestionInput(
        question_set_id=question_input.question_set_id,
        question_text=question_text or question_input.question_text,
        normalized_question_text=normalized_question_text
        or normalize_question_text(question_text or question_input.question_text),
        question_type=_validate_question_type(question_input.question_type),
        actor_user_id=question_input.actor_user_id,
        requested_search_scope=_validate_scope(question_input.requested_search_scope),
        document_group=_validate_nonblank(question_input.document_group, "document_group"),
        file_type=_validate_nonblank(question_input.file_type, "file_type"),
        chunk_policy_name=_validate_nonblank(question_input.chunk_policy_name, "chunk_policy_name"),
        top_k=question_input.top_k,
        metadata=_validate_metadata(question_input.metadata),
        created_by_user_id=question_input.created_by_user_id,
    )


def validate_expected_target_input(
    target_input: GoldenQuestionExpectedTargetInput,
) -> GoldenQuestionExpectedTargetInput:
    _require_positive_id(target_input.question_id, "question_id")
    _require_positive_id(target_input.chunk_id, "chunk_id")
    heading_path = _validate_heading_path(target_input.expected_heading_path)
    if target_input.chunk_id is None and not heading_path:
        raise InvalidGoldenQuestionError("chunk_id or expected_heading_path is required")
    if target_input.relevance_grade < 0 or target_input.relevance_grade > 3:
        raise InvalidGoldenQuestionError("relevance_grade must be between 0 and 3")
    return GoldenQuestionExpectedTargetInput(
        question_id=target_input.question_id,
        chunk_id=target_input.chunk_id,
        expected_heading_path=heading_path,
        expectation_type=_validate_expectation_type(target_input.expectation_type),
        relevance_grade=target_input.relevance_grade,
        notes=_validate_nonblank(target_input.notes, "notes"),
        metadata=_validate_metadata(target_input.metadata),
    )


def _row_to_question_set_record(row: dict[str, Any]) -> GoldenQuestionSetRecord:
    return GoldenQuestionSetRecord(
        question_set_id=int(row["question_set_id"]),
        set_name=str(row["set_name"]),
        description=row["description"],
        is_active=bool(row["is_active"]),
        metadata=dict(row["metadata"] or {}),
        created_by_user_id=(
            int(row["created_by_user_id"]) if row["created_by_user_id"] is not None else None
        ),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_question_record(row: dict[str, Any]) -> GoldenQuestionRecord:
    return GoldenQuestionRecord(
        question_id=int(row["question_id"]),
        question_set_id=int(row["question_set_id"]),
        question_text=str(row["question_text"]),
        normalized_question_text=row["normalized_question_text"],
        question_type=str(row["question_type"]),
        actor_user_id=int(row["actor_user_id"]) if row["actor_user_id"] is not None else None,
        requested_search_scope=str(row["requested_search_scope"]),
        document_group=row["document_group"],
        file_type=row["file_type"],
        chunk_policy_name=row["chunk_policy_name"],
        top_k=int(row["top_k"]),
        metadata=dict(row["metadata"] or {}),
        created_by_user_id=(
            int(row["created_by_user_id"]) if row["created_by_user_id"] is not None else None
        ),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_expected_target_record(row: dict[str, Any]) -> GoldenQuestionExpectedTargetRecord:
    return GoldenQuestionExpectedTargetRecord(
        expected_target_id=int(row["expected_target_id"]),
        question_id=int(row["question_id"]),
        chunk_id=int(row["chunk_id"]) if row["chunk_id"] is not None else None,
        expected_heading_path=tuple(row["expected_heading_path"] or ()),
        expectation_type=str(row["expectation_type"]),
        relevance_grade=int(row["relevance_grade"]),
        notes=row["notes"],
        metadata=dict(row["metadata"] or {}),
        created_at=row["created_at"],
    )


def create_golden_question_set_in_connection(
    connection: Connection,
    question_set_input: GoldenQuestionSetInput,
) -> GoldenQuestionSetRecord:
    validated = validate_golden_question_set_input(question_set_input)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO golden_question_sets (
                set_name,
                description,
                is_active,
                metadata,
                created_by_user_id
            )
            VALUES (%s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                validated.set_name,
                validated.description,
                validated.is_active,
                Json(validated.metadata),
                validated.created_by_user_id,
            ),
        )
        return _row_to_question_set_record(dict(cursor.fetchone()))


def create_golden_question_set(
    database_url: str,
    question_set_input: GoldenQuestionSetInput,
) -> GoldenQuestionSetRecord:
    with connect(database_url) as connection:
        return create_golden_question_set_in_connection(connection, question_set_input)


def get_golden_question_set(
    database_url: str,
    question_set_id: int,
) -> GoldenQuestionSetRecord | None:
    _require_positive_id(question_set_id, "question_set_id")
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM golden_question_sets WHERE question_set_id = %s",
                (question_set_id,),
            )
            row = cursor.fetchone()
    return _row_to_question_set_record(dict(row)) if row else None


def list_golden_question_sets(
    database_url: str,
    *,
    active_only: bool = True,
    limit: int = 100,
) -> list[GoldenQuestionSetRecord]:
    validated_limit = _validate_limit(limit)
    active_filter = "WHERE is_active" if active_only else ""
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM golden_question_sets
                {active_filter}
                ORDER BY created_at DESC, question_set_id DESC
                LIMIT %s
                """,
                (validated_limit,),
            )
            rows = cursor.fetchall()
    return [_row_to_question_set_record(dict(row)) for row in rows]


def create_golden_question_in_connection(
    connection: Connection,
    question_input: GoldenQuestionInput,
) -> GoldenQuestionRecord:
    validated = validate_golden_question_input(question_input)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO golden_questions (
                question_set_id,
                question_text,
                normalized_question_text,
                question_type,
                actor_user_id,
                requested_search_scope,
                document_group,
                file_type,
                chunk_policy_name,
                top_k,
                metadata,
                created_by_user_id
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                validated.question_set_id,
                validated.question_text,
                validated.normalized_question_text,
                validated.question_type,
                validated.actor_user_id,
                validated.requested_search_scope,
                validated.document_group,
                validated.file_type,
                validated.chunk_policy_name,
                validated.top_k,
                Json(validated.metadata),
                validated.created_by_user_id,
            ),
        )
        return _row_to_question_record(dict(cursor.fetchone()))


def create_golden_question(
    database_url: str,
    question_input: GoldenQuestionInput,
) -> GoldenQuestionRecord:
    with connect(database_url) as connection:
        return create_golden_question_in_connection(connection, question_input)


def get_golden_question(
    database_url: str,
    question_id: int,
) -> GoldenQuestionRecord | None:
    _require_positive_id(question_id, "question_id")
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM golden_questions WHERE question_id = %s",
                (question_id,),
            )
            row = cursor.fetchone()
    return _row_to_question_record(dict(row)) if row else None


def list_golden_questions(
    database_url: str,
    question_set_id: int,
    *,
    actor_user_id: int | None = None,
    requested_search_scope: str | None = None,
    limit: int = 100,
) -> list[GoldenQuestionRecord]:
    _require_positive_id(question_set_id, "question_set_id")
    _require_positive_id(actor_user_id, "actor_user_id")
    validated_limit = _validate_limit(limit)
    filters = ["question_set_id = %s"]
    params: list[object] = [question_set_id]
    if actor_user_id is not None:
        filters.append("actor_user_id = %s")
        params.append(actor_user_id)
    if requested_search_scope is not None:
        filters.append("requested_search_scope = %s")
        params.append(_validate_scope(requested_search_scope))

    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM golden_questions
                WHERE {' AND '.join(filters)}
                ORDER BY question_id ASC
                LIMIT %s
                """,
                [*params, validated_limit],
            )
            rows = cursor.fetchall()
    return [_row_to_question_record(dict(row)) for row in rows]


def create_expected_target_in_connection(
    connection: Connection,
    target_input: GoldenQuestionExpectedTargetInput,
) -> GoldenQuestionExpectedTargetRecord:
    validated = validate_expected_target_input(target_input)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO golden_question_expected_targets (
                question_id,
                chunk_id,
                expected_heading_path,
                expectation_type,
                relevance_grade,
                notes,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                validated.question_id,
                validated.chunk_id,
                list(validated.expected_heading_path) or None,
                validated.expectation_type,
                validated.relevance_grade,
                validated.notes,
                Json(validated.metadata),
            ),
        )
        return _row_to_expected_target_record(dict(cursor.fetchone()))


def create_expected_target(
    database_url: str,
    target_input: GoldenQuestionExpectedTargetInput,
) -> GoldenQuestionExpectedTargetRecord:
    with connect(database_url) as connection:
        return create_expected_target_in_connection(connection, target_input)


def list_expected_targets(
    database_url: str,
    question_id: int,
) -> list[GoldenQuestionExpectedTargetRecord]:
    _require_positive_id(question_id, "question_id")
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM golden_question_expected_targets
                WHERE question_id = %s
                ORDER BY expectation_type DESC, relevance_grade DESC, expected_target_id ASC
                """,
                (question_id,),
            )
            rows = cursor.fetchall()
    return [_row_to_expected_target_record(dict(row)) for row in rows]


def get_golden_question_detail(
    database_url: str,
    question_id: int,
) -> GoldenQuestionDetailRecord | None:
    question = get_golden_question(database_url, question_id)
    if question is None:
        return None
    return GoldenQuestionDetailRecord(
        question=question,
        expected_targets=tuple(list_expected_targets(database_url, question_id)),
    )
