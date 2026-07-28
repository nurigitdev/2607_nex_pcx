"""Conversational session, message, and artifact link repository helpers."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from psycopg.types.json import Json

from app.core.database import connect
from app.core.generation_runs import (
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
)
from app.core.search_logs import SEARCH_SCOPES

CHAT_SESSION_STATUS_ACTIVE = "active"
CHAT_SESSION_STATUS_ARCHIVE = "archive"
CHAT_SESSION_STATUSES = {CHAT_SESSION_STATUS_ACTIVE, CHAT_SESSION_STATUS_ARCHIVE}

CHAT_ROLE_USER = "user"
CHAT_ROLE_ASSISTANT = "assistant"
CHAT_ROLE_SYSTEM = "system"
CHAT_ROLE_TOOL = "tool"
CHAT_ROLES = {CHAT_ROLE_USER, CHAT_ROLE_ASSISTANT, CHAT_ROLE_SYSTEM, CHAT_ROLE_TOOL}

CHAT_INTENT_GENERAL_ANSWER = "general_answer"
CHAT_INTENT_DOCUMENT_SEARCH_SUMMARY = "document_search_summary"
CHAT_INTENT_GROUNDED_ANSWER = "grounded_answer"
CHAT_INTENT_DOCUMENT_GENERATION = "document_generation"
CHAT_INTENT_DOCUMENT_SUMMARY = "document_summary"
CHAT_INTENTS = {
    CHAT_INTENT_GENERAL_ANSWER,
    CHAT_INTENT_DOCUMENT_SEARCH_SUMMARY,
    CHAT_INTENT_GROUNDED_ANSWER,
    CHAT_INTENT_DOCUMENT_GENERATION,
    CHAT_INTENT_DOCUMENT_SUMMARY,
}

CHAT_MESSAGE_STATUS_PENDING = "pending"
CHAT_MESSAGE_STATUS_RUNNING = "running"
CHAT_MESSAGE_STATUS_COMPLETED = "completed"
CHAT_MESSAGE_STATUS_FAILED = "failed"
CHAT_MESSAGE_STATUS_BLOCKED = "blocked"
CHAT_MESSAGE_STATUSES = {
    CHAT_MESSAGE_STATUS_PENDING,
    CHAT_MESSAGE_STATUS_RUNNING,
    CHAT_MESSAGE_STATUS_COMPLETED,
    CHAT_MESSAGE_STATUS_FAILED,
    CHAT_MESSAGE_STATUS_BLOCKED,
}

CHAT_LINK_TYPE_SEARCH_LOG = "search_log"
CHAT_LINK_TYPE_GENERATION_RUN = "generation_run"
CHAT_LINK_TYPE_DOCUMENT_SUMMARY = "document_summary"
CHAT_LINK_TYPE_DOCUMENT = "document"
CHAT_LINK_TYPE_ARTIFACT = "artifact"
CHAT_LINK_TYPE_DOWNLOAD = "download"
CHAT_LINK_TYPES = {
    CHAT_LINK_TYPE_SEARCH_LOG,
    CHAT_LINK_TYPE_GENERATION_RUN,
    CHAT_LINK_TYPE_DOCUMENT_SUMMARY,
    CHAT_LINK_TYPE_DOCUMENT,
    CHAT_LINK_TYPE_ARTIFACT,
    CHAT_LINK_TYPE_DOWNLOAD,
}

CHAT_PROVIDER_MODES = {
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
}

DEFAULT_CHAT_SESSION_LIMIT = 50
MAX_CHAT_SESSION_LIMIT = 200


@dataclass(frozen=True)
class ChatSessionInput:
    session_title: str
    actor_user_id: int | None = None
    default_language: str = "ko"
    default_provider_mode: str = GENERATION_PROVIDER_MODE_MOCK
    default_search_profile_name: str | None = None
    default_search_scope: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatSessionRecord:
    chat_session_id: int
    actor_user_id: int | None
    session_title: str
    status: str
    default_language: str
    default_provider_mode: str
    default_search_profile_name: str | None
    default_search_scope: str | None
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ChatMessageInput:
    chat_session_id: int
    role: str
    content: str
    parent_message_id: int | None = None
    sequence_no: int | None = None
    intent: str | None = None
    status: str = CHAT_MESSAGE_STATUS_COMPLETED
    intent_confidence: float | None = None
    routing_metadata: Mapping[str, Any] = field(default_factory=dict)
    runtime_metadata: Mapping[str, Any] = field(default_factory=dict)
    error_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatMessageRecord:
    chat_message_id: int
    chat_session_id: int
    parent_message_id: int | None
    sequence_no: int
    role: str
    content: str
    intent: str | None
    status: str
    intent_confidence: float | None
    routing_metadata: dict[str, Any]
    runtime_metadata: dict[str, Any]
    error_metadata: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class ChatMessageLinkInput:
    chat_message_id: int
    link_type: str
    target_id: int | None = None
    target_url: str | None = None
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatMessageLinkRecord:
    chat_message_link_id: int
    chat_message_id: int
    link_type: str
    target_id: int | None
    target_url: str | None
    label: str | None
    metadata: dict[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class ChatMessageWithLinks:
    message: ChatMessageRecord
    links: tuple[ChatMessageLinkRecord, ...]


@dataclass(frozen=True)
class ChatThreadRecord:
    session: ChatSessionRecord
    messages: tuple[ChatMessageWithLinks, ...]


@dataclass(frozen=True)
class ChatIntentRoute:
    intent: str
    intent_confidence: float
    rationale: str
    suggested_action: str


class InvalidChatRepositoryError(ValueError):
    """Raised when chat repository input is invalid before reaching the DB."""


def _validate_nonblank(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise InvalidChatRepositoryError(f"{field_name} must not be blank")
    return normalized


def _validate_positive(value: int | None, field_name: str) -> int | None:
    if value is not None and value <= 0:
        raise InvalidChatRepositoryError(f"{field_name} must be greater than 0")
    return value


def _validate_choice(value: str, field_name: str, allowed_values: set[str]) -> str:
    normalized = (_validate_nonblank(value, field_name) or value).lower()
    if normalized not in allowed_values:
        raise InvalidChatRepositoryError(f"{field_name} is not supported")
    return normalized


def _validate_optional_choice(
    value: str | None,
    field_name: str,
    allowed_values: set[str],
) -> str | None:
    normalized = _validate_nonblank(value, field_name)
    if normalized is None:
        return None
    lowered = normalized.lower()
    if lowered not in allowed_values:
        raise InvalidChatRepositoryError(f"{field_name} is not supported")
    return lowered


def _validate_json_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidChatRepositoryError(f"{field_name} must be a JSON object")
    return dict(value)


def _validate_intent_confidence(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise InvalidChatRepositoryError("intent_confidence must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0 or normalized > 1:
        raise InvalidChatRepositoryError("intent_confidence must be between 0 and 1")
    return normalized


def validate_chat_session_input(session_input: ChatSessionInput) -> ChatSessionInput:
    _validate_positive(session_input.actor_user_id, "actor_user_id")
    return ChatSessionInput(
        session_title=_validate_nonblank(session_input.session_title, "session_title")
        or session_input.session_title,
        actor_user_id=session_input.actor_user_id,
        default_language=_validate_nonblank(session_input.default_language, "default_language")
        or session_input.default_language,
        default_provider_mode=_validate_choice(
            session_input.default_provider_mode,
            "default_provider_mode",
            CHAT_PROVIDER_MODES,
        ),
        default_search_profile_name=_validate_nonblank(
            session_input.default_search_profile_name,
            "default_search_profile_name",
        ),
        default_search_scope=_validate_optional_choice(
            session_input.default_search_scope,
            "default_search_scope",
            SEARCH_SCOPES,
        ),
        metadata=_validate_json_mapping(session_input.metadata, "metadata"),
    )


def validate_chat_message_input(message_input: ChatMessageInput) -> ChatMessageInput:
    _validate_positive(message_input.chat_session_id, "chat_session_id")
    _validate_positive(message_input.parent_message_id, "parent_message_id")
    _validate_positive(message_input.sequence_no, "sequence_no")
    return ChatMessageInput(
        chat_session_id=message_input.chat_session_id,
        parent_message_id=message_input.parent_message_id,
        sequence_no=message_input.sequence_no,
        role=_validate_choice(message_input.role, "role", CHAT_ROLES),
        content=_validate_nonblank(message_input.content, "content") or message_input.content,
        intent=_validate_optional_choice(message_input.intent, "intent", CHAT_INTENTS),
        status=_validate_choice(message_input.status, "status", CHAT_MESSAGE_STATUSES),
        intent_confidence=_validate_intent_confidence(message_input.intent_confidence),
        routing_metadata=_validate_json_mapping(
            message_input.routing_metadata,
            "routing_metadata",
        ),
        runtime_metadata=_validate_json_mapping(
            message_input.runtime_metadata,
            "runtime_metadata",
        ),
        error_metadata=_validate_json_mapping(message_input.error_metadata, "error_metadata"),
    )


def validate_chat_message_link_input(
    link_input: ChatMessageLinkInput,
) -> ChatMessageLinkInput:
    _validate_positive(link_input.chat_message_id, "chat_message_id")
    target_id = _validate_positive(link_input.target_id, "target_id")
    target_url = _validate_nonblank(link_input.target_url, "target_url")
    if target_id is None and target_url is None:
        raise InvalidChatRepositoryError("target_id or target_url is required")
    return ChatMessageLinkInput(
        chat_message_id=link_input.chat_message_id,
        link_type=_validate_choice(link_input.link_type, "link_type", CHAT_LINK_TYPES),
        target_id=target_id,
        target_url=target_url,
        label=_validate_nonblank(link_input.label, "label"),
        metadata=_validate_json_mapping(link_input.metadata, "metadata"),
    )


def validate_chat_session_limit(limit: int) -> int:
    if limit < 1 or limit > MAX_CHAT_SESSION_LIMIT:
        raise InvalidChatRepositoryError(f"limit must be between 1 and {MAX_CHAT_SESSION_LIMIT}")
    return limit


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def route_chat_intent_mock(content: str) -> ChatIntentRoute:
    normalized = _validate_nonblank(content, "content")
    assert normalized is not None
    lowered = normalized.lower()
    generation_keywords = (
        "보고서",
        "제안서",
        "요약서",
        "작성",
        "생성",
        "draft",
        "proposal",
        "report",
    )
    summary_keywords = ("요약", "summarize", "summary")
    search_keywords = ("검색", "찾아", "찾기", "관련 문서", "search", "retrieve")
    grounded_keywords = ("근거", "규정", "정책", "사규", "문서 기준", "based on", "citation")

    if _contains_any(lowered, generation_keywords):
        return ChatIntentRoute(
            intent=CHAT_INTENT_DOCUMENT_GENERATION,
            intent_confidence=0.88,
            rationale="document_generation_keyword_match",
            suggested_action="select_template_and_run_grounded_generation",
        )
    if _contains_any(lowered, summary_keywords) and _contains_any(lowered, search_keywords):
        return ChatIntentRoute(
            intent=CHAT_INTENT_DOCUMENT_SEARCH_SUMMARY,
            intent_confidence=0.84,
            rationale="summary_and_search_keyword_match",
            suggested_action="run_search_then_summarize_retrieved_chunks",
        )
    if _contains_any(lowered, summary_keywords):
        return ChatIntentRoute(
            intent=CHAT_INTENT_DOCUMENT_SUMMARY,
            intent_confidence=0.8,
            rationale="summary_keyword_match",
            suggested_action="select_document_or_corpus_summary_flow",
        )
    if _contains_any(lowered, grounded_keywords) or _contains_any(lowered, search_keywords):
        return ChatIntentRoute(
            intent=CHAT_INTENT_GROUNDED_ANSWER,
            intent_confidence=0.76,
            rationale="grounded_or_search_keyword_match",
            suggested_action="run_retrieval_context_package_then_generation",
        )
    return ChatIntentRoute(
        intent=CHAT_INTENT_GENERAL_ANSWER,
        intent_confidence=0.55,
        rationale="fallback_general_answer",
        suggested_action="call_llm_without_document_context",
    )


def _row_to_chat_session(row: dict[str, Any]) -> ChatSessionRecord:
    return ChatSessionRecord(
        chat_session_id=int(row["chat_session_id"]),
        actor_user_id=int(row["actor_user_id"]) if row["actor_user_id"] is not None else None,
        session_title=str(row["session_title"]),
        status=str(row["status"]),
        default_language=str(row["default_language"]),
        default_provider_mode=str(row["default_provider_mode"]),
        default_search_profile_name=row["default_search_profile_name"],
        default_search_scope=row["default_search_scope"],
        metadata=dict(row["metadata"] or {}),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_chat_message(row: dict[str, Any]) -> ChatMessageRecord:
    return ChatMessageRecord(
        chat_message_id=int(row["chat_message_id"]),
        chat_session_id=int(row["chat_session_id"]),
        parent_message_id=(
            int(row["parent_message_id"]) if row["parent_message_id"] is not None else None
        ),
        sequence_no=int(row["sequence_no"]),
        role=str(row["role"]),
        content=str(row["content"]),
        intent=row["intent"],
        status=str(row["status"]),
        intent_confidence=(
            float(row["intent_confidence"]) if row["intent_confidence"] is not None else None
        ),
        routing_metadata=dict(row["routing_metadata"] or {}),
        runtime_metadata=dict(row["runtime_metadata"] or {}),
        error_metadata=dict(row["error_metadata"] or {}),
        created_at=row["created_at"],
    )


def _row_to_chat_message_link(row: dict[str, Any]) -> ChatMessageLinkRecord:
    return ChatMessageLinkRecord(
        chat_message_link_id=int(row["chat_message_link_id"]),
        chat_message_id=int(row["chat_message_id"]),
        link_type=str(row["link_type"]),
        target_id=int(row["target_id"]) if row["target_id"] is not None else None,
        target_url=row["target_url"],
        label=row["label"],
        metadata=dict(row["metadata"] or {}),
        created_at=row["created_at"],
    )


def create_chat_session(
    database_url: str,
    session_input: ChatSessionInput,
) -> ChatSessionRecord:
    validated = validate_chat_session_input(session_input)
    with connect(database_url) as conn:
        row = conn.execute(
            """
            INSERT INTO chat_sessions (
                actor_user_id,
                session_title,
                default_language,
                default_provider_mode,
                default_search_profile_name,
                default_search_scope,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                validated.actor_user_id,
                validated.session_title,
                validated.default_language,
                validated.default_provider_mode,
                validated.default_search_profile_name,
                validated.default_search_scope,
                Json(validated.metadata),
            ),
        ).fetchone()
        conn.commit()
    assert row is not None
    return _row_to_chat_session(dict(row))


def get_chat_session(database_url: str, chat_session_id: int) -> ChatSessionRecord | None:
    _validate_positive(chat_session_id, "chat_session_id")
    with connect(database_url) as conn:
        row = conn.execute(
            """
            SELECT *
            FROM chat_sessions
            WHERE chat_session_id = %s
            """,
            (chat_session_id,),
        ).fetchone()
    return _row_to_chat_session(dict(row)) if row else None


def list_chat_sessions(
    database_url: str,
    *,
    limit: int = DEFAULT_CHAT_SESSION_LIMIT,
    status: str | None = CHAT_SESSION_STATUS_ACTIVE,
    actor_user_id: int | None = None,
) -> tuple[ChatSessionRecord, ...]:
    validated_limit = validate_chat_session_limit(limit)
    validated_status = _validate_optional_choice(status, "status", CHAT_SESSION_STATUSES)
    _validate_positive(actor_user_id, "actor_user_id")
    filters: list[str] = []
    params: list[object] = []
    if validated_status is not None:
        filters.append("status = %s")
        params.append(validated_status)
    if actor_user_id is not None:
        filters.append("actor_user_id = %s")
        params.append(actor_user_id)
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    params.append(validated_limit)
    with connect(database_url) as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM chat_sessions
            {where_clause}
            ORDER BY updated_at DESC, chat_session_id DESC
            LIMIT %s
            """,
            params,
        ).fetchall()
    return tuple(_row_to_chat_session(dict(row)) for row in rows)


def _next_sequence_no(conn, chat_session_id: int) -> int:
    row = conn.execute(
        """
        SELECT COALESCE(MAX(sequence_no), 0) + 1 AS next_sequence_no
        FROM chat_messages
        WHERE chat_session_id = %s
        """,
        (chat_session_id,),
    ).fetchone()
    assert row is not None
    return int(row["next_sequence_no"])


def _validate_parent_message_session(
    conn,
    *,
    parent_message_id: int | None,
    chat_session_id: int,
) -> None:
    if parent_message_id is None:
        return
    parent = conn.execute(
        """
        SELECT chat_session_id
        FROM chat_messages
        WHERE chat_message_id = %s
        """,
        (parent_message_id,),
    ).fetchone()
    if parent is None:
        raise InvalidChatRepositoryError("parent_message_id was not found")
    if int(parent["chat_session_id"]) != chat_session_id:
        raise InvalidChatRepositoryError("parent_message_id must belong to chat_session_id")


def append_chat_message(
    database_url: str,
    message_input: ChatMessageInput,
) -> ChatMessageRecord:
    validated = validate_chat_message_input(message_input)
    with connect(database_url) as conn:
        _validate_parent_message_session(
            conn,
            parent_message_id=validated.parent_message_id,
            chat_session_id=validated.chat_session_id,
        )
        sequence_no = validated.sequence_no or _next_sequence_no(conn, validated.chat_session_id)
        row = conn.execute(
            """
            INSERT INTO chat_messages (
                chat_session_id,
                parent_message_id,
                sequence_no,
                role,
                content,
                intent,
                status,
                intent_confidence,
                routing_metadata,
                runtime_metadata,
                error_metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                validated.chat_session_id,
                validated.parent_message_id,
                sequence_no,
                validated.role,
                validated.content,
                validated.intent,
                validated.status,
                validated.intent_confidence,
                Json(validated.routing_metadata),
                Json(validated.runtime_metadata),
                Json(validated.error_metadata),
            ),
        ).fetchone()
        conn.execute(
            """
            UPDATE chat_sessions
            SET updated_at = now()
            WHERE chat_session_id = %s
            """,
            (validated.chat_session_id,),
        )
        conn.commit()
    assert row is not None
    return _row_to_chat_message(dict(row))


def get_chat_message(database_url: str, chat_message_id: int) -> ChatMessageRecord | None:
    _validate_positive(chat_message_id, "chat_message_id")
    with connect(database_url) as conn:
        row = conn.execute(
            """
            SELECT *
            FROM chat_messages
            WHERE chat_message_id = %s
            """,
            (chat_message_id,),
        ).fetchone()
    return _row_to_chat_message(dict(row)) if row else None


def list_chat_messages(
    database_url: str,
    chat_session_id: int,
) -> tuple[ChatMessageRecord, ...]:
    _validate_positive(chat_session_id, "chat_session_id")
    with connect(database_url) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM chat_messages
            WHERE chat_session_id = %s
            ORDER BY sequence_no, chat_message_id
            """,
            (chat_session_id,),
        ).fetchall()
    return tuple(_row_to_chat_message(dict(row)) for row in rows)


def create_chat_message_link(
    database_url: str,
    link_input: ChatMessageLinkInput,
) -> ChatMessageLinkRecord:
    validated = validate_chat_message_link_input(link_input)
    with connect(database_url) as conn:
        row = conn.execute(
            """
            INSERT INTO chat_message_links (
                chat_message_id,
                link_type,
                target_id,
                target_url,
                label,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                validated.chat_message_id,
                validated.link_type,
                validated.target_id,
                validated.target_url,
                validated.label,
                Json(validated.metadata),
            ),
        ).fetchone()
        conn.commit()
    assert row is not None
    return _row_to_chat_message_link(dict(row))


def link_chat_message_to_search_log(
    database_url: str,
    *,
    chat_message_id: int,
    search_log_id: int,
    label: str | None = "검색 이력",
    metadata: Mapping[str, Any] | None = None,
) -> ChatMessageLinkRecord:
    return create_chat_message_link(
        database_url,
        ChatMessageLinkInput(
            chat_message_id=chat_message_id,
            link_type=CHAT_LINK_TYPE_SEARCH_LOG,
            target_id=search_log_id,
            label=label,
            metadata=metadata or {},
        ),
    )


def link_chat_message_to_generation_run(
    database_url: str,
    *,
    chat_message_id: int,
    generation_run_id: int,
    label: str | None = "생성 실행",
    metadata: Mapping[str, Any] | None = None,
) -> ChatMessageLinkRecord:
    return create_chat_message_link(
        database_url,
        ChatMessageLinkInput(
            chat_message_id=chat_message_id,
            link_type=CHAT_LINK_TYPE_GENERATION_RUN,
            target_id=generation_run_id,
            label=label,
            metadata=metadata or {},
        ),
    )


def link_chat_message_to_document_summary(
    database_url: str,
    *,
    chat_message_id: int,
    generation_run_id: int,
    label: str | None = "문서 요약",
    metadata: Mapping[str, Any] | None = None,
) -> ChatMessageLinkRecord:
    return create_chat_message_link(
        database_url,
        ChatMessageLinkInput(
            chat_message_id=chat_message_id,
            link_type=CHAT_LINK_TYPE_DOCUMENT_SUMMARY,
            target_id=generation_run_id,
            label=label,
            metadata=metadata or {},
        ),
    )


def link_chat_message_to_download_url(
    database_url: str,
    *,
    chat_message_id: int,
    target_url: str,
    label: str | None = "다운로드",
    metadata: Mapping[str, Any] | None = None,
) -> ChatMessageLinkRecord:
    return create_chat_message_link(
        database_url,
        ChatMessageLinkInput(
            chat_message_id=chat_message_id,
            link_type=CHAT_LINK_TYPE_DOWNLOAD,
            target_url=target_url,
            label=label,
            metadata=metadata or {},
        ),
    )


def list_chat_message_links(
    database_url: str,
    chat_message_id: int,
) -> tuple[ChatMessageLinkRecord, ...]:
    _validate_positive(chat_message_id, "chat_message_id")
    with connect(database_url) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM chat_message_links
            WHERE chat_message_id = %s
            ORDER BY chat_message_link_id
            """,
            (chat_message_id,),
        ).fetchall()
    return tuple(_row_to_chat_message_link(dict(row)) for row in rows)


def get_chat_thread(database_url: str, chat_session_id: int) -> ChatThreadRecord | None:
    session = get_chat_session(database_url, chat_session_id)
    if session is None:
        return None
    messages = list_chat_messages(database_url, chat_session_id)
    if not messages:
        return ChatThreadRecord(session=session, messages=())
    message_ids = [message.chat_message_id for message in messages]
    with connect(database_url) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM chat_message_links
            WHERE chat_message_id = ANY(%s)
            ORDER BY chat_message_id, chat_message_link_id
            """,
            (message_ids,),
        ).fetchall()
    links_by_message_id: dict[int, list[ChatMessageLinkRecord]] = {
        message.chat_message_id: [] for message in messages
    }
    for row in rows:
        link = _row_to_chat_message_link(dict(row))
        links_by_message_id.setdefault(link.chat_message_id, []).append(link)
    return ChatThreadRecord(
        session=session,
        messages=tuple(
            ChatMessageWithLinks(
                message=message,
                links=tuple(links_by_message_id.get(message.chat_message_id, ())),
            )
            for message in messages
        ),
    )
