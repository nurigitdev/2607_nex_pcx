"""Generation provider config and run repository helpers."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from psycopg.types.json import Json

from app.core.citation_readiness import (
    CITATION_READINESS_FAILED,
    CITATION_READINESS_READY,
    CITATION_READINESS_WARNING,
)
from app.core.database import connect
from app.core.retrieval_confidence import (
    RETRIEVAL_CONFIDENCE_ANSWERABLE,
    RETRIEVAL_CONFIDENCE_FAILED,
    RETRIEVAL_CONFIDENCE_LOW,
    RETRIEVAL_CONFIDENCE_NO_CONTEXT,
)

GENERATION_PROVIDER_MODE_MOCK = "mock"
GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE = "remote_openai_compatible"
GENERATION_PROVIDER_MODES = {
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
}

GENERATION_STATUS_PENDING = "pending"
GENERATION_STATUS_BLOCKED = "blocked"
GENERATION_STATUS_RUNNING = "running"
GENERATION_STATUS_SUCCEEDED = "succeeded"
GENERATION_STATUS_FAILED = "failed"
GENERATION_STATUS_CANCELED = "canceled"
GENERATION_STATUS_NO_ANSWER = "no_answer"
GENERATION_STATUSES = {
    GENERATION_STATUS_PENDING,
    GENERATION_STATUS_BLOCKED,
    GENERATION_STATUS_RUNNING,
    GENERATION_STATUS_SUCCEEDED,
    GENERATION_STATUS_FAILED,
    GENERATION_STATUS_CANCELED,
    GENERATION_STATUS_NO_ANSWER,
}

GENERATION_GUARDRAIL_ALLOWED = "allowed"
GENERATION_GUARDRAIL_BLOCKED = "blocked"
GENERATION_GUARDRAIL_NO_ANSWER = "no_answer"
GENERATION_GUARDRAIL_STATUSES = {
    GENERATION_GUARDRAIL_ALLOWED,
    GENERATION_GUARDRAIL_BLOCKED,
    GENERATION_GUARDRAIL_NO_ANSWER,
}
GENERATION_RETRIEVAL_CONFIDENCE_STATUSES = {
    RETRIEVAL_CONFIDENCE_ANSWERABLE,
    RETRIEVAL_CONFIDENCE_LOW,
    RETRIEVAL_CONFIDENCE_NO_CONTEXT,
    RETRIEVAL_CONFIDENCE_FAILED,
}
GENERATION_CITATION_READINESS_STATUSES = {
    CITATION_READINESS_READY,
    CITATION_READINESS_WARNING,
    CITATION_READINESS_FAILED,
}


@dataclass(frozen=True)
class GenerationProviderConfigRecord:
    provider_config_id: int
    provider_name: str
    provider_mode: str
    provider_base_url: str | None
    model_id: str
    is_default: bool
    is_active: bool
    request_timeout_seconds: int
    max_tokens: int
    temperature: float
    top_p: float
    runtime_options: dict[str, Any]
    created_by: str | None
    created_by_user_id: int | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class GenerationRunInput:
    search_log_id: int
    retrieval_package_key: str
    provider_name: str
    provider_mode: str
    model_id: str
    retrieval_confidence_status: str
    citation_readiness_status: str
    query_text: str
    provider_config_id: int | None = None
    prompt_version: str = "grounded_answer_v1"
    prompt_hash: str | None = None
    context_hash: str | None = None
    status: str = GENERATION_STATUS_PENDING
    guardrail_status: str = GENERATION_GUARDRAIL_ALLOWED
    answer_text: str | None = None
    finish_reason: str | None = None
    input_token_count: int | None = None
    output_token_count: int | None = None
    total_token_count: int | None = None
    elapsed_ms: int | None = None
    request_metadata: dict[str, Any] = field(default_factory=dict)
    response_metadata: dict[str, Any] = field(default_factory=dict)
    guardrail_metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    created_by: str | None = None
    created_by_user_id: int | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


@dataclass(frozen=True)
class GenerationRunRecord:
    generation_run_id: int
    search_log_id: int
    retrieval_package_key: str
    provider_config_id: int | None
    provider_name: str
    provider_mode: str
    model_id: str
    prompt_version: str
    prompt_hash: str | None
    context_hash: str | None
    status: str
    guardrail_status: str
    retrieval_confidence_status: str
    citation_readiness_status: str
    query_text: str
    answer_text: str | None
    finish_reason: str | None
    input_token_count: int | None
    output_token_count: int | None
    total_token_count: int | None
    elapsed_ms: int | None
    request_metadata: dict[str, Any]
    response_metadata: dict[str, Any]
    guardrail_metadata: dict[str, Any]
    error_message: str | None
    created_by: str | None
    created_by_user_id: int | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class GenerationRunCitationInput:
    generation_run_id: int
    citation_key: str
    citation_index: int
    search_log_result_id: int | None = None
    chunk_id: int | None = None
    document_id: int | None = None
    file_id: int | None = None
    source_label: str = ""
    source_anchor: dict[str, Any] = field(default_factory=dict)
    citation_payload: dict[str, Any] = field(default_factory=dict)
    was_cited: bool = False


@dataclass(frozen=True)
class GenerationRunCitationRecord:
    generation_run_citation_id: int
    generation_run_id: int
    citation_key: str
    citation_index: int
    search_log_result_id: int | None
    chunk_id: int | None
    document_id: int | None
    file_id: int | None
    source_label: str
    source_anchor: dict[str, Any]
    citation_payload: dict[str, Any]
    was_cited: bool
    created_at: datetime


class InvalidGenerationRunError(ValueError):
    """Raised when generation run repository input is invalid."""


def _validate_non_empty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidGenerationRunError(f"{field_name} must not be empty")
    return normalized


def _validate_positive(value: int, field_name: str) -> int:
    if value <= 0:
        raise InvalidGenerationRunError(f"{field_name} must be greater than 0")
    return value


def _validate_optional_non_negative(value: int | None, field_name: str) -> int | None:
    if value is not None and value < 0:
        raise InvalidGenerationRunError(f"{field_name} must be greater than or equal to 0")
    return value


def _validate_optional_positive(value: int | None, field_name: str) -> int | None:
    if value is not None:
        _validate_positive(value, field_name)
    return value


def _provider_config_from_row(row: dict[str, Any]) -> GenerationProviderConfigRecord:
    return GenerationProviderConfigRecord(
        provider_config_id=int(row["provider_config_id"]),
        provider_name=str(row["provider_name"]),
        provider_mode=str(row["provider_mode"]),
        provider_base_url=row["provider_base_url"],
        model_id=str(row["model_id"]),
        is_default=bool(row["is_default"]),
        is_active=bool(row["is_active"]),
        request_timeout_seconds=int(row["request_timeout_seconds"]),
        max_tokens=int(row["max_tokens"]),
        temperature=float(row["temperature"]),
        top_p=float(row["top_p"]),
        runtime_options=dict(row["runtime_options"] or {}),
        created_by=row["created_by"],
        created_by_user_id=row["created_by_user_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _generation_run_from_row(row: dict[str, Any]) -> GenerationRunRecord:
    return GenerationRunRecord(
        generation_run_id=int(row["generation_run_id"]),
        search_log_id=int(row["search_log_id"]),
        retrieval_package_key=str(row["retrieval_package_key"]),
        provider_config_id=row["provider_config_id"],
        provider_name=str(row["provider_name"]),
        provider_mode=str(row["provider_mode"]),
        model_id=str(row["model_id"]),
        prompt_version=str(row["prompt_version"]),
        prompt_hash=row["prompt_hash"],
        context_hash=row["context_hash"],
        status=str(row["status"]),
        guardrail_status=str(row["guardrail_status"]),
        retrieval_confidence_status=str(row["retrieval_confidence_status"]),
        citation_readiness_status=str(row["citation_readiness_status"]),
        query_text=str(row["query_text"]),
        answer_text=row["answer_text"],
        finish_reason=row["finish_reason"],
        input_token_count=row["input_token_count"],
        output_token_count=row["output_token_count"],
        total_token_count=row["total_token_count"],
        elapsed_ms=row["elapsed_ms"],
        request_metadata=dict(row["request_metadata"] or {}),
        response_metadata=dict(row["response_metadata"] or {}),
        guardrail_metadata=dict(row["guardrail_metadata"] or {}),
        error_message=row["error_message"],
        created_by=row["created_by"],
        created_by_user_id=row["created_by_user_id"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _generation_run_citation_from_row(row: dict[str, Any]) -> GenerationRunCitationRecord:
    return GenerationRunCitationRecord(
        generation_run_citation_id=int(row["generation_run_citation_id"]),
        generation_run_id=int(row["generation_run_id"]),
        citation_key=str(row["citation_key"]),
        citation_index=int(row["citation_index"]),
        search_log_result_id=row["search_log_result_id"],
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        file_id=row["file_id"],
        source_label=str(row["source_label"]),
        source_anchor=dict(row["source_anchor"] or {}),
        citation_payload=dict(row["citation_payload"] or {}),
        was_cited=bool(row["was_cited"]),
        created_at=row["created_at"],
    )


def get_default_generation_provider_config(
    database_url: str,
) -> GenerationProviderConfigRecord | None:
    with connect(database_url) as conn:
        row = conn.execute("""
            SELECT *
            FROM generation_provider_configs
            WHERE is_default
              AND is_active
            ORDER BY provider_config_id
            LIMIT 1
            """).fetchone()
    return _provider_config_from_row(row) if row else None


def get_generation_run(database_url: str, generation_run_id: int) -> GenerationRunRecord | None:
    _validate_positive(generation_run_id, "generation_run_id")
    with connect(database_url) as conn:
        row = conn.execute(
            """
            SELECT *
            FROM generation_runs
            WHERE generation_run_id = %s
            """,
            (generation_run_id,),
        ).fetchone()
    return _generation_run_from_row(dict(row)) if row else None


def create_generation_run(
    database_url: str,
    run_input: GenerationRunInput,
) -> GenerationRunRecord:
    validated_search_log_id = _validate_positive(run_input.search_log_id, "search_log_id")
    retrieval_package_key = _validate_non_empty(
        run_input.retrieval_package_key,
        "retrieval_package_key",
    )
    provider_name = _validate_non_empty(run_input.provider_name, "provider_name")
    if run_input.provider_mode not in GENERATION_PROVIDER_MODES:
        raise InvalidGenerationRunError("provider_mode is not supported")
    model_id = _validate_non_empty(run_input.model_id, "model_id")
    prompt_version = _validate_non_empty(run_input.prompt_version, "prompt_version")
    _validate_optional_positive(run_input.provider_config_id, "provider_config_id")
    if run_input.status not in GENERATION_STATUSES:
        raise InvalidGenerationRunError("status is not supported")
    if run_input.guardrail_status not in GENERATION_GUARDRAIL_STATUSES:
        raise InvalidGenerationRunError("guardrail_status is not supported")
    if run_input.retrieval_confidence_status not in GENERATION_RETRIEVAL_CONFIDENCE_STATUSES:
        raise InvalidGenerationRunError("retrieval_confidence_status is not supported")
    if run_input.citation_readiness_status not in GENERATION_CITATION_READINESS_STATUSES:
        raise InvalidGenerationRunError("citation_readiness_status is not supported")
    query_text = _validate_non_empty(run_input.query_text, "query_text")
    _validate_optional_non_negative(run_input.input_token_count, "input_token_count")
    _validate_optional_non_negative(run_input.output_token_count, "output_token_count")
    _validate_optional_non_negative(run_input.total_token_count, "total_token_count")
    _validate_optional_non_negative(run_input.elapsed_ms, "elapsed_ms")

    with connect(database_url) as conn:
        row = conn.execute(
            """
            INSERT INTO generation_runs (
                search_log_id,
                retrieval_package_key,
                provider_config_id,
                provider_name,
                provider_mode,
                model_id,
                prompt_version,
                prompt_hash,
                context_hash,
                status,
                guardrail_status,
                retrieval_confidence_status,
                citation_readiness_status,
                query_text,
                answer_text,
                finish_reason,
                input_token_count,
                output_token_count,
                total_token_count,
                elapsed_ms,
                request_metadata,
                response_metadata,
                guardrail_metadata,
                error_message,
                created_by,
                created_by_user_id,
                started_at,
                finished_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING *
            """,
            (
                validated_search_log_id,
                retrieval_package_key,
                run_input.provider_config_id,
                provider_name,
                run_input.provider_mode,
                model_id,
                prompt_version,
                run_input.prompt_hash,
                run_input.context_hash,
                run_input.status,
                run_input.guardrail_status,
                run_input.retrieval_confidence_status,
                run_input.citation_readiness_status,
                query_text,
                run_input.answer_text,
                run_input.finish_reason,
                run_input.input_token_count,
                run_input.output_token_count,
                run_input.total_token_count,
                run_input.elapsed_ms,
                Json(run_input.request_metadata),
                Json(run_input.response_metadata),
                Json(run_input.guardrail_metadata),
                run_input.error_message,
                run_input.created_by,
                run_input.created_by_user_id,
                run_input.started_at,
                run_input.finished_at,
            ),
        ).fetchone()
        conn.commit()
    assert row is not None
    return _generation_run_from_row(dict(row))


def create_generation_run_citation(
    database_url: str,
    citation_input: GenerationRunCitationInput,
) -> GenerationRunCitationRecord:
    generation_run_id = _validate_positive(citation_input.generation_run_id, "generation_run_id")
    citation_key = _validate_non_empty(citation_input.citation_key, "citation_key")
    citation_index = _validate_positive(citation_input.citation_index, "citation_index")
    _validate_optional_positive(citation_input.search_log_result_id, "search_log_result_id")
    _validate_optional_positive(citation_input.chunk_id, "chunk_id")
    _validate_optional_positive(citation_input.document_id, "document_id")
    _validate_optional_positive(citation_input.file_id, "file_id")

    with connect(database_url) as conn:
        row = conn.execute(
            """
            INSERT INTO generation_run_citations (
                generation_run_id,
                citation_key,
                citation_index,
                search_log_result_id,
                chunk_id,
                document_id,
                file_id,
                source_label,
                source_anchor,
                citation_payload,
                was_cited
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                generation_run_id,
                citation_key,
                citation_index,
                citation_input.search_log_result_id,
                citation_input.chunk_id,
                citation_input.document_id,
                citation_input.file_id,
                citation_input.source_label,
                Json(citation_input.source_anchor),
                Json(citation_input.citation_payload),
                citation_input.was_cited,
            ),
        ).fetchone()
        conn.commit()
    assert row is not None
    return _generation_run_citation_from_row(dict(row))


def list_generation_run_citations(
    database_url: str,
    generation_run_id: int,
) -> tuple[GenerationRunCitationRecord, ...]:
    _validate_positive(generation_run_id, "generation_run_id")
    with connect(database_url) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM generation_run_citations
            WHERE generation_run_id = %s
            ORDER BY citation_index, generation_run_citation_id
            """,
            (generation_run_id,),
        ).fetchall()
    return tuple(_generation_run_citation_from_row(dict(row)) for row in rows)
