"""Generation provider config and run repository helpers."""

from collections.abc import Mapping
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
from app.core.generation_answer_quality import GENERATION_ANSWER_QUALITY_STATUSES
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
DGX_VLLM_GENERATION_PROVIDER_NAME = "dgx_vllm_qwen36_27b_nvfp4"
DGX_VLLM_GENERATION_BASE_URL = "http://192.168.20.243:12000"
DGX_VLLM_GENERATION_MODEL_ID = "/home/nurivoice-dgx/models/nvidia/Qwen3.6-27B-NVFP4"
DGX_VLLM_GENERATION_API_KEY_ENV = "NEX_PCX_REMOTE_GENERATION_PROVIDER_API_KEY"
DGX_VLLM_GENERATION_TIMEOUT_SECONDS = 300
DGX_VLLM_GENERATION_MAX_TOKENS = 1024
DGX_VLLM_GENERATION_TEMPERATURE = 0.2
DGX_VLLM_GENERATION_TOP_P = 0.9
DEFAULT_GENERATION_RUN_HISTORY_LIMIT = 50
MAX_GENERATION_RUN_HISTORY_LIMIT = 500
GENERATION_RUN_HISTORY_FILTER_ALL = "all"
GENERATION_ANSWER_QUALITY_NOT_AVAILABLE = "not_available"
GENERATION_ANSWER_QUALITY_HISTORY_STATUSES = {
    *GENERATION_ANSWER_QUALITY_STATUSES,
    GENERATION_ANSWER_QUALITY_NOT_AVAILABLE,
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
class GenerationProviderConfigInput:
    provider_name: str
    provider_mode: str
    model_id: str
    provider_base_url: str | None = None
    is_default: bool = False
    is_active: bool = True
    request_timeout_seconds: int = 120
    max_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.9
    runtime_options: Mapping[str, Any] = field(default_factory=dict)
    created_by: str | None = None
    created_by_user_id: int | None = None


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
    generation_template_id: int | None = None
    provider_config_id: int | None = None
    prompt_version: str = "grounded_answer_v1_prompt_v1"
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
    generation_template_id: int | None
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


@dataclass(frozen=True)
class GenerationRunHistoryFilter:
    limit: int = DEFAULT_GENERATION_RUN_HISTORY_LIMIT
    answer_quality_status: str = GENERATION_RUN_HISTORY_FILTER_ALL
    provider_mode: str = GENERATION_RUN_HISTORY_FILTER_ALL
    run_status: str = GENERATION_RUN_HISTORY_FILTER_ALL


@dataclass(frozen=True)
class GenerationRunHistoryItem:
    run: GenerationRunRecord
    answer_quality_status: str
    answer_quality_reason_codes: tuple[str, ...]
    citation_coverage_percent: float | None
    expected_citation_count: int
    cited_citation_count: int
    missing_citation_count: int
    unrecognized_citation_count: int


@dataclass(frozen=True)
class GenerationRunHistorySummary:
    run_count: int
    passed_count: int
    warning_count: int
    failed_count: int
    not_evaluated_count: int
    not_available_count: int


@dataclass(frozen=True)
class GenerationRunHistory:
    filters: GenerationRunHistoryFilter
    summary: GenerationRunHistorySummary
    runs: tuple[GenerationRunHistoryItem, ...]


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


def _validate_provider_mode(value: str) -> str:
    normalized = _validate_non_empty(value, "provider_mode").lower()
    if normalized not in GENERATION_PROVIDER_MODES:
        raise InvalidGenerationRunError("provider_mode is not supported")
    return normalized


def _validate_provider_base_url(value: str | None, provider_mode: str) -> str | None:
    if value is None:
        if provider_mode == GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE:
            raise InvalidGenerationRunError("provider_base_url is required for remote provider")
        return None
    normalized = value.strip().rstrip("/")
    if not normalized:
        if provider_mode == GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE:
            raise InvalidGenerationRunError("provider_base_url is required for remote provider")
        return None
    return normalized


def _validate_temperature(value: float) -> float:
    normalized = float(value)
    if not 0 <= normalized <= 2:
        raise InvalidGenerationRunError("temperature must be between 0 and 2")
    return normalized


def _validate_top_p(value: float) -> float:
    normalized = float(value)
    if not 0 < normalized <= 1:
        raise InvalidGenerationRunError("top_p must be greater than 0 and less than or equal to 1")
    return normalized


def validate_generation_run_history_limit(limit: int) -> int:
    if limit < 1 or limit > MAX_GENERATION_RUN_HISTORY_LIMIT:
        raise InvalidGenerationRunError(
            "limit must be between 1 and " f"{MAX_GENERATION_RUN_HISTORY_LIMIT}"
        )
    return limit


def _validate_optional_filter(
    value: str,
    *,
    field_name: str,
    allowed_values: set[str],
) -> str:
    normalized = _validate_non_empty(value, field_name).lower()
    if normalized == GENERATION_RUN_HISTORY_FILTER_ALL:
        return normalized
    if normalized not in allowed_values:
        raise InvalidGenerationRunError(f"{field_name} is not supported")
    return normalized


def validate_generation_run_history_filter(
    history_filter: GenerationRunHistoryFilter,
) -> GenerationRunHistoryFilter:
    return GenerationRunHistoryFilter(
        limit=validate_generation_run_history_limit(history_filter.limit),
        answer_quality_status=_validate_optional_filter(
            history_filter.answer_quality_status,
            field_name="answer_quality_status",
            allowed_values=GENERATION_ANSWER_QUALITY_HISTORY_STATUSES,
        ),
        provider_mode=_validate_optional_filter(
            history_filter.provider_mode,
            field_name="provider_mode",
            allowed_values=GENERATION_PROVIDER_MODES,
        ),
        run_status=_validate_optional_filter(
            history_filter.run_status,
            field_name="run_status",
            allowed_values=GENERATION_STATUSES,
        ),
    )


def _validate_runtime_options(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidGenerationRunError("runtime_options must be a mapping")
    normalized = dict(value)
    _reject_secret_runtime_options(normalized)
    return normalized


def _reject_secret_runtime_options(value: object, *, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_name = str(key).strip()
            lowered_key = key_name.lower()
            child_path = (*path, key_name)
            if _is_forbidden_secret_runtime_option_key(lowered_key):
                dotted = ".".join(child_path)
                raise InvalidGenerationRunError(
                    f"runtime_options.{dotted} must reference an environment variable, "
                    "not a secret value"
                )
            _reject_secret_runtime_options(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secret_runtime_options(child, path=(*path, str(index)))


def _is_forbidden_secret_runtime_option_key(key: str) -> bool:
    if key.endswith("_env") or key in {"api_key_env", "secret_env"}:
        return False
    return key in {
        "api_key",
        "authorization",
        "bearer_token",
        "password",
        "secret",
        "token",
    }


def validate_generation_provider_config_input(
    config_input: GenerationProviderConfigInput,
) -> GenerationProviderConfigInput:
    provider_name = _validate_non_empty(config_input.provider_name, "provider_name")
    provider_mode = _validate_provider_mode(config_input.provider_mode)
    model_id = _validate_non_empty(config_input.model_id, "model_id")
    provider_base_url = _validate_provider_base_url(
        config_input.provider_base_url,
        provider_mode,
    )
    request_timeout_seconds = _validate_positive(
        config_input.request_timeout_seconds,
        "request_timeout_seconds",
    )
    max_tokens = _validate_positive(config_input.max_tokens, "max_tokens")
    runtime_options = _validate_runtime_options(config_input.runtime_options)
    created_by = (
        _validate_non_empty(config_input.created_by, "created_by")
        if config_input.created_by is not None
        else None
    )
    _validate_optional_positive(config_input.created_by_user_id, "created_by_user_id")
    return GenerationProviderConfigInput(
        provider_name=provider_name,
        provider_mode=provider_mode,
        provider_base_url=provider_base_url,
        model_id=model_id,
        is_default=bool(config_input.is_default),
        is_active=bool(config_input.is_active),
        request_timeout_seconds=request_timeout_seconds,
        max_tokens=max_tokens,
        temperature=_validate_temperature(config_input.temperature),
        top_p=_validate_top_p(config_input.top_p),
        runtime_options=runtime_options,
        created_by=created_by,
        created_by_user_id=config_input.created_by_user_id,
    )


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
        generation_template_id=row.get("generation_template_id"),
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


def get_generation_provider_config_for_mode(
    database_url: str,
    provider_mode: str,
) -> GenerationProviderConfigRecord | None:
    normalized_provider_mode = _validate_provider_mode(provider_mode)
    preferred_provider_name = (
        DGX_VLLM_GENERATION_PROVIDER_NAME
        if normalized_provider_mode == GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE
        else ""
    )
    with connect(database_url) as conn:
        row = conn.execute(
            """
            SELECT *
            FROM generation_provider_configs
            WHERE provider_mode = %s
              AND is_active
            ORDER BY
                is_default DESC,
                CASE WHEN provider_name = %s THEN 0 ELSE 1 END,
                provider_config_id
            LIMIT 1
            """,
            (normalized_provider_mode, preferred_provider_name),
        ).fetchone()
    return _provider_config_from_row(row) if row else None


def list_generation_provider_configs(
    database_url: str,
    *,
    include_inactive: bool = True,
) -> tuple[GenerationProviderConfigRecord, ...]:
    with connect(database_url) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM generation_provider_configs
            WHERE (%s OR is_active)
            ORDER BY is_default DESC, provider_name, provider_config_id
            """,
            (include_inactive,),
        ).fetchall()
    return tuple(_provider_config_from_row(dict(row)) for row in rows)


def get_generation_provider_config_by_name(
    database_url: str,
    provider_name: str,
) -> GenerationProviderConfigRecord | None:
    normalized_provider_name = _validate_non_empty(provider_name, "provider_name")
    with connect(database_url) as conn:
        row = conn.execute(
            """
            SELECT *
            FROM generation_provider_configs
            WHERE provider_name = %s
            """,
            (normalized_provider_name,),
        ).fetchone()
    return _provider_config_from_row(dict(row)) if row else None


def upsert_generation_provider_config(
    database_url: str,
    config_input: GenerationProviderConfigInput,
) -> GenerationProviderConfigRecord:
    validated = validate_generation_provider_config_input(config_input)
    with connect(database_url) as conn:
        if validated.is_default:
            conn.execute("""
                UPDATE generation_provider_configs
                SET is_default = false,
                    updated_at = now()
                WHERE is_default
                """)
        row = conn.execute(
            """
            INSERT INTO generation_provider_configs (
                provider_name,
                provider_mode,
                provider_base_url,
                model_id,
                is_default,
                is_active,
                request_timeout_seconds,
                max_tokens,
                temperature,
                top_p,
                runtime_options,
                created_by,
                created_by_user_id
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (provider_name) DO UPDATE
            SET provider_mode = EXCLUDED.provider_mode,
                provider_base_url = EXCLUDED.provider_base_url,
                model_id = EXCLUDED.model_id,
                is_default = EXCLUDED.is_default,
                is_active = EXCLUDED.is_active,
                request_timeout_seconds = EXCLUDED.request_timeout_seconds,
                max_tokens = EXCLUDED.max_tokens,
                temperature = EXCLUDED.temperature,
                top_p = EXCLUDED.top_p,
                runtime_options = EXCLUDED.runtime_options,
                created_by = COALESCE(EXCLUDED.created_by, generation_provider_configs.created_by),
                created_by_user_id = COALESCE(
                    EXCLUDED.created_by_user_id,
                    generation_provider_configs.created_by_user_id
                ),
                updated_at = now()
            RETURNING *
            """,
            (
                validated.provider_name,
                validated.provider_mode,
                validated.provider_base_url,
                validated.model_id,
                validated.is_default,
                validated.is_active,
                validated.request_timeout_seconds,
                validated.max_tokens,
                validated.temperature,
                validated.top_p,
                Json(validated.runtime_options),
                validated.created_by,
                validated.created_by_user_id,
            ),
        ).fetchone()
        conn.commit()
    assert row is not None
    return _provider_config_from_row(dict(row))


def build_dgx_vllm_generation_provider_config_input(
    *,
    provider_name: str = DGX_VLLM_GENERATION_PROVIDER_NAME,
    provider_base_url: str = DGX_VLLM_GENERATION_BASE_URL,
    model_id: str = DGX_VLLM_GENERATION_MODEL_ID,
    api_key_env: str = DGX_VLLM_GENERATION_API_KEY_ENV,
    request_timeout_seconds: int = DGX_VLLM_GENERATION_TIMEOUT_SECONDS,
    max_tokens: int = DGX_VLLM_GENERATION_MAX_TOKENS,
    temperature: float = DGX_VLLM_GENERATION_TEMPERATURE,
    top_p: float = DGX_VLLM_GENERATION_TOP_P,
    is_default: bool = False,
    is_active: bool = True,
    thinking_disabled: bool = True,
    created_by: str | None = "slice_349_seed",
    created_by_user_id: int | None = None,
) -> GenerationProviderConfigInput:
    return validate_generation_provider_config_input(
        GenerationProviderConfigInput(
            provider_name=provider_name,
            provider_mode=GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
            provider_base_url=provider_base_url,
            model_id=model_id,
            is_default=is_default,
            is_active=is_active,
            request_timeout_seconds=request_timeout_seconds,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            runtime_options={
                "contract": "openai_chat_completions",
                "endpoint": "/v1/chat/completions",
                "api_key_env": api_key_env,
                "extra_body": {
                    "chat_template_kwargs": {
                        "enable_thinking": not thinking_disabled,
                    }
                },
                "serving_max_model_len": "200k",
                "smoke_evidence": "docs/dgx_vllm_generation_smoke_result.md",
                "secret_storage": "environment_variable_only",
                "slice": 349,
            },
            created_by=created_by,
            created_by_user_id=created_by_user_id,
        )
    )


def seed_dgx_vllm_generation_provider_config(
    database_url: str,
    *,
    provider_name: str = DGX_VLLM_GENERATION_PROVIDER_NAME,
    provider_base_url: str = DGX_VLLM_GENERATION_BASE_URL,
    model_id: str = DGX_VLLM_GENERATION_MODEL_ID,
    api_key_env: str = DGX_VLLM_GENERATION_API_KEY_ENV,
    request_timeout_seconds: int = DGX_VLLM_GENERATION_TIMEOUT_SECONDS,
    max_tokens: int = DGX_VLLM_GENERATION_MAX_TOKENS,
    temperature: float = DGX_VLLM_GENERATION_TEMPERATURE,
    top_p: float = DGX_VLLM_GENERATION_TOP_P,
    is_default: bool = False,
    is_active: bool = True,
    thinking_disabled: bool = True,
    created_by: str | None = "slice_349_seed",
    created_by_user_id: int | None = None,
) -> GenerationProviderConfigRecord:
    config_input = build_dgx_vllm_generation_provider_config_input(
        provider_name=provider_name,
        provider_base_url=provider_base_url,
        model_id=model_id,
        api_key_env=api_key_env,
        request_timeout_seconds=request_timeout_seconds,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        is_default=is_default,
        is_active=is_active,
        thinking_disabled=thinking_disabled,
        created_by=created_by,
        created_by_user_id=created_by_user_id,
    )
    return upsert_generation_provider_config(database_url, config_input)


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


def _string_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    items: list[str] = []
    for item in value:
        normalized = str(item).strip()
        if normalized:
            items.append(normalized)
    return tuple(items)


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _answer_quality_mapping(run: GenerationRunRecord) -> dict[str, Any]:
    answer_quality = run.response_metadata.get("answer_quality")
    return dict(answer_quality) if isinstance(answer_quality, Mapping) else {}


def _answer_quality_status(run: GenerationRunRecord) -> str:
    answer_quality = _answer_quality_mapping(run)
    status = answer_quality.get("status") or run.guardrail_metadata.get("answer_quality_status")
    if isinstance(status, str) and status.strip():
        normalized = status.strip().lower()
        if normalized in GENERATION_ANSWER_QUALITY_HISTORY_STATUSES:
            return normalized
    return GENERATION_ANSWER_QUALITY_NOT_AVAILABLE


def _history_item_from_run(run: GenerationRunRecord) -> GenerationRunHistoryItem:
    answer_quality = _answer_quality_mapping(run)
    return GenerationRunHistoryItem(
        run=run,
        answer_quality_status=_answer_quality_status(run),
        answer_quality_reason_codes=(
            _string_list(answer_quality.get("reason_codes"))
            or _string_list(run.guardrail_metadata.get("answer_quality_reason_codes"))
        ),
        citation_coverage_percent=_optional_float(answer_quality.get("citation_coverage_percent")),
        expected_citation_count=len(_string_list(answer_quality.get("expected_citation_keys"))),
        cited_citation_count=len(_string_list(answer_quality.get("cited_citation_keys"))),
        missing_citation_count=len(_string_list(answer_quality.get("missing_citation_keys"))),
        unrecognized_citation_count=len(
            _string_list(answer_quality.get("unrecognized_citation_keys"))
        ),
    )


def _history_summary(
    runs: tuple[GenerationRunHistoryItem, ...],
) -> GenerationRunHistorySummary:
    return GenerationRunHistorySummary(
        run_count=len(runs),
        passed_count=sum(1 for item in runs if item.answer_quality_status == "passed"),
        warning_count=sum(1 for item in runs if item.answer_quality_status == "warning"),
        failed_count=sum(1 for item in runs if item.answer_quality_status == "failed"),
        not_evaluated_count=sum(
            1 for item in runs if item.answer_quality_status == "not_evaluated"
        ),
        not_available_count=sum(
            1 for item in runs if item.answer_quality_status == "not_available"
        ),
    )


def list_generation_run_history(
    database_url: str,
    *,
    history_filter: GenerationRunHistoryFilter | None = None,
) -> GenerationRunHistory:
    validated = validate_generation_run_history_filter(
        history_filter or GenerationRunHistoryFilter()
    )
    with connect(database_url) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM generation_runs
            WHERE (
                %s = 'all'
                OR CASE
                    WHEN lower(btrim(COALESCE(
                        response_metadata #>> '{answer_quality,status}',
                        guardrail_metadata ->> 'answer_quality_status',
                        ''
                    ))) = ANY(%s) THEN lower(btrim(COALESCE(
                        response_metadata #>> '{answer_quality,status}',
                        guardrail_metadata ->> 'answer_quality_status',
                        ''
                    )))
                    ELSE 'not_available'
                END = %s
            )
              AND (%s = 'all' OR provider_mode = %s)
              AND (%s = 'all' OR status = %s)
            ORDER BY created_at DESC, generation_run_id DESC
            LIMIT %s
            """,
            (
                validated.answer_quality_status,
                sorted(GENERATION_ANSWER_QUALITY_HISTORY_STATUSES),
                validated.answer_quality_status,
                validated.provider_mode,
                validated.provider_mode,
                validated.run_status,
                validated.run_status,
                validated.limit,
            ),
        ).fetchall()
    runs = tuple(_history_item_from_run(_generation_run_from_row(dict(row))) for row in rows)
    return GenerationRunHistory(
        filters=validated,
        summary=_history_summary(runs),
        runs=runs,
    )


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
    _validate_optional_positive(run_input.generation_template_id, "generation_template_id")
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
                generation_template_id,
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
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s
            )
            RETURNING *
            """,
            (
                validated_search_log_id,
                retrieval_package_key,
                run_input.generation_template_id,
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
