"""Deterministic generation executors used before remote LLM runtime wiring."""

from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter

from app.core.citation_readiness import (
    CITATION_READINESS_FAILED,
    assess_citation_readiness_package,
)
from app.core.generation_answer_quality import (
    assess_generation_answer_quality,
    generation_answer_quality_payload,
)
from app.core.generation_prompts import (
    GenerationPromptPackage,
    build_generation_prompt_package,
)
from app.core.generation_provider_metrics import (
    generation_provider_metrics_payload,
    parse_openai_chat_completion_metrics,
)
from app.core.generation_providers import (
    GenerationProvider,
    GenerationProviderRequestError,
    build_generation_provider_from_runtime_config,
    generation_chat_request_from_openai_messages,
    generation_provider_runtime_config_from_record,
)
from app.core.generation_runs import (
    GENERATION_GUARDRAIL_ALLOWED,
    GENERATION_GUARDRAIL_NO_ANSWER,
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
    GENERATION_STATUS_FAILED,
    GENERATION_STATUS_NO_ANSWER,
    GENERATION_STATUS_SUCCEEDED,
    GenerationProviderConfigRecord,
    GenerationRunCitationInput,
    GenerationRunCitationRecord,
    GenerationRunInput,
    GenerationRunRecord,
    InvalidGenerationRunError,
    create_generation_run,
    create_generation_run_citation,
    get_generation_provider_config_for_mode,
)
from app.core.generation_templates import (
    GenerationTemplateRecord,
    get_default_generation_template,
    get_generation_template_by_key,
)
from app.core.retrieval_confidence import (
    RETRIEVAL_CONFIDENCE_ANSWERABLE,
)
from app.core.retrieval_context import RetrievalContextCandidate, RetrievalContextPackage

MOCK_NO_ANSWER_TEXT = "제공된 문서 근거만으로는 답변할 수 없습니다."
MOCK_FINISH_REASON_COMPLETED = "mock_completed"
MOCK_FINISH_REASON_NO_ANSWER = "guardrail_no_answer"
REMOTE_FINISH_REASON_GUARDRAIL_NO_ANSWER = "guardrail_no_answer"


@dataclass(frozen=True)
class GenerationExecutionReport:
    provider: GenerationProviderConfigRecord
    prompt_package: GenerationPromptPackage
    run: GenerationRunRecord
    citations: tuple[GenerationRunCitationRecord, ...]


def _estimate_token_count(text: str) -> int:
    normalized = text.strip()
    if not normalized:
        return 0
    return len(normalized.split())


def _first_answerable_candidate(
    package: RetrievalContextPackage,
) -> RetrievalContextCandidate | None:
    return next(iter(package.included_candidates), None)


def _mock_answer(package: RetrievalContextPackage) -> str:
    candidate = _first_answerable_candidate(package)
    if candidate is None or not candidate.citation.citation_key:
        return MOCK_NO_ANSWER_TEXT
    chunk = next(iter(candidate.chunks), None)
    excerpt = chunk.chunk_preview if chunk is not None else candidate.context_text
    excerpt = " ".join(excerpt.split())
    citation_key = candidate.citation.citation_key
    return f"제공된 문서 근거에 따르면, {excerpt} [{citation_key}]"


def _retrieval_confidence_status(package: RetrievalContextPackage) -> str:
    if package.confidence_assessment is None:
        return RETRIEVAL_CONFIDENCE_ANSWERABLE
    return package.confidence_assessment.status


def _request_metadata(prompt_package: GenerationPromptPackage) -> dict[str, object]:
    return {
        "contract": "openai_chat_completions",
        "prompt_version": prompt_package.prompt_version,
        "prompt_hash": prompt_package.prompt_hash,
        "context_hash": prompt_package.context_hash,
        "response_language": prompt_package.response_language,
        "generation_template_id": prompt_package.generation_template_id,
        "template_key": prompt_package.template_key,
        "template_version": prompt_package.template_version,
        "document_type": prompt_package.document_type,
        "output_format": prompt_package.output_format,
        "generation_template": prompt_package.template_snapshot,
        "messages": prompt_package.openai_messages,
        "citation_keys": list(prompt_package.citation_keys),
        "blocked": prompt_package.blocked,
        "block_reason": prompt_package.block_reason,
    }


def _guardrail_metadata(
    *,
    prompt_package: GenerationPromptPackage,
    retrieval_confidence_status: str,
    citation_readiness_status: str,
    answer_quality: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "prompt_blocked": prompt_package.blocked,
        "prompt_block_reason": prompt_package.block_reason,
        "retrieval_confidence_status": retrieval_confidence_status,
        "citation_readiness_status": citation_readiness_status,
    }
    if answer_quality is not None:
        metadata["answer_quality_status"] = answer_quality["status"]
        metadata["answer_quality_reason_codes"] = answer_quality["reason_codes"]
    return metadata


def _answer_quality_metadata(
    *,
    answer_text: str | None,
    prompt_package: GenerationPromptPackage,
    guardrail_status: str,
    provider_error: bool = False,
) -> dict[str, object]:
    return generation_answer_quality_payload(
        assess_generation_answer_quality(
            answer_text=answer_text,
            expected_citation_keys=prompt_package.citation_keys,
            guardrail_status=guardrail_status,
            provider_error=provider_error,
        )
    )


def _runtime_extra_body(provider: GenerationProviderConfigRecord) -> dict[str, object]:
    extra_body = provider.runtime_options.get("extra_body", {})
    return dict(extra_body) if isinstance(extra_body, dict) else {}


def _resolve_generation_template(
    database_url: str,
    *,
    generation_template_key: str | None,
) -> GenerationTemplateRecord | None:
    normalized_key = generation_template_key.strip() if generation_template_key else ""
    if normalized_key:
        template = get_generation_template_by_key(database_url, normalized_key)
        if template is None:
            raise InvalidGenerationRunError("active generation template was not found")
        return template
    return get_default_generation_template(database_url)


def _create_citations_for_answer(
    database_url: str,
    *,
    run: GenerationRunRecord,
    package: RetrievalContextPackage,
    answer_text: str,
) -> tuple[GenerationRunCitationRecord, ...]:
    return tuple(
        create_generation_run_citation(
            database_url,
            _citation_input(
                run=run,
                candidate=candidate,
                citation_index=index,
                answer_text=answer_text,
            ),
        )
        for index, candidate in enumerate(package.included_candidates, start=1)
    )


def _citation_input(
    *,
    run: GenerationRunRecord,
    candidate: RetrievalContextCandidate,
    citation_index: int,
    answer_text: str,
) -> GenerationRunCitationInput:
    citation = candidate.citation
    citation_key = citation.citation_key or f"RCP-{citation_index:03d}"
    return GenerationRunCitationInput(
        generation_run_id=run.generation_run_id,
        citation_key=citation_key,
        citation_index=citation_index,
        search_log_result_id=candidate.primary_result.search_log_result_id,
        chunk_id=citation.chunk_id,
        document_id=citation.document_id,
        file_id=citation.file_id,
        source_label=citation.source_label,
        source_anchor=citation.source_anchor,
        citation_payload={
            "search_log_result_id": candidate.primary_result.search_log_result_id,
            "chunk_id": citation.chunk_id,
            "document_id": citation.document_id,
            "file_id": citation.file_id,
            "document_title": citation.document_title,
            "original_file_name": citation.original_file_name,
            "chunk_policy_name": citation.chunk_policy_name,
            "source_label": citation.source_label,
        },
        was_cited=f"[{citation_key}]" in answer_text,
    )


def _remote_request_metadata(
    prompt_package: GenerationPromptPackage,
    *,
    provider: GenerationProviderConfigRecord,
) -> dict[str, object]:
    return {
        **_request_metadata(prompt_package),
        "provider_config_id": provider.provider_config_id,
        "provider_name": provider.provider_name,
        "provider_mode": provider.provider_mode,
        "model_id": provider.model_id,
        "max_tokens": provider.max_tokens,
        "temperature": provider.temperature,
        "top_p": provider.top_p,
        "extra_body": _runtime_extra_body(provider),
    }


def _remote_response_metadata(
    *,
    provider: GenerationProviderConfigRecord,
    provider_metrics: object | None,
    response_metadata: dict[str, object] | None = None,
    provider_model_id: str | None = None,
    response_id: str | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "provider_mode": provider.provider_mode,
        "model_id": provider.model_id,
        "remote_base_url": provider.provider_base_url,
        "runtime_options": provider.runtime_options,
    }
    if provider_model_id:
        metadata["provider_model_id"] = provider_model_id
    if response_id:
        metadata["response_id"] = response_id
    if response_metadata:
        metadata.update(response_metadata)
    if provider_metrics is not None:
        metadata["provider_metrics"] = generation_provider_metrics_payload(provider_metrics)
    return metadata


def execute_mock_generation_run(
    database_url: str,
    package: RetrievalContextPackage,
    *,
    generation_template_key: str | None = None,
    created_by: str | None = None,
    created_by_user_id: int | None = None,
) -> GenerationExecutionReport:
    """Persist a deterministic mock generation run from a retrieval context package."""

    provider = get_generation_provider_config_for_mode(database_url, GENERATION_PROVIDER_MODE_MOCK)
    if provider is None:
        raise InvalidGenerationRunError("active mock generation provider config was not found")

    started_at = datetime.now(UTC)
    started_monotonic = perf_counter()
    generation_template = _resolve_generation_template(
        database_url,
        generation_template_key=generation_template_key,
    )
    prompt_package = build_generation_prompt_package(
        package,
        generation_template=generation_template,
    )
    citation_report = assess_citation_readiness_package(package)
    retrieval_status = _retrieval_confidence_status(package)
    citation_status = citation_report.summary.status
    guardrail_blocks = prompt_package.blocked or citation_status == CITATION_READINESS_FAILED
    answer_text = MOCK_NO_ANSWER_TEXT if guardrail_blocks else _mock_answer(package)
    status = GENERATION_STATUS_NO_ANSWER if guardrail_blocks else GENERATION_STATUS_SUCCEEDED
    guardrail_status = (
        GENERATION_GUARDRAIL_NO_ANSWER if guardrail_blocks else GENERATION_GUARDRAIL_ALLOWED
    )
    finish_reason = (
        MOCK_FINISH_REASON_NO_ANSWER if guardrail_blocks else MOCK_FINISH_REASON_COMPLETED
    )
    elapsed_ms = int((perf_counter() - started_monotonic) * 1000)
    input_token_count = sum(
        _estimate_token_count(message.content) for message in prompt_package.messages
    )
    output_token_count = _estimate_token_count(answer_text)
    finished_at = datetime.now(UTC)
    token_total = input_token_count + output_token_count
    provider_metrics = parse_openai_chat_completion_metrics(
        {
            "id": f"mock-generation-{prompt_package.search_log_id}",
            "object": "chat.completion",
            "created": int(finished_at.timestamp()),
            "model": provider.model_id,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {
                        "role": "assistant",
                        "content": answer_text,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": input_token_count,
                "completion_tokens": output_token_count,
                "total_tokens": token_total,
            },
        },
        provider_name=provider.provider_name,
        provider_mode=provider.provider_mode,
        requested_model_id=provider.model_id,
        http_status_code=200,
        elapsed_ms=elapsed_ms,
        provider_elapsed_ms=elapsed_ms,
    )
    answer_quality = _answer_quality_metadata(
        answer_text=answer_text,
        prompt_package=prompt_package,
        guardrail_status=guardrail_status,
    )

    run = create_generation_run(
        database_url,
        GenerationRunInput(
            search_log_id=prompt_package.search_log_id,
            retrieval_package_key=prompt_package.retrieval_package_key,
            generation_template_id=prompt_package.generation_template_id,
            provider_config_id=provider.provider_config_id,
            provider_name=provider.provider_name,
            provider_mode=provider.provider_mode,
            model_id=provider.model_id,
            prompt_version=prompt_package.prompt_version,
            prompt_hash=prompt_package.prompt_hash,
            context_hash=prompt_package.context_hash,
            status=status,
            guardrail_status=guardrail_status,
            retrieval_confidence_status=retrieval_status,
            citation_readiness_status=citation_status,
            query_text=prompt_package.query_text,
            answer_text=answer_text,
            finish_reason=finish_reason,
            input_token_count=input_token_count,
            output_token_count=output_token_count,
            total_token_count=token_total,
            elapsed_ms=elapsed_ms,
            request_metadata=_request_metadata(prompt_package),
            response_metadata={
                "provider_mode": provider.provider_mode,
                "model_id": provider.model_id,
                "deterministic": True,
                "provider_metrics": generation_provider_metrics_payload(provider_metrics),
                "answer_quality": answer_quality,
            },
            guardrail_metadata=_guardrail_metadata(
                prompt_package=prompt_package,
                retrieval_confidence_status=retrieval_status,
                citation_readiness_status=citation_status,
                answer_quality=answer_quality,
            ),
            created_by=created_by,
            created_by_user_id=created_by_user_id,
            started_at=started_at,
            finished_at=finished_at,
        ),
    )
    citations = (
        ()
        if guardrail_blocks
        else _create_citations_for_answer(
            database_url,
            run=run,
            package=package,
            answer_text=answer_text,
        )
    )
    return GenerationExecutionReport(
        provider=provider,
        prompt_package=prompt_package,
        run=run,
        citations=citations,
    )


def execute_remote_generation_run(
    database_url: str,
    package: RetrievalContextPackage,
    *,
    generation_template_key: str | None = None,
    provider_client: GenerationProvider | None = None,
    api_key: str | None = None,
    created_by: str | None = None,
    created_by_user_id: int | None = None,
) -> GenerationExecutionReport:
    """Persist a remote OpenAI-compatible generation run from a retrieval package."""

    provider = get_generation_provider_config_for_mode(
        database_url,
        GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
    )
    if provider is None:
        raise InvalidGenerationRunError(
            "active remote_openai_compatible generation provider config was not found"
        )

    started_at = datetime.now(UTC)
    started_monotonic = perf_counter()
    generation_template = _resolve_generation_template(
        database_url,
        generation_template_key=generation_template_key,
    )
    prompt_package = build_generation_prompt_package(
        package,
        generation_template=generation_template,
    )
    citation_report = assess_citation_readiness_package(package)
    retrieval_status = _retrieval_confidence_status(package)
    citation_status = citation_report.summary.status
    guardrail_blocks = prompt_package.blocked or citation_status == CITATION_READINESS_FAILED
    guardrail_status = (
        GENERATION_GUARDRAIL_NO_ANSWER if guardrail_blocks else GENERATION_GUARDRAIL_ALLOWED
    )

    if guardrail_blocks:
        finished_at = datetime.now(UTC)
        elapsed_ms = int((perf_counter() - started_monotonic) * 1000)
        answer_quality = _answer_quality_metadata(
            answer_text=MOCK_NO_ANSWER_TEXT,
            prompt_package=prompt_package,
            guardrail_status=guardrail_status,
        )
        run = create_generation_run(
            database_url,
            GenerationRunInput(
                search_log_id=prompt_package.search_log_id,
                retrieval_package_key=prompt_package.retrieval_package_key,
                generation_template_id=prompt_package.generation_template_id,
                provider_config_id=provider.provider_config_id,
                provider_name=provider.provider_name,
                provider_mode=provider.provider_mode,
                model_id=provider.model_id,
                prompt_version=prompt_package.prompt_version,
                prompt_hash=prompt_package.prompt_hash,
                context_hash=prompt_package.context_hash,
                status=GENERATION_STATUS_NO_ANSWER,
                guardrail_status=guardrail_status,
                retrieval_confidence_status=retrieval_status,
                citation_readiness_status=citation_status,
                query_text=prompt_package.query_text,
                answer_text=MOCK_NO_ANSWER_TEXT,
                finish_reason=REMOTE_FINISH_REASON_GUARDRAIL_NO_ANSWER,
                elapsed_ms=elapsed_ms,
                request_metadata=_remote_request_metadata(prompt_package, provider=provider),
                response_metadata=_remote_response_metadata(
                    provider=provider,
                    provider_metrics=None,
                    response_metadata={
                        "skipped_provider_call": True,
                        "answer_quality": answer_quality,
                    },
                ),
                guardrail_metadata=_guardrail_metadata(
                    prompt_package=prompt_package,
                    retrieval_confidence_status=retrieval_status,
                    citation_readiness_status=citation_status,
                    answer_quality=answer_quality,
                ),
                created_by=created_by,
                created_by_user_id=created_by_user_id,
                started_at=started_at,
                finished_at=finished_at,
            ),
        )
        return GenerationExecutionReport(
            provider=provider,
            prompt_package=prompt_package,
            run=run,
            citations=(),
        )

    runtime_config = generation_provider_runtime_config_from_record(provider, api_key=api_key)
    owns_provider_client = provider_client is None
    generation_provider = provider_client or build_generation_provider_from_runtime_config(
        runtime_config,
        provider_name=provider.provider_name,
    )
    chat_request = generation_chat_request_from_openai_messages(
        prompt_package.openai_messages,
        model_id=provider.model_id,
        max_tokens=provider.max_tokens,
        temperature=provider.temperature,
        top_p=provider.top_p,
        trace_id=f"generation-run-search-log-{prompt_package.search_log_id}",
        extra_body=_runtime_extra_body(provider),
        runtime_metadata={
            "search_log_id": prompt_package.search_log_id,
            "retrieval_package_key": prompt_package.retrieval_package_key,
            "template_key": prompt_package.template_key,
            "template_version": prompt_package.template_version,
            "provider_config_id": provider.provider_config_id,
        },
    )

    try:
        response = generation_provider.complete(chat_request)
    except GenerationProviderRequestError as exc:
        finished_at = datetime.now(UTC)
        elapsed_ms = (
            exc.metrics.elapsed_ms
            if exc.metrics is not None and exc.metrics.elapsed_ms is not None
            else int((perf_counter() - started_monotonic) * 1000)
        )
        answer_quality = _answer_quality_metadata(
            answer_text=None,
            prompt_package=prompt_package,
            guardrail_status=guardrail_status,
            provider_error=True,
        )
        run = create_generation_run(
            database_url,
            GenerationRunInput(
                search_log_id=prompt_package.search_log_id,
                retrieval_package_key=prompt_package.retrieval_package_key,
                generation_template_id=prompt_package.generation_template_id,
                provider_config_id=provider.provider_config_id,
                provider_name=provider.provider_name,
                provider_mode=provider.provider_mode,
                model_id=provider.model_id,
                prompt_version=prompt_package.prompt_version,
                prompt_hash=prompt_package.prompt_hash,
                context_hash=prompt_package.context_hash,
                status=GENERATION_STATUS_FAILED,
                guardrail_status=guardrail_status,
                retrieval_confidence_status=retrieval_status,
                citation_readiness_status=citation_status,
                query_text=prompt_package.query_text,
                finish_reason=exc.metrics.finish_reason if exc.metrics is not None else None,
                input_token_count=(
                    exc.metrics.input_token_count if exc.metrics is not None else None
                ),
                output_token_count=(
                    exc.metrics.output_token_count if exc.metrics is not None else None
                ),
                total_token_count=(
                    exc.metrics.total_token_count if exc.metrics is not None else None
                ),
                elapsed_ms=elapsed_ms,
                request_metadata=_remote_request_metadata(prompt_package, provider=provider),
                response_metadata=_remote_response_metadata(
                    provider=provider,
                    provider_metrics=exc.metrics,
                    response_metadata={
                        "provider_error": True,
                        "error_payload": exc.payload,
                        "answer_quality": answer_quality,
                    },
                ),
                guardrail_metadata=_guardrail_metadata(
                    prompt_package=prompt_package,
                    retrieval_confidence_status=retrieval_status,
                    citation_readiness_status=citation_status,
                    answer_quality=answer_quality,
                ),
                error_message=str(exc),
                created_by=created_by,
                created_by_user_id=created_by_user_id,
                started_at=started_at,
                finished_at=finished_at,
            ),
        )
        return GenerationExecutionReport(
            provider=provider,
            prompt_package=prompt_package,
            run=run,
            citations=(),
        )
    finally:
        if owns_provider_client and hasattr(generation_provider, "close"):
            generation_provider.close()  # type: ignore[attr-defined]

    finished_at = datetime.now(UTC)
    answer_quality = _answer_quality_metadata(
        answer_text=response.answer_text,
        prompt_package=prompt_package,
        guardrail_status=guardrail_status,
    )
    run = create_generation_run(
        database_url,
        GenerationRunInput(
            search_log_id=prompt_package.search_log_id,
            retrieval_package_key=prompt_package.retrieval_package_key,
            generation_template_id=prompt_package.generation_template_id,
            provider_config_id=provider.provider_config_id,
            provider_name=provider.provider_name,
            provider_mode=provider.provider_mode,
            model_id=provider.model_id,
            prompt_version=prompt_package.prompt_version,
            prompt_hash=prompt_package.prompt_hash,
            context_hash=prompt_package.context_hash,
            status=GENERATION_STATUS_SUCCEEDED,
            guardrail_status=guardrail_status,
            retrieval_confidence_status=retrieval_status,
            citation_readiness_status=citation_status,
            query_text=prompt_package.query_text,
            answer_text=response.answer_text,
            finish_reason=response.finish_reason,
            input_token_count=response.input_token_count,
            output_token_count=response.output_token_count,
            total_token_count=response.total_token_count,
            elapsed_ms=response.elapsed_ms,
            request_metadata=_remote_request_metadata(prompt_package, provider=provider),
            response_metadata=_remote_response_metadata(
                provider=provider,
                provider_metrics=response.provider_metrics,
                response_metadata={
                    **dict(response.response_metadata),
                    "answer_quality": answer_quality,
                },
                provider_model_id=response.provider_model_id,
                response_id=response.response_id,
            ),
            guardrail_metadata=_guardrail_metadata(
                prompt_package=prompt_package,
                retrieval_confidence_status=retrieval_status,
                citation_readiness_status=citation_status,
                answer_quality=answer_quality,
            ),
            created_by=created_by,
            created_by_user_id=created_by_user_id,
            started_at=started_at,
            finished_at=finished_at,
        ),
    )
    citations = _create_citations_for_answer(
        database_url,
        run=run,
        package=package,
        answer_text=response.answer_text,
    )
    return GenerationExecutionReport(
        provider=provider,
        prompt_package=prompt_package,
        run=run,
        citations=citations,
    )
