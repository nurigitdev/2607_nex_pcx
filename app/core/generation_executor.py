"""Deterministic generation executors used before remote LLM runtime wiring."""

from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter

from app.core.citation_readiness import (
    CITATION_READINESS_FAILED,
    assess_citation_readiness_package,
)
from app.core.generation_prompts import (
    GenerationPromptPackage,
    build_generation_prompt_package,
)
from app.core.generation_runs import (
    GENERATION_GUARDRAIL_ALLOWED,
    GENERATION_GUARDRAIL_NO_ANSWER,
    GENERATION_PROVIDER_MODE_MOCK,
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
    get_default_generation_provider_config,
)
from app.core.retrieval_confidence import (
    RETRIEVAL_CONFIDENCE_ANSWERABLE,
)
from app.core.retrieval_context import RetrievalContextCandidate, RetrievalContextPackage

MOCK_NO_ANSWER_TEXT = "제공된 문서 근거만으로는 답변할 수 없습니다."
MOCK_FINISH_REASON_COMPLETED = "mock_completed"
MOCK_FINISH_REASON_NO_ANSWER = "guardrail_no_answer"


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
) -> dict[str, object]:
    return {
        "prompt_blocked": prompt_package.blocked,
        "prompt_block_reason": prompt_package.block_reason,
        "retrieval_confidence_status": retrieval_confidence_status,
        "citation_readiness_status": citation_readiness_status,
    }


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


def execute_mock_generation_run(
    database_url: str,
    package: RetrievalContextPackage,
    *,
    created_by: str | None = None,
    created_by_user_id: int | None = None,
) -> GenerationExecutionReport:
    """Persist a deterministic mock generation run from a retrieval context package."""

    provider = get_default_generation_provider_config(database_url)
    if provider is None:
        raise InvalidGenerationRunError("default generation provider config was not found")
    if provider.provider_mode != GENERATION_PROVIDER_MODE_MOCK:
        raise InvalidGenerationRunError("default generation provider is not mock")

    started_at = datetime.now(UTC)
    started_monotonic = perf_counter()
    prompt_package = build_generation_prompt_package(package)
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

    run = create_generation_run(
        database_url,
        GenerationRunInput(
            search_log_id=prompt_package.search_log_id,
            retrieval_package_key=prompt_package.retrieval_package_key,
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
            total_token_count=input_token_count + output_token_count,
            elapsed_ms=elapsed_ms,
            request_metadata=_request_metadata(prompt_package),
            response_metadata={
                "provider_mode": provider.provider_mode,
                "model_id": provider.model_id,
                "deterministic": True,
            },
            guardrail_metadata=_guardrail_metadata(
                prompt_package=prompt_package,
                retrieval_confidence_status=retrieval_status,
                citation_readiness_status=citation_status,
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
        else tuple(
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
    )
    return GenerationExecutionReport(
        provider=provider,
        prompt_package=prompt_package,
        run=run,
        citations=citations,
    )
