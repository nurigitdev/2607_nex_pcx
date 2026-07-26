"""Prompt package builder for grounded generation runs."""

from dataclasses import dataclass
from hashlib import sha256

from app.core.retrieval_confidence import (
    RETRIEVAL_CONFIDENCE_ANSWERABLE,
)
from app.core.retrieval_context import RetrievalContextCandidate, RetrievalContextPackage

DEFAULT_GENERATION_PROMPT_VERSION = "grounded_answer_v1"
DEFAULT_GENERATION_RESPONSE_LANGUAGE = "ko"
GENERATION_BLOCK_NO_INCLUDED_CONTEXT = "no_included_context"
GENERATION_BLOCK_LOW_CONFIDENCE = "retrieval_confidence_blocked"
GENERATION_BLOCK_EMPTY_QUERY = "empty_query"


@dataclass(frozen=True)
class GenerationPromptMessage:
    role: str
    content: str


@dataclass(frozen=True)
class GenerationPromptPackage:
    prompt_version: str
    response_language: str
    query_text: str
    retrieval_package_key: str
    search_log_id: int
    messages: tuple[GenerationPromptMessage, ...]
    citation_keys: tuple[str, ...]
    context_text: str
    prompt_hash: str
    context_hash: str
    blocked: bool = False
    block_reason: str | None = None

    @property
    def openai_messages(self) -> list[dict[str, str]]:
        return [{"role": message.role, "content": message.content} for message in self.messages]


class InvalidGenerationPromptError(ValueError):
    """Raised when a generation prompt package cannot be built."""


def _hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _normalize_language(language: str | None) -> str:
    normalized = (language or DEFAULT_GENERATION_RESPONSE_LANGUAGE).strip().lower()
    if not normalized:
        return DEFAULT_GENERATION_RESPONSE_LANGUAGE
    return normalized


def _query_text(package: RetrievalContextPackage) -> str:
    return package.search_log.search_log.query_text.strip()


def _citation_keys(candidates: tuple[RetrievalContextCandidate, ...]) -> tuple[str, ...]:
    return tuple(
        candidate.citation.citation_key
        for candidate in candidates
        if candidate.citation.citation_key
    )


def _blocked_reason(package: RetrievalContextPackage, query_text: str) -> str | None:
    if not query_text:
        return GENERATION_BLOCK_EMPTY_QUERY
    confidence = package.confidence_assessment
    if confidence is not None and (
        confidence.withhold_generation_context
        or confidence.status != RETRIEVAL_CONFIDENCE_ANSWERABLE
    ):
        return GENERATION_BLOCK_LOW_CONFIDENCE
    if not package.included_candidates or not package.generation_context_text.strip():
        return GENERATION_BLOCK_NO_INCLUDED_CONTEXT
    return None


def _system_instruction(language: str) -> str:
    return "\n".join(
        (
            "You are NeX-PCX grounded generation assistant.",
            "Answer only from the provided retrieval context.",
            "Use citation keys such as [RCP-001] for every factual claim.",
            "If the context is insufficient, say that the answer cannot be determined "
            "from the provided documents.",
            f"Write the answer in language code: {language}.",
        )
    )


def _user_instruction(
    *,
    query_text: str,
    context_text: str,
    citation_keys: tuple[str, ...],
    package: RetrievalContextPackage,
) -> str:
    search_log = package.search_log.search_log
    citation_line = ", ".join(citation_keys) if citation_keys else "(none)"
    return "\n".join(
        (
            f"Question: {query_text}",
            "",
            "Retrieval metadata:",
            f"- search_log_id: {search_log.search_log_id}",
            f"- retrieval_package_key: {package.package_key}",
            f"- requested_search_scope: {search_log.requested_search_scope or '-'}",
            f"- effective_search_scope: {search_log.effective_search_scope or '-'}",
            f"- chunk_policy_name: {search_log.chunk_policy_name or '-'}",
            f"- citations: {citation_line}",
            "",
            "Context:",
            context_text.strip(),
            "",
            "Answer requirements:",
            "- Use only the context above.",
            "- Keep the answer concise.",
            "- Include citation keys inline.",
            "- Do not invent policy, date, amount, role, or source details.",
        )
    )


def _blocked_instruction(
    *,
    query_text: str,
    block_reason: str,
    package: RetrievalContextPackage,
) -> str:
    return "\n".join(
        (
            f"Question: {query_text or '(empty)'}",
            "",
            "Generation is blocked before LLM execution.",
            f"- search_log_id: {package.search_log.search_log.search_log_id}",
            f"- retrieval_package_key: {package.package_key}",
            f"- reason: {block_reason}",
            "",
            "Return a no-answer response without using unsupported context.",
        )
    )


def build_generation_prompt_package(
    package: RetrievalContextPackage,
    *,
    prompt_version: str = DEFAULT_GENERATION_PROMPT_VERSION,
    response_language: str | None = DEFAULT_GENERATION_RESPONSE_LANGUAGE,
) -> GenerationPromptPackage:
    """Build OpenAI-compatible chat messages from a retrieval context package."""

    normalized_version = prompt_version.strip()
    if not normalized_version:
        raise InvalidGenerationPromptError("prompt_version must not be empty")

    language = _normalize_language(response_language)
    query_text = _query_text(package)
    block_reason = _blocked_reason(package, query_text)
    citation_keys = _citation_keys(package.included_candidates)
    context_text = package.generation_context_text.strip()
    system_message = GenerationPromptMessage(
        role="system",
        content=_system_instruction(language),
    )
    user_content = (
        _blocked_instruction(
            query_text=query_text,
            block_reason=block_reason,
            package=package,
        )
        if block_reason
        else _user_instruction(
            query_text=query_text,
            context_text=context_text,
            citation_keys=citation_keys,
            package=package,
        )
    )
    user_message = GenerationPromptMessage(role="user", content=user_content)
    messages = (system_message, user_message)
    rendered_prompt = "\n\n".join(f"{message.role}:\n{message.content}" for message in messages)

    return GenerationPromptPackage(
        prompt_version=normalized_version,
        response_language=language,
        query_text=query_text,
        retrieval_package_key=package.package_key,
        search_log_id=package.search_log.search_log.search_log_id,
        messages=messages,
        citation_keys=citation_keys,
        context_text=context_text if not block_reason else "",
        prompt_hash=_hash_text(f"{normalized_version}\n{language}\n{rendered_prompt}"),
        context_hash=_hash_text(context_text),
        blocked=block_reason is not None,
        block_reason=block_reason,
    )
