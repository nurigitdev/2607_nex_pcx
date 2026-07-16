"""Execute golden question sets through the search pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.core.embedding_providers import (
    EmbeddingProviderRuntimeConfig,
    build_embedding_provider_from_runtime_config,
)
from app.core.evaluation_runs import (
    EvaluationRunInput,
    EvaluationRunReport,
    run_golden_evaluation_from_search_logs,
)
from app.core.golden_questions import (
    GoldenQuestionRecord,
    GoldenQuestionSetRecord,
    get_golden_question_set,
    list_golden_questions,
)
from app.core.query_embeddings import QueryEmbeddingProviderBuilder
from app.core.search_compare import SearchCompareInput, run_search_compare


@dataclass(frozen=True)
class GoldenEvaluationExecutionInput:
    question_set_id: int
    profile_name: str
    run_name: str | None = None
    chunk_policy_name: str | None = None
    top_k: int = 5
    runtime_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GoldenEvaluationExecutionReport:
    evaluation: EvaluationRunReport
    question_set: GoldenQuestionSetRecord
    search_log_ids_by_question: dict[int, int]


class InvalidGoldenEvaluationExecutionError(ValueError):
    """Raised when a golden evaluation execution request is invalid."""


def _require_positive_id(value: int | None, field_name: str) -> None:
    if value is None or value <= 0:
        raise InvalidGoldenEvaluationExecutionError(f"{field_name} must be greater than 0")


def _validate_nonblank(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise InvalidGoldenEvaluationExecutionError(f"{field_name} must not be blank")
    return normalized


def _validate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise InvalidGoldenEvaluationExecutionError("runtime_metadata must be a JSON object")
    return dict(metadata)


def validate_golden_evaluation_execution_input(
    execution_input: GoldenEvaluationExecutionInput,
) -> GoldenEvaluationExecutionInput:
    _require_positive_id(execution_input.question_set_id, "question_set_id")
    if execution_input.top_k <= 0:
        raise InvalidGoldenEvaluationExecutionError("top_k must be greater than 0")
    return GoldenEvaluationExecutionInput(
        question_set_id=execution_input.question_set_id,
        profile_name=_validate_nonblank(execution_input.profile_name, "profile_name")
        or execution_input.profile_name,
        run_name=_validate_nonblank(execution_input.run_name, "run_name"),
        chunk_policy_name=_validate_nonblank(
            execution_input.chunk_policy_name,
            "chunk_policy_name",
        ),
        top_k=execution_input.top_k,
        runtime_metadata=_validate_metadata(execution_input.runtime_metadata),
    )


def _default_run_name(question_set: GoldenQuestionSetRecord, profile_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{question_set.set_name} / {profile_name} / {timestamp}"


def _question_search_input(
    question: GoldenQuestionRecord,
    execution_input: GoldenEvaluationExecutionInput,
) -> SearchCompareInput:
    if question.actor_user_id is None:
        raise InvalidGoldenEvaluationExecutionError(
            f"question_id {question.question_id} has no actor_user_id"
        )
    return SearchCompareInput(
        query_text=question.question_text,
        actor_user_id=question.actor_user_id,
        requested_search_scope=question.requested_search_scope,
        top_k=execution_input.top_k,
        profiles=(execution_input.profile_name,),
        chunk_policy_name=execution_input.chunk_policy_name or question.chunk_policy_name,
        document_group=question.document_group,
        file_type=question.file_type,
    )


def execute_golden_evaluation(
    database_url: str,
    execution_input: GoldenEvaluationExecutionInput,
    *,
    fallback_runtime_config: EmbeddingProviderRuntimeConfig | None = None,
    query_embedding_provider_builder: QueryEmbeddingProviderBuilder = (
        build_embedding_provider_from_runtime_config
    ),
) -> GoldenEvaluationExecutionReport | None:
    validated = validate_golden_evaluation_execution_input(execution_input)
    question_set = get_golden_question_set(database_url, validated.question_set_id)
    if question_set is None:
        return None

    questions = list_golden_questions(
        database_url,
        validated.question_set_id,
        limit=500,
    )
    if not questions:
        raise InvalidGoldenEvaluationExecutionError("question_set has no golden questions")

    search_log_ids_by_question: dict[int, int] = {}
    for question in questions:
        search_result = run_search_compare(
            database_url,
            _question_search_input(question, validated),
            fallback_runtime_config=fallback_runtime_config,
            query_embedding_provider_builder=query_embedding_provider_builder,
        )
        search_log_ids_by_question[question.question_id] = search_result.search_log_id

    metadata = {
        **validated.runtime_metadata,
        "executor": "api",
        "search_log_ids_by_question": {
            str(question_id): search_log_id
            for question_id, search_log_id in search_log_ids_by_question.items()
        },
    }
    evaluation = run_golden_evaluation_from_search_logs(
        database_url,
        EvaluationRunInput(
            question_set_id=validated.question_set_id,
            run_name=validated.run_name or _default_run_name(question_set, validated.profile_name),
            profile_name=validated.profile_name,
            chunk_policy_name=validated.chunk_policy_name,
            top_k=validated.top_k,
            runtime_metadata=metadata,
        ),
        search_log_ids_by_question=search_log_ids_by_question,
    )
    return GoldenEvaluationExecutionReport(
        evaluation=evaluation,
        question_set=question_set,
        search_log_ids_by_question=search_log_ids_by_question,
    )
