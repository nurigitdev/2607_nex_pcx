"""Run search experiments from golden question sets."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.core.golden_questions import (
    GoldenQuestionRecord,
    GoldenQuestionSetRecord,
    get_golden_question_set,
    list_golden_questions,
)
from app.core.search_experiment_runner import (
    SearchExperimentExecutionInput,
    SearchExperimentExecutionReport,
    execute_search_experiment,
)


@dataclass(frozen=True)
class GoldenSearchExperimentBatchInput:
    question_set_id: int
    run_name_prefix: str | None = None
    profiles: tuple[str, ...] | None = None
    strategy_name: str = "vector_cosine"
    top_k: int | None = None
    score_threshold: float | None = None
    chunk_policy_name: str | None = None
    runtime_metadata: dict[str, Any] = field(default_factory=dict)
    created_by: str | None = "golden-search-experiment-batch"
    created_by_user_id: int | None = None


@dataclass(frozen=True)
class GoldenSearchExperimentQuestionReport:
    question: GoldenQuestionRecord
    experiment: SearchExperimentExecutionReport


@dataclass(frozen=True)
class GoldenSearchExperimentBatchReport:
    question_set: GoldenQuestionSetRecord
    question_reports: tuple[GoldenSearchExperimentQuestionReport, ...]
    total_elapsed_ms: int

    @property
    def experiment_run_ids_by_question(self) -> dict[int, int]:
        return {
            item.question.question_id: item.experiment.run.experiment_run_id
            for item in self.question_reports
        }


class InvalidGoldenSearchExperimentError(ValueError):
    """Raised when a golden question search experiment batch cannot run."""


def _require_positive_id(value: int | None, field_name: str) -> None:
    if value is None or value <= 0:
        raise InvalidGoldenSearchExperimentError(f"{field_name} must be greater than 0")


def _require_optional_positive_id(value: int | None, field_name: str) -> None:
    if value is not None and value <= 0:
        raise InvalidGoldenSearchExperimentError(f"{field_name} must be greater than 0")


def _validate_nonblank(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise InvalidGoldenSearchExperimentError(f"{field_name} must not be blank")
    return normalized


def _validate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise InvalidGoldenSearchExperimentError("runtime_metadata must be a JSON object")
    return dict(metadata)


def validate_golden_search_experiment_batch_input(
    batch_input: GoldenSearchExperimentBatchInput,
) -> GoldenSearchExperimentBatchInput:
    _require_positive_id(batch_input.question_set_id, "question_set_id")
    _require_optional_positive_id(batch_input.created_by_user_id, "created_by_user_id")
    if batch_input.top_k is not None and batch_input.top_k <= 0:
        raise InvalidGoldenSearchExperimentError("top_k must be greater than 0")
    return GoldenSearchExperimentBatchInput(
        question_set_id=batch_input.question_set_id,
        run_name_prefix=_validate_nonblank(batch_input.run_name_prefix, "run_name_prefix"),
        profiles=batch_input.profiles,
        strategy_name=_validate_nonblank(batch_input.strategy_name, "strategy_name")
        or batch_input.strategy_name,
        top_k=batch_input.top_k,
        score_threshold=batch_input.score_threshold,
        chunk_policy_name=_validate_nonblank(
            batch_input.chunk_policy_name,
            "chunk_policy_name",
        ),
        runtime_metadata=_validate_metadata(batch_input.runtime_metadata),
        created_by=_validate_nonblank(batch_input.created_by, "created_by"),
        created_by_user_id=batch_input.created_by_user_id,
    )


def _default_run_name_prefix(question_set: GoldenQuestionSetRecord) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{question_set.set_name} / search experiment / {timestamp}"


def _question_run_name(prefix: str, question: GoldenQuestionRecord) -> str:
    return f"{prefix} / Q{question.question_id}"


def _question_execution_input(
    *,
    question_set: GoldenQuestionSetRecord,
    question: GoldenQuestionRecord,
    batch_input: GoldenSearchExperimentBatchInput,
    run_name_prefix: str,
) -> SearchExperimentExecutionInput:
    if question.actor_user_id is None:
        raise InvalidGoldenSearchExperimentError(
            f"question_id {question.question_id} has no actor_user_id"
        )
    return SearchExperimentExecutionInput(
        run_name=_question_run_name(run_name_prefix, question),
        query_text=question.question_text,
        actor_user_id=question.actor_user_id,
        requested_search_scope=question.requested_search_scope,
        profiles=batch_input.profiles,
        strategy_name=batch_input.strategy_name,
        top_k=batch_input.top_k or question.top_k,
        score_threshold=batch_input.score_threshold,
        chunk_policy_name=batch_input.chunk_policy_name or question.chunk_policy_name,
        document_group=question.document_group,
        file_type=question.file_type,
        runtime_metadata={
            **batch_input.runtime_metadata,
            "question_set_id": question_set.question_set_id,
            "question_set_name": question_set.set_name,
            "question_id": question.question_id,
            "question_type": question.question_type,
            "golden_question_batch": True,
        },
        created_by=batch_input.created_by,
        created_by_user_id=batch_input.created_by_user_id,
    )


def execute_golden_search_experiment_batch(
    database_url: str,
    batch_input: GoldenSearchExperimentBatchInput,
) -> GoldenSearchExperimentBatchReport | None:
    validated = validate_golden_search_experiment_batch_input(batch_input)
    question_set = get_golden_question_set(database_url, validated.question_set_id)
    if question_set is None:
        return None

    questions = tuple(
        list_golden_questions(
            database_url,
            question_set.question_set_id,
            limit=500,
        )
    )
    if not questions:
        raise InvalidGoldenSearchExperimentError("question_set has no golden questions")

    run_name_prefix = validated.run_name_prefix or _default_run_name_prefix(question_set)
    question_reports: list[GoldenSearchExperimentQuestionReport] = []
    total_elapsed_ms = 0
    for question in questions:
        experiment = execute_search_experiment(
            database_url,
            _question_execution_input(
                question_set=question_set,
                question=question,
                batch_input=validated,
                run_name_prefix=run_name_prefix,
            ),
        )
        total_elapsed_ms += experiment.run.total_elapsed_ms or 0
        question_reports.append(
            GoldenSearchExperimentQuestionReport(
                question=question,
                experiment=experiment,
            )
        )

    return GoldenSearchExperimentBatchReport(
        question_set=question_set,
        question_reports=tuple(question_reports),
        total_elapsed_ms=total_elapsed_ms,
    )
