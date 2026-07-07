"""Import and export helpers for portable golden question sets."""

from dataclasses import dataclass, field
from typing import Any

from app.core.database import connect
from app.core.golden_questions import (
    GoldenQuestionExpectedTargetInput,
    GoldenQuestionExpectedTargetRecord,
    GoldenQuestionInput,
    GoldenQuestionRecord,
    GoldenQuestionSetInput,
    GoldenQuestionSetRecord,
    create_expected_target_in_connection,
    create_golden_question_in_connection,
    create_golden_question_set_in_connection,
    get_golden_question_set,
    list_expected_targets,
    list_golden_questions,
)

GOLDEN_QUESTION_EXPORT_VERSION = 1


@dataclass(frozen=True)
class GoldenQuestionImportTargetInput:
    chunk_id: int | None = None
    expected_heading_path: tuple[str, ...] = ()
    expectation_type: str = "visible"
    relevance_grade: int = 3
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GoldenQuestionImportQuestionInput:
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
    expected_targets: tuple[GoldenQuestionImportTargetInput, ...] = ()


@dataclass(frozen=True)
class GoldenQuestionImportInput:
    question_set: GoldenQuestionSetInput
    questions: tuple[GoldenQuestionImportQuestionInput, ...] = ()
    version: int = GOLDEN_QUESTION_EXPORT_VERSION


@dataclass(frozen=True)
class GoldenQuestionImportRecord:
    question_set: GoldenQuestionSetRecord
    questions: tuple[GoldenQuestionRecord, ...]
    expected_targets: tuple[GoldenQuestionExpectedTargetRecord, ...]


class InvalidGoldenQuestionExchangeError(ValueError):
    """Raised when import/export input is invalid."""


def _question_set_export_payload(question_set: GoldenQuestionSetRecord) -> dict[str, object]:
    return {
        "set_name": question_set.set_name,
        "description": question_set.description,
        "is_active": question_set.is_active,
        "metadata": question_set.metadata,
        "created_by_user_id": question_set.created_by_user_id,
    }


def _target_export_payload(target: GoldenQuestionExpectedTargetRecord) -> dict[str, object]:
    return {
        "chunk_id": target.chunk_id,
        "expected_heading_path": list(target.expected_heading_path),
        "expectation_type": target.expectation_type,
        "relevance_grade": target.relevance_grade,
        "notes": target.notes,
        "metadata": target.metadata,
    }


def _question_export_payload(
    database_url: str,
    question: GoldenQuestionRecord,
) -> dict[str, object]:
    return {
        "question_text": question.question_text,
        "normalized_question_text": question.normalized_question_text,
        "question_type": question.question_type,
        "actor_user_id": question.actor_user_id,
        "requested_search_scope": question.requested_search_scope,
        "document_group": question.document_group,
        "file_type": question.file_type,
        "chunk_policy_name": question.chunk_policy_name,
        "top_k": question.top_k,
        "metadata": question.metadata,
        "created_by_user_id": question.created_by_user_id,
        "expected_targets": [
            _target_export_payload(target)
            for target in list_expected_targets(database_url, question.question_id)
        ],
    }


def export_golden_question_set(
    database_url: str,
    question_set_id: int,
) -> dict[str, object] | None:
    question_set = get_golden_question_set(database_url, question_set_id)
    if question_set is None:
        return None
    questions = list_golden_questions(database_url, question_set.question_set_id, limit=500)
    return {
        "version": GOLDEN_QUESTION_EXPORT_VERSION,
        "question_set": _question_set_export_payload(question_set),
        "questions": [_question_export_payload(database_url, question) for question in questions],
    }


def _validate_import_input(import_input: GoldenQuestionImportInput) -> GoldenQuestionImportInput:
    if import_input.version != GOLDEN_QUESTION_EXPORT_VERSION:
        raise InvalidGoldenQuestionExchangeError(
            f"Unsupported golden question export version: {import_input.version}"
        )
    return import_input


def import_golden_question_set(
    database_url: str,
    import_input: GoldenQuestionImportInput,
) -> GoldenQuestionImportRecord:
    validated = _validate_import_input(import_input)
    questions: list[GoldenQuestionRecord] = []
    expected_targets: list[GoldenQuestionExpectedTargetRecord] = []
    with connect(database_url) as connection:
        question_set = create_golden_question_set_in_connection(
            connection,
            validated.question_set,
        )
        for question_input in validated.questions:
            question = create_golden_question_in_connection(
                connection,
                GoldenQuestionInput(
                    question_set_id=question_set.question_set_id,
                    question_text=question_input.question_text,
                    normalized_question_text=question_input.normalized_question_text,
                    question_type=question_input.question_type,
                    actor_user_id=question_input.actor_user_id,
                    requested_search_scope=question_input.requested_search_scope,
                    document_group=question_input.document_group,
                    file_type=question_input.file_type,
                    chunk_policy_name=question_input.chunk_policy_name,
                    top_k=question_input.top_k,
                    metadata=question_input.metadata,
                    created_by_user_id=(
                        question_input.created_by_user_id or question_set.created_by_user_id
                    ),
                ),
            )
            questions.append(question)
            for target_input in question_input.expected_targets:
                expected_targets.append(
                    create_expected_target_in_connection(
                        connection,
                        GoldenQuestionExpectedTargetInput(
                            question_id=question.question_id,
                            chunk_id=target_input.chunk_id,
                            expected_heading_path=target_input.expected_heading_path,
                            expectation_type=target_input.expectation_type,
                            relevance_grade=target_input.relevance_grade,
                            notes=target_input.notes,
                            metadata=target_input.metadata,
                        ),
                    )
                )
    return GoldenQuestionImportRecord(
        question_set=question_set,
        questions=tuple(questions),
        expected_targets=tuple(expected_targets),
    )
