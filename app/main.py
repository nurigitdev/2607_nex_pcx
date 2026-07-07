"""FastAPI application factory for NeX_PCX."""

import traceback as traceback_module
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.core.admin_logging import list_logs, log_event
from app.core.chunks import (
    ChunkRecord,
    InvalidChunkError,
    list_document_chunks,
)
from app.core.config import Settings, get_settings
from app.core.database import connect
from app.core.document_inventory import (
    DocumentInventoryItem,
    InvalidDocumentInventoryError,
    get_document_inventory_item,
    list_document_inventory,
)
from app.core.embedding_jobs import (
    EmbeddingJobRecord,
    InvalidEmbeddingJobError,
    get_embedding_job,
    list_active_embedding_profiles,
    list_embedding_jobs,
    retry_embedding_job,
)
from app.core.embedding_vectors import (
    EmbeddingVectorRecord,
    InvalidEmbeddingVectorError,
    get_chunk_embedding,
)
from app.core.evaluation_dashboard import (
    EvaluationDashboardRecentRun,
    EvaluationDashboardStatusCount,
    EvaluationDashboardSummary,
    InvalidEvaluationDashboardError,
    get_evaluation_dashboard_summary,
)
from app.core.evaluation_executor import (
    GoldenEvaluationExecutionInput,
    GoldenEvaluationExecutionReport,
    InvalidGoldenEvaluationExecutionError,
    execute_golden_evaluation,
)
from app.core.evaluation_reports import (
    InvalidEvaluationReportError,
    ProfileComparisonRecord,
    get_latest_profile_comparison,
)
from app.core.evaluation_runs import (
    EvaluationResultRecord,
    EvaluationRunRecord,
    InvalidEvaluationRunError,
    get_evaluation_run,
    list_evaluation_results,
    list_evaluation_runs,
)
from app.core.file_metadata import (
    SUPPORTED_FILE_EXTENSIONS,
    InvalidFileMetadataError,
    UnsupportedFileExtensionError,
)
from app.core.file_uploads import InvalidUploadFileNameError, store_upload
from app.core.golden_question_exchange import (
    GoldenQuestionImportInput,
    GoldenQuestionImportQuestionInput,
    GoldenQuestionImportRecord,
    GoldenQuestionImportTargetInput,
    InvalidGoldenQuestionExchangeError,
    export_golden_question_set,
    import_golden_question_set,
)
from app.core.golden_question_promotions import (
    GoldenQuestionPromotionInput,
    GoldenQuestionPromotionRecord,
    InvalidGoldenQuestionPromotionError,
    promote_search_result_to_golden_question,
)
from app.core.golden_questions import (
    GoldenQuestionDetailRecord,
    GoldenQuestionExpectedTargetInput,
    GoldenQuestionExpectedTargetRecord,
    GoldenQuestionInput,
    GoldenQuestionRecord,
    GoldenQuestionSetInput,
    GoldenQuestionSetRecord,
    InvalidGoldenQuestionError,
    create_expected_target,
    create_golden_question,
    create_golden_question_set,
    delete_expected_target,
    delete_golden_question,
    delete_golden_question_set,
    get_expected_target,
    get_golden_question_detail,
    get_golden_question_set,
    list_expected_targets,
    list_golden_question_sets,
    list_golden_questions,
    update_expected_target,
    update_golden_question,
    update_golden_question_set,
)
from app.core.permissions import InvalidPermissionError
from app.core.pipeline_jobs import (
    InvalidPipelineJobError,
    PipelineJobEventRecord,
    PipelineJobListItem,
    PipelineJobRecord,
    get_pipeline_job,
    list_pipeline_job_events,
    list_pipeline_jobs,
    retry_pipeline_job,
)
from app.core.search_compare import (
    InvalidSearchCompareError,
    SearchCompareInput,
    SearchCompareProfileResult,
    SearchCompareResult,
    run_search_compare,
)
from app.core.search_logs import (
    InvalidSearchLogError,
    SearchFeedbackProfileSummaryRecord,
    SearchLogDetailRecord,
    SearchLogListItem,
    SearchLogResultDetailRecord,
    SearchResultFeedbackInput,
    SearchResultFeedbackRecord,
    create_search_result_feedback,
    get_search_log,
    get_search_log_detail,
    get_search_log_result,
    list_search_logs,
    summarize_search_feedback,
)
from app.core.vector_search import InvalidVectorSearchError, VectorSearchResult

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=BASE_DIR / "web" / "templates")
UPLOAD_FILE_FORM = File(...)
DOCUMENT_GROUP_FORM = Form("default")
SECURITY_LEVEL_FORM = Form("internal")
UPLOADED_BY_FORM = Form(None)


class SearchCompareRequest(BaseModel):
    query_text: str
    actor_user_id: int
    requested_search_scope: str = "company"
    top_k: int = Field(default=5, ge=1)
    profiles: list[str] | None = None
    chunk_policy_name: str | None = None
    document_group: str | None = None
    file_type: str | None = None


class SearchFeedbackRequest(BaseModel):
    search_log_result_id: int = Field(ge=1)
    relevance_label: str
    comment: str | None = None


class GoldenQuestionSetRequest(BaseModel):
    set_name: str
    description: str | None = None
    is_active: bool = True
    metadata: dict[str, object] = Field(default_factory=dict)
    created_by_user_id: int | None = Field(default=None, ge=1)


class GoldenQuestionRequest(BaseModel):
    question_set_id: int = Field(ge=1)
    question_text: str
    normalized_question_text: str | None = None
    question_type: str = "single_fact"
    actor_user_id: int | None = Field(default=None, ge=1)
    requested_search_scope: str = "company"
    document_group: str | None = None
    file_type: str | None = None
    chunk_policy_name: str | None = None
    top_k: int = Field(default=5, ge=1)
    metadata: dict[str, object] = Field(default_factory=dict)
    created_by_user_id: int | None = Field(default=None, ge=1)


class ExpectedTargetRequest(BaseModel):
    question_id: int = Field(ge=1)
    chunk_id: int | None = Field(default=None, ge=1)
    expected_heading_path: list[str] = Field(default_factory=list)
    expectation_type: str = "visible"
    relevance_grade: int = Field(default=3, ge=0, le=3)
    notes: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class GoldenQuestionPromotionRequest(BaseModel):
    question_set_id: int = Field(ge=1)
    question_type: str = "single_fact"
    expectation_type: str = "visible"
    relevance_grade: int = Field(default=3, ge=0, le=3)
    notes: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    created_by_user_id: int | None = Field(default=None, ge=1)


class GoldenEvaluationExecuteRequest(BaseModel):
    question_set_id: int = Field(ge=1)
    profile_name: str
    run_name: str | None = None
    chunk_policy_name: str | None = None
    top_k: int = Field(default=5, ge=1)
    runtime_metadata: dict[str, object] = Field(default_factory=dict)


class GoldenQuestionImportTargetRequest(BaseModel):
    chunk_id: int | None = Field(default=None, ge=1)
    expected_heading_path: list[str] = Field(default_factory=list)
    expectation_type: str = "visible"
    relevance_grade: int = Field(default=3, ge=0, le=3)
    notes: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class GoldenQuestionImportQuestionRequest(BaseModel):
    question_text: str
    normalized_question_text: str | None = None
    question_type: str = "single_fact"
    actor_user_id: int | None = Field(default=None, ge=1)
    requested_search_scope: str = "company"
    document_group: str | None = None
    file_type: str | None = None
    chunk_policy_name: str | None = None
    top_k: int = Field(default=5, ge=1)
    metadata: dict[str, object] = Field(default_factory=dict)
    created_by_user_id: int | None = Field(default=None, ge=1)
    expected_targets: list[GoldenQuestionImportTargetRequest] = Field(default_factory=list)


class GoldenQuestionSetImportRequest(BaseModel):
    version: int = 1
    question_set: GoldenQuestionSetRequest
    questions: list[GoldenQuestionImportQuestionRequest] = Field(default_factory=list)


def list_search_actor_options(database_url: str) -> list[dict[str, object]]:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT user_id, login_id, display_name
                FROM app_users
                WHERE is_active
                ORDER BY login_id ASC
                """)
            return [
                {
                    "user_id": int(row["user_id"]),
                    "login_id": str(row["login_id"]),
                    "display_name": str(row["display_name"]),
                }
                for row in cursor.fetchall()
            ]


def pipeline_job_response_payload(
    pipeline_job: PipelineJobRecord | None,
) -> dict[str, object] | None:
    if pipeline_job is None:
        return None
    return {
        "job_id": pipeline_job.job_id,
        "status": pipeline_job.status,
        "stage": pipeline_job.stage,
        "progress_percent": str(pipeline_job.progress_percent),
    }


def _datetime_response(value: object | None) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else None


def pipeline_job_detail_payload(pipeline_job: PipelineJobRecord) -> dict[str, object]:
    return {
        "job_id": pipeline_job.job_id,
        "job_type": pipeline_job.job_type,
        "file_id": pipeline_job.file_id,
        "document_id": pipeline_job.document_id,
        "parent_job_id": pipeline_job.parent_job_id,
        "requested_by_user_id": pipeline_job.requested_by_user_id,
        "status": pipeline_job.status,
        "stage": pipeline_job.stage,
        "priority": pipeline_job.priority,
        "total_units": pipeline_job.total_units,
        "processed_units": pipeline_job.processed_units,
        "progress_percent": str(pipeline_job.progress_percent),
        "current_message": pipeline_job.current_message,
        "attempts": pipeline_job.attempts,
        "max_attempts": pipeline_job.max_attempts,
        "lease_owner": pipeline_job.lease_owner,
        "lease_expires_at": _datetime_response(pipeline_job.lease_expires_at),
        "heartbeat_at": _datetime_response(pipeline_job.heartbeat_at),
        "error_code": pipeline_job.error_code,
        "error_message": pipeline_job.error_message,
        "metadata": pipeline_job.metadata,
        "queued_at": _datetime_response(pipeline_job.queued_at),
        "started_at": _datetime_response(pipeline_job.started_at),
        "finished_at": _datetime_response(pipeline_job.finished_at),
        "updated_at": _datetime_response(pipeline_job.updated_at),
    }


def pipeline_job_list_item_payload(item: PipelineJobListItem) -> dict[str, object]:
    return {
        "job": pipeline_job_detail_payload(item.job),
        "original_file_name": item.original_file_name,
        "document_title": item.document_title,
        "requested_by_login_id": item.requested_by_login_id,
        "requested_by_display_name": item.requested_by_display_name,
    }


def pipeline_job_event_payload(event: PipelineJobEventRecord) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "job_id": event.job_id,
        "event_type": event.event_type,
        "stage": event.stage,
        "status": event.status,
        "message": event.message,
        "event_metadata": event.event_metadata,
        "created_at": _datetime_response(event.created_at),
    }


def embedding_job_payload(job: EmbeddingJobRecord) -> dict[str, object]:
    return {
        "job_id": job.job_id,
        "chunk_id": job.chunk_id,
        "profile_name": job.profile_name,
        "status": job.status,
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "lease_owner": job.lease_owner,
        "lease_expires_at": _datetime_response(job.lease_expires_at),
        "error_code": job.error_code,
        "error_message": job.error_message,
        "last_error_at": _datetime_response(job.last_error_at),
        "runtime_metadata": job.runtime_metadata,
        "created_at": _datetime_response(job.created_at),
        "started_at": _datetime_response(job.started_at),
        "finished_at": _datetime_response(job.finished_at),
        "updated_at": _datetime_response(job.updated_at),
    }


def embedding_vector_payload(vector: EmbeddingVectorRecord | None) -> dict[str, object] | None:
    if vector is None:
        return None
    return {
        "chunk_id": vector.chunk_id,
        "profile_name": vector.profile_name,
        "table_name": vector.table_name,
        "dimension": vector.dimension,
        "storage_type": vector.storage_type,
        "elapsed_ms": vector.elapsed_ms,
        "created_at": _datetime_response(vector.created_at),
    }


def vector_search_result_payload(result: VectorSearchResult) -> dict[str, object]:
    return {
        "profile_name": result.profile_name,
        "rank": result.rank,
        "chunk_id": result.chunk_id,
        "document_id": result.document_id,
        "file_id": result.file_id,
        "distance": result.distance,
        "score": result.score,
        "chunk_preview": result.chunk_preview,
        "content_hash": result.content_hash,
        "chunk_policy_name": result.chunk_policy_name,
        "heading_path": list(result.heading_path),
        "page_no": result.page_no,
        "slide_no": result.slide_no,
        "sheet_name": result.sheet_name,
        "cell_range": result.cell_range,
        "document_title": result.document_title,
        "document_group": result.document_group,
        "original_file_name": result.original_file_name,
        "file_ext": result.file_ext,
        "embedding_elapsed_ms": result.embedding_elapsed_ms,
    }


def search_compare_profile_payload(profile: SearchCompareProfileResult) -> dict[str, object]:
    return {
        "profile_name": profile.profile_name,
        "elapsed_ms": profile.elapsed_ms,
        "results": [
            {
                **vector_search_result_payload(result.vector_result),
                "search_log_result_id": result.search_log_result_id,
            }
            for result in profile.results
        ],
    }


def search_compare_payload(result: SearchCompareResult) -> dict[str, object]:
    return {
        "search_log_id": result.search_log_id,
        "query_text": result.query_text,
        "actor_user_id": result.actor_user_id,
        "requested_search_scope": result.requested_search_scope,
        "effective_search_scope": result.effective_search_scope,
        "permission_filter_metadata": result.permission_filter.metadata,
        "top_k": result.top_k,
        "total_elapsed_ms": result.total_elapsed_ms,
        "profiles": [search_compare_profile_payload(profile) for profile in result.profiles],
    }


def search_feedback_payload(feedback: SearchResultFeedbackRecord) -> dict[str, object]:
    return {
        "feedback_id": feedback.feedback_id,
        "search_log_result_id": feedback.search_log_result_id,
        "relevance_label": feedback.relevance_label,
        "comment": feedback.comment,
        "created_by": feedback.created_by,
        "created_by_user_id": feedback.created_by_user_id,
        "created_at": _datetime_response(feedback.created_at),
    }


def search_log_record_payload(item: SearchLogListItem) -> dict[str, object]:
    search_log = item.search_log
    return {
        "search_log_id": search_log.search_log_id,
        "query_text": search_log.query_text,
        "normalized_query_text": search_log.normalized_query_text,
        "actor_user_id": search_log.actor_user_id,
        "actor_login_id": item.actor_login_id,
        "actor_display_name": item.actor_display_name,
        "requested_search_scope": search_log.requested_search_scope,
        "effective_search_scope": search_log.effective_search_scope,
        "permission_filter_metadata": search_log.permission_filter_metadata,
        "document_group": search_log.document_group,
        "file_type": search_log.file_type,
        "chunk_policy_name": search_log.chunk_policy_name,
        "top_k": search_log.top_k,
        "similarity_metric": search_log.similarity_metric,
        "profiles": list(search_log.profiles),
        "query_runtime_metadata": search_log.query_runtime_metadata,
        "total_elapsed_ms": search_log.total_elapsed_ms,
        "result_count": item.result_count,
        "feedback_count": item.feedback_count,
        "correct_count": item.correct_count,
        "partial_count": item.partial_count,
        "wrong_count": item.wrong_count,
        "duplicate_count": item.duplicate_count,
        "insufficient_context_count": item.insufficient_context_count,
        "created_by": search_log.created_by,
        "created_by_user_id": search_log.created_by_user_id,
        "created_at": _datetime_response(search_log.created_at),
        "latest_feedback_at": _datetime_response(item.latest_feedback_at),
    }


def search_log_result_detail_payload(
    result: SearchLogResultDetailRecord,
) -> dict[str, object]:
    search_result = result.search_log_result
    return {
        "search_log_result_id": search_result.search_log_result_id,
        "search_log_id": search_result.search_log_id,
        "profile_name": search_result.profile_name,
        "rank": search_result.rank,
        "chunk_id": search_result.chunk_id,
        "document_id": result.document_id,
        "file_id": result.file_id,
        "distance": search_result.distance,
        "score": search_result.score,
        "profile_elapsed_ms": search_result.profile_elapsed_ms,
        "chunk_preview": result.chunk_preview,
        "content_hash": result.content_hash,
        "chunk_policy_name": result.chunk_policy_name,
        "heading_path": list(result.heading_path),
        "page_no": result.page_no,
        "slide_no": result.slide_no,
        "sheet_name": result.sheet_name,
        "cell_range": result.cell_range,
        "document_title": result.document_title,
        "document_group": result.document_group,
        "original_file_name": result.original_file_name,
        "file_ext": result.file_ext,
        "created_at": _datetime_response(search_result.created_at),
        "feedback": [search_feedback_payload(feedback) for feedback in result.feedback],
    }


def search_log_detail_payload(detail: SearchLogDetailRecord) -> dict[str, object]:
    list_item = SearchLogListItem(
        search_log=detail.search_log,
        actor_login_id=detail.actor_login_id,
        actor_display_name=detail.actor_display_name,
        result_count=len(detail.results),
        feedback_count=sum(len(result.feedback) for result in detail.results),
        correct_count=sum(
            1
            for result in detail.results
            for feedback in result.feedback
            if feedback.relevance_label == "correct"
        ),
        partial_count=sum(
            1
            for result in detail.results
            for feedback in result.feedback
            if feedback.relevance_label == "partial"
        ),
        wrong_count=sum(
            1
            for result in detail.results
            for feedback in result.feedback
            if feedback.relevance_label == "wrong"
        ),
        duplicate_count=sum(
            1
            for result in detail.results
            for feedback in result.feedback
            if feedback.relevance_label == "duplicate"
        ),
        insufficient_context_count=sum(
            1
            for result in detail.results
            for feedback in result.feedback
            if feedback.relevance_label == "insufficient_context"
        ),
        latest_feedback_at=max(
            (feedback.created_at for result in detail.results for feedback in result.feedback),
            default=None,
        ),
    )
    return {
        "search_log": search_log_record_payload(list_item),
        "results": [search_log_result_detail_payload(result) for result in detail.results],
    }


def search_feedback_profile_summary_payload(
    profile: SearchFeedbackProfileSummaryRecord,
) -> dict[str, object]:
    correct_rate = (
        round(profile.correct_count / profile.feedback_count, 4) if profile.feedback_count else None
    )
    relevant_rate = (
        round(profile.relevant_count / profile.feedback_count, 4)
        if profile.feedback_count
        else None
    )
    return {
        "profile_name": profile.profile_name,
        "feedback_count": profile.feedback_count,
        "search_log_count": profile.search_log_count,
        "result_count": profile.result_count,
        "correct_count": profile.correct_count,
        "partial_count": profile.partial_count,
        "wrong_count": profile.wrong_count,
        "duplicate_count": profile.duplicate_count,
        "insufficient_context_count": profile.insufficient_context_count,
        "relevant_count": profile.relevant_count,
        "correct_rate": correct_rate,
        "relevant_rate": relevant_rate,
        "average_rank": profile.average_rank,
        "average_score": profile.average_score,
        "average_profile_elapsed_ms": profile.average_profile_elapsed_ms,
        "latest_feedback_at": _datetime_response(profile.latest_feedback_at),
    }


def golden_question_set_payload(question_set: GoldenQuestionSetRecord) -> dict[str, object]:
    return {
        "question_set_id": question_set.question_set_id,
        "set_name": question_set.set_name,
        "description": question_set.description,
        "is_active": question_set.is_active,
        "metadata": question_set.metadata,
        "created_by_user_id": question_set.created_by_user_id,
        "created_at": _datetime_response(question_set.created_at),
        "updated_at": _datetime_response(question_set.updated_at),
    }


def golden_question_payload(question: GoldenQuestionRecord) -> dict[str, object]:
    return {
        "question_id": question.question_id,
        "question_set_id": question.question_set_id,
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
        "created_at": _datetime_response(question.created_at),
        "updated_at": _datetime_response(question.updated_at),
    }


def expected_target_payload(target: GoldenQuestionExpectedTargetRecord) -> dict[str, object]:
    return {
        "expected_target_id": target.expected_target_id,
        "question_id": target.question_id,
        "chunk_id": target.chunk_id,
        "expected_heading_path": list(target.expected_heading_path),
        "expectation_type": target.expectation_type,
        "relevance_grade": target.relevance_grade,
        "notes": target.notes,
        "metadata": target.metadata,
        "created_at": _datetime_response(target.created_at),
    }


def golden_question_detail_payload(detail: GoldenQuestionDetailRecord) -> dict[str, object]:
    return {
        "question": golden_question_payload(detail.question),
        "expected_targets": [expected_target_payload(target) for target in detail.expected_targets],
    }


def golden_question_set_input_from_request(
    payload: GoldenQuestionSetRequest,
) -> GoldenQuestionSetInput:
    return GoldenQuestionSetInput(
        set_name=payload.set_name,
        description=payload.description,
        is_active=payload.is_active,
        metadata=dict(payload.metadata),
        created_by_user_id=payload.created_by_user_id,
    )


def golden_question_input_from_request(payload: GoldenQuestionRequest) -> GoldenQuestionInput:
    return GoldenQuestionInput(
        question_set_id=payload.question_set_id,
        question_text=payload.question_text,
        normalized_question_text=payload.normalized_question_text,
        question_type=payload.question_type,
        actor_user_id=payload.actor_user_id,
        requested_search_scope=payload.requested_search_scope,
        document_group=payload.document_group,
        file_type=payload.file_type,
        chunk_policy_name=payload.chunk_policy_name,
        top_k=payload.top_k,
        metadata=dict(payload.metadata),
        created_by_user_id=payload.created_by_user_id,
    )


def expected_target_input_from_request(
    payload: ExpectedTargetRequest,
) -> GoldenQuestionExpectedTargetInput:
    return GoldenQuestionExpectedTargetInput(
        question_id=payload.question_id,
        chunk_id=payload.chunk_id,
        expected_heading_path=tuple(payload.expected_heading_path),
        expectation_type=payload.expectation_type,
        relevance_grade=payload.relevance_grade,
        notes=payload.notes,
        metadata=dict(payload.metadata),
    )


def golden_question_promotion_input_from_request(
    search_log_result_id: int,
    payload: GoldenQuestionPromotionRequest,
) -> GoldenQuestionPromotionInput:
    return GoldenQuestionPromotionInput(
        question_set_id=payload.question_set_id,
        search_log_result_id=search_log_result_id,
        question_type=payload.question_type,
        expectation_type=payload.expectation_type,
        relevance_grade=payload.relevance_grade,
        notes=payload.notes,
        metadata=dict(payload.metadata),
        created_by_user_id=payload.created_by_user_id,
    )


def golden_question_import_input_from_request(
    payload: GoldenQuestionSetImportRequest,
) -> GoldenQuestionImportInput:
    return GoldenQuestionImportInput(
        version=payload.version,
        question_set=golden_question_set_input_from_request(payload.question_set),
        questions=tuple(
            GoldenQuestionImportQuestionInput(
                question_text=question.question_text,
                normalized_question_text=question.normalized_question_text,
                question_type=question.question_type,
                actor_user_id=question.actor_user_id,
                requested_search_scope=question.requested_search_scope,
                document_group=question.document_group,
                file_type=question.file_type,
                chunk_policy_name=question.chunk_policy_name,
                top_k=question.top_k,
                metadata=dict(question.metadata),
                created_by_user_id=question.created_by_user_id,
                expected_targets=tuple(
                    GoldenQuestionImportTargetInput(
                        chunk_id=target.chunk_id,
                        expected_heading_path=tuple(target.expected_heading_path),
                        expectation_type=target.expectation_type,
                        relevance_grade=target.relevance_grade,
                        notes=target.notes,
                        metadata=dict(target.metadata),
                    )
                    for target in question.expected_targets
                ),
            )
            for question in payload.questions
        ),
    )


def golden_question_promotion_payload(
    promotion: GoldenQuestionPromotionRecord,
) -> dict[str, object]:
    source_result = promotion.source_result.search_log_result
    return {
        "question": golden_question_payload(promotion.question),
        "expected_target": expected_target_payload(promotion.expected_target),
        "source": {
            "search_log_id": promotion.source_search_log.search_log_id,
            "search_log_result_id": source_result.search_log_result_id,
            "chunk_id": source_result.chunk_id,
            "profile_name": source_result.profile_name,
            "rank": source_result.rank,
        },
    }


def golden_question_import_payload(imported: GoldenQuestionImportRecord) -> dict[str, object]:
    return {
        "question_set": golden_question_set_payload(imported.question_set),
        "questions": [golden_question_payload(question) for question in imported.questions],
        "expected_targets": [
            expected_target_payload(target) for target in imported.expected_targets
        ],
    }


def evaluation_run_payload(run: EvaluationRunRecord) -> dict[str, object]:
    return {
        "evaluation_run_id": run.evaluation_run_id,
        "question_set_id": run.question_set_id,
        "run_name": run.run_name,
        "profile_name": run.profile_name,
        "chunk_policy_name": run.chunk_policy_name,
        "similarity_metric": run.similarity_metric,
        "top_k": run.top_k,
        "status": run.status,
        "question_count": run.question_count,
        "recall_question_count": run.recall_question_count,
        "ndcg_question_count": run.ndcg_question_count,
        "no_answer_question_count": run.no_answer_question_count,
        "hidden_violation_count": run.hidden_violation_count,
        "mean_recall_at_k": run.mean_recall_at_k,
        "mean_reciprocal_rank": run.mean_reciprocal_rank,
        "mean_ndcg": run.mean_ndcg,
        "no_answer_success_rate": run.no_answer_success_rate,
        "runtime_metadata": run.runtime_metadata,
        "error_message": run.error_message,
        "started_at": _datetime_response(run.started_at),
        "finished_at": _datetime_response(run.finished_at),
        "created_at": _datetime_response(run.created_at),
        "updated_at": _datetime_response(run.updated_at),
    }


def evaluation_result_payload(result: EvaluationResultRecord) -> dict[str, object]:
    return {
        "evaluation_result_id": result.evaluation_result_id,
        "evaluation_run_id": result.evaluation_run_id,
        "question_id": result.question_id,
        "search_log_id": result.search_log_id,
        "top_k": result.top_k,
        "visible_expected_count": result.visible_expected_count,
        "retrieved_count": result.retrieved_count,
        "matched_visible_count": result.matched_visible_count,
        "hidden_violation_count": result.hidden_violation_count,
        "matched_chunk_ids": list(result.matched_chunk_ids),
        "hidden_violation_chunk_ids": list(result.hidden_violation_chunk_ids),
        "recall_at_k": result.recall_at_k,
        "reciprocal_rank": result.reciprocal_rank,
        "dcg": result.dcg,
        "ideal_dcg": result.ideal_dcg,
        "ndcg": result.ndcg,
        "no_answer_success": result.no_answer_success,
        "metadata": result.metadata,
        "created_at": _datetime_response(result.created_at),
    }


def golden_evaluation_execution_input_from_request(
    payload: GoldenEvaluationExecuteRequest,
) -> GoldenEvaluationExecutionInput:
    return GoldenEvaluationExecutionInput(
        question_set_id=payload.question_set_id,
        profile_name=payload.profile_name,
        run_name=payload.run_name,
        chunk_policy_name=payload.chunk_policy_name,
        top_k=payload.top_k,
        runtime_metadata=dict(payload.runtime_metadata),
    )


def golden_evaluation_execution_payload(
    execution: GoldenEvaluationExecutionReport,
) -> dict[str, object]:
    return {
        "run": evaluation_run_payload(execution.evaluation.run),
        "question_set": golden_question_set_payload(execution.question_set),
        "results": [evaluation_result_payload(result) for result in execution.evaluation.results],
        "search_log_ids_by_question": execution.search_log_ids_by_question,
        "summary": {
            "question_count": execution.evaluation.summary.question_count,
            "recall_question_count": execution.evaluation.summary.recall_question_count,
            "ndcg_question_count": execution.evaluation.summary.ndcg_question_count,
            "no_answer_question_count": execution.evaluation.summary.no_answer_question_count,
            "hidden_violation_count": execution.evaluation.summary.hidden_violation_count,
            "mean_recall_at_k": execution.evaluation.summary.mean_recall_at_k,
            "mean_reciprocal_rank": execution.evaluation.summary.mean_reciprocal_rank,
            "mean_ndcg": execution.evaluation.summary.mean_ndcg,
            "no_answer_success_rate": execution.evaluation.summary.no_answer_success_rate,
        },
    }


def profile_comparison_payload(comparison: ProfileComparisonRecord) -> dict[str, object]:
    return {
        "evaluation_run_id": comparison.evaluation_run_id,
        "question_set_id": comparison.question_set_id,
        "run_name": comparison.run_name,
        "profile_name": comparison.profile_name,
        "chunk_policy_name": comparison.chunk_policy_name,
        "top_k": comparison.top_k,
        "question_count": comparison.question_count,
        "recall_question_count": comparison.recall_question_count,
        "ndcg_question_count": comparison.ndcg_question_count,
        "no_answer_question_count": comparison.no_answer_question_count,
        "hidden_violation_count": comparison.hidden_violation_count,
        "mean_recall_at_k": comparison.mean_recall_at_k,
        "mean_reciprocal_rank": comparison.mean_reciprocal_rank,
        "mean_ndcg": comparison.mean_ndcg,
        "no_answer_success_rate": comparison.no_answer_success_rate,
        "finished_at": _datetime_response(comparison.finished_at),
        "created_at": _datetime_response(comparison.created_at),
    }


def evaluation_dashboard_status_count_payload(
    status_count: EvaluationDashboardStatusCount,
) -> dict[str, object]:
    return {
        "status": status_count.status,
        "count": status_count.count,
    }


def evaluation_dashboard_recent_run_payload(
    recent_run: EvaluationDashboardRecentRun,
) -> dict[str, object]:
    return {
        "evaluation_run_id": recent_run.evaluation_run_id,
        "question_set_id": recent_run.question_set_id,
        "question_set_name": recent_run.question_set_name,
        "run_name": recent_run.run_name,
        "profile_name": recent_run.profile_name,
        "status": recent_run.status,
        "question_count": recent_run.question_count,
        "hidden_violation_count": recent_run.hidden_violation_count,
        "mean_recall_at_k": recent_run.mean_recall_at_k,
        "mean_ndcg": recent_run.mean_ndcg,
        "no_answer_success_rate": recent_run.no_answer_success_rate,
        "created_at": _datetime_response(recent_run.created_at),
        "finished_at": _datetime_response(recent_run.finished_at),
    }


def evaluation_dashboard_summary_payload(
    summary: EvaluationDashboardSummary,
) -> dict[str, object]:
    return {
        "question_set_count": summary.question_set_count,
        "active_question_set_count": summary.active_question_set_count,
        "question_count": summary.question_count,
        "expected_target_count": summary.expected_target_count,
        "evaluation_run_count": summary.evaluation_run_count,
        "status_counts": [
            evaluation_dashboard_status_count_payload(status_count)
            for status_count in summary.status_counts
        ],
        "recent_runs": [
            evaluation_dashboard_recent_run_payload(recent_run)
            for recent_run in summary.recent_runs
        ],
    }


def document_inventory_item_payload(item: DocumentInventoryItem) -> dict[str, object]:
    return {
        "document_id": item.document_id,
        "file_id": item.file_id,
        "document_title": item.document_title,
        "original_file_name": item.original_file_name,
        "file_ext": item.file_ext,
        "mime_type": item.mime_type,
        "file_size_bytes": item.file_size_bytes,
        "document_group": item.document_group,
        "security_level": item.security_level,
        "document_status": item.document_status,
        "parse_status": item.parse_status,
        "owner_user_id": item.owner_user_id,
        "owner_login_id": item.owner_login_id,
        "owner_display_name": item.owner_display_name,
        "owner_org_unit_id": item.owner_org_unit_id,
        "owner_org_unit_name": item.owner_org_unit_name,
        "access_scope": item.access_scope,
        "uploaded_by": item.uploaded_by,
        "uploaded_by_user_id": item.uploaded_by_user_id,
        "uploaded_by_login_id": item.uploaded_by_login_id,
        "uploaded_by_display_name": item.uploaded_by_display_name,
        "chunk_count": item.chunk_count,
        "total_token_count": item.total_token_count,
        "total_char_count": item.total_char_count,
        "latest_pipeline_job_id": item.latest_pipeline_job_id,
        "latest_pipeline_status": item.latest_pipeline_status,
        "latest_pipeline_stage": item.latest_pipeline_stage,
        "latest_pipeline_progress_percent": (
            str(item.latest_pipeline_progress_percent)
            if item.latest_pipeline_progress_percent is not None
            else None
        ),
        "uploaded_at": _datetime_response(item.uploaded_at),
        "updated_at": _datetime_response(item.updated_at),
    }


def chunk_payload(chunk: ChunkRecord) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "chunk_seq": chunk.chunk_seq,
        "chunk_text": chunk.chunk_text,
        "content_hash": chunk.content_hash,
        "chunk_policy_name": chunk.chunk_policy_name,
        "parser_name": chunk.parser_name,
        "parser_version": chunk.parser_version,
        "heading_path": list(chunk.heading_path),
        "page_no": chunk.page_no,
        "slide_no": chunk.slide_no,
        "sheet_name": chunk.sheet_name,
        "cell_range": chunk.cell_range,
        "token_count": chunk.token_count,
        "char_count": chunk.char_count,
        "prev_chunk_id": chunk.prev_chunk_id,
        "next_chunk_id": chunk.next_chunk_id,
        "metadata": chunk.metadata,
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version)
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

    def template_context(request: Request, **context: object) -> dict[str, object]:
        return {
            "request": request,
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "environment": settings.environment,
            **context,
        }

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        if settings.database_url:
            try:
                log_event(
                    settings.database_url,
                    level="ERROR",
                    event_type="unhandled_exception",
                    source="fastapi",
                    message=str(exc) or exc.__class__.__name__,
                    traceback=traceback_module.format_exc(),
                    request_path=request.url.path,
                )
            except Exception:
                pass
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {
            "status": "ok",
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
        }

    @app.get("/api/dashboard/evaluations")
    def api_get_evaluation_dashboard(recent_limit: int = 5) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            summary = get_evaluation_dashboard_summary(
                settings.database_url,
                recent_limit=recent_limit,
            )
        except InvalidEvaluationDashboardError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content={"evaluations": evaluation_dashboard_summary_payload(summary)})

    @app.get("/api/documents")
    def api_list_documents(
        parse_status: str | None = None,
        document_group: str | None = None,
        limit: int = 100,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            documents = list_document_inventory(
                settings.database_url,
                parse_status=parse_status,
                document_group=document_group,
                limit=limit,
            )
        except InvalidDocumentInventoryError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "documents": [document_inventory_item_payload(document) for document in documents],
            },
        )

    @app.get("/api/documents/{document_id}")
    def api_get_document_detail(
        document_id: int,
        chunk_policy_name: str | None = None,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            document = get_document_inventory_item(settings.database_url, document_id)
            if document is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Document not found.",
                )
            chunks = list_document_chunks(
                settings.database_url,
                document_id,
                chunk_policy_name=chunk_policy_name,
            )
        except InvalidDocumentInventoryError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except InvalidChunkError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "document": document_inventory_item_payload(document),
                "chunks": [chunk_payload(chunk) for chunk in chunks],
            },
        )

    def upload_template_context(
        request: Request,
        *,
        result: dict[str, object] | None = None,
        duplicate: bool = False,
        error_message: str | None = None,
        form_values: dict[str, str] | None = None,
    ) -> dict[str, object]:
        return template_context(
            request,
            database_configured=bool(settings.database_url),
            supported_file_extensions=sorted(SUPPORTED_FILE_EXTENSIONS),
            result=result,
            duplicate=duplicate,
            error_message=error_message,
            form_values=form_values
            or {
                "document_group": "default",
                "security_level": "internal",
                "uploaded_by": "",
            },
        )

    @app.post("/api/files")
    async def api_upload_file(
        file: UploadFile = UPLOAD_FILE_FORM,
        document_group: str = DOCUMENT_GROUP_FORM,
        security_level: str = SECURITY_LEVEL_FORM,
        uploaded_by: str | None = UPLOADED_BY_FORM,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            result = store_upload(
                database_url=settings.database_url,
                upload_stream=file.file,
                original_file_name=file.filename,
                storage_dir=settings.upload_storage_dir,
                mime_type=file.content_type,
                document_group=document_group.strip() or "default",
                security_level=security_level.strip() or "internal",
                uploaded_by=uploaded_by.strip() if uploaded_by and uploaded_by.strip() else None,
            )
        except UnsupportedFileExtensionError as exc:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=str(exc),
            ) from exc
        except (InvalidUploadFileNameError, InvalidFileMetadataError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            status_code=status.HTTP_200_OK if result.duplicate else status.HTTP_201_CREATED,
            content={
                "duplicate": result.duplicate,
                "file": asdict(result.file),
                "pipeline_job_id": (
                    result.pipeline_job.job_id if result.pipeline_job is not None else None
                ),
                "pipeline_job": pipeline_job_response_payload(result.pipeline_job),
            },
        )

    @app.get("/api/pipeline/jobs")
    def api_list_pipeline_jobs(
        status_filter: str | None = Query(default=None, alias="status"),
        limit: int = 100,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            jobs = list_pipeline_jobs(
                settings.database_url,
                status=status_filter,
                limit=limit,
            )
        except InvalidPipelineJobError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content={"jobs": [pipeline_job_list_item_payload(job) for job in jobs]})

    @app.get("/api/pipeline/jobs/{job_id}")
    def api_get_pipeline_job(job_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            job = get_pipeline_job(settings.database_url, job_id)
            if job is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Pipeline job not found.",
                )
            events = list_pipeline_job_events(settings.database_url, job_id)
        except InvalidPipelineJobError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "job": pipeline_job_detail_payload(job),
                "events": [pipeline_job_event_payload(event) for event in events],
            },
        )

    @app.post("/api/pipeline/jobs/{job_id}/retry")
    def api_retry_pipeline_job(job_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            existing = get_pipeline_job(settings.database_url, job_id)
            if existing is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Pipeline job not found.",
                )
            retried = retry_pipeline_job(
                settings.database_url,
                job_id,
                message="Manual retry requested",
            )
        except InvalidPipelineJobError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        if retried is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Pipeline job is not retryable.",
            )
        return JSONResponse(content={"job": pipeline_job_detail_payload(retried)})

    @app.get("/api/embedding/jobs")
    def api_list_embedding_jobs(
        status_filter: str | None = Query(default=None, alias="status"),
        profile_name: str | None = None,
        limit: int = 100,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            jobs = list_embedding_jobs(
                settings.database_url,
                status=status_filter,
                profile_name=profile_name,
                limit=limit,
            )
        except InvalidEmbeddingJobError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content={"jobs": [embedding_job_payload(job) for job in jobs]})

    @app.get("/api/embedding/jobs/{job_id}")
    def api_get_embedding_job(job_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            job = get_embedding_job(settings.database_url, job_id)
            if job is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Embedding job not found.",
                )
            vector = get_chunk_embedding(
                settings.database_url,
                profile_name=job.profile_name,
                chunk_id=job.chunk_id,
            )
        except InvalidEmbeddingJobError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except InvalidEmbeddingVectorError:
            vector = None

        return JSONResponse(
            content={
                "job": embedding_job_payload(job),
                "embedding": embedding_vector_payload(vector),
            },
        )

    @app.post("/api/embedding/jobs/{job_id}/retry")
    def api_retry_embedding_job(job_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            existing = get_embedding_job(settings.database_url, job_id)
            if existing is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Embedding job not found.",
                )
            retried = retry_embedding_job(settings.database_url, job_id)
        except InvalidEmbeddingJobError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        if retried is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Embedding job is not retryable.",
            )
        return JSONResponse(content={"job": embedding_job_payload(retried)})

    @app.post("/api/search/compare")
    def api_search_compare(payload: SearchCompareRequest) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            result = run_search_compare(
                settings.database_url,
                SearchCompareInput(
                    query_text=payload.query_text,
                    actor_user_id=payload.actor_user_id,
                    requested_search_scope=payload.requested_search_scope,
                    top_k=payload.top_k,
                    profiles=tuple(payload.profiles) if payload.profiles is not None else None,
                    chunk_policy_name=payload.chunk_policy_name,
                    document_group=payload.document_group,
                    file_type=payload.file_type,
                ),
            )
        except (
            InvalidSearchCompareError,
            InvalidPermissionError,
            InvalidVectorSearchError,
            InvalidSearchLogError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content=search_compare_payload(result))

    @app.get("/api/search/logs")
    def api_list_search_logs(
        actor_user_id: int | None = None,
        requested_search_scope: str | None = None,
        document_group: str | None = None,
        limit: int = 50,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            logs = list_search_logs(
                settings.database_url,
                actor_user_id=actor_user_id,
                requested_search_scope=requested_search_scope,
                document_group=document_group,
                limit=limit,
            )
        except InvalidSearchLogError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content={"logs": [search_log_record_payload(log) for log in logs]})

    @app.get("/api/search/logs/{search_log_id}")
    def api_get_search_log_detail(search_log_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            detail = get_search_log_detail(settings.database_url, search_log_id)
        except InvalidSearchLogError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if detail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Search log not found.",
            )

        return JSONResponse(content=search_log_detail_payload(detail))

    @app.post("/api/search/feedback")
    def api_search_feedback(payload: SearchFeedbackRequest) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            search_result = get_search_log_result(
                settings.database_url,
                payload.search_log_result_id,
            )
            if search_result is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Search result not found.",
                )
            search_log = get_search_log(settings.database_url, search_result.search_log_id)
            feedback = create_search_result_feedback(
                settings.database_url,
                SearchResultFeedbackInput(
                    search_log_result_id=payload.search_log_result_id,
                    relevance_label=payload.relevance_label,
                    comment=payload.comment,
                    created_by="search-compare-ui",
                    created_by_user_id=search_log.actor_user_id if search_log else None,
                ),
            )
        except InvalidSearchLogError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"feedback": search_feedback_payload(feedback)},
        )

    @app.get("/api/search/feedback/summary")
    def api_search_feedback_summary(document_group: str | None = None) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            summary = summarize_search_feedback(
                settings.database_url,
                document_group=document_group,
            )
        except InvalidSearchLogError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "feedback_count": summary.feedback_count,
                "search_log_count": summary.search_log_count,
                "result_count": summary.result_count,
                "latest_feedback_at": _datetime_response(summary.latest_feedback_at),
                "profiles": [
                    search_feedback_profile_summary_payload(profile) for profile in summary.profiles
                ],
            },
        )

    @app.post("/api/search/results/{search_log_result_id}/promote-golden-question")
    def api_promote_search_result_to_golden_question(
        search_log_result_id: int,
        payload: GoldenQuestionPromotionRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            promotion = promote_search_result_to_golden_question(
                settings.database_url,
                golden_question_promotion_input_from_request(search_log_result_id, payload),
            )
        except (
            InvalidGoldenQuestionPromotionError,
            InvalidGoldenQuestionError,
            InvalidSearchLogError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if promotion is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Search result not found.",
            )

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"promotion": golden_question_promotion_payload(promotion)},
        )

    @app.get("/api/evaluations/question-sets")
    def api_list_golden_question_sets(
        active_only: bool = True,
        limit: int = 100,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            question_sets = list_golden_question_sets(
                settings.database_url,
                active_only=active_only,
                limit=limit,
            )
        except InvalidGoldenQuestionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "question_sets": [
                    golden_question_set_payload(question_set) for question_set in question_sets
                ],
            },
        )

    @app.post("/api/evaluations/question-sets")
    def api_create_golden_question_set(payload: GoldenQuestionSetRequest) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            question_set = create_golden_question_set(
                settings.database_url,
                golden_question_set_input_from_request(payload),
            )
        except InvalidGoldenQuestionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"question_set": golden_question_set_payload(question_set)},
        )

    @app.post("/api/evaluations/question-sets/import")
    def api_import_golden_question_set(payload: GoldenQuestionSetImportRequest) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            imported = import_golden_question_set(
                settings.database_url,
                golden_question_import_input_from_request(payload),
            )
        except (InvalidGoldenQuestionExchangeError, InvalidGoldenQuestionError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"imported": golden_question_import_payload(imported)},
        )

    @app.get("/api/evaluations/question-sets/{question_set_id}/export")
    def api_export_golden_question_set(question_set_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            exported = export_golden_question_set(settings.database_url, question_set_id)
        except InvalidGoldenQuestionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if exported is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Golden question set not found.",
            )

        return JSONResponse(content=exported)

    @app.get("/api/evaluations/question-sets/{question_set_id}")
    def api_get_golden_question_set(question_set_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            question_set = get_golden_question_set(settings.database_url, question_set_id)
        except InvalidGoldenQuestionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if question_set is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Golden question set not found.",
            )

        return JSONResponse(content={"question_set": golden_question_set_payload(question_set)})

    @app.put("/api/evaluations/question-sets/{question_set_id}")
    def api_update_golden_question_set(
        question_set_id: int,
        payload: GoldenQuestionSetRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            question_set = update_golden_question_set(
                settings.database_url,
                question_set_id,
                golden_question_set_input_from_request(payload),
            )
        except InvalidGoldenQuestionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if question_set is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Golden question set not found.",
            )

        return JSONResponse(content={"question_set": golden_question_set_payload(question_set)})

    @app.delete("/api/evaluations/question-sets/{question_set_id}")
    def api_delete_golden_question_set(question_set_id: int) -> Response:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            deleted = delete_golden_question_set(settings.database_url, question_set_id)
        except InvalidGoldenQuestionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Golden question set not found.",
            )

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/api/evaluations/question-sets/{question_set_id}/questions")
    def api_list_golden_questions(
        question_set_id: int,
        actor_user_id: int | None = None,
        requested_search_scope: str | None = None,
        limit: int = 100,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            questions = list_golden_questions(
                settings.database_url,
                question_set_id,
                actor_user_id=actor_user_id,
                requested_search_scope=requested_search_scope,
                limit=limit,
            )
        except InvalidGoldenQuestionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={"questions": [golden_question_payload(question) for question in questions]},
        )

    @app.post("/api/evaluations/questions")
    def api_create_golden_question(payload: GoldenQuestionRequest) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            question = create_golden_question(
                settings.database_url,
                golden_question_input_from_request(payload),
            )
        except InvalidGoldenQuestionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"question": golden_question_payload(question)},
        )

    @app.get("/api/evaluations/questions/{question_id}")
    def api_get_golden_question(question_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            detail = get_golden_question_detail(settings.database_url, question_id)
        except InvalidGoldenQuestionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if detail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Golden question not found.",
            )

        return JSONResponse(content=golden_question_detail_payload(detail))

    @app.put("/api/evaluations/questions/{question_id}")
    def api_update_golden_question(
        question_id: int,
        payload: GoldenQuestionRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            question = update_golden_question(
                settings.database_url,
                question_id,
                golden_question_input_from_request(payload),
            )
        except InvalidGoldenQuestionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if question is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Golden question not found.",
            )

        return JSONResponse(content={"question": golden_question_payload(question)})

    @app.delete("/api/evaluations/questions/{question_id}")
    def api_delete_golden_question(question_id: int) -> Response:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            deleted = delete_golden_question(settings.database_url, question_id)
        except InvalidGoldenQuestionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Golden question not found.",
            )

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/api/evaluations/questions/{question_id}/expected-targets")
    def api_list_expected_targets(question_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            targets = list_expected_targets(settings.database_url, question_id)
        except InvalidGoldenQuestionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={"expected_targets": [expected_target_payload(target) for target in targets]},
        )

    @app.post("/api/evaluations/expected-targets")
    def api_create_expected_target(payload: ExpectedTargetRequest) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            target = create_expected_target(
                settings.database_url,
                expected_target_input_from_request(payload),
            )
        except InvalidGoldenQuestionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"expected_target": expected_target_payload(target)},
        )

    @app.get("/api/evaluations/expected-targets/{expected_target_id}")
    def api_get_expected_target(expected_target_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            target = get_expected_target(settings.database_url, expected_target_id)
        except InvalidGoldenQuestionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if target is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Expected target not found.",
            )

        return JSONResponse(content={"expected_target": expected_target_payload(target)})

    @app.put("/api/evaluations/expected-targets/{expected_target_id}")
    def api_update_expected_target(
        expected_target_id: int,
        payload: ExpectedTargetRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            target = update_expected_target(
                settings.database_url,
                expected_target_id,
                expected_target_input_from_request(payload),
            )
        except InvalidGoldenQuestionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if target is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Expected target not found.",
            )

        return JSONResponse(content={"expected_target": expected_target_payload(target)})

    @app.delete("/api/evaluations/expected-targets/{expected_target_id}")
    def api_delete_expected_target(expected_target_id: int) -> Response:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            deleted = delete_expected_target(settings.database_url, expected_target_id)
        except InvalidGoldenQuestionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Expected target not found.",
            )

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/api/evaluations/runs")
    def api_list_evaluation_runs(
        question_set_id: int | None = None,
        profile_name: str | None = None,
        status_filter: str | None = Query(default=None, alias="status"),
        limit: int = 100,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            runs = list_evaluation_runs(
                settings.database_url,
                question_set_id=question_set_id,
                profile_name=profile_name,
                status=status_filter,
                limit=limit,
            )
        except InvalidEvaluationRunError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content={"runs": [evaluation_run_payload(run) for run in runs]})

    @app.post("/api/evaluations/runs/execute")
    def api_execute_golden_evaluation(payload: GoldenEvaluationExecuteRequest) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            execution = execute_golden_evaluation(
                settings.database_url,
                golden_evaluation_execution_input_from_request(payload),
            )
        except (
            InvalidGoldenEvaluationExecutionError,
            InvalidEvaluationRunError,
            InvalidGoldenQuestionError,
            InvalidSearchCompareError,
            InvalidPermissionError,
            InvalidVectorSearchError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if execution is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Golden question set not found.",
            )

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"execution": golden_evaluation_execution_payload(execution)},
        )

    @app.get("/api/evaluations/profile-comparison")
    def api_get_profile_comparison(question_set_id: int, limit: int = 20) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            question_set = get_golden_question_set(settings.database_url, question_set_id)
            if question_set is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Golden question set not found.",
                )
            profile_comparison = get_latest_profile_comparison(
                settings.database_url,
                question_set_id,
                limit=limit,
            )
        except InvalidEvaluationReportError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except InvalidGoldenQuestionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "question_set": golden_question_set_payload(question_set),
                "profiles": [
                    profile_comparison_payload(comparison) for comparison in profile_comparison
                ],
            },
        )

    @app.get("/api/evaluations/runs/{evaluation_run_id}")
    def api_get_evaluation_run_detail(evaluation_run_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            run = get_evaluation_run(settings.database_url, evaluation_run_id)
            if run is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Evaluation run not found.",
                )
            question_set = get_golden_question_set(settings.database_url, run.question_set_id)
            results = list_evaluation_results(settings.database_url, evaluation_run_id)
        except InvalidEvaluationRunError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except InvalidGoldenQuestionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "run": evaluation_run_payload(run),
                "question_set": (
                    golden_question_set_payload(question_set) if question_set is not None else None
                ),
                "results": [evaluation_result_payload(result) for result in results],
            },
        )

    @app.get("/files/upload", response_class=HTMLResponse)
    def upload_file_page(request: Request) -> HTMLResponse:
        error_message = None
        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."

        return TEMPLATES.TemplateResponse(
            request,
            "file_upload.html",
            upload_template_context(request, error_message=error_message),
        )

    @app.post("/files/upload", response_class=HTMLResponse)
    async def submit_upload_file(
        request: Request,
        file: UploadFile = UPLOAD_FILE_FORM,
        document_group: str = DOCUMENT_GROUP_FORM,
        security_level: str = SECURITY_LEVEL_FORM,
        uploaded_by: str | None = UPLOADED_BY_FORM,
    ) -> HTMLResponse:
        form_values = {
            "document_group": document_group.strip() or "default",
            "security_level": security_level.strip() or "internal",
            "uploaded_by": uploaded_by.strip() if uploaded_by and uploaded_by.strip() else "",
        }
        result_payload = None
        duplicate = False
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                result = store_upload(
                    database_url=settings.database_url,
                    upload_stream=file.file,
                    original_file_name=file.filename,
                    storage_dir=settings.upload_storage_dir,
                    mime_type=file.content_type,
                    document_group=form_values["document_group"],
                    security_level=form_values["security_level"],
                    uploaded_by=form_values["uploaded_by"] or None,
                )
                result_payload = asdict(result.file)
                result_payload["pipeline_job"] = pipeline_job_response_payload(
                    result.pipeline_job,
                )
                result_payload["pipeline_job_id"] = (
                    result.pipeline_job.job_id if result.pipeline_job is not None else None
                )
                duplicate = result.duplicate
            except (
                UnsupportedFileExtensionError,
                InvalidUploadFileNameError,
                InvalidFileMetadataError,
            ) as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "file_upload.html",
            upload_template_context(
                request,
                result=result_payload,
                duplicate=duplicate,
                error_message=error_message,
                form_values=form_values,
            ),
        )

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        evaluation_dashboard: EvaluationDashboardSummary | None = None
        error_message = None
        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                evaluation_dashboard = get_evaluation_dashboard_summary(
                    settings.database_url,
                    recent_limit=5,
                )
            except InvalidEvaluationDashboardError as exc:
                error_message = str(exc)
            except Exception as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "dashboard.html",
            template_context(
                request,
                database_configured=bool(settings.database_url),
                evaluation_dashboard=evaluation_dashboard,
                error_message=error_message,
            ),
        )

    @app.get("/documents", response_class=HTMLResponse)
    def documents_page(
        request: Request,
        parse_status: str | None = None,
        document_group: str | None = None,
    ) -> HTMLResponse:
        documents: list[DocumentInventoryItem] = []
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                documents = list_document_inventory(
                    settings.database_url,
                    parse_status=parse_status,
                    document_group=document_group,
                    limit=100,
                )
            except InvalidDocumentInventoryError as exc:
                error_message = str(exc)
            except Exception as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "documents.html",
            template_context(
                request,
                database_configured=bool(settings.database_url),
                documents=documents,
                selected_parse_status=parse_status,
                selected_document_group=document_group,
                error_message=error_message,
            ),
        )

    @app.get("/documents/{document_id}", response_class=HTMLResponse)
    def document_detail_page(
        request: Request,
        document_id: int,
        chunk_policy_name: str | None = None,
    ) -> HTMLResponse:
        document: DocumentInventoryItem | None = None
        chunks: list[ChunkRecord] = []
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                document = get_document_inventory_item(settings.database_url, document_id)
                if document is None:
                    error_message = f"Document not found: {document_id}"
                else:
                    chunks = list_document_chunks(
                        settings.database_url,
                        document_id,
                        chunk_policy_name=chunk_policy_name,
                    )
            except (InvalidDocumentInventoryError, InvalidChunkError) as exc:
                error_message = str(exc)
            except Exception as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "document_detail.html",
            template_context(
                request,
                database_configured=bool(settings.database_url),
                document=document,
                chunks=chunks,
                selected_chunk_policy_name=chunk_policy_name,
                error_message=error_message,
            ),
        )

    @app.get("/search", response_class=HTMLResponse)
    def search_compare_page(request: Request) -> HTMLResponse:
        actor_options: list[dict[str, object]] = []
        profile_options: list[str] = []
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                actor_options = list_search_actor_options(settings.database_url)
                profile_options = [
                    profile.profile_name
                    for profile in list_active_embedding_profiles(settings.database_url)
                ]
            except Exception as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "search_compare.html",
            template_context(
                request,
                actor_options=actor_options,
                profile_options=profile_options,
                default_actor_id=actor_options[0]["user_id"] if actor_options else "",
                error_message=error_message,
                database_configured=bool(settings.database_url),
            ),
        )

    @app.get("/search/logs", response_class=HTMLResponse)
    def search_logs_page(
        request: Request,
        actor_user_id: str | None = None,
        requested_search_scope: str | None = None,
        document_group: str | None = None,
        search_log_id: int | None = None,
        limit: int = 50,
    ) -> HTMLResponse:
        actor_options: list[dict[str, object]] = []
        question_sets: list[GoldenQuestionSetRecord] = []
        logs: list[SearchLogListItem] = []
        selected_log: SearchLogDetailRecord | None = None
        error_message = None
        actor_user_id_value: int | None = None
        scope_value = requested_search_scope.strip() if requested_search_scope else None
        document_group_value = document_group.strip() if document_group else None
        if scope_value == "":
            scope_value = None
        if document_group_value == "":
            document_group_value = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                if actor_user_id and actor_user_id.strip():
                    actor_user_id_value = int(actor_user_id)
                actor_options = list_search_actor_options(settings.database_url)
                question_sets = list_golden_question_sets(
                    settings.database_url,
                    active_only=True,
                )
                logs = list_search_logs(
                    settings.database_url,
                    actor_user_id=actor_user_id_value,
                    requested_search_scope=scope_value,
                    document_group=document_group_value,
                    limit=limit,
                )
                if search_log_id is not None:
                    selected_log = get_search_log_detail(settings.database_url, search_log_id)
                    if selected_log is None:
                        error_message = f"Search log not found: {search_log_id}"
            except Exception as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "search_history.html",
            template_context(
                request,
                actor_options=actor_options,
                question_sets=question_sets,
                logs=logs,
                selected_log=selected_log,
                selected_actor_user_id=actor_user_id_value or "",
                selected_scope=scope_value or "",
                selected_document_group=document_group_value or "",
                selected_search_log_id=search_log_id,
                selected_limit=limit,
                error_message=error_message,
                database_configured=bool(settings.database_url),
            ),
        )

    @app.get("/evaluations/questions", response_class=HTMLResponse)
    def golden_questions_page(request: Request) -> HTMLResponse:
        question_sets: list[GoldenQuestionSetRecord] = []
        actor_options: list[dict[str, object]] = []
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                question_sets = list_golden_question_sets(
                    settings.database_url,
                    active_only=False,
                )
                actor_options = list_search_actor_options(settings.database_url)
            except Exception as exc:
                error_message = str(exc)

        question_set_payloads = [
            golden_question_set_payload(question_set) for question_set in question_sets
        ]

        return TEMPLATES.TemplateResponse(
            request,
            "golden_questions.html",
            template_context(
                request,
                question_sets=question_sets,
                question_sets_payload=question_set_payloads,
                actor_options=actor_options,
                default_actor_id=actor_options[0]["user_id"] if actor_options else "",
                error_message=error_message,
                database_configured=bool(settings.database_url),
            ),
        )

    @app.get("/evaluations", response_class=HTMLResponse)
    def golden_evaluations_page(
        request: Request,
        question_set_id: int | None = None,
        profile_name: str | None = None,
        status_filter: str | None = Query(default=None, alias="status"),
        evaluation_run_id: int | None = None,
        limit: int = 100,
    ) -> HTMLResponse:
        question_sets: list[GoldenQuestionSetRecord] = []
        profile_options: list[str] = []
        runs: list[EvaluationRunRecord] = []
        profile_comparison: list[ProfileComparisonRecord] = []
        selected_run: EvaluationRunRecord | None = None
        selected_question_set: GoldenQuestionSetRecord | None = None
        selected_results: list[EvaluationResultRecord] = []
        error_message = None
        profile_value = profile_name.strip() if profile_name else None
        if profile_value == "":
            profile_value = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                question_sets = list_golden_question_sets(
                    settings.database_url,
                    active_only=False,
                )
                profile_options = [
                    profile.profile_name
                    for profile in list_active_embedding_profiles(settings.database_url)
                ]
                runs = list_evaluation_runs(
                    settings.database_url,
                    question_set_id=question_set_id,
                    profile_name=profile_value,
                    status=status_filter,
                    limit=limit,
                )
                if question_set_id is not None:
                    profile_comparison = get_latest_profile_comparison(
                        settings.database_url,
                        question_set_id,
                    )
                if evaluation_run_id is not None:
                    selected_run = get_evaluation_run(settings.database_url, evaluation_run_id)
                    if selected_run is None:
                        error_message = f"Evaluation run not found: {evaluation_run_id}"
                    else:
                        selected_question_set = get_golden_question_set(
                            settings.database_url,
                            selected_run.question_set_id,
                        )
                        selected_results = list_evaluation_results(
                            settings.database_url,
                            selected_run.evaluation_run_id,
                        )
            except (
                InvalidEvaluationRunError,
                InvalidEvaluationReportError,
                InvalidGoldenQuestionError,
            ) as exc:
                error_message = str(exc)

        question_set_names = {
            question_set.question_set_id: question_set.set_name for question_set in question_sets
        }

        return TEMPLATES.TemplateResponse(
            request,
            "golden_evaluations.html",
            template_context(
                request,
                question_sets=question_sets,
                question_set_names=question_set_names,
                profile_options=profile_options,
                runs=runs,
                profile_comparison=profile_comparison,
                selected_run=selected_run,
                selected_question_set=selected_question_set,
                selected_results=selected_results,
                selected_question_set_id=question_set_id,
                selected_profile_name=profile_value or "",
                selected_status=status_filter or "",
                selected_evaluation_run_id=evaluation_run_id,
                error_message=error_message,
                database_configured=bool(settings.database_url),
            ),
        )

    @app.get("/admin/jobs", response_class=HTMLResponse)
    def pipeline_jobs_page(
        request: Request,
        status_filter: str | None = Query(default=None, alias="status"),
        job_id: int | None = None,
    ) -> HTMLResponse:
        jobs: list[PipelineJobListItem] = []
        selected_job = None
        selected_events: list[PipelineJobEventRecord] = []
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                jobs = list_pipeline_jobs(settings.database_url, status=status_filter)
                if job_id is not None:
                    selected_job = get_pipeline_job(settings.database_url, job_id)
                    if selected_job is None:
                        error_message = f"Pipeline job not found: {job_id}"
                    else:
                        selected_events = list_pipeline_job_events(settings.database_url, job_id)
            except InvalidPipelineJobError as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "pipeline_jobs.html",
            template_context(
                request,
                jobs=jobs,
                selected_job=selected_job,
                selected_events=selected_events,
                selected_status=status_filter or "",
                selected_job_id=job_id,
                error_message=error_message,
                database_configured=bool(settings.database_url),
            ),
        )

    @app.get("/admin/embedding-jobs", response_class=HTMLResponse)
    def embedding_jobs_page(
        request: Request,
        status_filter: str | None = Query(default=None, alias="status"),
        profile_name: str | None = None,
        job_id: int | None = None,
    ) -> HTMLResponse:
        jobs: list[EmbeddingJobRecord] = []
        selected_job = None
        selected_vector = None
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                jobs = list_embedding_jobs(
                    settings.database_url,
                    status=status_filter,
                    profile_name=profile_name,
                )
                if job_id is not None:
                    selected_job = get_embedding_job(settings.database_url, job_id)
                    if selected_job is None:
                        error_message = f"Embedding job not found: {job_id}"
                    else:
                        try:
                            selected_vector = get_chunk_embedding(
                                settings.database_url,
                                profile_name=selected_job.profile_name,
                                chunk_id=selected_job.chunk_id,
                            )
                        except InvalidEmbeddingVectorError:
                            selected_vector = None
            except InvalidEmbeddingJobError as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "embedding_jobs.html",
            template_context(
                request,
                jobs=jobs,
                selected_job=selected_job,
                selected_vector=selected_vector,
                selected_status=status_filter or "",
                selected_profile_name=profile_name or "",
                selected_job_id=job_id,
                error_message=error_message,
                database_configured=bool(settings.database_url),
            ),
        )

    @app.get("/admin/logs", response_class=HTMLResponse)
    def admin_logs(request: Request, level: str | None = None) -> HTMLResponse:
        logs = []
        error_message = None
        if settings.database_url:
            try:
                logs = list_logs(settings.database_url, level=level)
            except Exception as exc:
                error_message = str(exc)
        else:
            error_message = "NEX_PCX_DATABASE_URL is not configured."

        return TEMPLATES.TemplateResponse(
            request,
            "admin_logs.html",
            template_context(
                request,
                logs=logs,
                selected_level=level or "",
                error_message=error_message,
            ),
        )

    return app


app = create_app()
