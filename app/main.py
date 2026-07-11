"""FastAPI application factory for NeX_PCX."""

import csv
import hashlib
import io
import json
import traceback as traceback_module
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from urllib.parse import urlencode

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.core.admin_logging import (
    InvalidAdminLogError,
    acknowledge_log,
    list_logs,
    list_provider_route_alert_logs,
    log_event,
)
from app.core.chunk_policies import (
    ChunkPolicySummaryRecord,
    InvalidChunkPolicyManagementError,
    get_chunk_policy_summary,
    list_chunk_policy_summaries,
)
from app.core.chunks import (
    DEFAULT_CHUNK_POLICY_NAME,
    ChunkRecord,
    InvalidChunkError,
    list_document_chunks,
)
from app.core.config import Settings, get_settings
from app.core.database import connect
from app.core.document_inventory import (
    DocumentInventoryItem,
    DocumentPermissionUpdateInput,
    InvalidDocumentInventoryError,
    get_document_inventory_item,
    list_document_inventory,
    update_document_permission,
)
from app.core.embedding_jobs import (
    EmbeddingJobRecord,
    EmbeddingProfileRecord,
    InvalidEmbeddingJobError,
    get_embedding_job,
    list_active_embedding_profiles,
    list_embedding_jobs,
    retry_embedding_job,
)
from app.core.embedding_model_distribution import (
    EmbeddingModelReadiness,
    audit_embedding_model_readiness,
)
from app.core.embedding_provider_contract_sample_sets import (
    EmbeddingProviderContractSampleSetInput,
    EmbeddingProviderContractSampleSetRecord,
    InvalidEmbeddingProviderContractSampleSetError,
    delete_embedding_provider_contract_sample_set,
    get_default_embedding_provider_contract_sample_set,
    get_embedding_provider_contract_sample_set,
    list_embedding_provider_contract_sample_sets,
    upsert_embedding_provider_contract_sample_set,
)
from app.core.embedding_provider_health import get_embedding_provider_health_status
from app.core.embedding_provider_preflight_runs import (
    EmbeddingProviderPreflightRunInput,
    EmbeddingProviderPreflightRunRecord,
    InvalidEmbeddingProviderPreflightRunError,
    list_embedding_provider_preflight_runs,
    record_embedding_provider_preflight_run,
)
from app.core.embedding_provider_preflight_schedules import (
    EmbeddingProviderPreflightScheduleInput,
    EmbeddingProviderPreflightScheduleRecord,
    InvalidEmbeddingProviderPreflightScheduleError,
    ScheduledProviderRoutePreflightRun,
    get_embedding_provider_preflight_schedule,
    list_due_embedding_provider_preflight_schedules,
    list_embedding_provider_preflight_schedules,
    run_due_embedding_provider_preflight_schedules,
    upsert_embedding_provider_preflight_schedule,
)
from app.core.embedding_provider_route_contract_snapshots import (
    EmbeddingProviderRouteContractSnapshotRecord,
    InvalidEmbeddingProviderRouteContractSnapshotError,
    list_embedding_provider_route_contract_snapshots,
    record_embedding_provider_route_contract_snapshot,
)
from app.core.embedding_provider_route_contracts import (
    EmbeddingProviderRouteContractResult,
    check_embedding_provider_route_contract,
)
from app.core.embedding_provider_route_health import (
    EmbeddingProviderRouteHealthResult,
    check_embedding_provider_route_health,
    get_embedding_provider_route_health_summary,
)
from app.core.embedding_provider_route_health_snapshots import (
    EmbeddingProviderRouteHealthSnapshotRecord,
    InvalidEmbeddingProviderRouteHealthSnapshotError,
    list_embedding_provider_route_health_snapshots,
    record_embedding_provider_route_health_snapshot,
    record_embedding_provider_route_health_summary,
)
from app.core.embedding_provider_route_readiness import (
    EmbeddingProviderRouteReadinessItem,
    EmbeddingProviderRouteReadinessSummary,
    get_embedding_provider_route_readiness_summary,
)
from app.core.embedding_provider_route_retention import (
    InvalidProviderRouteRetentionError,
    ProviderRouteCleanupResult,
    ProviderRouteRetentionSettings,
    ProviderRouteRetentionSettingsInput,
    cleanup_expired_provider_route_records,
    load_provider_route_retention_settings,
    update_provider_route_retention_settings,
)
from app.core.embedding_provider_routes import (
    EmbeddingProviderRouteInput,
    EmbeddingProviderRouteRecord,
    InvalidEmbeddingProviderRouteError,
    get_embedding_provider_route,
    list_embedding_provider_routes,
    upsert_embedding_provider_route,
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
    EvaluationPermissionAuditRecord,
    InvalidEvaluationReportError,
    ProfileComparisonRecord,
    get_evaluation_permission_audit,
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
    DOCUMENT_ACCESS_SCOPES,
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
    GoldenQuestionBatchPromotionInput,
    GoldenQuestionBatchPromotionRecord,
    GoldenQuestionCandidateRecord,
    GoldenQuestionPromotionInput,
    GoldenQuestionPromotionRecord,
    InvalidGoldenQuestionPromotionError,
    list_golden_question_candidates,
    promote_search_result_to_golden_question,
    promote_search_results_to_golden_questions,
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
from app.core.i18n import (
    LANGUAGE_COOKIE_NAME,
    LANGUAGE_OPTIONS,
    get_translator,
    normalize_language,
    resolve_language,
)
from app.core.permission_inventory import (
    InvalidPermissionInventoryError,
    PermissionInventory,
    PermissionInventorySummary,
    PermissionMembershipInventoryRecord,
    PermissionOrgUnitInventoryRecord,
    PermissionReadinessIssueRecord,
    PermissionReadinessSummary,
    PermissionUserInventoryRecord,
    get_permission_inventory,
    get_permission_inventory_summary,
    get_permission_readiness_summary,
    list_permission_memberships,
    list_permission_org_units,
    list_permission_users,
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
    SearchPermissionMatrixEntryInput,
    SearchPermissionMatrixEntryResult,
    SearchPermissionMatrixInput,
    SearchPermissionMatrixResult,
    run_permission_search_matrix,
    run_search_compare,
)
from app.core.search_logs import (
    InvalidSearchLogError,
    SearchFeedbackCommentRecord,
    SearchFeedbackProfileSummaryRecord,
    SearchLogCleanupResult,
    SearchLogDetailRecord,
    SearchLogListItem,
    SearchLogRecord,
    SearchLogResultDetailRecord,
    SearchLogRetentionSettings,
    SearchLogRetentionSettingsInput,
    SearchLogReviewMetadataInput,
    SearchResultFeedbackInput,
    SearchResultFeedbackRecord,
    cleanup_expired_search_logs,
    create_search_result_feedback,
    get_search_log,
    get_search_log_detail,
    get_search_log_result,
    list_search_feedback_comments,
    list_search_logs,
    load_search_log_retention_settings,
    summarize_search_feedback,
    update_search_log_retention_settings,
    update_search_log_review_metadata,
)
from app.core.vector_search import InvalidVectorSearchError, VectorSearchResult

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=BASE_DIR / "web" / "templates")
UPLOAD_FILE_FORM = File(...)
DOCUMENT_GROUP_FORM = Form("default")
SECURITY_LEVEL_FORM = Form("internal")
UPLOADED_BY_FORM = Form(None)
UPLOADED_BY_USER_ID_FORM = Form(None)
OWNER_USER_ID_FORM = Form(None)
OWNER_ORG_UNIT_ID_FORM = Form(None)
ACCESS_SCOPE_FORM = Form("personal")
UPDATED_BY_USER_ID_FORM = Form(None)
EVALUATION_EXPORT_VERSION = 1
SEARCH_LOG_EXPORT_VERSION = 1
SEARCH_EXPERIMENT_REPORT_VERSION = 1
SEARCH_COMPARE_FILE_TYPES = (".md", ".pdf", ".docx", ".hwpx", ".pptx", ".xlsx")
SEARCH_COMPARE_SCOPE_OPTIONS = ("mine", "team", "managed_org", "company")


class SearchCompareRequest(BaseModel):
    query_text: str
    actor_user_id: int
    requested_search_scope: str = "company"
    top_k: int = Field(default=5, ge=1)
    profiles: list[str] | None = None
    chunk_policy_name: str | None = None
    document_group: str | None = None
    file_type: str | None = None


class SearchPermissionMatrixEntryRequest(BaseModel):
    actor_user_id: int = Field(ge=1)
    requested_search_scope: str = "company"


class SearchPermissionMatrixRequest(BaseModel):
    query_text: str
    entries: list[SearchPermissionMatrixEntryRequest]
    top_k: int = Field(default=5, ge=1)
    profiles: list[str] | None = None
    chunk_policy_name: str | None = None
    document_group: str | None = None
    file_type: str | None = None


class SearchFeedbackRequest(BaseModel):
    search_log_result_id: int = Field(ge=1)
    relevance_label: str
    comment: str | None = None


class SearchLogReviewMetadataRequest(BaseModel):
    review_tags: list[str] = Field(default_factory=list, max_length=12)
    review_memo: str | None = Field(default=None, max_length=2000)
    reviewed_by_user_id: int | None = Field(default=None, ge=1)


class SearchLogRetentionSettingsRequest(BaseModel):
    enabled: bool = True
    retention_days: int = Field(default=30, ge=1, le=3650)
    cleanup_batch_size: int = Field(default=1000, ge=1, le=100000)


class SearchLogCleanupRequest(BaseModel):
    dry_run: bool = True


class ProviderRouteRetentionSettingsRequest(BaseModel):
    enabled: bool = True
    retention_days: int = Field(default=30, ge=1, le=3650)
    cleanup_batch_size: int = Field(default=1000, ge=1, le=100000)


class ProviderRouteCleanupRequest(BaseModel):
    dry_run: bool = True


class ProviderPreflightScheduleRequest(BaseModel):
    description: str | None = Field(default=None, max_length=500)
    profile_name: str | None = Field(default=None, max_length=120)
    active_only: bool = True
    interval_minutes: int = Field(default=60, ge=1, le=10080)
    is_enabled: bool = False
    next_run_at: datetime | None = None


class ProviderPreflightScheduleRunDueRequest(BaseModel):
    schedule_name: str | None = Field(default=None, max_length=120)
    limit: int = Field(default=20, ge=1, le=100)


class DocumentPermissionUpdateRequest(BaseModel):
    owner_user_id: int | None = Field(default=None, ge=1)
    owner_org_unit_id: int | None = Field(default=None, ge=1)
    access_scope: str = "personal"
    updated_by_user_id: int | None = Field(default=None, ge=1)


class EmbeddingProviderRouteRequest(BaseModel):
    profile_name: str
    provider_name: str
    provider_mode: str = "remote"
    provider_base_url: str | None = None
    timeout_seconds: float = Field(default=30.0, gt=0)
    priority: int = Field(default=100, ge=0)
    is_active: bool = True
    health_check_enabled: bool = True
    runtime_metadata: dict[str, object] = Field(default_factory=dict)


class EmbeddingProviderContractSampleSetRequest(BaseModel):
    sample_set_name: str
    description: str | None = Field(default=None, max_length=1000)
    input_type: str = "document"
    sample_texts: list[str] = Field(min_length=1, max_length=20)
    is_active: bool = True
    is_default: bool = False


class ProviderRouteAlertAcknowledgeRequest(BaseModel):
    acknowledged_by: str = "operator"
    acknowledgement_note: str | None = Field(default=None, max_length=1000)


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


class GoldenQuestionBatchPromotionRequest(GoldenQuestionPromotionRequest):
    search_log_result_ids: list[int] = Field(min_length=1, max_length=50)


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


def parse_optional_positive_int_form(value: str | int | None, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        parsed = value
    else:
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = int(stripped)
        except ValueError as exc:
            raise InvalidFileMetadataError(f"{field_name} must be an integer") from exc
    if parsed <= 0:
        raise InvalidFileMetadataError(f"{field_name} must be greater than 0")
    return parsed


def upload_permission_user_option_payload(
    user: PermissionUserInventoryRecord,
) -> dict[str, object]:
    return {
        "user_id": user.user_id,
        "login_id": user.login_id,
        "display_name": user.display_name,
        "primary_role_name": user.primary_role_name,
        "primary_org_unit_id": user.primary_org_unit_id,
        "primary_org_unit_name": user.primary_org_unit_name,
    }


def upload_permission_org_unit_option_payload(
    org_unit: PermissionOrgUnitInventoryRecord,
) -> dict[str, object]:
    return {
        "org_unit_id": org_unit.org_unit_id,
        "org_unit_name": org_unit.org_unit_name,
        "org_unit_type": org_unit.org_unit_type,
        "org_path": org_unit.org_path,
    }


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


def admin_log_payload(log: dict[str, object]) -> dict[str, object]:
    return {
        "log_id": log["log_id"],
        "occurred_at": _datetime_response(log.get("occurred_at")),
        "level": log["level"],
        "event_type": log["event_type"],
        "source": log.get("source"),
        "message": log["message"],
        "detail": dict(log.get("detail") or {}),
        "traceback": log.get("traceback"),
        "request_path": log.get("request_path"),
        "correlation_id": log.get("correlation_id"),
        "acknowledged_at": _datetime_response(log.get("acknowledged_at")),
        "acknowledged_by": log.get("acknowledged_by"),
        "acknowledgement_note": log.get("acknowledgement_note"),
    }


def parse_acknowledged_filter(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"", "all"}:
        return None
    if normalized in {"1", "true", "yes", "acknowledged"}:
        return True
    if normalized in {"0", "false", "no", "unacknowledged"}:
        return False
    raise ValueError("acknowledged must be one of false, true, or all")


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


def embedding_job_provider_route_failover_payload(
    job: EmbeddingJobRecord,
) -> dict[str, object] | None:
    metadata = job.runtime_metadata or {}
    candidate_count = metadata.get("provider_route_failover_candidate_count")
    attempt = metadata.get("provider_route_failover_attempt")
    failed_attempts = metadata.get("provider_route_failed_attempts")
    if candidate_count is None and attempt is None and not failed_attempts:
        return None
    return {
        "candidate_count": candidate_count,
        "succeeded_attempt": attempt,
        "selected_route_id": metadata.get("provider_route_id"),
        "selected_provider_name": metadata.get("provider_route_name"),
        "selected_route_priority": metadata.get("provider_route_priority"),
        "failed_attempts": failed_attempts if isinstance(failed_attempts, list) else [],
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
        "provider_route_failover": embedding_job_provider_route_failover_payload(job),
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


def embedding_model_readiness_payload(readiness: EmbeddingModelReadiness) -> dict[str, object]:
    distribution = readiness.distribution
    return {
        "model_key": distribution.model_key,
        "repo_id": distribution.repo_id,
        "revision": distribution.default_revision,
        "local_dir": str(readiness.local_dir),
        "profile_names": list(distribution.profile_names),
        "adapter_name": distribution.adapter_name,
        "note": distribution.note,
        "exists": readiness.exists,
        "ready": readiness.ready,
        "has_config": readiness.has_config,
        "has_tokenizer": readiness.has_tokenizer,
        "has_model_weights": readiness.has_model_weights,
        "file_count": readiness.file_count,
        "total_size_bytes": readiness.total_size_bytes,
    }


def embedding_provider_contract_sample_set_payload(
    sample_set: EmbeddingProviderContractSampleSetRecord,
) -> dict[str, object]:
    return {
        "sample_set_name": sample_set.sample_set_name,
        "description": sample_set.description,
        "input_type": sample_set.input_type,
        "sample_texts": list(sample_set.sample_texts),
        "sample_text_count": len(sample_set.sample_texts),
        "is_active": sample_set.is_active,
        "is_default": sample_set.is_default,
        "created_at": _datetime_response(sample_set.created_at),
        "updated_at": _datetime_response(sample_set.updated_at),
    }


def embedding_provider_contract_sample_set_input_from_request(
    payload: EmbeddingProviderContractSampleSetRequest,
    *,
    sample_set_name: str | None = None,
) -> EmbeddingProviderContractSampleSetInput:
    return EmbeddingProviderContractSampleSetInput(
        sample_set_name=sample_set_name or payload.sample_set_name,
        description=payload.description,
        input_type=payload.input_type,
        sample_texts=tuple(payload.sample_texts),
        is_active=payload.is_active,
        is_default=payload.is_default,
    )


def embedding_provider_route_payload(route: EmbeddingProviderRouteRecord) -> dict[str, object]:
    return {
        "route_id": route.route_id,
        "profile_name": route.profile_name,
        "provider_name": route.provider_name,
        "provider_mode": route.provider_mode,
        "provider_base_url": route.provider_base_url,
        "timeout_seconds": route.timeout_seconds,
        "priority": route.priority,
        "is_active": route.is_active,
        "health_check_enabled": route.health_check_enabled,
        "runtime_metadata": route.runtime_metadata,
        "created_at": _datetime_response(route.created_at),
        "updated_at": _datetime_response(route.updated_at),
    }


def embedding_provider_route_health_payload(
    route_health: EmbeddingProviderRouteHealthResult,
) -> dict[str, object]:
    return {
        "route": embedding_provider_route_payload(route_health.route),
        "checked": route_health.checked,
        "ready": route_health.ready,
        "status": route_health.status,
        "elapsed_ms": route_health.elapsed_ms,
        "provider_type": route_health.provider_type,
        "provider_model_id": route_health.provider_model_id,
        "model_key": route_health.model_key,
        "profile_names": list(route_health.profile_names),
        "dimension": route_health.dimension,
        "device": route_health.device,
        "runtime_metadata": route_health.runtime_metadata,
        "validation_errors": list(route_health.validation_errors),
        "error_message": route_health.error_message,
    }


def embedding_provider_route_health_snapshot_payload(
    snapshot: EmbeddingProviderRouteHealthSnapshotRecord,
) -> dict[str, object]:
    return {
        "snapshot_id": snapshot.snapshot_id,
        "route_id": snapshot.route_id,
        "profile_name": snapshot.profile_name,
        "provider_name": snapshot.provider_name,
        "provider_mode": snapshot.provider_mode,
        "checked": snapshot.checked,
        "ready": snapshot.ready,
        "status": snapshot.status,
        "elapsed_ms": snapshot.elapsed_ms,
        "provider_type": snapshot.provider_type,
        "provider_model_id": snapshot.provider_model_id,
        "model_key": snapshot.model_key,
        "profile_names": list(snapshot.profile_names),
        "dimension": snapshot.dimension,
        "device": snapshot.device,
        "runtime_metadata": snapshot.runtime_metadata,
        "validation_errors": list(snapshot.validation_errors),
        "error_message": snapshot.error_message,
        "checked_at": _datetime_response(snapshot.checked_at),
    }


def embedding_provider_route_contract_payload(
    contract: EmbeddingProviderRouteContractResult,
) -> dict[str, object]:
    return {
        "route": embedding_provider_route_payload(contract.route),
        "passed": contract.passed,
        "status": contract.status,
        "elapsed_ms": contract.elapsed_ms,
        "health": (
            embedding_provider_route_health_payload(contract.health)
            if contract.health is not None
            else None
        ),
        "input_type": contract.input_type,
        "sample_text_count": contract.sample_text_count,
        "expected_dimension": contract.expected_dimension,
        "provider_type": contract.provider_type,
        "provider_model_id": contract.provider_model_id,
        "model_key": contract.model_key,
        "dimension": contract.dimension,
        "input_count": contract.input_count,
        "runtime_metadata": contract.runtime_metadata,
        "validation_errors": list(contract.validation_errors),
        "error_message": contract.error_message,
    }


def embedding_provider_route_contract_snapshot_payload(
    snapshot: EmbeddingProviderRouteContractSnapshotRecord,
) -> dict[str, object]:
    return {
        "snapshot_id": snapshot.snapshot_id,
        "route_id": snapshot.route_id,
        "profile_name": snapshot.profile_name,
        "provider_name": snapshot.provider_name,
        "provider_mode": snapshot.provider_mode,
        "passed": snapshot.passed,
        "status": snapshot.status,
        "elapsed_ms": snapshot.elapsed_ms,
        "input_type": snapshot.input_type,
        "sample_text_count": snapshot.sample_text_count,
        "expected_dimension": snapshot.expected_dimension,
        "provider_type": snapshot.provider_type,
        "provider_model_id": snapshot.provider_model_id,
        "model_key": snapshot.model_key,
        "dimension": snapshot.dimension,
        "input_count": snapshot.input_count,
        "runtime_metadata": snapshot.runtime_metadata,
        "validation_errors": list(snapshot.validation_errors),
        "error_message": snapshot.error_message,
        "checked_at": _datetime_response(snapshot.checked_at),
    }


def embedding_provider_route_readiness_item_payload(
    item: EmbeddingProviderRouteReadinessItem,
) -> dict[str, object]:
    return {
        "route": embedding_provider_route_payload(item.route),
        "ready": item.ready,
        "status": item.status,
        "reasons": list(item.reasons),
        "latest_health_snapshot": (
            embedding_provider_route_health_snapshot_payload(item.latest_health_snapshot)
            if item.latest_health_snapshot is not None
            else None
        ),
        "latest_contract_snapshot": (
            embedding_provider_route_contract_snapshot_payload(item.latest_contract_snapshot)
            if item.latest_contract_snapshot is not None
            else None
        ),
    }


def embedding_provider_route_readiness_summary_payload(
    summary: EmbeddingProviderRouteReadinessSummary,
) -> dict[str, object]:
    return {
        "route_count": summary.route_count,
        "active_count": summary.active_count,
        "ready_count": summary.ready_count,
        "blocked_count": summary.blocked_count,
        "needs_preflight_count": summary.needs_preflight_count,
        "status_counts": summary.status_counts,
        "routes": [
            embedding_provider_route_readiness_item_payload(item) for item in summary.routes
        ],
    }


def embedding_provider_route_operations_summary_payload(
    *,
    readiness: EmbeddingProviderRouteReadinessSummary,
    schedules: list[EmbeddingProviderPreflightScheduleRecord],
    due_schedules: list[EmbeddingProviderPreflightScheduleRecord],
    latest_run: EmbeddingProviderPreflightRunRecord | None,
) -> dict[str, object]:
    failed_schedule_count = sum(
        1 for schedule in schedules if schedule.last_status in {"failed", "error"}
    )
    return {
        "route_count": readiness.route_count,
        "active_route_count": readiness.active_count,
        "ready_route_count": readiness.ready_count,
        "blocked_route_count": readiness.blocked_count,
        "needs_preflight_count": readiness.needs_preflight_count,
        "schedule_count": len(schedules),
        "enabled_schedule_count": sum(1 for schedule in schedules if schedule.is_enabled),
        "due_schedule_count": len(due_schedules),
        "failed_schedule_count": failed_schedule_count,
        "latest_preflight_run": (
            embedding_provider_preflight_run_payload(latest_run) if latest_run is not None else None
        ),
    }


def embedding_provider_preflight_run_payload(
    run: EmbeddingProviderPreflightRunRecord,
) -> dict[str, object]:
    return {
        "run_id": run.run_id,
        "schedule_name": run.schedule_name,
        "trigger_source": run.trigger_source,
        "profile_name": run.profile_name,
        "active_only": run.active_only,
        "status": run.status,
        "route_count": run.route_count,
        "passed_count": run.passed_count,
        "failed_count": run.failed_count,
        "sample_set_name": run.sample_set_name,
        "input_type": run.input_type,
        "sample_text_count": run.sample_text_count,
        "elapsed_ms": run.elapsed_ms,
        "result": run.result,
        "error_message": run.error_message,
        "started_at": _datetime_response(run.started_at),
        "completed_at": _datetime_response(run.completed_at),
    }


def embedding_provider_preflight_schedule_payload(
    schedule: EmbeddingProviderPreflightScheduleRecord,
) -> dict[str, object]:
    return {
        "schedule_name": schedule.schedule_name,
        "description": schedule.description,
        "profile_name": schedule.profile_name,
        "active_only": schedule.active_only,
        "interval_minutes": schedule.interval_minutes,
        "is_enabled": schedule.is_enabled,
        "next_run_at": _datetime_response(schedule.next_run_at),
        "last_run_at": _datetime_response(schedule.last_run_at),
        "last_status": schedule.last_status,
        "last_result": schedule.last_result,
        "run_count": schedule.run_count,
        "failure_count": schedule.failure_count,
        "created_at": _datetime_response(schedule.created_at),
        "updated_at": _datetime_response(schedule.updated_at),
    }


def embedding_provider_preflight_schedule_input_from_request(
    schedule_name: str,
    payload: ProviderPreflightScheduleRequest,
) -> EmbeddingProviderPreflightScheduleInput:
    return EmbeddingProviderPreflightScheduleInput(
        schedule_name=schedule_name,
        description=payload.description,
        profile_name=payload.profile_name,
        active_only=payload.active_only,
        interval_minutes=payload.interval_minutes,
        is_enabled=payload.is_enabled,
        next_run_at=payload.next_run_at,
    )


def scheduled_provider_route_preflight_run_payload(
    run: ScheduledProviderRoutePreflightRun,
) -> dict[str, object]:
    return {
        "schedule": embedding_provider_preflight_schedule_payload(run.schedule),
        "status": run.status,
        "result": run.result,
        "updated_schedule": embedding_provider_preflight_schedule_payload(run.updated_schedule),
        "run_record": embedding_provider_preflight_run_payload(run.run_record),
    }


def provider_route_retention_settings_payload(
    retention_settings: ProviderRouteRetentionSettings,
) -> dict[str, object]:
    return {
        "enabled": retention_settings.enabled,
        "retention_days": retention_settings.retention_days,
        "cleanup_batch_size": retention_settings.cleanup_batch_size,
    }


def provider_route_retention_settings_input_from_request(
    payload: ProviderRouteRetentionSettingsRequest,
) -> ProviderRouteRetentionSettingsInput:
    return ProviderRouteRetentionSettingsInput(
        enabled=payload.enabled,
        retention_days=payload.retention_days,
        cleanup_batch_size=payload.cleanup_batch_size,
    )


def provider_route_cleanup_result_payload(
    result: ProviderRouteCleanupResult,
) -> dict[str, object]:
    return {
        "enabled": result.enabled,
        "dry_run": result.dry_run,
        "retention_days": result.retention_days,
        "cleanup_batch_size": result.cleanup_batch_size,
        "expired_count": result.expired_count,
        "deleted_count": result.deleted_count,
        "expired_health_snapshot_count": result.expired_health_snapshot_count,
        "expired_contract_snapshot_count": result.expired_contract_snapshot_count,
        "expired_preflight_run_count": result.expired_preflight_run_count,
        "deleted_health_snapshot_count": result.deleted_health_snapshot_count,
        "deleted_contract_snapshot_count": result.deleted_contract_snapshot_count,
        "deleted_preflight_run_count": result.deleted_preflight_run_count,
        "cutoff_at": _datetime_response(result.cutoff_at),
    }


def log_embedding_provider_route_health_alert(
    database_url: str,
    route_health: EmbeddingProviderRouteHealthResult,
) -> int | None:
    if route_health.ready or not route_health.checked:
        return None
    level = "ERROR" if route_health.status in {"unreachable", "unsupported"} else "WARNING"
    route = route_health.route
    return log_event(
        database_url,
        level=level,
        event_type="embedding_provider_route_health_alert",
        source="embedding_provider_routes",
        message=(
            f"Embedding provider route {route.provider_name} health is " f"{route_health.status}."
        ),
        detail={
            "route_id": route.route_id,
            "profile_name": route.profile_name,
            "provider_name": route.provider_name,
            "provider_mode": route.provider_mode,
            "status": route_health.status,
            "ready": route_health.ready,
            "checked": route_health.checked,
            "provider_type": route_health.provider_type,
            "provider_model_id": route_health.provider_model_id,
            "model_key": route_health.model_key,
            "dimension": route_health.dimension,
            "validation_errors": list(route_health.validation_errors),
            "error_message": route_health.error_message,
        },
        correlation_id=f"embedding-provider-route:{route.route_id}:health",
    )


def log_embedding_provider_route_contract_alert(
    database_url: str,
    contract: EmbeddingProviderRouteContractResult,
) -> int | None:
    if contract.passed:
        return None
    level = (
        "ERROR"
        if contract.status in {"embedding_failed", "invalid_route"}
        or contract.status.startswith("health_unreachable")
        else "WARNING"
    )
    route = contract.route
    return log_event(
        database_url,
        level=level,
        event_type="embedding_provider_route_contract_alert",
        source="embedding_provider_routes",
        message=(
            f"Embedding provider route {route.provider_name} contract check "
            f"failed with status {contract.status}."
        ),
        detail={
            "route_id": route.route_id,
            "profile_name": route.profile_name,
            "provider_name": route.provider_name,
            "provider_mode": route.provider_mode,
            "status": contract.status,
            "passed": contract.passed,
            "expected_dimension": contract.expected_dimension,
            "provider_type": contract.provider_type,
            "provider_model_id": contract.provider_model_id,
            "model_key": contract.model_key,
            "dimension": contract.dimension,
            "input_count": contract.input_count,
            "validation_errors": list(contract.validation_errors),
            "error_message": contract.error_message,
        },
        correlation_id=f"embedding-provider-route:{route.route_id}:contract",
    )


def embedding_provider_route_input_from_request(
    payload: EmbeddingProviderRouteRequest,
) -> EmbeddingProviderRouteInput:
    return EmbeddingProviderRouteInput(
        profile_name=payload.profile_name,
        provider_name=payload.provider_name,
        provider_mode=payload.provider_mode,
        provider_base_url=payload.provider_base_url,
        timeout_seconds=payload.timeout_seconds,
        priority=payload.priority,
        is_active=payload.is_active,
        health_check_enabled=payload.health_check_enabled,
        runtime_metadata=payload.runtime_metadata,
    )


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
        "permission_summary": result.permission_filter.metadata.get(
            "permission_explainability",
            {},
        ),
        "top_k": result.top_k,
        "total_elapsed_ms": result.total_elapsed_ms,
        "profiles": [search_compare_profile_payload(profile) for profile in result.profiles],
    }


def search_permission_matrix_entry_payload(
    entry: SearchPermissionMatrixEntryResult,
) -> dict[str, object]:
    return {
        "search_log_id": entry.search_log_id,
        "actor_user_id": entry.actor_user_id,
        "requested_search_scope": entry.requested_search_scope,
        "effective_search_scope": entry.effective_search_scope,
        "permission_filter_metadata": entry.permission_filter.metadata,
        "permission_summary": entry.permission_filter.metadata.get(
            "permission_explainability",
            {},
        ),
        "result_count": entry.result_count,
        "unique_chunk_count": entry.unique_chunk_count,
        "top_result": (
            vector_search_result_payload(entry.top_result) if entry.top_result is not None else None
        ),
        "profiles": [search_compare_profile_payload(profile) for profile in entry.profiles],
        "total_elapsed_ms": entry.total_elapsed_ms,
    }


def search_permission_matrix_payload(
    result: SearchPermissionMatrixResult,
) -> dict[str, object]:
    return {
        "query_text": result.query_text,
        "top_k": result.top_k,
        "total_elapsed_ms": result.total_elapsed_ms,
        "entries": [search_permission_matrix_entry_payload(entry) for entry in result.entries],
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


def search_log_retention_settings_payload(
    retention_settings: SearchLogRetentionSettings,
) -> dict[str, object]:
    return {
        "enabled": retention_settings.enabled,
        "retention_days": retention_settings.retention_days,
        "cleanup_batch_size": retention_settings.cleanup_batch_size,
    }


def search_log_retention_settings_input_from_request(
    payload: SearchLogRetentionSettingsRequest,
) -> SearchLogRetentionSettingsInput:
    return SearchLogRetentionSettingsInput(
        enabled=payload.enabled,
        retention_days=payload.retention_days,
        cleanup_batch_size=payload.cleanup_batch_size,
    )


def search_log_cleanup_result_payload(result: SearchLogCleanupResult) -> dict[str, object]:
    return {
        "enabled": result.enabled,
        "dry_run": result.dry_run,
        "retention_days": result.retention_days,
        "cleanup_batch_size": result.cleanup_batch_size,
        "expired_count": result.expired_count,
        "deleted_count": result.deleted_count,
        "cutoff_at": _datetime_response(result.cutoff_at),
    }


def permission_summary_from_metadata(metadata: dict[str, object]) -> dict[str, object]:
    summary = metadata.get("permission_explainability", {})
    return dict(summary) if isinstance(summary, dict) else {}


def search_reproducibility_payload(search_log: SearchLogRecord) -> dict[str, object]:
    runtime_metadata = dict(search_log.query_runtime_metadata)
    return {
        "fingerprint": search_reproducibility_fingerprint(search_log),
        "fingerprint_algorithm": "sha256:16",
        "query_text": search_log.query_text,
        "normalized_query_text": search_log.normalized_query_text,
        "actor_user_id": search_log.actor_user_id,
        "requested_search_scope": search_log.requested_search_scope,
        "effective_search_scope": search_log.effective_search_scope,
        "top_k": search_log.top_k,
        "profiles": list(search_log.profiles),
        "profile_count": len(search_log.profiles),
        "chunk_policy_name": search_log.chunk_policy_name,
        "document_group": search_log.document_group,
        "file_type": search_log.file_type,
        "similarity_metric": search_log.similarity_metric,
        "query_runtime_metadata": runtime_metadata,
        "runtime_metadata_keys": sorted(str(key) for key in runtime_metadata),
    }


def search_reproducibility_fingerprint(search_log: SearchLogRecord) -> str:
    fingerprint_payload = {
        "query_text": search_log.query_text,
        "normalized_query_text": search_log.normalized_query_text,
        "actor_user_id": search_log.actor_user_id,
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
    }
    canonical = json.dumps(
        fingerprint_payload,
        default=str,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def normalize_search_fingerprint(value: str | None) -> str | None:
    normalized = value.strip().lower() if value else None
    return normalized or None


def filter_search_logs_by_fingerprint(
    logs: list[SearchLogListItem],
    fingerprint: str | None,
) -> list[SearchLogListItem]:
    normalized = normalize_search_fingerprint(fingerprint)
    if normalized is None:
        return logs
    return [
        item for item in logs if search_reproducibility_fingerprint(item.search_log) == normalized
    ]


def search_log_replay_url(search_log: SearchLogRecord) -> str:
    query_params: dict[str, object] = {
        "replay_search_log_id": search_log.search_log_id,
        "query_text": search_log.query_text,
        "actor_user_id": search_log.actor_user_id,
        "requested_search_scope": search_log.requested_search_scope,
        "top_k": search_log.top_k,
        "profiles": list(search_log.profiles),
    }
    optional_params = {
        "document_group": search_log.document_group,
        "file_type": search_log.file_type,
        "chunk_policy_name": search_log.chunk_policy_name,
    }
    for key, value in optional_params.items():
        if value:
            query_params[key] = value
    return f"/search?{urlencode(query_params, doseq=True)}"


def search_compare_prefill_payload(
    request: Request,
    default_actor_id: object,
) -> dict[str, object]:
    query_params = request.query_params
    profiles = tuple(
        profile.strip() for profile in query_params.getlist("profiles") if profile.strip()
    )
    scope = (query_params.get("requested_search_scope") or "company").strip()
    if scope not in SEARCH_COMPARE_SCOPE_OPTIONS:
        scope = "company"

    top_k_text = (query_params.get("top_k") or "5").strip()
    try:
        top_k = max(1, min(20, int(top_k_text)))
    except ValueError:
        top_k = 5

    actor_user_id_text = (query_params.get("actor_user_id") or "").strip()
    try:
        actor_user_id: object = int(actor_user_id_text)
    except ValueError:
        actor_user_id = default_actor_id
    if actor_user_id == "":
        actor_user_id = default_actor_id

    return {
        "replay_search_log_id": (query_params.get("replay_search_log_id") or "").strip(),
        "query_text": (query_params.get("query_text") or "").strip(),
        "actor_user_id": actor_user_id,
        "requested_search_scope": scope,
        "top_k": top_k,
        "document_group": (query_params.get("document_group") or "").strip(),
        "file_type": (query_params.get("file_type") or "").strip(),
        "chunk_policy_name": (query_params.get("chunk_policy_name") or "").strip(),
        "profiles": profiles,
        "profile_selection_explicit": bool(query_params.getlist("profiles")),
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
        "permission_summary": permission_summary_from_metadata(
            search_log.permission_filter_metadata
        ),
        "document_group": search_log.document_group,
        "file_type": search_log.file_type,
        "chunk_policy_name": search_log.chunk_policy_name,
        "top_k": search_log.top_k,
        "similarity_metric": search_log.similarity_metric,
        "profiles": list(search_log.profiles),
        "query_runtime_metadata": search_log.query_runtime_metadata,
        "reproducibility_summary": search_reproducibility_payload(search_log),
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
        "review_tags": list(search_log.review_tags),
        "review_memo": search_log.review_memo,
        "reviewed_by_user_id": search_log.reviewed_by_user_id,
        "reviewed_at": _datetime_response(search_log.reviewed_at),
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


def search_log_export_payload(detail: SearchLogDetailRecord) -> dict[str, object]:
    payload = search_log_detail_payload(detail)
    return {
        "version": SEARCH_LOG_EXPORT_VERSION,
        "exported_at": _datetime_response(datetime.now(UTC)),
        **payload,
    }


def search_log_results_csv(detail: SearchLogDetailRecord) -> str:
    fieldnames = [
        "search_log_id",
        "query_text",
        "actor_login_id",
        "requested_search_scope",
        "effective_search_scope",
        "document_group",
        "file_type",
        "chunk_policy_name",
        "top_k",
        "profile_name",
        "rank",
        "search_log_result_id",
        "chunk_id",
        "document_id",
        "file_id",
        "distance",
        "score",
        "profile_elapsed_ms",
        "document_title",
        "original_file_name",
        "heading_path",
        "feedback_count",
        "feedback_labels",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for result in detail.results:
        payload = search_log_result_detail_payload(result)
        feedback = payload["feedback"]
        writer.writerow(
            {
                "search_log_id": detail.search_log.search_log_id,
                "query_text": detail.search_log.query_text,
                "actor_login_id": detail.actor_login_id or "",
                "requested_search_scope": detail.search_log.requested_search_scope,
                "effective_search_scope": detail.search_log.effective_search_scope,
                "document_group": detail.search_log.document_group or "",
                "file_type": detail.search_log.file_type or "",
                "chunk_policy_name": detail.search_log.chunk_policy_name or "",
                "top_k": detail.search_log.top_k,
                "profile_name": payload["profile_name"],
                "rank": payload["rank"],
                "search_log_result_id": payload["search_log_result_id"],
                "chunk_id": payload["chunk_id"],
                "document_id": payload["document_id"],
                "file_id": payload["file_id"],
                "distance": payload["distance"],
                "score": payload["score"],
                "profile_elapsed_ms": payload["profile_elapsed_ms"],
                "document_title": payload["document_title"] or "",
                "original_file_name": payload["original_file_name"] or "",
                "heading_path": " / ".join(payload["heading_path"]),
                "feedback_count": len(feedback),
                "feedback_labels": "|".join(item["relevance_label"] for item in feedback),
            }
        )
    return output.getvalue()


def _markdown_text(value: object) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_markdown_text(item) for item in value) or "-"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _markdown_cell(value: object) -> str:
    return _markdown_text(value).replace("\n", " ").replace("|", "\\|")


def _markdown_row(values: list[object]) -> str:
    return "| " + " | ".join(_markdown_cell(value) for value in values) + " |"


def _markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    return [
        _markdown_row(headers),
        _markdown_row(["---"] * len(headers)),
        *[_markdown_row(row) for row in rows],
    ]


def _search_log_feedback_counts(detail: SearchLogDetailRecord) -> dict[str, int]:
    counts = {
        "correct": 0,
        "partial": 0,
        "wrong": 0,
        "duplicate": 0,
        "insufficient_context": 0,
    }
    for result in detail.results:
        for feedback in result.feedback:
            counts[feedback.relevance_label] = counts.get(feedback.relevance_label, 0) + 1
    return counts


def search_experiment_report_markdown(
    detail: SearchLogDetailRecord,
    comparison: dict[str, object] | None = None,
) -> str:
    search_log = detail.search_log
    reproducibility = search_reproducibility_payload(search_log)
    permission_summary = permission_summary_from_metadata(search_log.permission_filter_metadata)
    feedback_counts = _search_log_feedback_counts(detail)
    exported_at = _datetime_response(datetime.now(UTC))
    actor = detail.actor_display_name or detail.actor_login_id or search_log.actor_user_id

    lines = [
        "# Search Experiment Report",
        "",
        *[
            f"- Report Version: {SEARCH_EXPERIMENT_REPORT_VERSION}",
            f"- Exported At: {exported_at}",
            f"- Search Log ID: {search_log.search_log_id}",
        ],
        "",
        "## Experiment Conditions",
        "",
        *_markdown_table(
            ["Field", "Value"],
            [
                ["Query", search_log.query_text],
                ["Normalized Query", search_log.normalized_query_text],
                ["Actor", actor],
                ["Requested Scope", search_log.requested_search_scope],
                ["Effective Scope", search_log.effective_search_scope],
                ["Document Group", search_log.document_group],
                ["File Type", search_log.file_type],
                ["Chunk Policy", search_log.chunk_policy_name],
                ["Top K", search_log.top_k],
                ["Similarity Metric", search_log.similarity_metric],
                ["Profiles", list(search_log.profiles)],
                ["Fingerprint", reproducibility["fingerprint"]],
                ["Total Elapsed ms", search_log.total_elapsed_ms],
                ["Runtime Metadata", search_log.query_runtime_metadata],
            ],
        ),
        "",
        "## Permission Summary",
        "",
    ]
    if permission_summary:
        lines.extend(
            _markdown_table(
                ["Field", "Value"],
                [
                    ["Actor Login", permission_summary.get("actor_login_id")],
                    ["Role", permission_summary.get("role_name")],
                    ["Primary Org", permission_summary.get("primary_org_unit_name")],
                    ["Candidate Documents", permission_summary.get("candidate_document_count")],
                    ["Visible Documents", permission_summary.get("visible_document_count")],
                    ["Excluded Documents", permission_summary.get("excluded_document_count")],
                    ["Included Access Scopes", permission_summary.get("included_access_scopes")],
                ],
            )
        )
    else:
        lines.append("_No permission summary recorded._")

    lines.extend(
        [
            "",
            "## Review Metadata",
            "",
            *_markdown_table(
                ["Field", "Value"],
                [
                    ["Review Tags", list(search_log.review_tags)],
                    ["Review Memo", search_log.review_memo],
                    ["Reviewed By User ID", search_log.reviewed_by_user_id],
                    ["Reviewed At", _datetime_response(search_log.reviewed_at)],
                ],
            ),
            "",
            "## Feedback Summary",
            "",
            *_markdown_table(
                ["Label", "Count"],
                [[label, count] for label, count in feedback_counts.items()],
            ),
            "",
            "## Results",
            "",
        ]
    )
    if detail.results:
        lines.extend(
            _markdown_table(
                ["Rank", "Profile", "Chunk ID", "Document", "Score", "Feedback"],
                [
                    [
                        result.search_log_result.rank,
                        result.search_log_result.profile_name,
                        result.search_log_result.chunk_id,
                        result.document_title or result.original_file_name,
                        result.search_log_result.score,
                        [feedback.relevance_label for feedback in result.feedback],
                    ]
                    for result in detail.results
                ],
            )
        )
    else:
        lines.append("_No result rows recorded._")

    if comparison is not None:
        result_overlap = comparison["result_overlap"]
        reproducibility_compare = comparison["reproducibility"]
        right_log = comparison["right"]
        lines.extend(
            [
                "",
                "## Compare Summary",
                "",
                *_markdown_table(
                    ["Field", "Value"],
                    [
                        ["Target Search Log ID", f"#{right_log['search_log_id']}"],
                        ["Same Fingerprint", reproducibility_compare["same_fingerprint"]],
                        ["Left Fingerprint", reproducibility_compare["left_fingerprint"]],
                        ["Right Fingerprint", reproducibility_compare["right_fingerprint"]],
                        ["Left Result Count", result_overlap["left_result_count"]],
                        ["Right Result Count", result_overlap["right_result_count"]],
                        ["Shared Chunks", result_overlap["shared_chunk_count"]],
                        ["Left Only Chunks", result_overlap["left_only_chunk_count"]],
                        ["Right Only Chunks", result_overlap["right_only_chunk_count"]],
                    ],
                ),
                "",
                "## Reproducibility Field Compare",
                "",
                *_markdown_table(
                    [
                        "Field",
                        f"#{comparison['left']['search_log_id']}",
                        f"#{right_log['search_log_id']}",
                        "Match",
                    ],
                    [
                        [
                            field["field"],
                            field["left"],
                            field["right"],
                            field["matches"],
                        ]
                        for field in reproducibility_compare["fields"]
                    ],
                ),
            ]
        )

    return "\n".join(lines) + "\n"


def _stable_compare_value(value: object) -> object:
    if isinstance(value, tuple):
        return list(value)
    return value


def search_log_compare_field_payload(
    field_name: str,
    left_value: object,
    right_value: object,
) -> dict[str, object]:
    left_stable = _stable_compare_value(left_value)
    right_stable = _stable_compare_value(right_value)
    return {
        "field": field_name,
        "left": left_stable,
        "right": right_stable,
        "matches": left_stable == right_stable,
    }


def search_log_comparison_payload(
    left: SearchLogDetailRecord,
    right: SearchLogDetailRecord,
) -> dict[str, object]:
    left_summary = search_reproducibility_payload(left.search_log)
    right_summary = search_reproducibility_payload(right.search_log)
    compare_fields = [
        "query_text",
        "normalized_query_text",
        "actor_user_id",
        "requested_search_scope",
        "effective_search_scope",
        "document_group",
        "file_type",
        "chunk_policy_name",
        "top_k",
        "similarity_metric",
        "profiles",
        "query_runtime_metadata",
    ]
    left_chunks = {result.search_log_result.chunk_id for result in left.results}
    right_chunks = {result.search_log_result.chunk_id for result in right.results}
    shared_chunks = left_chunks & right_chunks
    return {
        "left": search_log_record_payload(
            SearchLogListItem(
                search_log=left.search_log,
                actor_login_id=left.actor_login_id,
                actor_display_name=left.actor_display_name,
                result_count=len(left.results),
                feedback_count=sum(len(result.feedback) for result in left.results),
                correct_count=0,
                partial_count=0,
                wrong_count=0,
                duplicate_count=0,
                insufficient_context_count=0,
                latest_feedback_at=None,
            )
        ),
        "right": search_log_record_payload(
            SearchLogListItem(
                search_log=right.search_log,
                actor_login_id=right.actor_login_id,
                actor_display_name=right.actor_display_name,
                result_count=len(right.results),
                feedback_count=sum(len(result.feedback) for result in right.results),
                correct_count=0,
                partial_count=0,
                wrong_count=0,
                duplicate_count=0,
                insufficient_context_count=0,
                latest_feedback_at=None,
            )
        ),
        "reproducibility": {
            "same_fingerprint": left_summary["fingerprint"] == right_summary["fingerprint"],
            "left_fingerprint": left_summary["fingerprint"],
            "right_fingerprint": right_summary["fingerprint"],
            "fields": [
                search_log_compare_field_payload(
                    field_name,
                    left_summary[field_name],
                    right_summary[field_name],
                )
                for field_name in compare_fields
            ],
        },
        "result_overlap": {
            "left_result_count": len(left.results),
            "right_result_count": len(right.results),
            "left_unique_chunk_count": len(left_chunks),
            "right_unique_chunk_count": len(right_chunks),
            "shared_chunk_count": len(shared_chunks),
            "left_only_chunk_count": len(left_chunks - right_chunks),
            "right_only_chunk_count": len(right_chunks - left_chunks),
            "shared_chunk_ids": sorted(shared_chunks),
        },
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


def search_feedback_comment_payload(comment: SearchFeedbackCommentRecord) -> dict[str, object]:
    return {
        "feedback_id": comment.feedback_id,
        "search_log_result_id": comment.search_log_result_id,
        "search_log_id": comment.search_log_id,
        "query_text": comment.query_text,
        "document_group": comment.document_group,
        "actor_login_id": comment.actor_login_id,
        "actor_display_name": comment.actor_display_name,
        "profile_name": comment.profile_name,
        "rank": comment.rank,
        "chunk_id": comment.chunk_id,
        "document_title": comment.document_title,
        "original_file_name": comment.original_file_name,
        "relevance_label": comment.relevance_label,
        "comment": comment.comment,
        "created_by_user_id": comment.created_by_user_id,
        "created_at": _datetime_response(comment.created_at),
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


def golden_question_batch_promotion_input_from_request(
    payload: GoldenQuestionBatchPromotionRequest,
) -> GoldenQuestionBatchPromotionInput:
    return GoldenQuestionBatchPromotionInput(
        question_set_id=payload.question_set_id,
        search_log_result_ids=tuple(payload.search_log_result_ids),
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


def golden_question_candidate_payload(
    candidate: GoldenQuestionCandidateRecord,
) -> dict[str, object]:
    return {
        "search_log_result_id": candidate.search_log_result_id,
        "search_log_id": candidate.search_log_id,
        "query_text": candidate.query_text,
        "actor_user_id": candidate.actor_user_id,
        "actor_login_id": candidate.actor_login_id,
        "actor_display_name": candidate.actor_display_name,
        "requested_search_scope": candidate.requested_search_scope,
        "document_group": candidate.document_group,
        "file_type": candidate.file_type,
        "chunk_policy_name": candidate.chunk_policy_name,
        "top_k": candidate.top_k,
        "profile_name": candidate.profile_name,
        "rank": candidate.rank,
        "chunk_id": candidate.chunk_id,
        "score": candidate.score,
        "document_id": candidate.document_id,
        "document_title": candidate.document_title,
        "original_file_name": candidate.original_file_name,
        "heading_path": list(candidate.heading_path),
        "chunk_preview": candidate.chunk_preview,
        "feedback_count": candidate.feedback_count,
        "correct_count": candidate.correct_count,
        "partial_count": candidate.partial_count,
        "feedback_labels": list(candidate.feedback_labels),
        "latest_feedback_comment": candidate.latest_feedback_comment,
        "latest_feedback_at": _datetime_response(candidate.latest_feedback_at),
        "already_promoted": candidate.already_promoted,
    }


def golden_question_batch_promotion_payload(
    batch: GoldenQuestionBatchPromotionRecord,
) -> dict[str, object]:
    return {
        "promoted_count": len(batch.promotions),
        "skipped_count": len(batch.skipped_search_log_result_ids),
        "missing_count": len(batch.missing_search_log_result_ids),
        "promotions": [
            golden_question_promotion_payload(promotion) for promotion in batch.promotions
        ],
        "skipped_search_log_result_ids": list(batch.skipped_search_log_result_ids),
        "missing_search_log_result_ids": list(batch.missing_search_log_result_ids),
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


def evaluation_run_export_payload(
    run: EvaluationRunRecord,
    question_set: GoldenQuestionSetRecord | None,
    results: list[EvaluationResultRecord],
) -> dict[str, object]:
    return {
        "version": EVALUATION_EXPORT_VERSION,
        "run": evaluation_run_payload(run),
        "question_set": (
            golden_question_set_payload(question_set) if question_set is not None else None
        ),
        "results": [evaluation_result_payload(result) for result in results],
    }


def evaluation_results_csv(
    run: EvaluationRunRecord,
    question_set: GoldenQuestionSetRecord | None,
    results: list[EvaluationResultRecord],
) -> str:
    fieldnames = [
        "evaluation_run_id",
        "run_name",
        "question_set_id",
        "question_set_name",
        "profile_name",
        "chunk_policy_name",
        "status",
        "question_id",
        "search_log_id",
        "top_k",
        "visible_expected_count",
        "retrieved_count",
        "matched_visible_count",
        "hidden_violation_count",
        "matched_chunk_ids",
        "hidden_violation_chunk_ids",
        "recall_at_k",
        "reciprocal_rank",
        "ndcg",
        "no_answer_success",
        "result_created_at",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow(
            {
                "evaluation_run_id": run.evaluation_run_id,
                "run_name": run.run_name,
                "question_set_id": run.question_set_id,
                "question_set_name": question_set.set_name if question_set is not None else "",
                "profile_name": run.profile_name,
                "chunk_policy_name": run.chunk_policy_name or "",
                "status": run.status,
                "question_id": result.question_id,
                "search_log_id": result.search_log_id or "",
                "top_k": result.top_k,
                "visible_expected_count": result.visible_expected_count,
                "retrieved_count": result.retrieved_count,
                "matched_visible_count": result.matched_visible_count,
                "hidden_violation_count": result.hidden_violation_count,
                "matched_chunk_ids": ",".join(
                    str(chunk_id) for chunk_id in result.matched_chunk_ids
                ),
                "hidden_violation_chunk_ids": ",".join(
                    str(chunk_id) for chunk_id in result.hidden_violation_chunk_ids
                ),
                "recall_at_k": result.recall_at_k if result.recall_at_k is not None else "",
                "reciprocal_rank": (
                    result.reciprocal_rank if result.reciprocal_rank is not None else ""
                ),
                "ndcg": result.ndcg if result.ndcg is not None else "",
                "no_answer_success": (
                    result.no_answer_success if result.no_answer_success is not None else ""
                ),
                "result_created_at": _datetime_response(result.created_at),
            }
        )
    return output.getvalue()


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


def evaluation_permission_audit_payload(
    audit: EvaluationPermissionAuditRecord,
) -> dict[str, object]:
    return {
        "evaluation_result_id": audit.evaluation_result_id,
        "evaluation_run_id": audit.evaluation_run_id,
        "question_id": audit.question_id,
        "question_text": audit.question_text,
        "actor_user_id": audit.actor_user_id,
        "actor_login_id": audit.actor_login_id,
        "actor_display_name": audit.actor_display_name,
        "requested_search_scope": audit.requested_search_scope,
        "effective_search_scope": audit.effective_search_scope,
        "permission_filter_metadata": audit.permission_filter_metadata,
        "search_log_id": audit.search_log_id,
        "top_k": audit.top_k,
        "retrieved_count": audit.retrieved_count,
        "visible_expected_count": audit.visible_expected_count,
        "matched_visible_count": audit.matched_visible_count,
        "hidden_violation_count": audit.hidden_violation_count,
        "matched_chunk_ids": list(audit.matched_chunk_ids),
        "hidden_violation_chunk_ids": list(audit.hidden_violation_chunk_ids),
        "no_answer_success": audit.no_answer_success,
        "permission_status": "violation" if audit.hidden_violation_count else "clean",
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


def chunk_policy_summary_payload(policy: ChunkPolicySummaryRecord) -> dict[str, object]:
    return {
        "chunk_policy_name": policy.chunk_policy_name,
        "target_token_size": policy.target_token_size,
        "overlap_token_size": policy.overlap_token_size,
        "split_strategy": policy.split_strategy,
        "preserve_table": policy.preserve_table,
        "preserve_code_block": policy.preserve_code_block,
        "description": policy.description,
        "is_default": policy.is_default,
        "chunk_count": policy.chunk_count,
        "document_count": policy.document_count,
        "total_token_count": policy.total_token_count,
        "total_char_count": policy.total_char_count,
        "average_token_count": (
            str(policy.average_token_count) if policy.average_token_count is not None else None
        ),
        "average_char_count": (
            str(policy.average_char_count) if policy.average_char_count is not None else None
        ),
        "created_at": _datetime_response(policy.created_at),
    }


def permission_inventory_summary_payload(
    summary: PermissionInventorySummary,
) -> dict[str, object]:
    return {
        "active_user_count": summary.active_user_count,
        "inactive_user_count": summary.inactive_user_count,
        "org_unit_count": summary.org_unit_count,
        "active_org_unit_count": summary.active_org_unit_count,
        "membership_count": summary.membership_count,
        "document_count": summary.document_count,
        "access_scope_counts": [
            {
                "access_scope": access_scope.access_scope,
                "document_count": access_scope.document_count,
            }
            for access_scope in summary.access_scope_counts
        ],
    }


def permission_readiness_issue_payload(
    issue: PermissionReadinessIssueRecord,
) -> dict[str, object]:
    return {
        "document_id": issue.document_id,
        "file_id": issue.file_id,
        "document_title": issue.document_title,
        "original_file_name": issue.original_file_name,
        "access_scope": issue.access_scope,
        "issue_codes": list(issue.issue_codes),
    }


def permission_readiness_summary_payload(
    summary: PermissionReadinessSummary,
) -> dict[str, object]:
    return {
        "document_count": summary.document_count,
        "ready_document_count": summary.ready_document_count,
        "issue_document_count": summary.issue_document_count,
        "missing_uploader_count": summary.missing_uploader_count,
        "personal_missing_owner_count": summary.personal_missing_owner_count,
        "scoped_missing_org_count": summary.scoped_missing_org_count,
        "readiness_percent": summary.readiness_percent,
        "issues": [permission_readiness_issue_payload(issue) for issue in summary.issues],
    }


def permission_user_inventory_payload(
    user: PermissionUserInventoryRecord,
) -> dict[str, object]:
    return {
        "user_id": user.user_id,
        "login_id": user.login_id,
        "display_name": user.display_name,
        "email": user.email,
        "is_active": user.is_active,
        "primary_role_name": user.primary_role_name,
        "primary_org_unit_id": user.primary_org_unit_id,
        "primary_org_unit_name": user.primary_org_unit_name,
        "primary_org_unit_type": user.primary_org_unit_type,
        "membership_count": user.membership_count,
        "uploaded_file_count": user.uploaded_file_count,
        "owned_document_count": user.owned_document_count,
        "managed_org_unit_count": user.managed_org_unit_count,
        "ancestor_org_unit_count": user.ancestor_org_unit_count,
        "created_at": _datetime_response(user.created_at),
        "updated_at": _datetime_response(user.updated_at),
    }


def permission_org_unit_inventory_payload(
    org_unit: PermissionOrgUnitInventoryRecord,
) -> dict[str, object]:
    return {
        "org_unit_id": org_unit.org_unit_id,
        "parent_org_unit_id": org_unit.parent_org_unit_id,
        "parent_org_unit_name": org_unit.parent_org_unit_name,
        "org_unit_name": org_unit.org_unit_name,
        "org_unit_type": org_unit.org_unit_type,
        "is_active": org_unit.is_active,
        "depth": org_unit.depth,
        "org_path": org_unit.org_path,
        "membership_count": org_unit.membership_count,
        "primary_membership_count": org_unit.primary_membership_count,
        "owned_document_count": org_unit.owned_document_count,
        "child_org_unit_count": org_unit.child_org_unit_count,
        "created_at": _datetime_response(org_unit.created_at),
        "updated_at": _datetime_response(org_unit.updated_at),
    }


def permission_membership_inventory_payload(
    membership: PermissionMembershipInventoryRecord,
) -> dict[str, object]:
    return {
        "membership_id": membership.membership_id,
        "user_id": membership.user_id,
        "login_id": membership.login_id,
        "display_name": membership.display_name,
        "org_unit_id": membership.org_unit_id,
        "org_unit_name": membership.org_unit_name,
        "org_unit_type": membership.org_unit_type,
        "role_name": membership.role_name,
        "is_primary": membership.is_primary,
        "created_at": _datetime_response(membership.created_at),
        "updated_at": _datetime_response(membership.updated_at),
    }


def permission_inventory_payload(inventory: PermissionInventory) -> dict[str, object]:
    return {
        "summary": permission_inventory_summary_payload(inventory.summary),
        "users": [permission_user_inventory_payload(user) for user in inventory.users],
        "org_units": [
            permission_org_unit_inventory_payload(org_unit) for org_unit in inventory.org_units
        ],
        "memberships": [
            permission_membership_inventory_payload(membership)
            for membership in inventory.memberships
        ],
    }


def document_permission_update_input_from_request(
    payload: DocumentPermissionUpdateRequest,
) -> DocumentPermissionUpdateInput:
    return DocumentPermissionUpdateInput(
        owner_user_id=payload.owner_user_id,
        owner_org_unit_id=payload.owner_org_unit_id,
        access_scope=payload.access_scope,
        updated_by_user_id=payload.updated_by_user_id,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version)
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

    @app.middleware("http")
    async def language_middleware(request: Request, call_next):
        language = resolve_language(
            query_language=request.query_params.get("lang"),
            cookie_language=request.cookies.get(LANGUAGE_COOKIE_NAME),
        )
        request.state.language = language
        response = await call_next(request)
        try:
            query_language = normalize_language(request.query_params.get("lang"))
        except ValueError:
            query_language = None
        if query_language is not None:
            response.set_cookie(
                LANGUAGE_COOKIE_NAME,
                query_language,
                max_age=60 * 60 * 24 * 365,
                samesite="lax",
            )
        return response

    def template_context(request: Request, **context: object) -> dict[str, object]:
        language = getattr(
            request.state,
            "language",
            resolve_language(
                query_language=request.query_params.get("lang"),
                cookie_language=request.cookies.get(LANGUAGE_COOKIE_NAME),
            ),
        )
        translator = get_translator(language)

        def language_url(language_code: str) -> str:
            query_items = [
                (key, value) for key, value in request.query_params.multi_items() if key != "lang"
            ]
            query_items.append(("lang", language_code))
            return f"{request.url.path}?{urlencode(query_items)}"

        return {
            "request": request,
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "environment": settings.environment,
            "current_language": language,
            "language_options": LANGUAGE_OPTIONS,
            "language_url": language_url,
            "t": translator,
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

    @app.get("/api/chunk-policies")
    def api_list_chunk_policies() -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        policies = list_chunk_policy_summaries(settings.database_url)
        return JSONResponse(
            content={
                "default_chunk_policy_name": DEFAULT_CHUNK_POLICY_NAME,
                "chunk_policies": [chunk_policy_summary_payload(policy) for policy in policies],
            },
        )

    @app.get("/api/chunk-policies/{chunk_policy_name}")
    def api_get_chunk_policy(chunk_policy_name: str) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            policy = get_chunk_policy_summary(settings.database_url, chunk_policy_name)
        except InvalidChunkPolicyManagementError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if policy is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chunk policy not found.",
            )

        return JSONResponse(content={"chunk_policy": chunk_policy_summary_payload(policy)})

    @app.get("/api/admin/permissions")
    def api_get_permission_inventory() -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            inventory = get_permission_inventory(settings.database_url)
        except InvalidPermissionInventoryError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={"permission_inventory": permission_inventory_payload(inventory)}
        )

    @app.get("/api/admin/permissions/readiness")
    def api_get_permission_readiness(issue_limit: int = 20) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            readiness = get_permission_readiness_summary(
                settings.database_url,
                issue_limit=issue_limit,
            )
        except InvalidPermissionInventoryError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={"permission_readiness": permission_readiness_summary_payload(readiness)}
        )

    @app.get("/api/admin/permissions/users")
    def api_list_permission_users(limit: int = 100) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            users = list_permission_users(settings.database_url, limit=limit)
        except InvalidPermissionInventoryError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={"users": [permission_user_inventory_payload(user) for user in users]},
        )

    @app.get("/api/admin/permissions/org-units")
    def api_list_permission_org_units(limit: int = 100) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            org_units = list_permission_org_units(settings.database_url, limit=limit)
        except InvalidPermissionInventoryError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "org_units": [
                    permission_org_unit_inventory_payload(org_unit) for org_unit in org_units
                ],
            },
        )

    @app.get("/api/admin/permissions/memberships")
    def api_list_permission_memberships(limit: int = 200) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            memberships = list_permission_memberships(settings.database_url, limit=limit)
        except InvalidPermissionInventoryError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "memberships": [
                    permission_membership_inventory_payload(membership)
                    for membership in memberships
                ],
            },
        )

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

    @app.put("/api/documents/{document_id}/permissions")
    def api_update_document_permission(
        document_id: int,
        payload: DocumentPermissionUpdateRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            document = update_document_permission(
                settings.database_url,
                document_id,
                document_permission_update_input_from_request(payload),
            )
        except InvalidDocumentInventoryError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found.",
            )

        return JSONResponse(content={"document": document_inventory_item_payload(document)})

    def upload_template_context(
        request: Request,
        *,
        result: dict[str, object] | None = None,
        duplicate: bool = False,
        error_message: str | None = None,
        form_values: dict[str, str] | None = None,
        permission_users: tuple[PermissionUserInventoryRecord, ...] = (),
        permission_org_units: tuple[PermissionOrgUnitInventoryRecord, ...] = (),
    ) -> dict[str, object]:
        return template_context(
            request,
            database_configured=bool(settings.database_url),
            supported_file_extensions=sorted(SUPPORTED_FILE_EXTENSIONS),
            access_scope_options=tuple(
                scope
                for scope in ("personal", "team", "org_tree", "company")
                if scope in DOCUMENT_ACCESS_SCOPES
            ),
            permission_user_options=[
                upload_permission_user_option_payload(user) for user in permission_users
            ],
            permission_org_unit_options=[
                upload_permission_org_unit_option_payload(org_unit)
                for org_unit in permission_org_units
            ],
            result=result,
            duplicate=duplicate,
            error_message=error_message,
            form_values=form_values
            or {
                "document_group": "default",
                "security_level": "internal",
                "uploaded_by": "",
                "uploaded_by_user_id": "",
                "owner_user_id": "",
                "owner_org_unit_id": "",
                "access_scope": "personal",
            },
        )

    @app.post("/api/files")
    async def api_upload_file(
        file: UploadFile = UPLOAD_FILE_FORM,
        document_group: str = DOCUMENT_GROUP_FORM,
        security_level: str = SECURITY_LEVEL_FORM,
        uploaded_by: str | None = UPLOADED_BY_FORM,
        uploaded_by_user_id: str | None = UPLOADED_BY_USER_ID_FORM,
        owner_user_id: str | None = OWNER_USER_ID_FORM,
        owner_org_unit_id: str | None = OWNER_ORG_UNIT_ID_FORM,
        access_scope: str = ACCESS_SCOPE_FORM,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            access_scope_value = access_scope.strip() or "personal"
            result = store_upload(
                database_url=settings.database_url,
                upload_stream=file.file,
                original_file_name=file.filename,
                storage_dir=settings.upload_storage_dir,
                mime_type=file.content_type,
                document_group=document_group.strip() or "default",
                security_level=security_level.strip() or "internal",
                uploaded_by=uploaded_by.strip() if uploaded_by and uploaded_by.strip() else None,
                uploaded_by_user_id=parse_optional_positive_int_form(
                    uploaded_by_user_id,
                    "uploaded_by_user_id",
                ),
                owner_user_id=parse_optional_positive_int_form(
                    owner_user_id,
                    "owner_user_id",
                ),
                owner_org_unit_id=parse_optional_positive_int_form(
                    owner_org_unit_id,
                    "owner_org_unit_id",
                ),
                access_scope=access_scope_value,
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

    @app.get("/api/embedding/models/readiness")
    def api_embedding_model_readiness() -> JSONResponse:
        readiness = audit_embedding_model_readiness(settings.embedding_models_dir)
        ready_count = sum(1 for item in readiness if item.ready)
        return JSONResponse(
            content={
                "models_dir": str(settings.embedding_models_dir),
                "ready_count": ready_count,
                "model_count": len(readiness),
                "models": [embedding_model_readiness_payload(item) for item in readiness],
            }
        )

    @app.get("/api/embedding/providers/health")
    def api_embedding_provider_health() -> JSONResponse:
        provider_health = get_embedding_provider_health_status(settings)
        return JSONResponse(
            status_code=provider_health.status_code,
            content=provider_health.payload,
        )

    @app.get("/api/admin/embedding-provider-routes/contract-sample-sets")
    def api_list_embedding_provider_contract_sample_sets(
        active_only: bool = False,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        sample_sets = list_embedding_provider_contract_sample_sets(
            settings.database_url,
            active_only=active_only,
        )
        default_sample_set = next(
            (sample_set for sample_set in sample_sets if sample_set.is_default),
            None,
        )
        return JSONResponse(
            content={
                "sample_set_count": len(sample_sets),
                "default_sample_set": (
                    embedding_provider_contract_sample_set_payload(default_sample_set)
                    if default_sample_set is not None
                    else None
                ),
                "sample_sets": [
                    embedding_provider_contract_sample_set_payload(sample_set)
                    for sample_set in sample_sets
                ],
            }
        )

    @app.post("/api/admin/embedding-provider-routes/contract-sample-sets")
    def api_save_embedding_provider_contract_sample_set(
        payload: EmbeddingProviderContractSampleSetRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            sample_set = upsert_embedding_provider_contract_sample_set(
                settings.database_url,
                embedding_provider_contract_sample_set_input_from_request(payload),
            )
        except InvalidEmbeddingProviderContractSampleSetError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse(
            content={"sample_set": embedding_provider_contract_sample_set_payload(sample_set)}
        )

    @app.get("/api/admin/embedding-provider-routes/contract-sample-sets/{sample_set_name}")
    def api_get_embedding_provider_contract_sample_set(sample_set_name: str) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            sample_set = get_embedding_provider_contract_sample_set(
                settings.database_url,
                sample_set_name,
            )
        except InvalidEmbeddingProviderContractSampleSetError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if sample_set is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Embedding provider contract sample set not found.",
            )
        return JSONResponse(
            content={"sample_set": embedding_provider_contract_sample_set_payload(sample_set)}
        )

    @app.put("/api/admin/embedding-provider-routes/contract-sample-sets/{sample_set_name}")
    def api_update_embedding_provider_contract_sample_set(
        sample_set_name: str,
        payload: EmbeddingProviderContractSampleSetRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            existing = get_embedding_provider_contract_sample_set(
                settings.database_url,
                sample_set_name,
            )
            if existing is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Embedding provider contract sample set not found.",
                )
            sample_set = upsert_embedding_provider_contract_sample_set(
                settings.database_url,
                embedding_provider_contract_sample_set_input_from_request(
                    payload,
                    sample_set_name=sample_set_name,
                ),
            )
        except InvalidEmbeddingProviderContractSampleSetError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse(
            content={"sample_set": embedding_provider_contract_sample_set_payload(sample_set)}
        )

    @app.delete("/api/admin/embedding-provider-routes/contract-sample-sets/{sample_set_name}")
    def api_delete_embedding_provider_contract_sample_set(sample_set_name: str) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            deleted_sample_set = delete_embedding_provider_contract_sample_set(
                settings.database_url,
                sample_set_name,
            )
        except InvalidEmbeddingProviderContractSampleSetError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if deleted_sample_set is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Embedding provider contract sample set not found.",
            )
        return JSONResponse(
            content={
                "deleted_sample_set": embedding_provider_contract_sample_set_payload(
                    deleted_sample_set
                )
            }
        )

    @app.get("/api/admin/embedding-provider-routes")
    def api_list_embedding_provider_routes(
        profile_name: str | None = None,
        active_only: bool = False,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            routes = list_embedding_provider_routes(
                settings.database_url,
                profile_name=profile_name,
                active_only=active_only,
            )
        except InvalidEmbeddingProviderRouteError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "routes": [embedding_provider_route_payload(route) for route in routes],
            }
        )

    @app.get("/api/admin/embedding-provider-routes/health")
    def api_embedding_provider_route_health(
        profile_name: str | None = None,
        active_only: bool = True,
        persist: bool = False,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        snapshots: list[EmbeddingProviderRouteHealthSnapshotRecord] = []
        try:
            summary = get_embedding_provider_route_health_summary(
                settings.database_url,
                profile_name=profile_name,
                active_only=active_only,
            )
            if persist:
                snapshots = record_embedding_provider_route_health_summary(
                    settings.database_url,
                    summary,
                )
                for route_health in summary.routes:
                    log_embedding_provider_route_health_alert(
                        settings.database_url,
                        route_health,
                    )
        except (
            InvalidEmbeddingProviderRouteError,
            InvalidEmbeddingProviderRouteHealthSnapshotError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "route_count": summary.route_count,
                "checked_count": summary.checked_count,
                "ready_count": summary.ready_count,
                "snapshot_count": len(snapshots),
                "routes": [
                    embedding_provider_route_health_payload(route_health)
                    for route_health in summary.routes
                ],
                "snapshots": [
                    embedding_provider_route_health_snapshot_payload(snapshot)
                    for snapshot in snapshots
                ],
            }
        )

    @app.get("/api/admin/embedding-provider-routes/health-snapshots")
    def api_list_embedding_provider_route_health_snapshots(
        profile_name: str | None = None,
        route_id: int | None = None,
        limit: int = 20,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            snapshots = list_embedding_provider_route_health_snapshots(
                settings.database_url,
                profile_name=profile_name,
                route_id=route_id,
                limit=limit,
            )
        except InvalidEmbeddingProviderRouteHealthSnapshotError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "snapshot_count": len(snapshots),
                "snapshots": [
                    embedding_provider_route_health_snapshot_payload(snapshot)
                    for snapshot in snapshots
                ],
            }
        )

    @app.get("/api/admin/embedding-provider-routes/contract-snapshots")
    def api_list_embedding_provider_route_contract_snapshots(
        profile_name: str | None = None,
        route_id: int | None = None,
        limit: int = 20,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            snapshots = list_embedding_provider_route_contract_snapshots(
                settings.database_url,
                profile_name=profile_name,
                route_id=route_id,
                limit=limit,
            )
        except InvalidEmbeddingProviderRouteContractSnapshotError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "snapshot_count": len(snapshots),
                "snapshots": [
                    embedding_provider_route_contract_snapshot_payload(snapshot)
                    for snapshot in snapshots
                ],
            }
        )

    @app.get("/api/admin/embedding-provider-routes/readiness")
    def api_embedding_provider_route_readiness(
        profile_name: str | None = None,
        active_only: bool = False,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            readiness = get_embedding_provider_route_readiness_summary(
                settings.database_url,
                profile_name=profile_name,
                active_only=active_only,
            )
        except (
            InvalidEmbeddingProviderRouteError,
            InvalidEmbeddingProviderRouteHealthSnapshotError,
            InvalidEmbeddingProviderRouteContractSnapshotError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content=embedding_provider_route_readiness_summary_payload(readiness))

    @app.get("/api/admin/embedding-provider-routes/operations-summary")
    def api_embedding_provider_route_operations_summary() -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            readiness = get_embedding_provider_route_readiness_summary(settings.database_url)
            schedules = list_embedding_provider_preflight_schedules(settings.database_url)
            due_schedules = list_due_embedding_provider_preflight_schedules(
                settings.database_url,
                limit=100,
            )
            latest_runs = list_embedding_provider_preflight_runs(
                settings.database_url,
                limit=1,
            )
        except (
            InvalidEmbeddingProviderRouteError,
            InvalidEmbeddingProviderRouteHealthSnapshotError,
            InvalidEmbeddingProviderRouteContractSnapshotError,
            InvalidEmbeddingProviderPreflightScheduleError,
            InvalidEmbeddingProviderPreflightRunError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "operations_summary": embedding_provider_route_operations_summary_payload(
                    readiness=readiness,
                    schedules=schedules,
                    due_schedules=due_schedules,
                    latest_run=latest_runs[0] if latest_runs else None,
                )
            }
        )

    @app.get("/api/admin/embedding-provider-routes/alerts")
    def api_list_embedding_provider_route_alerts(
        level: str | None = None,
        acknowledged: str = "false",
        limit: int = 20,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            alerts = list_provider_route_alert_logs(
                settings.database_url,
                level=level,
                acknowledged=parse_acknowledged_filter(acknowledged),
                limit=limit,
            )
        except (InvalidAdminLogError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "alert_count": len(alerts),
                "alerts": [admin_log_payload(alert) for alert in alerts],
            }
        )

    @app.post("/api/admin/embedding-provider-routes/alerts/{log_id}/acknowledge")
    def api_acknowledge_embedding_provider_route_alert(
        log_id: int,
        payload: ProviderRouteAlertAcknowledgeRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            alert = acknowledge_log(
                settings.database_url,
                log_id,
                acknowledged_by=payload.acknowledged_by,
                acknowledgement_note=payload.acknowledgement_note,
            )
        except InvalidAdminLogError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if alert is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Provider route alert not found.",
            )

        return JSONResponse(content={"alert": admin_log_payload(alert)})

    @app.get("/api/admin/embedding-provider-routes/retention-settings")
    def api_get_provider_route_retention_settings() -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        retention_settings = load_provider_route_retention_settings(settings.database_url)
        return JSONResponse(
            content={
                "settings": provider_route_retention_settings_payload(retention_settings),
            }
        )

    @app.put("/api/admin/embedding-provider-routes/retention-settings")
    def api_update_provider_route_retention_settings(
        payload: ProviderRouteRetentionSettingsRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            retention_settings = update_provider_route_retention_settings(
                settings.database_url,
                provider_route_retention_settings_input_from_request(payload),
            )
        except InvalidProviderRouteRetentionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "settings": provider_route_retention_settings_payload(retention_settings),
            }
        )

    @app.post("/api/admin/embedding-provider-routes/cleanup")
    def api_cleanup_provider_route_operational_records(
        payload: ProviderRouteCleanupRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            result = cleanup_expired_provider_route_records(
                settings.database_url,
                dry_run=payload.dry_run,
            )
        except InvalidProviderRouteRetentionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content={"cleanup": provider_route_cleanup_result_payload(result)})

    @app.get("/api/admin/embedding-provider-routes/preflight-schedules")
    def api_list_embedding_provider_preflight_schedules(
        enabled_only: bool = False,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        schedules = list_embedding_provider_preflight_schedules(
            settings.database_url,
            enabled_only=enabled_only,
        )
        return JSONResponse(
            content={
                "schedule_count": len(schedules),
                "schedules": [
                    embedding_provider_preflight_schedule_payload(schedule)
                    for schedule in schedules
                ],
            }
        )

    @app.get("/api/admin/embedding-provider-routes/preflight-schedules/due")
    def api_list_due_embedding_provider_preflight_schedules(
        schedule_name: str | None = None,
        limit: int = 20,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            schedules = list_due_embedding_provider_preflight_schedules(
                settings.database_url,
                schedule_name=schedule_name,
                limit=limit,
            )
        except InvalidEmbeddingProviderPreflightScheduleError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse(
            content={
                "schedule_count": len(schedules),
                "schedules": [
                    embedding_provider_preflight_schedule_payload(schedule)
                    for schedule in schedules
                ],
            }
        )

    @app.post("/api/admin/embedding-provider-routes/preflight-schedules/run-due")
    def api_run_due_embedding_provider_preflight_schedules(
        payload: ProviderPreflightScheduleRunDueRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            runs = run_due_embedding_provider_preflight_schedules(
                settings.database_url,
                schedule_name=payload.schedule_name,
                limit=payload.limit,
            )
        except InvalidEmbeddingProviderPreflightScheduleError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        failed_count = sum(1 for run in runs if run.status != "succeeded")
        return JSONResponse(
            content={
                "run_count": len(runs),
                "failed_count": failed_count,
                "runs": [scheduled_provider_route_preflight_run_payload(run) for run in runs],
            }
        )

    @app.get("/api/admin/embedding-provider-routes/preflight-schedules/{schedule_name}")
    def api_get_embedding_provider_preflight_schedule(schedule_name: str) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            schedule = get_embedding_provider_preflight_schedule(
                settings.database_url,
                schedule_name,
            )
        except InvalidEmbeddingProviderPreflightScheduleError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if schedule is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Provider preflight schedule not found.",
            )
        return JSONResponse(
            content={"schedule": embedding_provider_preflight_schedule_payload(schedule)}
        )

    @app.put("/api/admin/embedding-provider-routes/preflight-schedules/{schedule_name}")
    def api_upsert_embedding_provider_preflight_schedule(
        schedule_name: str,
        payload: ProviderPreflightScheduleRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            schedule = upsert_embedding_provider_preflight_schedule(
                settings.database_url,
                embedding_provider_preflight_schedule_input_from_request(
                    schedule_name,
                    payload,
                ),
            )
        except InvalidEmbeddingProviderPreflightScheduleError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse(
            content={"schedule": embedding_provider_preflight_schedule_payload(schedule)}
        )

    @app.get("/api/admin/embedding-provider-routes/preflight-runs")
    def api_list_embedding_provider_preflight_runs(
        schedule_name: str | None = None,
        status_filter: str | None = Query(default=None, alias="status"),
        limit: int = 10,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            runs = list_embedding_provider_preflight_runs(
                settings.database_url,
                schedule_name=schedule_name,
                status=status_filter,
                limit=limit,
            )
        except InvalidEmbeddingProviderPreflightRunError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse(
            content={
                "run_count": len(runs),
                "runs": [embedding_provider_preflight_run_payload(run) for run in runs],
            }
        )

    @app.post("/api/admin/embedding-provider-routes/preflight")
    def api_preflight_embedding_provider_routes(
        profile_name: str | None = None,
        active_only: bool = True,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        started_at = datetime.now(UTC)
        started_perf = perf_counter()
        try:
            sample_set = get_default_embedding_provider_contract_sample_set(settings.database_url)
            routes = list_embedding_provider_routes(
                settings.database_url,
                profile_name=profile_name,
                active_only=active_only,
            )
            results = []
            for route in routes:
                contract = check_embedding_provider_route_contract(
                    route,
                    sample_texts=sample_set.sample_texts,
                    input_type=sample_set.input_type,
                    sample_set_name=sample_set.sample_set_name,
                )
                health_snapshot = None
                if contract.health is not None:
                    health_snapshot = record_embedding_provider_route_health_snapshot(
                        settings.database_url,
                        contract.health,
                    )
                    log_embedding_provider_route_health_alert(
                        settings.database_url, contract.health
                    )
                contract_snapshot = record_embedding_provider_route_contract_snapshot(
                    settings.database_url,
                    contract,
                )
                log_embedding_provider_route_contract_alert(settings.database_url, contract)
                results.append(
                    {
                        "route": embedding_provider_route_payload(route),
                        "health": (
                            embedding_provider_route_health_payload(contract.health)
                            if contract.health is not None
                            else None
                        ),
                        "health_snapshot": (
                            embedding_provider_route_health_snapshot_payload(health_snapshot)
                            if health_snapshot is not None
                            else None
                        ),
                        "contract": embedding_provider_route_contract_payload(contract),
                        "contract_snapshot": embedding_provider_route_contract_snapshot_payload(
                            contract_snapshot,
                        ),
                    }
                )
        except (
            InvalidEmbeddingProviderContractSampleSetError,
            InvalidEmbeddingProviderRouteError,
            InvalidEmbeddingProviderRouteHealthSnapshotError,
            InvalidEmbeddingProviderRouteContractSnapshotError,
            InvalidEmbeddingProviderPreflightRunError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        passed_count = sum(1 for result in results if result["contract"]["passed"])
        response_content = {
            "route_count": len(routes),
            "passed_count": passed_count,
            "failed_count": len(routes) - passed_count,
            "sample_set": embedding_provider_contract_sample_set_payload(sample_set),
            "results": results,
        }
        completed_at = datetime.now(UTC)
        elapsed_ms = int((perf_counter() - started_perf) * 1000)
        try:
            preflight_run = record_embedding_provider_preflight_run(
                settings.database_url,
                EmbeddingProviderPreflightRunInput(
                    trigger_source="manual_api",
                    status="succeeded" if response_content["failed_count"] == 0 else "failed",
                    result=response_content,
                    profile_name=profile_name,
                    active_only=active_only,
                    elapsed_ms=elapsed_ms,
                    started_at=started_at,
                    completed_at=completed_at,
                ),
            )
        except InvalidEmbeddingProviderPreflightRunError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        response_content["preflight_run"] = embedding_provider_preflight_run_payload(preflight_run)
        return JSONResponse(content=response_content)

    @app.post("/api/admin/embedding-provider-routes/{route_id}/health-check")
    def api_check_embedding_provider_route_health(route_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            route = get_embedding_provider_route(settings.database_url, route_id)
            if route is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Embedding provider route not found.",
                )
            route_health = check_embedding_provider_route_health(route)
            snapshot = record_embedding_provider_route_health_snapshot(
                settings.database_url,
                route_health,
            )
            log_embedding_provider_route_health_alert(settings.database_url, route_health)
        except (
            InvalidEmbeddingProviderRouteError,
            InvalidEmbeddingProviderRouteHealthSnapshotError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "route_health": embedding_provider_route_health_payload(route_health),
                "snapshot": embedding_provider_route_health_snapshot_payload(snapshot),
            }
        )

    @app.post("/api/admin/embedding-provider-routes/{route_id}/contract-check")
    def api_check_embedding_provider_route_contract(route_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            route = get_embedding_provider_route(settings.database_url, route_id)
            if route is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Embedding provider route not found.",
                )
            sample_set = get_default_embedding_provider_contract_sample_set(settings.database_url)
            contract = check_embedding_provider_route_contract(
                route,
                sample_texts=sample_set.sample_texts,
                input_type=sample_set.input_type,
                sample_set_name=sample_set.sample_set_name,
            )
            snapshot = record_embedding_provider_route_contract_snapshot(
                settings.database_url,
                contract,
            )
            log_embedding_provider_route_contract_alert(settings.database_url, contract)
        except (
            InvalidEmbeddingProviderContractSampleSetError,
            InvalidEmbeddingProviderRouteError,
            InvalidEmbeddingProviderRouteContractSnapshotError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "sample_set": embedding_provider_contract_sample_set_payload(sample_set),
                "contract": embedding_provider_route_contract_payload(contract),
                "snapshot": embedding_provider_route_contract_snapshot_payload(snapshot),
            }
        )

    @app.post("/api/admin/embedding-provider-routes")
    def api_upsert_embedding_provider_route(
        payload: EmbeddingProviderRouteRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            route = upsert_embedding_provider_route(
                settings.database_url,
                embedding_provider_route_input_from_request(payload),
            )
        except InvalidEmbeddingProviderRouteError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content={"route": embedding_provider_route_payload(route)})

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

    @app.post("/api/search/permission-matrix")
    def api_search_permission_matrix(payload: SearchPermissionMatrixRequest) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            result = run_permission_search_matrix(
                settings.database_url,
                SearchPermissionMatrixInput(
                    query_text=payload.query_text,
                    entries=tuple(
                        SearchPermissionMatrixEntryInput(
                            actor_user_id=entry.actor_user_id,
                            requested_search_scope=entry.requested_search_scope,
                        )
                        for entry in payload.entries
                    ),
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

        return JSONResponse(content=search_permission_matrix_payload(result))

    @app.get("/api/search/logs")
    def api_list_search_logs(
        actor_user_id: int | None = None,
        requested_search_scope: str | None = None,
        document_group: str | None = None,
        fingerprint: str | None = None,
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
            logs = filter_search_logs_by_fingerprint(logs, fingerprint)
        except InvalidSearchLogError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content={"logs": [search_log_record_payload(log) for log in logs]})

    @app.get("/api/search/logs/compare")
    def api_compare_search_logs(
        left_search_log_id: int = Query(ge=1),
        right_search_log_id: int = Query(ge=1),
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            left = get_search_log_detail(settings.database_url, left_search_log_id)
            right = get_search_log_detail(settings.database_url, right_search_log_id)
        except InvalidSearchLogError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if left is None or right is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Search log not found.",
            )

        return JSONResponse(content=search_log_comparison_payload(left, right))

    @app.get("/api/search/logs/retention-settings")
    def api_get_search_log_retention_settings() -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        retention_settings = load_search_log_retention_settings(settings.database_url)
        return JSONResponse(
            content={
                "settings": search_log_retention_settings_payload(retention_settings),
            },
        )

    @app.put("/api/search/logs/retention-settings")
    def api_update_search_log_retention_settings(
        payload: SearchLogRetentionSettingsRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            retention_settings = update_search_log_retention_settings(
                settings.database_url,
                search_log_retention_settings_input_from_request(payload),
            )
        except InvalidSearchLogError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "settings": search_log_retention_settings_payload(retention_settings),
            },
        )

    @app.post("/api/search/logs/cleanup")
    def api_cleanup_search_logs(payload: SearchLogCleanupRequest) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            result = cleanup_expired_search_logs(settings.database_url, dry_run=payload.dry_run)
        except InvalidSearchLogError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content={"cleanup": search_log_cleanup_result_payload(result)})

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

    @app.patch("/api/search/logs/{search_log_id}/metadata")
    def api_update_search_log_review_metadata(
        search_log_id: int,
        payload: SearchLogReviewMetadataRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            updated = update_search_log_review_metadata(
                settings.database_url,
                SearchLogReviewMetadataInput(
                    search_log_id=search_log_id,
                    review_tags=tuple(payload.review_tags),
                    review_memo=payload.review_memo,
                    reviewed_by_user_id=payload.reviewed_by_user_id,
                ),
            )
            detail = (
                get_search_log_detail(settings.database_url, updated.search_log_id)
                if updated is not None
                else None
            )
        except InvalidSearchLogError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if detail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Search log not found.",
            )

        return JSONResponse(content={"search_log": search_log_detail_payload(detail)["search_log"]})

    @app.get("/api/search/logs/{search_log_id}/export")
    def api_export_search_log(
        search_log_id: int,
        export_format: str = Query(default="json", alias="format"),
    ) -> Response:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        normalized_format = export_format.strip().lower()
        if normalized_format not in {"json", "csv"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="format must be json or csv.",
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

        filename_base = f"search-log-{search_log_id}"
        if normalized_format == "csv":
            return Response(
                content=search_log_results_csv(detail),
                media_type="text/csv; charset=utf-8",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename_base}.csv"',
                },
            )

        return JSONResponse(
            content=search_log_export_payload(detail),
            headers={
                "Content-Disposition": f'attachment; filename="{filename_base}.json"',
            },
        )

    @app.get("/api/search/logs/{search_log_id}/experiment-report")
    def api_export_search_experiment_report(
        search_log_id: int,
        compare_search_log_id: int | None = Query(default=None, ge=1),
    ) -> Response:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            detail = get_search_log_detail(settings.database_url, search_log_id)
            compare_detail = (
                get_search_log_detail(settings.database_url, compare_search_log_id)
                if compare_search_log_id is not None
                else None
            )
        except InvalidSearchLogError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if detail is None or (compare_search_log_id is not None and compare_detail is None):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Search log not found.",
            )

        comparison = (
            search_log_comparison_payload(detail, compare_detail)
            if compare_detail is not None
            else None
        )
        return Response(
            content=search_experiment_report_markdown(detail, comparison),
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="search-log-{search_log_id}-experiment-report.md"'
                ),
            },
        )

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

    @app.get("/api/search/feedback/comments")
    def api_search_feedback_comments(
        document_group: str | None = None,
        limit: int = Query(default=20, ge=1, le=100),
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            comments = list_search_feedback_comments(
                settings.database_url,
                document_group=document_group,
                limit=limit,
            )
        except InvalidSearchLogError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "comments": [search_feedback_comment_payload(comment) for comment in comments],
            },
        )

    @app.get("/api/evaluations/golden-question-candidates")
    def api_list_golden_question_candidates(
        document_group: str | None = None,
        include_promoted: bool = False,
        limit: int = Query(default=20, ge=1, le=100),
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            candidates = list_golden_question_candidates(
                settings.database_url,
                document_group=document_group,
                include_promoted=include_promoted,
                limit=limit,
            )
        except InvalidGoldenQuestionPromotionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "candidates": [
                    golden_question_candidate_payload(candidate) for candidate in candidates
                ],
            },
        )

    @app.post("/api/evaluations/golden-question-candidates/promote")
    def api_promote_golden_question_candidates(
        payload: GoldenQuestionBatchPromotionRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            batch = promote_search_results_to_golden_questions(
                settings.database_url,
                golden_question_batch_promotion_input_from_request(payload),
            )
        except (
            InvalidGoldenQuestionPromotionError,
            InvalidGoldenQuestionError,
            InvalidSearchLogError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"batch": golden_question_batch_promotion_payload(batch)},
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

    @app.get("/api/evaluations/runs/{evaluation_run_id}/permission-audit")
    def api_get_evaluation_permission_audit(
        evaluation_run_id: int,
        limit: int = 500,
    ) -> JSONResponse:
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
            audit = get_evaluation_permission_audit(
                settings.database_url,
                evaluation_run_id,
                limit=limit,
            )
        except InvalidEvaluationRunError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except InvalidEvaluationReportError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "run": evaluation_run_payload(run),
                "audit": [evaluation_permission_audit_payload(item) for item in audit],
            },
        )

    @app.get("/api/evaluations/runs/{evaluation_run_id}/export")
    def api_export_evaluation_run(
        evaluation_run_id: int,
        export_format: str = Query(default="json", alias="format"),
    ) -> Response:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        normalized_format = export_format.strip().lower()
        if normalized_format not in {"json", "csv"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="format must be json or csv.",
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

        filename_base = f"golden-evaluation-run-{run.evaluation_run_id}"
        if normalized_format == "csv":
            return Response(
                content=evaluation_results_csv(run, question_set, results),
                media_type="text/csv; charset=utf-8",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename_base}.csv"',
                },
            )

        return JSONResponse(
            content=evaluation_run_export_payload(run, question_set, results),
            headers={
                "Content-Disposition": f'attachment; filename="{filename_base}.json"',
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
        permission_users: tuple[PermissionUserInventoryRecord, ...] = ()
        permission_org_units: tuple[PermissionOrgUnitInventoryRecord, ...] = ()
        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                permission_users = tuple(list_permission_users(settings.database_url))
                permission_org_units = tuple(list_permission_org_units(settings.database_url))
            except InvalidPermissionInventoryError as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "file_upload.html",
            upload_template_context(
                request,
                error_message=error_message,
                permission_users=permission_users,
                permission_org_units=permission_org_units,
            ),
        )

    @app.post("/files/upload", response_class=HTMLResponse)
    async def submit_upload_file(
        request: Request,
        file: UploadFile = UPLOAD_FILE_FORM,
        document_group: str = DOCUMENT_GROUP_FORM,
        security_level: str = SECURITY_LEVEL_FORM,
        uploaded_by: str | None = UPLOADED_BY_FORM,
        uploaded_by_user_id: str | None = UPLOADED_BY_USER_ID_FORM,
        owner_user_id: str | None = OWNER_USER_ID_FORM,
        owner_org_unit_id: str | None = OWNER_ORG_UNIT_ID_FORM,
        access_scope: str = ACCESS_SCOPE_FORM,
    ) -> HTMLResponse:
        form_values = {
            "document_group": document_group.strip() or "default",
            "security_level": security_level.strip() or "internal",
            "uploaded_by": uploaded_by.strip() if uploaded_by and uploaded_by.strip() else "",
            "uploaded_by_user_id": (uploaded_by_user_id.strip() if uploaded_by_user_id else ""),
            "owner_user_id": owner_user_id.strip() if owner_user_id else "",
            "owner_org_unit_id": owner_org_unit_id.strip() if owner_org_unit_id else "",
            "access_scope": access_scope.strip() or "personal",
        }
        permission_users: tuple[PermissionUserInventoryRecord, ...] = ()
        permission_org_units: tuple[PermissionOrgUnitInventoryRecord, ...] = ()
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
                    uploaded_by_user_id=parse_optional_positive_int_form(
                        form_values["uploaded_by_user_id"],
                        "uploaded_by_user_id",
                    ),
                    owner_user_id=parse_optional_positive_int_form(
                        form_values["owner_user_id"],
                        "owner_user_id",
                    ),
                    owner_org_unit_id=parse_optional_positive_int_form(
                        form_values["owner_org_unit_id"],
                        "owner_org_unit_id",
                    ),
                    access_scope=form_values["access_scope"],
                )
                permission_users = tuple(list_permission_users(settings.database_url))
                permission_org_units = tuple(list_permission_org_units(settings.database_url))
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
                InvalidPermissionInventoryError,
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
                permission_users=permission_users,
                permission_org_units=permission_org_units,
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
        permission_users: tuple[PermissionUserInventoryRecord, ...] = ()
        permission_org_units: tuple[PermissionOrgUnitInventoryRecord, ...] = ()
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                permission_users = tuple(list_permission_users(settings.database_url))
                permission_org_units = tuple(list_permission_org_units(settings.database_url))
                document = get_document_inventory_item(settings.database_url, document_id)
                if document is None:
                    error_message = f"Document not found: {document_id}"
                else:
                    chunks = list_document_chunks(
                        settings.database_url,
                        document_id,
                        chunk_policy_name=chunk_policy_name,
                    )
            except (
                InvalidDocumentInventoryError,
                InvalidChunkError,
                InvalidPermissionInventoryError,
            ) as exc:
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
                access_scope_options=tuple(
                    scope
                    for scope in ("personal", "team", "org_tree", "company")
                    if scope in DOCUMENT_ACCESS_SCOPES
                ),
                permission_user_options=[
                    upload_permission_user_option_payload(user) for user in permission_users
                ],
                permission_org_unit_options=[
                    upload_permission_org_unit_option_payload(org_unit)
                    for org_unit in permission_org_units
                ],
                error_message=error_message,
            ),
        )

    @app.post("/documents/{document_id}/permissions", response_class=HTMLResponse)
    def submit_document_permission_update(
        request: Request,
        document_id: int,
        owner_user_id: str | None = OWNER_USER_ID_FORM,
        owner_org_unit_id: str | None = OWNER_ORG_UNIT_ID_FORM,
        access_scope: str = ACCESS_SCOPE_FORM,
        updated_by_user_id: str | None = UPDATED_BY_USER_ID_FORM,
    ) -> HTMLResponse:
        document: DocumentInventoryItem | None = None
        chunks: list[ChunkRecord] = []
        permission_users: tuple[PermissionUserInventoryRecord, ...] = ()
        permission_org_units: tuple[PermissionOrgUnitInventoryRecord, ...] = ()
        error_message = None
        success_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                permission_users = tuple(list_permission_users(settings.database_url))
                permission_org_units = tuple(list_permission_org_units(settings.database_url))
                document = update_document_permission(
                    settings.database_url,
                    document_id,
                    DocumentPermissionUpdateInput(
                        owner_user_id=parse_optional_positive_int_form(
                            owner_user_id,
                            "owner_user_id",
                        ),
                        owner_org_unit_id=parse_optional_positive_int_form(
                            owner_org_unit_id,
                            "owner_org_unit_id",
                        ),
                        access_scope=access_scope.strip() or "personal",
                        updated_by_user_id=parse_optional_positive_int_form(
                            updated_by_user_id,
                            "updated_by_user_id",
                        ),
                    ),
                )
                if document is None:
                    error_message = f"Document not found: {document_id}"
                else:
                    chunks = list_document_chunks(settings.database_url, document_id)
                    success_message = "document_permissions.updated"
            except (
                InvalidDocumentInventoryError,
                InvalidChunkError,
                InvalidPermissionInventoryError,
                InvalidFileMetadataError,
            ) as exc:
                error_message = str(exc)
                document = get_document_inventory_item(settings.database_url, document_id)
                if document is not None:
                    chunks = list_document_chunks(settings.database_url, document_id)

        return TEMPLATES.TemplateResponse(
            request,
            "document_detail.html",
            template_context(
                request,
                database_configured=bool(settings.database_url),
                document=document,
                chunks=chunks,
                selected_chunk_policy_name="",
                access_scope_options=tuple(
                    scope
                    for scope in ("personal", "team", "org_tree", "company")
                    if scope in DOCUMENT_ACCESS_SCOPES
                ),
                permission_user_options=[
                    upload_permission_user_option_payload(user) for user in permission_users
                ],
                permission_org_unit_options=[
                    upload_permission_org_unit_option_payload(org_unit)
                    for org_unit in permission_org_units
                ],
                error_message=error_message,
                success_message=success_message,
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

        default_actor_id = actor_options[0]["user_id"] if actor_options else ""
        return TEMPLATES.TemplateResponse(
            request,
            "search_compare.html",
            template_context(
                request,
                actor_options=actor_options,
                profile_options=profile_options,
                default_actor_id=default_actor_id,
                search_prefill=search_compare_prefill_payload(request, default_actor_id),
                search_scope_options=SEARCH_COMPARE_SCOPE_OPTIONS,
                search_file_type_options=SEARCH_COMPARE_FILE_TYPES,
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
        fingerprint: str | None = None,
        search_log_id: int | None = None,
        compare_search_log_id: int | None = None,
        limit: int = 50,
    ) -> HTMLResponse:
        actor_options: list[dict[str, object]] = []
        question_sets: list[GoldenQuestionSetRecord] = []
        logs: list[SearchLogListItem] = []
        selected_log: SearchLogDetailRecord | None = None
        selected_log_comparison: dict[str, object] | None = None
        retention_settings = SearchLogRetentionSettings()
        comparison_error_message = None
        error_message = None
        actor_user_id_value: int | None = None
        scope_value = requested_search_scope.strip() if requested_search_scope else None
        document_group_value = document_group.strip() if document_group else None
        fingerprint_value = normalize_search_fingerprint(fingerprint)
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
                retention_settings = load_search_log_retention_settings(settings.database_url)
                logs = list_search_logs(
                    settings.database_url,
                    actor_user_id=actor_user_id_value,
                    requested_search_scope=scope_value,
                    document_group=document_group_value,
                    limit=limit,
                )
                logs = filter_search_logs_by_fingerprint(logs, fingerprint_value)
                if search_log_id is not None:
                    selected_log = get_search_log_detail(settings.database_url, search_log_id)
                    if selected_log is None:
                        error_message = f"Search log not found: {search_log_id}"
                if selected_log is not None and compare_search_log_id is not None:
                    compare_log = get_search_log_detail(
                        settings.database_url,
                        compare_search_log_id,
                    )
                    if compare_log is None:
                        comparison_error_message = f"Search log not found: {compare_search_log_id}"
                    else:
                        selected_log_comparison = search_log_comparison_payload(
                            selected_log,
                            compare_log,
                        )
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
                selected_log_comparison=selected_log_comparison,
                selected_log_reproducibility=(
                    search_reproducibility_payload(selected_log.search_log) if selected_log else {}
                ),
                selected_log_replay_url=(
                    search_log_replay_url(selected_log.search_log) if selected_log else ""
                ),
                search_log_retention_settings=retention_settings,
                selected_actor_user_id=actor_user_id_value or "",
                selected_scope=scope_value or "",
                selected_document_group=document_group_value or "",
                selected_fingerprint=fingerprint_value or "",
                selected_search_log_id=search_log_id,
                selected_compare_search_log_id=compare_search_log_id or "",
                selected_limit=limit,
                comparison_error_message=comparison_error_message,
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
        selected_permission_audit: list[EvaluationPermissionAuditRecord] = []
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
                        selected_permission_audit = get_evaluation_permission_audit(
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
                selected_permission_audit=selected_permission_audit,
                selected_question_set_id=question_set_id,
                selected_profile_name=profile_value or "",
                selected_status=status_filter or "",
                selected_evaluation_run_id=evaluation_run_id,
                error_message=error_message,
                database_configured=bool(settings.database_url),
            ),
        )

    @app.get("/admin/permissions", response_class=HTMLResponse)
    def permission_inventory_page(request: Request) -> HTMLResponse:
        summary: PermissionInventorySummary | None = None
        readiness: PermissionReadinessSummary | None = None
        users: tuple[PermissionUserInventoryRecord, ...] = ()
        org_units: tuple[PermissionOrgUnitInventoryRecord, ...] = ()
        memberships: tuple[PermissionMembershipInventoryRecord, ...] = ()
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                summary = get_permission_inventory_summary(settings.database_url)
                readiness = get_permission_readiness_summary(settings.database_url)
                users = tuple(list_permission_users(settings.database_url))
                org_units = tuple(list_permission_org_units(settings.database_url))
                memberships = tuple(list_permission_memberships(settings.database_url))
            except InvalidPermissionInventoryError as exc:
                error_message = str(exc)
            except Exception as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "permission_inventory.html",
            template_context(
                request,
                summary=summary,
                readiness=readiness,
                users=users,
                org_units=org_units,
                memberships=memberships,
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

    @app.get("/admin/embedding-models", response_class=HTMLResponse)
    def embedding_models_page(request: Request) -> HTMLResponse:
        readiness = audit_embedding_model_readiness(settings.embedding_models_dir)
        return TEMPLATES.TemplateResponse(
            request,
            "embedding_models.html",
            template_context(
                request,
                models_dir=settings.embedding_models_dir,
                model_readiness=readiness,
                ready_count=sum(1 for item in readiness if item.ready),
                model_count=len(readiness),
                profile_count=sum(len(item.distribution.profile_names) for item in readiness),
            ),
        )

    @app.get("/admin/embedding-provider", response_class=HTMLResponse)
    def embedding_provider_page(request: Request) -> HTMLResponse:
        provider_health = get_embedding_provider_health_status(settings)
        return TEMPLATES.TemplateResponse(
            request,
            "embedding_provider.html",
            template_context(
                request,
                provider_health=provider_health.payload,
                provider_status_code=provider_health.status_code,
            ),
        )

    @app.get("/admin/embedding-provider-routes", response_class=HTMLResponse)
    def embedding_provider_routes_page(request: Request) -> HTMLResponse:
        routes: list[EmbeddingProviderRouteRecord] = []
        profiles: list[EmbeddingProfileRecord] = []
        error_message = None
        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                routes = list_embedding_provider_routes(settings.database_url)
                profiles = list_active_embedding_profiles(settings.database_url)
            except (InvalidEmbeddingProviderRouteError, InvalidEmbeddingJobError) as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "embedding_provider_routes.html",
            template_context(
                request,
                routes=routes,
                profiles=profiles,
                error_message=error_message,
                success_message=None,
                database_configured=bool(settings.database_url),
            ),
        )

    @app.post("/admin/embedding-provider-routes", response_class=HTMLResponse)
    def embedding_provider_routes_upsert_page(
        request: Request,
        profile_name: str = Form(...),
        provider_name: str = Form(...),
        provider_mode: str = Form("remote"),
        provider_base_url: str | None = Form(None),
        timeout_seconds: float = Form(30.0),
        priority: int = Form(100),
        is_active: bool = Form(False),
        health_check_enabled: bool = Form(False),
    ) -> HTMLResponse:
        routes: list[EmbeddingProviderRouteRecord] = []
        profiles: list[EmbeddingProfileRecord] = []
        error_message = None
        success_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                upsert_embedding_provider_route(
                    settings.database_url,
                    EmbeddingProviderRouteInput(
                        profile_name=profile_name,
                        provider_name=provider_name,
                        provider_mode=provider_mode,
                        provider_base_url=provider_base_url,
                        timeout_seconds=timeout_seconds,
                        priority=priority,
                        is_active=is_active,
                        health_check_enabled=health_check_enabled,
                    ),
                )
                success_message = "Embedding provider route saved."
            except InvalidEmbeddingProviderRouteError as exc:
                error_message = str(exc)

            try:
                routes = list_embedding_provider_routes(settings.database_url)
                profiles = list_active_embedding_profiles(settings.database_url)
            except (InvalidEmbeddingProviderRouteError, InvalidEmbeddingJobError) as exc:
                error_message = error_message or str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "embedding_provider_routes.html",
            template_context(
                request,
                routes=routes,
                profiles=profiles,
                error_message=error_message,
                success_message=success_message,
                database_configured=bool(settings.database_url),
            ),
        )

    @app.get("/admin/chunk-policies", response_class=HTMLResponse)
    def chunk_policies_page(request: Request) -> HTMLResponse:
        policies: list[ChunkPolicySummaryRecord] = []
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                policies = list_chunk_policy_summaries(settings.database_url)
            except Exception as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "chunk_policies.html",
            template_context(
                request,
                policies=policies,
                default_chunk_policy_name=DEFAULT_CHUNK_POLICY_NAME,
                error_message=error_message,
                database_configured=bool(settings.database_url),
            ),
        )

    @app.get("/admin/design-system", response_class=HTMLResponse)
    def design_system_page(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request,
            "design_system.html",
            template_context(request),
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
