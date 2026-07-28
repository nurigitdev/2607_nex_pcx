"""FastAPI application factory for NeX_PCX."""

import csv
import hashlib
import io
import json
import os
import traceback as traceback_module
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from urllib.parse import urlencode

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.core.admin_logging import (
    InvalidAdminLogError,
    acknowledge_log,
    count_provider_route_alert_logs,
    get_provider_route_change_log,
    list_logs,
    list_provider_route_alert_logs,
    list_provider_route_change_logs,
    log_event,
)
from app.core.bm25_index_coverage import (
    BM25IndexCoverageMatrix,
    BM25IndexCoveragePolicySummary,
    BM25IndexCoverageRow,
    InvalidBM25IndexCoverageError,
    get_bm25_index_coverage_matrix,
)
from app.core.bm25_index_refresh import (
    BM25IndexRefreshOptions,
    bm25_index_refresh_report_payload,
    refresh_bm25_keyword_indexes,
)
from app.core.bm25_keyword_index import (
    DEFAULT_BM25_TOKENIZER_NAME,
    InvalidBM25KeywordIndexError,
    list_bm25_tokenizers,
    validate_bm25_tokenizer_name,
)
from app.core.bm25_search import BM25_SEARCH_PROFILE_NAME
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
from app.core.citation_readiness import (
    CitationReadinessCandidate,
    CitationReadinessInput,
    CitationReadinessIssue,
    CitationReadinessReport,
    build_citation_readiness_report,
)
from app.core.config import Settings, get_settings
from app.core.dashboard_failures import (
    DashboardFailureDetail,
    DashboardFailureRecord,
    DashboardFailureSummary,
    InvalidDashboardFailureError,
    get_dashboard_failure_detail,
    get_dashboard_recent_failures,
)
from app.core.dashboard_health import (
    DashboardHealthSignal,
    DashboardOperationalHealth,
    summarize_dashboard_operational_health,
)
from app.core.dashboard_health_settings import (
    DEFAULT_DASHBOARD_HEALTH_THRESHOLDS,
    DashboardHealthThresholdSettings,
    DashboardHealthThresholdSettingsInput,
    InvalidDashboardHealthThresholdSettingsError,
    load_dashboard_health_threshold_settings,
    reset_dashboard_health_threshold_settings,
    update_dashboard_health_threshold_settings,
)
from app.core.dashboard_metrics import (
    DashboardChunkPolicySummary,
    DashboardCoreMetrics,
    DashboardDocumentGroupSummary,
    DashboardFileTypeSummary,
    get_dashboard_core_metrics,
)
from app.core.dashboard_throughput import (
    DEFAULT_DASHBOARD_THROUGHPUT_LOOKBACK_HOURS,
    DashboardEmbeddingProfileThroughput,
    DashboardEmbeddingThroughput,
    DashboardPipelineStageLatency,
    DashboardPipelineThroughput,
    DashboardSearchLatency,
    DashboardSearchProfileLatency,
    DashboardThroughputLatencySnapshot,
    InvalidDashboardThroughputError,
    get_dashboard_throughput_latency_snapshot,
    validate_lookback_hours,
)
from app.core.database import connect
from app.core.dgx_ingestion_benchmarks import (
    DgxIngestionBenchmarkDetail,
    DgxIngestionBenchmarkJobRecord,
    DgxIngestionBenchmarkProfileRecord,
    DgxIngestionBenchmarkRunRecord,
    InvalidDgxIngestionBenchmarkError,
    get_dgx_ingestion_benchmark_detail,
    list_dgx_ingestion_benchmark_runs,
)
from app.core.direct_generation import (
    DirectGenerationInput,
    DirectGenerationResult,
    InvalidDirectGenerationError,
    run_direct_generation_query,
)
from app.core.document_inventory import (
    DocumentInventoryItem,
    DocumentPermissionUpdateInput,
    InvalidDocumentInventoryError,
    get_document_inventory_item,
    list_document_inventory,
    update_document_permission,
)
from app.core.document_summary import (
    DEFAULT_DOCUMENT_SUMMARY_HISTORY_LIMIT,
    DEFAULT_DOCUMENT_SUMMARY_MAX_CHUNKS,
    DEFAULT_DOCUMENT_SUMMARY_TEMPLATE_KEY,
    MAX_DOCUMENT_SUMMARY_HISTORY_LIMIT,
    DocumentSummaryHistory,
    DocumentSummaryHistoryFilter,
    DocumentSummaryHistoryItem,
    DocumentSummaryInput,
    DocumentSummaryResult,
    InvalidDocumentSummaryError,
    list_document_summary_history,
    run_document_summary_generation,
)
from app.core.embedding_coverage import (
    EmbeddingCoverageDocument,
    EmbeddingCoverageMatrix,
    EmbeddingCoverageProfileCell,
    EmbeddingCoverageProfileSummary,
    InvalidEmbeddingCoverageError,
    MultiPolicyIngestionCoverageChunkDetail,
    MultiPolicyIngestionCoverageDetail,
    MultiPolicyIngestionCoverageMatrix,
    MultiPolicyIngestionCoveragePolicySummary,
    MultiPolicyIngestionCoverageRow,
    get_embedding_coverage_matrix,
    get_multi_policy_ingestion_coverage_detail,
    get_multi_policy_ingestion_coverage_matrix,
)
from app.core.embedding_jobs import (
    EmbeddingJobBacklogProfileSummary,
    EmbeddingJobBacklogSummary,
    EmbeddingJobRecord,
    EmbeddingProfileRecord,
    FailedEmbeddingJobRetryResult,
    InvalidEmbeddingJobError,
    MissingEmbeddingJobReconcileResult,
    get_embedding_job,
    get_embedding_job_backlog_summary,
    list_active_embedding_profiles,
    list_embedding_jobs,
    list_stale_embedding_jobs,
    reconcile_missing_embedding_jobs_for_document_policy_profile,
    release_stale_embedding_job_lease,
    retry_embedding_job,
    retry_failed_embedding_jobs_for_document_policy_profile,
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
from app.core.embedding_provider_model_availability import (
    ProviderModelAvailabilityDrilldown,
    ProviderModelAvailabilityMatrix,
    ProviderModelAvailabilityRow,
    get_provider_model_availability_drilldown,
    get_provider_model_availability_matrix,
)
from app.core.embedding_provider_preflight_runs import (
    EmbeddingProviderPreflightRunInput,
    EmbeddingProviderPreflightRunRecord,
    InvalidEmbeddingProviderPreflightRunError,
    get_embedding_provider_preflight_run,
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
from app.core.embedding_provider_presets import (
    EmbeddingProviderLaunchPlan,
    EmbeddingProviderPreset,
    EmbeddingProviderPresetRoutePlan,
    InvalidEmbeddingProviderPresetError,
    build_embedding_provider_launch_plan,
    build_embedding_provider_preset_route_plans,
    get_embedding_provider_preset,
    list_embedding_provider_presets,
)
from app.core.embedding_provider_route_auth import (
    AUTH_TYPE_API_KEY,
    AUTH_TYPE_BEARER,
    AUTH_TYPE_NONE,
    InvalidEmbeddingProviderRouteAuthError,
    describe_embedding_provider_route_request_metadata,
    normalize_embedding_provider_route_metadata,
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
    set_embedding_provider_route_active,
    update_embedding_provider_route,
    upsert_embedding_provider_route,
    validate_embedding_provider_route_input,
)
from app.core.embedding_providers import (
    InvalidEmbeddingProviderError,
    embedding_provider_runtime_config_from_settings,
)
from app.core.embedding_vectors import (
    EmbeddingVectorRecord,
    InvalidEmbeddingVectorError,
    get_chunk_embedding,
)
from app.core.embedding_worker import ERROR_CODE_EMBEDDING_PROVIDER_ROUTE_NOT_READY
from app.core.embedding_worker_batch_run_retention import (
    EmbeddingBatchRunCleanupResult,
    EmbeddingBatchRunRetentionSettings,
    EmbeddingBatchRunRetentionSettingsInput,
    InvalidEmbeddingBatchRunRetentionError,
    cleanup_expired_embedding_batch_run_records,
    load_embedding_batch_run_retention_settings,
    update_embedding_batch_run_retention_settings,
)
from app.core.embedding_worker_batch_runs import (
    EmbeddingWorkerBatchRunRecord,
    InvalidEmbeddingWorkerBatchRunError,
    get_embedding_worker_batch_run,
    list_embedding_worker_batch_runs,
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
from app.core.extraction_runtime import ExtractionRuntimeRequest
from app.core.file_metadata import (
    DOCUMENT_ACCESS_SCOPES,
    SUPPORTED_FILE_EXTENSIONS,
    FileMetadataRecord,
    InvalidFileMetadataError,
    UnsupportedFileExtensionError,
    get_file_metadata,
)
from app.core.file_uploads import InvalidUploadFileNameError, store_upload
from app.core.foreground_worker_runtime import (
    build_foreground_worker_runtime_report,
    foreground_worker_runtime_report_payload,
)
from app.core.generation_docx_export import (
    GENERATION_DOCX_MEDIA_TYPE,
    assess_generation_docx_export_readiness,
    generation_docx_export_evidence_from_run,
    generation_docx_export_evidence_payload,
    generation_docx_export_readiness_payload,
    markdown_to_docx_bytes,
)
from app.core.generation_executor import (
    GenerationExecutionReport,
    execute_mock_generation_run,
    execute_remote_generation_run,
)
from app.core.generation_prompts import (
    GenerationPromptPackage,
    InvalidGenerationPromptError,
    build_generation_prompt_package,
)
from app.core.generation_provider_metric_snapshots import (
    DEFAULT_GENERATION_PROVIDER_METRIC_SNAPSHOT_LIMIT,
    MAX_GENERATION_PROVIDER_METRIC_SNAPSHOT_LIMIT,
    InvalidGenerationProviderMetricSnapshotError,
    generation_provider_metric_snapshot_payload,
    get_generation_provider_metric_snapshot,
)
from app.core.generation_providers import (
    InvalidGenerationProviderError,
    generation_provider_runtime_config_from_record,
)
from app.core.generation_runs import (
    DEFAULT_GENERATION_RUN_HISTORY_LIMIT,
    DGX_VLLM_GENERATION_API_KEY_ENV,
    DGX_VLLM_GENERATION_BASE_URL,
    DGX_VLLM_GENERATION_MAX_TOKENS,
    DGX_VLLM_GENERATION_MODEL_ID,
    DGX_VLLM_GENERATION_PROVIDER_NAME,
    DGX_VLLM_GENERATION_TEMPERATURE,
    DGX_VLLM_GENERATION_TIMEOUT_SECONDS,
    DGX_VLLM_GENERATION_TOP_P,
    GENERATION_ANSWER_QUALITY_NOT_AVAILABLE,
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
    GENERATION_RUN_HISTORY_FILTER_ALL,
    MAX_GENERATION_RUN_HISTORY_LIMIT,
    GenerationProviderConfigRecord,
    GenerationRunCitationRecord,
    GenerationRunHistory,
    GenerationRunHistoryFilter,
    GenerationRunHistoryItem,
    GenerationRunRecord,
    InvalidGenerationRunError,
    get_default_generation_provider_config,
    get_generation_provider_config_for_mode,
    get_generation_run,
    list_generation_provider_configs,
    list_generation_run_citations,
    list_generation_run_history,
    seed_dgx_vllm_generation_provider_config,
)
from app.core.generation_template_completeness import (
    GenerationTemplateCompletenessAssessment,
    assess_generation_template_completeness,
    generation_template_completeness_payload,
)
from app.core.generation_templates import (
    GENERATION_TEMPLATE_DOCUMENT_TYPES,
    GENERATION_TEMPLATE_LANGUAGES,
    GENERATION_TEMPLATE_OUTPUT_FORMAT_MARKDOWN,
    GenerationTemplateCloneInput,
    GenerationTemplateInput,
    GenerationTemplateRecord,
    InvalidGenerationTemplateError,
    clone_generation_template_version,
    get_default_generation_template,
    get_generation_template_by_key,
    list_generation_templates,
    rollback_generation_template_version,
    set_generation_template_active,
    set_generation_template_default,
    suggest_generation_template_clone_key,
    suggest_generation_template_next_version,
    upsert_generation_template,
)
from app.core.go_live_readiness import (
    build_go_live_readiness_report,
    go_live_readiness_report_payload,
)
from app.core.golden_batch_metric_snapshots import (
    GoldenBatchMetricSnapshotComparison,
    GoldenBatchMetricSnapshotDetail,
    GoldenBatchMetricSnapshotRecord,
    GoldenBatchMetricSnapshotTrend,
    GoldenBatchMetricSnapshotTrendPoint,
    GoldenBatchProfileMetricSnapshotComparison,
    GoldenBatchProfileMetricSnapshotRecord,
    GoldenBatchQuestionMetricSnapshotComparison,
    GoldenBatchQuestionMetricSnapshotRecord,
    InvalidGoldenBatchMetricSnapshotError,
    compare_golden_batch_metric_snapshots,
    get_golden_batch_metric_snapshot_detail,
    get_golden_batch_metric_snapshot_trend,
    get_latest_golden_batch_metric_snapshot,
    list_golden_batch_metric_snapshots,
    record_golden_batch_metric_snapshot,
)
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
from app.core.golden_search_experiments import (
    GoldenSearchExperimentBatchInput,
    GoldenSearchExperimentBatchReport,
    InvalidGoldenSearchExperimentError,
    execute_golden_search_experiment_batch,
)
from app.core.hybrid_search import HYBRID_SEARCH_PROFILE_NAME
from app.core.i18n import (
    LANGUAGE_COOKIE_NAME,
    LANGUAGE_OPTIONS,
    get_translator,
    normalize_language,
    resolve_language,
)
from app.core.ingestion_artifacts import (
    DocumentBlockRecord,
    ExtractionArtifactRecord,
    ExtractionQualitySnapshotInput,
    ExtractionQualitySnapshotRecord,
    ExtractionQualitySnapshotSummary,
    ExtractionRunRecord,
    InvalidIngestionArtifactError,
    create_extraction_quality_snapshot,
    get_extraction_quality_snapshot_summary,
    list_document_blocks,
    list_document_extraction_artifacts,
    list_document_extraction_runs,
    list_extraction_quality_snapshots,
)
from app.core.local_extraction import (
    PersistedExtractionRuntimeResult,
    persist_extraction_runtime_result,
    run_local_extraction,
    select_local_extraction_profile_name,
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
    PipelineQueueStageSummary,
    PipelineQueueSummary,
    PipelineQueueTypeSummary,
    get_pipeline_job,
    get_pipeline_queue_summary,
    list_pipeline_job_events,
    list_pipeline_jobs,
    retry_pipeline_job,
)
from app.core.query_embeddings import InvalidQueryEmbeddingError
from app.core.remote_reranker_operations import get_remote_reranker_operations_status
from app.core.reranked_search import RERANKED_SEARCH_PROFILE_NAME
from app.core.rerankers import (
    DEFAULT_RERANKER_MODEL_ID,
    DEFAULT_RERANKER_PROFILE_NAME,
    InvalidRerankerError,
    reranker_runtime_config_from_settings,
)
from app.core.retrieval_confidence import retrieval_confidence_assessment_payload
from app.core.retrieval_context import (
    DEFAULT_CONTEXT_CHAR_BUDGET,
    DEFAULT_CONTEXT_MAX_ITEMS,
    InvalidRetrievalContextError,
    RetrievalContextCandidate,
    RetrievalContextChunkEntry,
    RetrievalContextInput,
    RetrievalContextPackage,
    RetrievalContextResultReference,
    build_retrieval_context_package,
)
from app.core.search_compare import (
    InvalidSearchCompareError,
    SearchCompareCoverageReconcileInput,
    SearchCompareCoverageReconcileResult,
    SearchCompareInput,
    SearchCompareProfileResult,
    SearchCompareReadinessInput,
    SearchCompareReadinessProfile,
    SearchCompareReadinessResult,
    SearchCompareResult,
    SearchPermissionMatrixEntryInput,
    SearchPermissionMatrixEntryResult,
    SearchPermissionMatrixInput,
    SearchPermissionMatrixResult,
    get_search_compare_readiness,
    reconcile_search_compare_policy_coverage,
    run_permission_search_matrix,
    run_search_compare,
)
from app.core.search_experiment_runner import (
    InvalidSearchExperimentExecutionError,
    SearchExperimentExecutionInput,
    SearchExperimentExecutionReport,
    SearchExperimentProfileExecutionSummary,
    execute_search_experiment,
)
from app.core.search_experiments import (
    SEARCH_EXPERIMENT_RUN_STATUSES,
    GoldenSearchExperimentBatchDetail,
    GoldenSearchExperimentBatchMetricSummary,
    GoldenSearchExperimentBatchQuestionMetricSummary,
    GoldenSearchExperimentBatchQuestionSummary,
    GoldenSearchExperimentBatchSummary,
    InvalidSearchExperimentError,
    SearchExperimentProfileRunRecord,
    SearchExperimentRunDetail,
    SearchExperimentRunRecord,
    get_golden_search_experiment_batch_detail,
    get_golden_search_experiment_batch_metric_summary,
    get_search_experiment_run_detail,
    golden_search_experiment_batch_key_from_run,
    list_golden_search_experiment_batch_summaries,
    list_search_experiment_runs,
)
from app.core.search_logs import (
    InvalidSearchLogError,
    SearchDuplicateFingerprintRecord,
    SearchFeedbackCommentRecord,
    SearchFeedbackProfileSummaryRecord,
    SearchLatencyOutlierRecord,
    SearchLogCleanupResult,
    SearchLogDetailRecord,
    SearchLogListItem,
    SearchLogRecord,
    SearchLogResultDetailRecord,
    SearchLogRetentionSettings,
    SearchLogRetentionSettingsInput,
    SearchLogReviewMetadataInput,
    SearchNoResultRecord,
    SearchOperationsSummaryRecord,
    SearchResultFeedbackInput,
    SearchResultFeedbackRecord,
    SearchRuntimeFailureRecord,
    cleanup_expired_search_logs,
    create_search_result_feedback,
    get_search_log,
    get_search_log_detail,
    get_search_log_result,
    get_search_operations_summary,
    list_search_duplicate_fingerprints,
    list_search_feedback_comments,
    list_search_latency_outliers,
    list_search_logs,
    list_search_no_result_logs,
    list_search_runtime_failures,
    load_search_log_retention_settings,
    summarize_search_feedback,
    update_search_log_retention_settings,
    update_search_log_review_metadata,
)
from app.core.search_result_context import (
    InvalidSearchResultContextError,
    SearchResultContextChunk,
    SearchResultSourceArtifact,
    SearchResultSourceBlock,
    SearchResultSourceContext,
    get_search_result_source_context,
)
from app.core.vector_search import InvalidVectorSearchError, VectorSearchResult

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=BASE_DIR / "web" / "templates")
PROVIDER_OPERATIONS_PLAYBOOK_PATH = BASE_DIR.parent / "docs" / "provider_operations_playbook.md"
OPERATIONS_RUNBOOK_PATH = BASE_DIR.parent / "docs" / "operations_runbook.md"
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
MAX_SEARCH_CHUNK_POLICY_COMPARE_POLICIES = 8


class SearchCompareRequest(BaseModel):
    query_text: str
    actor_user_id: int
    requested_search_scope: str = "company"
    top_k: int = Field(default=5, ge=1)
    profiles: list[str] | None = None
    chunk_policy_name: str | None = None
    document_group: str | None = None
    file_type: str | None = None
    bm25_tokenizer_name: str | None = None
    hybrid_vector_profile_name: str | None = None
    reranked_vector_profile_name: str | None = None
    allow_mock_fallback: bool = True


class DirectGenerationRequest(BaseModel):
    query_text: str = Field(min_length=1)
    actor_user_id: int = Field(ge=1)
    requested_search_scope: str = "company"
    provider_mode: str = GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE
    generation_template_key: str | None = None
    top_k: int = Field(default=5, ge=1)
    profiles: list[str] | None = None
    chunk_policy_name: str | None = None
    document_group: str | None = None
    file_type: str | None = None
    bm25_tokenizer_name: str | None = None
    hybrid_vector_profile_name: str | None = None
    reranked_vector_profile_name: str | None = None
    allow_mock_fallback: bool = True
    max_context_chars: int = Field(default=DEFAULT_CONTEXT_CHAR_BUDGET, ge=500, le=50000)
    include_neighbors: bool = True
    max_items: int = Field(default=DEFAULT_CONTEXT_MAX_ITEMS, ge=1, le=100)


class DocumentSummaryRunRequest(BaseModel):
    actor_user_id: int = Field(ge=1)
    summary_instruction: str = ""
    provider_mode: str = GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE
    generation_template_key: str | None = DEFAULT_DOCUMENT_SUMMARY_TEMPLATE_KEY
    max_chunks: int = Field(default=DEFAULT_DOCUMENT_SUMMARY_MAX_CHUNKS, ge=1, le=100)
    max_context_chars: int = Field(default=DEFAULT_CONTEXT_CHAR_BUDGET, ge=500, le=50000)
    include_neighbors: bool = False
    chunk_policy_name: str | None = None


class GenerationTemplateManagementRequest(BaseModel):
    template_key: str = Field(min_length=1, max_length=64)
    template_name: str = Field(min_length=1, max_length=200)
    template_version: str = Field(default="v1", min_length=1, max_length=80)
    template_family: str | None = Field(default=None, max_length=64)
    document_type: str = Field(default="grounded_answer", min_length=1, max_length=80)
    language: str = Field(default="ko", min_length=2, max_length=16)
    output_format: str = Field(default="markdown", min_length=1, max_length=40)
    section_schema: list[dict[str, object]] = Field(default_factory=list)
    system_instruction: str = Field(min_length=1)
    user_instruction_suffix: str = ""
    style_guidance: dict[str, object] = Field(default_factory=dict)
    citation_policy: dict[str, object] = Field(default_factory=dict)
    is_default: bool = False
    is_active: bool = True
    clone_source_template_id: int | None = Field(default=None, ge=1)
    change_note: str = Field(default="", max_length=1000)
    created_by: str | None = Field(default="generation-template-api", max_length=120)
    created_by_user_id: int | None = Field(default=None, ge=1)


class GenerationTemplateCloneRequest(BaseModel):
    target_template_key: str = Field(min_length=1, max_length=64)
    target_template_version: str = Field(min_length=1, max_length=80)
    target_template_name: str | None = Field(default=None, max_length=200)
    make_default: bool = False
    is_active: bool = True
    change_note: str = Field(default="", max_length=1000)
    created_by: str | None = Field(default="generation-template-api", max_length=120)
    created_by_user_id: int | None = Field(default=None, ge=1)


class GenerationTemplateActiveRequest(BaseModel):
    is_active: bool


class SearchChunkPolicyCompareRequest(BaseModel):
    query_text: str
    actor_user_id: int
    requested_search_scope: str = "company"
    top_k: int = Field(default=5, ge=1)
    profiles: list[str] | None = None
    chunk_policy_names: list[str] = Field(
        min_length=2,
        max_length=MAX_SEARCH_CHUNK_POLICY_COMPARE_POLICIES,
    )
    document_group: str | None = None
    file_type: str | None = None
    bm25_tokenizer_name: str | None = None
    hybrid_vector_profile_name: str | None = None
    reranked_vector_profile_name: str | None = None
    allow_mock_fallback: bool = True


class SearchCompareReadinessRequest(BaseModel):
    actor_user_id: int
    requested_search_scope: str = "company"
    profiles: list[str] | None = None
    chunk_policy_name: str | None = None
    chunk_policy_names: list[str] | None = None
    document_group: str | None = None
    file_type: str | None = None


class SearchCompareReadinessCoverageReconcileRequest(BaseModel):
    actor_user_id: int = Field(ge=1)
    requested_search_scope: str = "company"
    profile_name: str
    chunk_policy_name: str
    document_group: str | None = None
    file_type: str | None = None
    max_jobs: int = Field(default=500, ge=1, le=500)


class BM25IndexBackfillRequest(BaseModel):
    tokenizer_name: str = DEFAULT_BM25_TOKENIZER_NAME
    chunk_policy_names: list[str] | None = Field(default=None, max_length=20)
    continue_on_error: bool = True


class SearchProfileRetryRequest(BaseModel):
    profile_name: str


class MissingEmbeddingJobReconcileRequest(BaseModel):
    document_id: int = Field(ge=1)
    chunk_policy_name: str
    profile_name: str
    max_jobs: int = Field(default=500, ge=1, le=500)


class FailedEmbeddingJobRetryRequest(BaseModel):
    document_id: int = Field(ge=1)
    chunk_policy_name: str
    profile_name: str
    max_jobs: int = Field(default=500, ge=1, le=500)


class SearchRuntimeFailureRetryItemRequest(BaseModel):
    search_log_id: int = Field(ge=1)
    profile_name: str


class SearchRuntimeFailureRetryRequest(BaseModel):
    failures: list[SearchRuntimeFailureRetryItemRequest]


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
    bm25_tokenizer_name: str | None = None
    hybrid_vector_profile_name: str | None = None
    reranked_vector_profile_name: str | None = None
    allow_mock_fallback: bool = True


class SearchExperimentRunRequest(BaseModel):
    run_name: str
    query_text: str
    actor_user_id: int = Field(ge=1)
    requested_search_scope: str = "company"
    profiles: list[str] | None = None
    strategy_name: str = "vector_cosine"
    top_k: int = Field(default=5, ge=1)
    score_threshold: float | None = None
    chunk_policy_name: str | None = None
    document_group: str | None = None
    file_type: str | None = None
    runtime_metadata: dict[str, object] = Field(default_factory=dict)
    created_by: str | None = "search-experiment-api"
    created_by_user_id: int | None = Field(default=None, ge=1)
    allow_mock_fallback: bool = True


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


class EmbeddingFailedJobBulkRetryRequest(BaseModel):
    profile_name: str | None = Field(default=None, max_length=120)
    limit: int = Field(default=100, ge=1, le=500)


class EmbeddingBatchRunRetentionSettingsRequest(BaseModel):
    enabled: bool = True
    retention_days: int = Field(default=30, ge=1, le=3650)
    cleanup_batch_size: int = Field(default=1000, ge=1, le=100000)


class EmbeddingBatchRunCleanupRequest(BaseModel):
    dry_run: bool = True


class DashboardHealthThresholdSettingsRequest(BaseModel):
    thresholds: dict[str, int] = Field(default_factory=dict)


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


class GenerationProviderDgxVllmSeedRequest(BaseModel):
    provider_name: str = Field(
        default=DGX_VLLM_GENERATION_PROVIDER_NAME,
        min_length=1,
        max_length=120,
    )
    provider_base_url: str = Field(
        default=DGX_VLLM_GENERATION_BASE_URL,
        min_length=1,
        max_length=500,
    )
    model_id: str = Field(
        default=DGX_VLLM_GENERATION_MODEL_ID,
        min_length=1,
        max_length=500,
    )
    api_key_env: str = Field(
        default=DGX_VLLM_GENERATION_API_KEY_ENV,
        min_length=1,
        max_length=120,
    )
    request_timeout_seconds: int = Field(
        default=DGX_VLLM_GENERATION_TIMEOUT_SECONDS,
        ge=1,
        le=3600,
    )
    max_tokens: int = Field(default=DGX_VLLM_GENERATION_MAX_TOKENS, ge=1, le=200000)
    temperature: float = Field(default=DGX_VLLM_GENERATION_TEMPERATURE, ge=0, le=2)
    top_p: float = Field(default=DGX_VLLM_GENERATION_TOP_P, gt=0, le=1)
    is_default: bool = False
    is_active: bool = True
    thinking_disabled: bool = True
    created_by: str | None = Field(default="generation-provider-config-api", max_length=120)
    created_by_user_id: int | None = Field(default=None, ge=1)


class DocumentPermissionUpdateRequest(BaseModel):
    owner_user_id: int | None = Field(default=None, ge=1)
    owner_org_unit_id: int | None = Field(default=None, ge=1)
    access_scope: str = "personal"
    updated_by_user_id: int | None = Field(default=None, ge=1)


class ExtractionQualitySnapshotCreateRequest(BaseModel):
    artifact_id: int | None = Field(default=None, ge=1)
    created_by: str | None = Field(default="extraction-quality-api", max_length=120)
    created_by_user_id: int | None = Field(default=None, ge=1)


class ExtractionRerunRequest(BaseModel):
    extraction_profile_name: str | None = Field(default=None, max_length=120)
    requested_by: str | None = Field(default="extraction-rerun-api", max_length=120)
    options: dict[str, object] = Field(default_factory=dict)


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


class EmbeddingProviderRouteActivationRequest(BaseModel):
    is_active: bool


class EmbeddingProviderRouteImportRequest(BaseModel):
    routes: list[EmbeddingProviderRouteRequest] = Field(min_length=1, max_length=200)
    dry_run: bool = True


class EmbeddingProviderRoutePresetRegistrationRequest(BaseModel):
    preset_name: str
    host: str | None = Field(default=None, max_length=255)
    port: int | None = Field(default=None, ge=1, le=65535)
    base_url: str | None = Field(default=None, max_length=500)
    provider_name: str | None = Field(default=None, max_length=120)
    timeout_seconds: float = Field(default=30.0, gt=0)
    priority: int = Field(default=100, ge=0)
    is_active: bool = True
    health_check_enabled: bool = True
    run_preflight: bool = False
    runtime_metadata: dict[str, object] = Field(default_factory=dict)


class EmbeddingProviderLaunchPlanRequest(BaseModel):
    preset_name: str
    python_bin: str = Field(default="./.venv/bin/python", max_length=500)
    host: str | None = Field(default=None, max_length=255)
    port: int | None = Field(default=None, ge=1, le=65535)
    device: str = Field(default="cpu", max_length=120)
    models_dir: str | None = Field(default=None, max_length=1000)
    provider_model_id: str | None = Field(default=None, max_length=255)
    reload: bool = False


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


class GoldenSearchExperimentBatchRequest(BaseModel):
    question_set_id: int = Field(ge=1)
    run_name_prefix: str | None = None
    profiles: list[str] | None = None
    strategy_name: str = "vector_cosine"
    top_k: int | None = Field(default=None, ge=1)
    score_threshold: float | None = None
    chunk_policy_name: str | None = None
    runtime_metadata: dict[str, object] = Field(default_factory=dict)
    created_by: str | None = "golden-search-experiment-api"
    created_by_user_id: int | None = Field(default=None, ge=1)
    allow_mock_fallback: bool = True


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


DASHBOARD_TIME_WINDOW_OPTIONS = (
    {"hours": 1, "label": "1h"},
    {"hours": 6, "label": "6h"},
    {"hours": 24, "label": "24h"},
    {"hours": 168, "label": "7d"},
    {"hours": 720, "label": "30d"},
)

DASHBOARD_REFRESH_INTERVAL_OPTIONS = (
    {"seconds": 0, "label": "Off"},
    {"seconds": 30, "label": "30s"},
    {"seconds": 60, "label": "60s"},
)

DASHBOARD_HEALTH_THRESHOLD_UI_ROWS = (
    ("pipeline_stale", "critical"),
    ("pipeline_exhausted", "critical"),
    ("pipeline_retryable", "warning"),
    ("embedding_stale", "critical"),
    ("embedding_exhausted", "critical"),
    ("embedding_retryable", "warning"),
    ("provider_alert", "warning"),
    ("app_error", "warning"),
    ("parsing_failure", "warning"),
)


def dashboard_query_url(
    request: Request,
    updates: dict[str, object | None],
) -> str:
    query_items = [
        (key, value) for key, value in request.query_params.multi_items() if key not in updates
    ]
    for key, value in updates.items():
        if value is not None:
            query_items.append((key, str(value)))
    if not query_items:
        return request.url.path
    return f"{request.url.path}?{urlencode(query_items)}"


def dashboard_time_window_options(
    request: Request,
    *,
    selected_lookback_hours: int,
) -> list[dict[str, object]]:
    options: list[dict[str, object]] = []
    for option in DASHBOARD_TIME_WINDOW_OPTIONS:
        hours = int(option["hours"])
        options.append(
            {
                "hours": hours,
                "label": option["label"],
                "active": hours == selected_lookback_hours,
                "url": dashboard_query_url(
                    request,
                    {"lookback_hours": hours},
                ),
            }
        )
    return options


def dashboard_refresh_interval_options(
    request: Request,
    *,
    selected_refresh_seconds: int,
) -> list[dict[str, object]]:
    options: list[dict[str, object]] = []
    for option in DASHBOARD_REFRESH_INTERVAL_OPTIONS:
        seconds = int(option["seconds"])
        options.append(
            {
                "seconds": seconds,
                "label": option["label"],
                "active": seconds == selected_refresh_seconds,
                "url": dashboard_query_url(
                    request,
                    {"refresh_seconds": seconds},
                ),
            }
        )
    return options


def validate_dashboard_refresh_seconds(refresh_seconds: int) -> int:
    valid_intervals = {int(option["seconds"]) for option in DASHBOARD_REFRESH_INTERVAL_OPTIONS}
    if refresh_seconds not in valid_intervals:
        raise ValueError("refresh_seconds must be one of 0, 30, or 60")
    return refresh_seconds


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
        "progress_percent": _percent_value(pipeline_job.progress_percent),
        "progress_label": _percent_label(pipeline_job.progress_percent),
    }


def _datetime_response(value: object | None) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else None


def _datetime_label(value: object | None) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S") if hasattr(value, "strftime") else "-"


def _json_safe_dashboard_raw_value(value: object) -> object:
    if isinstance(value, datetime):
        return _datetime_response(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_dashboard_raw_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_dashboard_raw_value(item) for item in value]
    return value


def _json_safe_dashboard_display_value(value: object) -> object:
    if isinstance(value, datetime):
        return _datetime_label(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_dashboard_display_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_dashboard_display_value(item) for item in value]
    return value


def _byte_count_label(value: int | float | None) -> str:
    if value is None:
        return "-"
    amount = float(value)
    units = ("B", "KB", "MB", "GB", "TB")
    unit_index = 0
    while abs(amount) >= 1024 and unit_index < len(units) - 1:
        amount /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(amount):,} {units[unit_index]}"
    return f"{amount:,.1f} {units[unit_index]}"


def _duration_ms_label(value: int | float | None) -> str:
    if value is None:
        return "-"
    amount = float(value)
    if amount >= 1000:
        return f"{amount / 1000:.2f}s"
    return f"{amount:.1f} ms"


def _percent_value(value: object | None) -> str | None:
    if value is None:
        return None
    return f"{float(value):.2f}"


def _percent_label(value: object | None) -> str:
    formatted = _percent_value(value)
    if formatted is None:
        return "-"
    return f"{formatted}%"


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


def provider_route_change_diff_payload(log: dict[str, object]) -> dict[str, object]:
    detail = dict(log.get("detail") or {})
    previous = detail.get("previous") if isinstance(detail.get("previous"), dict) else {}
    current = detail.get("current") if isinstance(detail.get("current"), dict) else {}
    changed_fields = {
        str(field) for field in detail.get("changed_fields", []) if isinstance(field, str) and field
    }
    field_names = sorted(set(previous) | set(current) | changed_fields)
    fields = [
        {
            "field": field,
            "previous_value": previous.get(field),
            "current_value": current.get(field),
            "changed": field in changed_fields or previous.get(field) != current.get(field),
        }
        for field in field_names
    ]
    return {
        "change": admin_log_payload(log),
        "diff": {
            "field_count": len(fields),
            "changed_count": sum(1 for field in fields if field["changed"]),
            "changed_fields": sorted(changed_fields),
            "fields": fields,
        },
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
        "progress_percent": _percent_value(pipeline_job.progress_percent),
        "progress_label": _percent_label(pipeline_job.progress_percent),
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


def pipeline_queue_stage_summary_payload(
    summary: PipelineQueueStageSummary,
) -> dict[str, object]:
    return {
        "stage": summary.stage,
        "total_count": summary.total_count,
        "queued_count": summary.queued_count,
        "running_count": summary.running_count,
        "failed_count": summary.failed_count,
        "average_progress_percent": _percent_value(summary.average_progress_percent),
        "average_progress_label": _percent_label(summary.average_progress_percent),
        "oldest_queued_at": _datetime_response(summary.oldest_queued_at),
        "oldest_queued_label": _datetime_label(summary.oldest_queued_at),
    }


def pipeline_queue_type_summary_payload(
    summary: PipelineQueueTypeSummary,
) -> dict[str, object]:
    return {
        "job_type": summary.job_type,
        "total_count": summary.total_count,
        "queued_count": summary.queued_count,
        "running_count": summary.running_count,
        "failed_count": summary.failed_count,
    }


def pipeline_queue_summary_payload(summary: PipelineQueueSummary) -> dict[str, object]:
    return {
        "total_count": summary.total_count,
        "queued_count": summary.queued_count,
        "running_count": summary.running_count,
        "stale_running_count": summary.stale_running_count,
        "reclaimable_stale_running_count": summary.reclaimable_stale_running_count,
        "failed_count": summary.failed_count,
        "retryable_failed_count": summary.retryable_failed_count,
        "exhausted_failed_count": summary.exhausted_failed_count,
        "canceled_count": summary.canceled_count,
        "retryable_canceled_count": summary.retryable_canceled_count,
        "exhausted_canceled_count": summary.exhausted_canceled_count,
        "succeeded_count": summary.succeeded_count,
        "skipped_count": summary.skipped_count,
        "claimable_count": summary.claimable_count,
        "attention_count": summary.attention_count,
        "oldest_queued_at": _datetime_response(summary.oldest_queued_at),
        "oldest_queued_label": _datetime_label(summary.oldest_queued_at),
        "oldest_stale_lease_expires_at": _datetime_response(summary.oldest_stale_lease_expires_at),
        "oldest_stale_lease_expires_label": _datetime_label(summary.oldest_stale_lease_expires_at),
        "stages": [
            pipeline_queue_stage_summary_payload(stage_summary)
            for stage_summary in summary.stage_summaries
        ],
        "job_types": [
            pipeline_queue_type_summary_payload(type_summary)
            for type_summary in summary.type_summaries
        ],
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


def embedding_job_readiness_gate_payload(
    job: EmbeddingJobRecord,
) -> dict[str, object] | None:
    metadata = job.runtime_metadata or {}
    blocked_routes = metadata.get("provider_route_readiness_blocked_routes")
    if job.error_code != ERROR_CODE_EMBEDDING_PROVIDER_ROUTE_NOT_READY and not blocked_routes:
        return None
    return {
        "gate": metadata.get("provider_route_readiness_gate", "blocked_all_routes"),
        "blocked_count": metadata.get("provider_route_readiness_blocked_count", 0),
        "blocked_routes": blocked_routes if isinstance(blocked_routes, list) else [],
        "error_code": job.error_code,
        "error_message": job.error_message,
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
        "provider_route_readiness_gate": embedding_job_readiness_gate_payload(job),
        "created_at": _datetime_response(job.created_at),
        "started_at": _datetime_response(job.started_at),
        "finished_at": _datetime_response(job.finished_at),
        "updated_at": _datetime_response(job.updated_at),
    }


def embedding_job_backlog_profile_payload(
    profile_summary: EmbeddingJobBacklogProfileSummary,
) -> dict[str, object]:
    return {
        "profile_name": profile_summary.profile_name,
        "total_count": profile_summary.total_count,
        "pending_count": profile_summary.pending_count,
        "running_count": profile_summary.running_count,
        "stale_running_count": profile_summary.stale_running_count,
        "reclaimable_stale_running_count": profile_summary.reclaimable_stale_running_count,
        "failed_count": profile_summary.failed_count,
        "retryable_failed_count": profile_summary.retryable_failed_count,
        "exhausted_failed_count": profile_summary.exhausted_failed_count,
        "succeeded_count": profile_summary.succeeded_count,
        "skipped_count": profile_summary.skipped_count,
        "claimable_count": profile_summary.claimable_count,
        "attention_count": profile_summary.attention_count,
        "oldest_pending_at": _datetime_response(profile_summary.oldest_pending_at),
        "oldest_stale_lease_expires_at": _datetime_response(
            profile_summary.oldest_stale_lease_expires_at
        ),
    }


def embedding_job_backlog_summary_payload(
    summary: EmbeddingJobBacklogSummary,
) -> dict[str, object]:
    return {
        "total_count": summary.total_count,
        "pending_count": summary.pending_count,
        "running_count": summary.running_count,
        "stale_running_count": summary.stale_running_count,
        "reclaimable_stale_running_count": summary.reclaimable_stale_running_count,
        "failed_count": summary.failed_count,
        "retryable_failed_count": summary.retryable_failed_count,
        "exhausted_failed_count": summary.exhausted_failed_count,
        "succeeded_count": summary.succeeded_count,
        "skipped_count": summary.skipped_count,
        "claimable_count": summary.claimable_count,
        "attention_count": summary.attention_count,
        "profiles": [
            embedding_job_backlog_profile_payload(profile_summary)
            for profile_summary in summary.profile_summaries
        ],
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


def embedding_worker_batch_run_payload(
    batch_run: EmbeddingWorkerBatchRunRecord,
) -> dict[str, object]:
    return {
        "batch_run_id": batch_run.batch_run_id,
        "worker_name": batch_run.worker_name,
        "profile_name": batch_run.profile_name,
        "provider_source": batch_run.provider_source,
        "provider_mode": batch_run.provider_mode,
        "remote_provider_url": batch_run.remote_provider_url,
        "require_route_readiness": batch_run.require_route_readiness,
        "readiness_gate_failure_mode": batch_run.readiness_gate_failure_mode,
        "readiness_gate_defer_seconds": batch_run.readiness_gate_defer_seconds,
        "limit_requested": batch_run.limit_requested,
        "result_count": batch_run.result_count,
        "processed_count": batch_run.processed_count,
        "succeeded_count": batch_run.succeeded_count,
        "failed_count": batch_run.failed_count,
        "deferred_count": batch_run.deferred_count,
        "idle_count": batch_run.idle_count,
        "stopped_reason": batch_run.stopped_reason,
        "job_ids": list(batch_run.job_ids),
        "runtime_metadata": batch_run.runtime_metadata,
        "elapsed_ms": batch_run.elapsed_ms,
        "started_at": _datetime_response(batch_run.started_at),
        "completed_at": _datetime_response(batch_run.completed_at),
        "created_at": _datetime_response(batch_run.created_at),
    }


def embedding_worker_batch_run_summary(
    batch_runs: list[EmbeddingWorkerBatchRunRecord],
) -> dict[str, object]:
    elapsed_values = [run.elapsed_ms for run in batch_runs if run.elapsed_ms is not None]
    return {
        "run_count": len(batch_runs),
        "processed_count": sum(run.processed_count for run in batch_runs),
        "succeeded_count": sum(run.succeeded_count for run in batch_runs),
        "failed_count": sum(run.failed_count for run in batch_runs),
        "deferred_count": sum(run.deferred_count for run in batch_runs),
        "idle_count": sum(run.idle_count for run in batch_runs),
        "avg_elapsed_ms": (
            round(sum(elapsed_values) / len(elapsed_values), 1) if elapsed_values else 0
        ),
    }


def embedding_worker_batch_run_throughput_summary(
    batch_runs: list[EmbeddingWorkerBatchRunRecord],
) -> dict[str, object]:
    def metrics_for_runs(runs: list[EmbeddingWorkerBatchRunRecord]) -> dict[str, object]:
        elapsed_ms = sum(max(0, run.elapsed_ms or 0) for run in runs)
        processed_count = sum(run.processed_count for run in runs)
        succeeded_count = sum(run.succeeded_count for run in runs)
        failed_count = sum(run.failed_count for run in runs)
        run_count = len(runs)
        elapsed_seconds = elapsed_ms / 1000
        success_rate_pct = (
            round((succeeded_count / processed_count) * 100, 2) if processed_count > 0 else 0.0
        )
        return {
            "run_count": run_count,
            "processed_count": processed_count,
            "succeeded_count": succeeded_count,
            "failed_count": failed_count,
            "elapsed_ms": elapsed_ms,
            "throughput_per_second": (
                round(processed_count / elapsed_seconds, 2) if elapsed_seconds > 0 else 0
            ),
            "success_rate_pct": success_rate_pct,
            "success_rate_label": _percent_label(success_rate_pct),
            "avg_processed_per_run": (
                round(processed_count / run_count, 1) if run_count > 0 else 0
            ),
        }

    grouped_runs: dict[tuple[str, str, str], list[EmbeddingWorkerBatchRunRecord]] = {}
    for run in batch_runs:
        key = (
            run.profile_name or "all",
            run.provider_source,
            run.provider_mode,
        )
        grouped_runs.setdefault(key, []).append(run)

    groups = []
    for (profile_name, provider_source, provider_mode), runs in grouped_runs.items():
        metrics = metrics_for_runs(runs)
        groups.append(
            {
                "profile_name": profile_name,
                "provider_source": provider_source,
                "provider_mode": provider_mode,
                **metrics,
            }
        )

    groups.sort(
        key=lambda item: (
            -int(item["processed_count"]),
            str(item["profile_name"]),
            str(item["provider_source"]),
            str(item["provider_mode"]),
        )
    )
    return {
        "overall": metrics_for_runs(batch_runs),
        "groups": groups,
    }


def dgx_ingestion_benchmark_run_payload(
    run: DgxIngestionBenchmarkRunRecord,
) -> dict[str, object]:
    return {
        "benchmark_run_id": run.benchmark_run_id,
        "benchmark_run_key": run.benchmark_run_key,
        "script_name": run.script_name,
        "provider_names": list(run.provider_names),
        "profile_names": list(run.profile_names),
        "chunk_count": run.chunk_count,
        "expected_job_count": run.expected_job_count,
        "processed_count": run.processed_count,
        "succeeded_count": run.succeeded_count,
        "failed_count": run.failed_count,
        "vector_count": run.vector_count,
        "passed": run.passed,
        "preflight_before_worker": run.preflight_before_worker,
        "active_only_preflight": run.active_only_preflight,
        "cleanup_attempted": run.cleanup_attempted,
        "cleanup_confirmed": run.cleanup_confirmed,
        "total_elapsed_seconds": run.total_elapsed_seconds,
        "total_provider_elapsed_ms": run.total_provider_elapsed_ms,
        "total_worker_elapsed_ms": run.total_worker_elapsed_ms,
        "fixture_file_id": run.fixture_file_id,
        "fixture_document_id": run.fixture_document_id,
        "fixture_chunk_ids": list(run.fixture_chunk_ids),
        "plan_payload": _json_safe_dashboard_raw_value(run.plan_payload),
        "fixture_payload": _json_safe_dashboard_raw_value(run.fixture_payload),
        "report_payload": _json_safe_dashboard_raw_value(run.report_payload),
        "created_by": run.created_by,
        "created_by_user_id": run.created_by_user_id,
        "created_at": _datetime_response(run.created_at),
        "created_at_label": _datetime_label(run.created_at),
    }


def dgx_ingestion_benchmark_profile_payload(
    profile: DgxIngestionBenchmarkProfileRecord,
) -> dict[str, object]:
    return {
        "benchmark_profile_id": profile.benchmark_profile_id,
        "benchmark_run_id": profile.benchmark_run_id,
        "provider": profile.provider,
        "profile_name": profile.profile_name,
        "expected_job_count": profile.expected_job_count,
        "processed_count": profile.processed_count,
        "succeeded_count": profile.succeeded_count,
        "failed_count": profile.failed_count,
        "vector_count": profile.vector_count,
        "passed": profile.passed,
        "vector_table_name": profile.vector_table_name,
        "vector_dimension": profile.vector_dimension,
        "vector_storage_type": profile.vector_storage_type,
        "provider_route_id": profile.provider_route_id,
        "provider_route_name": profile.provider_route_name,
        "provider_runtime_base_url": profile.provider_runtime_base_url,
        "provider_model_id": profile.provider_model_id,
        "provider_type": profile.provider_type,
        "readiness_status": profile.readiness_status,
        "readiness_health_snapshot_id": profile.readiness_health_snapshot_id,
        "readiness_contract_snapshot_id": profile.readiness_contract_snapshot_id,
        "total_provider_elapsed_ms": profile.total_provider_elapsed_ms,
        "avg_provider_elapsed_ms": profile.avg_provider_elapsed_ms,
        "max_provider_elapsed_ms": profile.max_provider_elapsed_ms,
        "total_worker_elapsed_ms": profile.total_worker_elapsed_ms,
        "avg_worker_elapsed_ms": profile.avg_worker_elapsed_ms,
        "max_worker_elapsed_ms": profile.max_worker_elapsed_ms,
        "errors": list(profile.errors),
        "created_at": _datetime_response(profile.created_at),
    }


def dgx_ingestion_benchmark_job_payload(
    job: DgxIngestionBenchmarkJobRecord,
) -> dict[str, object]:
    return {
        "benchmark_job_result_id": job.benchmark_job_result_id,
        "benchmark_run_id": job.benchmark_run_id,
        "benchmark_profile_id": job.benchmark_profile_id,
        "provider": job.provider,
        "profile_name": job.profile_name,
        "source_job_id": job.source_job_id,
        "source_chunk_id": job.source_chunk_id,
        "processed": job.processed,
        "job_status": job.job_status,
        "vector_table_name": job.vector_table_name,
        "vector_dimension": job.vector_dimension,
        "vector_storage_type": job.vector_storage_type,
        "provider_route_id": job.provider_route_id,
        "provider_route_name": job.provider_route_name,
        "provider_runtime_base_url": job.provider_runtime_base_url,
        "provider_model_id": job.provider_model_id,
        "provider_type": job.provider_type,
        "provider_elapsed_ms": job.provider_elapsed_ms,
        "worker_elapsed_ms": job.worker_elapsed_ms,
        "readiness_status": job.readiness_status,
        "readiness_health_snapshot_id": job.readiness_health_snapshot_id,
        "readiness_contract_snapshot_id": job.readiness_contract_snapshot_id,
        "message": job.message,
        "error": job.error,
        "passed": job.passed,
        "created_at": _datetime_response(job.created_at),
    }


def dgx_ingestion_benchmark_detail_payload(
    detail: DgxIngestionBenchmarkDetail,
) -> dict[str, object]:
    return {
        "run": dgx_ingestion_benchmark_run_payload(detail.run),
        "profiles": [
            dgx_ingestion_benchmark_profile_payload(profile) for profile in detail.profiles
        ],
        "jobs": [dgx_ingestion_benchmark_job_payload(job) for job in detail.jobs],
    }


def dgx_ingestion_benchmark_summary(
    runs: list[DgxIngestionBenchmarkRunRecord],
) -> dict[str, object]:
    elapsed_values = [run.total_elapsed_seconds for run in runs if run.total_elapsed_seconds]
    passed_count = sum(1 for run in runs if run.passed)
    return {
        "run_count": len(runs),
        "passed_count": passed_count,
        "failed_count": len(runs) - passed_count,
        "expected_job_count": sum(run.expected_job_count for run in runs),
        "processed_count": sum(run.processed_count for run in runs),
        "vector_count": sum(run.vector_count for run in runs),
        "avg_elapsed_seconds": (
            round(sum(elapsed_values) / len(elapsed_values), 2) if elapsed_values else 0
        ),
    }


def _format_dgx_benchmark_metric_value(
    value: object,
    unit: str | None = None,
    *,
    signed: bool = False,
) -> str:
    if value is None:
        return "-"
    sign = ""
    if signed:
        try:
            sign = "+" if float(value) > 0 else ""
        except (TypeError, ValueError):
            sign = ""
    if unit == "s":
        return f"{sign}{float(value):.2f} s"
    if unit == "ms":
        return f"{sign}{float(value):.2f} ms"
    if isinstance(value, float) and not value.is_integer():
        return f"{sign}{value:.2f}"
    return f"{sign}{value}"


def _dgx_benchmark_metric_comparison(
    metric_key: str,
    left_value: int | float | None,
    right_value: int | float | None,
    *,
    unit: str | None = None,
    better_when: str = "neutral",
) -> dict[str, object]:
    delta_value = None
    status_name = "missing"
    if left_value is not None and right_value is not None:
        delta_value = right_value - left_value
        if abs(delta_value) < 0.000001:
            status_name = "same"
        elif better_when == "higher":
            status_name = "better" if delta_value > 0 else "worse"
        elif better_when == "lower":
            status_name = "better" if delta_value < 0 else "worse"
        else:
            status_name = "changed"
    return {
        "metric_key": metric_key,
        "left_value": left_value,
        "right_value": right_value,
        "delta_value": delta_value,
        "unit": unit,
        "status": status_name,
        "left_label": _format_dgx_benchmark_metric_value(left_value, unit),
        "right_label": _format_dgx_benchmark_metric_value(right_value, unit),
        "delta_label": _format_dgx_benchmark_metric_value(delta_value, unit, signed=True),
    }


def dgx_ingestion_benchmark_compare_payload(
    left: DgxIngestionBenchmarkDetail,
    right: DgxIngestionBenchmarkDetail,
) -> dict[str, object]:
    left_profiles = {(profile.provider, profile.profile_name): profile for profile in left.profiles}
    right_profiles = {
        (profile.provider, profile.profile_name): profile for profile in right.profiles
    }
    profile_comparisons = []
    for provider, profile_name in sorted(set(left_profiles) | set(right_profiles)):
        left_profile = left_profiles.get((provider, profile_name))
        right_profile = right_profiles.get((provider, profile_name))
        if left_profile is None:
            profile_status = "added"
        elif right_profile is None:
            profile_status = "removed"
        else:
            profile_status = "common"
        profile_comparisons.append(
            {
                "provider": provider,
                "profile_name": profile_name,
                "status": profile_status,
                "left": (
                    dgx_ingestion_benchmark_profile_payload(left_profile)
                    if left_profile is not None
                    else None
                ),
                "right": (
                    dgx_ingestion_benchmark_profile_payload(right_profile)
                    if right_profile is not None
                    else None
                ),
                "metrics": {
                    "vector_count": _dgx_benchmark_metric_comparison(
                        "vector_count",
                        left_profile.vector_count if left_profile is not None else None,
                        right_profile.vector_count if right_profile is not None else None,
                        better_when="higher",
                    ),
                    "failed_count": _dgx_benchmark_metric_comparison(
                        "failed_count",
                        left_profile.failed_count if left_profile is not None else None,
                        right_profile.failed_count if right_profile is not None else None,
                        better_when="lower",
                    ),
                    "avg_provider_elapsed_ms": _dgx_benchmark_metric_comparison(
                        "avg_provider_elapsed_ms",
                        left_profile.avg_provider_elapsed_ms if left_profile is not None else None,
                        (
                            right_profile.avg_provider_elapsed_ms
                            if right_profile is not None
                            else None
                        ),
                        unit="ms",
                        better_when="lower",
                    ),
                    "avg_worker_elapsed_ms": _dgx_benchmark_metric_comparison(
                        "avg_worker_elapsed_ms",
                        left_profile.avg_worker_elapsed_ms if left_profile is not None else None,
                        right_profile.avg_worker_elapsed_ms if right_profile is not None else None,
                        unit="ms",
                        better_when="lower",
                    ),
                },
            }
        )

    return {
        "left": dgx_ingestion_benchmark_detail_payload(left),
        "right": dgx_ingestion_benchmark_detail_payload(right),
        "run_metrics": [
            _dgx_benchmark_metric_comparison(
                "chunk_count",
                left.run.chunk_count,
                right.run.chunk_count,
            ),
            _dgx_benchmark_metric_comparison(
                "expected_job_count",
                left.run.expected_job_count,
                right.run.expected_job_count,
            ),
            _dgx_benchmark_metric_comparison(
                "processed_count",
                left.run.processed_count,
                right.run.processed_count,
                better_when="higher",
            ),
            _dgx_benchmark_metric_comparison(
                "succeeded_count",
                left.run.succeeded_count,
                right.run.succeeded_count,
                better_when="higher",
            ),
            _dgx_benchmark_metric_comparison(
                "failed_count",
                left.run.failed_count,
                right.run.failed_count,
                better_when="lower",
            ),
            _dgx_benchmark_metric_comparison(
                "vector_count",
                left.run.vector_count,
                right.run.vector_count,
                better_when="higher",
            ),
            _dgx_benchmark_metric_comparison(
                "total_elapsed_seconds",
                left.run.total_elapsed_seconds,
                right.run.total_elapsed_seconds,
                unit="s",
                better_when="lower",
            ),
            _dgx_benchmark_metric_comparison(
                "total_provider_elapsed_ms",
                left.run.total_provider_elapsed_ms,
                right.run.total_provider_elapsed_ms,
                unit="ms",
                better_when="lower",
            ),
            _dgx_benchmark_metric_comparison(
                "total_worker_elapsed_ms",
                left.run.total_worker_elapsed_ms,
                right.run.total_worker_elapsed_ms,
                unit="ms",
                better_when="lower",
            ),
        ],
        "profile_comparisons": profile_comparisons,
    }


def dgx_ingestion_benchmark_trend_summary_payload(
    details: list[DgxIngestionBenchmarkDetail],
) -> dict[str, object]:
    sorted_details = sorted(
        details,
        key=lambda detail: (
            detail.run.created_at,
            detail.run.benchmark_run_id,
        ),
    )
    points_by_profile: dict[
        tuple[str, str],
        list[dict[str, object]],
    ] = {}
    for detail in sorted_details:
        for profile in detail.profiles:
            point = {
                "benchmark_run_id": detail.run.benchmark_run_id,
                "benchmark_run_key": detail.run.benchmark_run_key,
                "created_at": _datetime_response(detail.run.created_at),
                "created_at_label": _datetime_label(detail.run.created_at),
                "passed": profile.passed,
                "expected_job_count": profile.expected_job_count,
                "processed_count": profile.processed_count,
                "succeeded_count": profile.succeeded_count,
                "failed_count": profile.failed_count,
                "vector_count": profile.vector_count,
                "avg_provider_elapsed_ms": profile.avg_provider_elapsed_ms,
                "avg_worker_elapsed_ms": profile.avg_worker_elapsed_ms,
                "provider_route_name": profile.provider_route_name,
                "readiness_status": profile.readiness_status,
            }
            points_by_profile.setdefault((profile.provider, profile.profile_name), []).append(point)

    profile_trends = []
    for (provider, profile_name), points in sorted(points_by_profile.items()):
        oldest = points[0]
        latest = points[-1]
        passed_count = sum(1 for point in points if point["passed"])
        vector_values = [int(point["vector_count"]) for point in points]
        provider_latency_values = [
            float(point["avg_provider_elapsed_ms"])
            for point in points
            if point["avg_provider_elapsed_ms"] is not None
        ]
        worker_latency_values = [
            float(point["avg_worker_elapsed_ms"])
            for point in points
            if point["avg_worker_elapsed_ms"] is not None
        ]
        profile_trends.append(
            {
                "provider": provider,
                "profile_name": profile_name,
                "run_count": len(points),
                "passed_count": passed_count,
                "failed_run_count": len(points) - passed_count,
                "total_vectors": sum(vector_values),
                "avg_vectors": (
                    round(sum(vector_values) / len(vector_values), 2) if vector_values else 0
                ),
                "avg_provider_elapsed_ms": (
                    round(sum(provider_latency_values) / len(provider_latency_values), 2)
                    if provider_latency_values
                    else None
                ),
                "avg_worker_elapsed_ms": (
                    round(sum(worker_latency_values) / len(worker_latency_values), 2)
                    if worker_latency_values
                    else None
                ),
                "oldest_point": oldest,
                "latest_point": latest,
                "deltas": {
                    "vector_count": _dgx_benchmark_metric_comparison(
                        "vector_count",
                        int(oldest["vector_count"]),
                        int(latest["vector_count"]),
                        better_when="higher",
                    ),
                    "failed_count": _dgx_benchmark_metric_comparison(
                        "failed_count",
                        int(oldest["failed_count"]),
                        int(latest["failed_count"]),
                        better_when="lower",
                    ),
                    "avg_provider_elapsed_ms": _dgx_benchmark_metric_comparison(
                        "avg_provider_elapsed_ms",
                        oldest["avg_provider_elapsed_ms"],
                        latest["avg_provider_elapsed_ms"],
                        unit="ms",
                        better_when="lower",
                    ),
                    "avg_worker_elapsed_ms": _dgx_benchmark_metric_comparison(
                        "avg_worker_elapsed_ms",
                        oldest["avg_worker_elapsed_ms"],
                        latest["avg_worker_elapsed_ms"],
                        unit="ms",
                        better_when="lower",
                    ),
                },
                "points": points,
            }
        )

    profile_trends.sort(
        key=lambda trend: (
            -int(trend["run_count"]),
            str(trend["provider"]),
            str(trend["profile_name"]),
        )
    )
    return {
        "run_count": len(sorted_details),
        "profile_count": len(profile_trends),
        "latest_created_at": (
            _datetime_response(sorted_details[-1].run.created_at) if sorted_details else None
        ),
        "latest_created_at_label": (
            _datetime_label(sorted_details[-1].run.created_at) if sorted_details else "-"
        ),
        "profiles": profile_trends,
    }


def embedding_worker_batch_run_failed_job_ids(
    batch_run: EmbeddingWorkerBatchRunRecord,
) -> list[int]:
    failed_job_ids: list[int] = []
    seen_job_ids: set[int] = set()

    def append_job_id(raw_job_id: object) -> None:
        if isinstance(raw_job_id, bool):
            return
        try:
            job_id = int(raw_job_id)
        except (TypeError, ValueError):
            return
        if job_id <= 0 or job_id in seen_job_ids:
            return
        seen_job_ids.add(job_id)
        failed_job_ids.append(job_id)

    results = batch_run.runtime_metadata.get("results")
    if isinstance(results, list):
        for result in results:
            if not isinstance(result, dict):
                continue
            if result.get("status") == "failed":
                append_job_id(result.get("job_id"))

    if not failed_job_ids and batch_run.failed_count > 0:
        for job_id in batch_run.job_ids:
            append_job_id(job_id)

    return failed_job_ids


def retry_failed_embedding_worker_batch_run_jobs(
    database_url: str,
    batch_run: EmbeddingWorkerBatchRunRecord,
) -> dict[str, object]:
    failed_job_ids = embedding_worker_batch_run_failed_job_ids(batch_run)
    retried_jobs: list[EmbeddingJobRecord] = []
    skipped_jobs: list[dict[str, object]] = []

    def skip_job(
        job_id: int,
        reason: str,
        job: EmbeddingJobRecord | None = None,
    ) -> None:
        skipped_jobs.append(
            {
                "job_id": job_id,
                "reason": reason,
                "status": job.status if job is not None else None,
                "attempts": job.attempts if job is not None else None,
                "max_attempts": job.max_attempts if job is not None else None,
            }
        )

    for job_id in failed_job_ids:
        job = get_embedding_job(database_url, job_id)
        if job is None:
            skip_job(job_id, "missing")
            continue
        if job.status != "failed":
            skip_job(job_id, "not_failed", job)
            continue
        if job.attempts >= job.max_attempts:
            skip_job(job_id, "max_attempts_reached", job)
            continue

        retried = retry_embedding_job(database_url, job_id)
        if retried is None:
            skip_job(job_id, "not_retryable", job)
            continue
        retried_jobs.append(retried)

    return {
        "batch_run_id": batch_run.batch_run_id,
        "failed_job_ids": failed_job_ids,
        "retried_count": len(retried_jobs),
        "skipped_count": len(skipped_jobs),
        "retried_jobs": [embedding_job_payload(job) for job in retried_jobs],
        "skipped_jobs": skipped_jobs,
    }


def retry_failed_embedding_jobs_for_scope(
    database_url: str,
    *,
    profile_name: str | None = None,
    limit: int = 100,
) -> dict[str, object]:
    failed_jobs = list_embedding_jobs(
        database_url,
        status="failed",
        profile_name=profile_name,
        limit=limit,
    )
    retried_jobs: list[EmbeddingJobRecord] = []
    skipped_jobs: list[dict[str, object]] = []

    def skip_job(job: EmbeddingJobRecord, reason: str) -> None:
        skipped_jobs.append(
            {
                "job_id": job.job_id,
                "reason": reason,
                "status": job.status,
                "attempts": job.attempts,
                "max_attempts": job.max_attempts,
            }
        )

    for job in failed_jobs:
        if job.attempts >= job.max_attempts:
            skip_job(job, "max_attempts_reached")
            continue

        retried = retry_embedding_job(database_url, job.job_id)
        if retried is None:
            skip_job(job, "not_retryable")
            continue
        retried_jobs.append(retried)

    return {
        "profile_name": profile_name,
        "limit": limit,
        "failed_job_count": len(failed_jobs),
        "retried_count": len(retried_jobs),
        "skipped_count": len(skipped_jobs),
        "retried_jobs": [embedding_job_payload(job) for job in retried_jobs],
        "skipped_jobs": skipped_jobs,
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


def provider_model_availability_row_payload(
    row: ProviderModelAvailabilityRow,
) -> dict[str, object]:
    return {
        "profile_name": row.profile_name,
        "model_key": row.model_key,
        "repo_id": row.repo_id,
        "local_dir": str(row.local_dir),
        "model_ready": row.model_ready,
        "model_exists": row.model_exists,
        "model_status": row.model_status,
        "route_count": row.route_count,
        "active_route_count": row.active_route_count,
        "ready_route_count": row.ready_route_count,
        "blocked_route_count": row.blocked_route_count,
        "status": row.status,
        "severity": row.severity,
        "action_code": row.action_code,
        "route_status_counts": row.route_status_counts,
        "provider_names": list(row.provider_names),
    }


def provider_model_availability_matrix_payload(
    matrix: ProviderModelAvailabilityMatrix,
) -> dict[str, object]:
    return {
        "profile_count": matrix.profile_count,
        "ready_count": matrix.ready_count,
        "blocked_count": matrix.blocked_count,
        "status_counts": matrix.status_counts,
        "rows": [provider_model_availability_row_payload(row) for row in matrix.rows],
    }


def provider_model_availability_drilldown_payload(
    drilldown: ProviderModelAvailabilityDrilldown,
) -> dict[str, object]:
    return {
        "profile_name": drilldown.row.profile_name,
        "availability": provider_model_availability_row_payload(drilldown.row),
        "route_count": drilldown.route_count,
        "routes": [
            embedding_provider_route_readiness_item_payload(item)
            for item in drilldown.route_readiness
        ],
        "latest_preflight_run": (
            embedding_provider_preflight_run_payload(drilldown.latest_preflight_run)
            if drilldown.latest_preflight_run is not None
            else None
        ),
    }


def embedding_coverage_profile_cell_payload(
    cell: EmbeddingCoverageProfileCell,
) -> dict[str, object]:
    return {
        "profile_name": cell.profile_name,
        "model_name": cell.model_name,
        "dimension": cell.dimension,
        "storage_type": cell.storage_type,
        "is_active": cell.is_active,
        "chunk_count": cell.chunk_count,
        "job_count": cell.job_count,
        "pending_count": cell.pending_count,
        "running_count": cell.running_count,
        "failed_count": cell.failed_count,
        "retryable_failed_count": cell.retryable_failed_count,
        "exhausted_failed_count": cell.exhausted_failed_count,
        "succeeded_job_count": cell.succeeded_job_count,
        "skipped_count": cell.skipped_count,
        "embedded_chunk_count": cell.embedded_chunk_count,
        "coverage_percent": _percent_value(cell.coverage_percent),
        "coverage_label": _percent_label(cell.coverage_percent),
        "status": cell.status,
        "latest_job_updated_at": _datetime_response(cell.latest_job_updated_at),
        "latest_embedding_at": _datetime_response(cell.latest_embedding_at),
        "average_embedding_elapsed_ms": (
            str(cell.average_embedding_elapsed_ms)
            if cell.average_embedding_elapsed_ms is not None
            else None
        ),
    }


def bm25_index_coverage_row_payload(row: BM25IndexCoverageRow) -> dict[str, object]:
    return {
        "document_id": row.document_id,
        "file_id": row.file_id,
        "document_title": row.document_title,
        "original_file_name": row.original_file_name,
        "file_ext": row.file_ext,
        "document_group": row.document_group,
        "parse_status": row.parse_status,
        "access_scope": row.access_scope,
        "uploaded_at": _datetime_response(row.uploaded_at),
        "chunk_policy_name": row.chunk_policy_name,
        "target_token_size": row.target_token_size,
        "overlap_token_size": row.overlap_token_size,
        "split_strategy": row.split_strategy,
        "tokenizer_name": row.tokenizer_name,
        "policy_chunk_count": row.policy_chunk_count,
        "chunk_count": row.chunk_count,
        "indexed_chunk_count": row.indexed_chunk_count,
        "missing_chunk_count": row.missing_chunk_count,
        "term_row_count": row.term_row_count,
        "statistics_term_count": row.statistics_term_count,
        "statistics_corpus_chunk_count": row.statistics_corpus_chunk_count,
        "average_document_length": (
            str(row.average_document_length) if row.average_document_length is not None else None
        ),
        "coverage_percent": _percent_value(row.coverage_percent),
        "coverage_label": _percent_label(row.coverage_percent),
        "status": row.status,
        "latest_term_created_at": _datetime_response(row.latest_term_created_at),
        "latest_statistics_updated_at": _datetime_response(row.latest_statistics_updated_at),
    }


def bm25_index_coverage_policy_summary_payload(
    policy: BM25IndexCoveragePolicySummary,
) -> dict[str, object]:
    return {
        "chunk_policy_name": policy.chunk_policy_name,
        "tokenizer_name": policy.tokenizer_name,
        "document_count": policy.document_count,
        "chunked_document_count": policy.chunked_document_count,
        "complete_document_count": policy.complete_document_count,
        "partial_document_count": policy.partial_document_count,
        "missing_document_count": policy.missing_document_count,
        "stale_document_count": policy.stale_document_count,
        "not_chunked_document_count": policy.not_chunked_document_count,
        "total_chunk_count": policy.total_chunk_count,
        "indexed_chunk_count": policy.indexed_chunk_count,
        "missing_chunk_count": policy.missing_chunk_count,
        "term_row_count": policy.term_row_count,
        "statistics_term_count": policy.statistics_term_count,
        "statistics_corpus_chunk_count": policy.statistics_corpus_chunk_count,
        "coverage_percent": _percent_value(policy.coverage_percent),
        "coverage_label": _percent_label(policy.coverage_percent),
        "latest_term_created_at": _datetime_response(policy.latest_term_created_at),
        "latest_statistics_updated_at": _datetime_response(policy.latest_statistics_updated_at),
    }


def bm25_index_coverage_matrix_payload(
    matrix: BM25IndexCoverageMatrix,
) -> dict[str, object]:
    summary = matrix.summary
    return {
        "summary": {
            "document_count": summary.document_count,
            "policy_count": summary.policy_count,
            "document_policy_count": summary.document_policy_count,
            "total_chunk_count": summary.total_chunk_count,
            "indexed_chunk_count": summary.indexed_chunk_count,
            "missing_chunk_count": summary.missing_chunk_count,
            "term_row_count": summary.term_row_count,
            "statistics_term_count": summary.statistics_term_count,
            "complete_row_count": summary.complete_row_count,
            "attention_row_count": summary.attention_row_count,
            "stale_row_count": summary.stale_row_count,
            "missing_row_count": summary.missing_row_count,
            "coverage_percent": _percent_value(summary.coverage_percent),
            "coverage_label": _percent_label(summary.coverage_percent),
            "latest_term_created_at": _datetime_response(summary.latest_term_created_at),
            "latest_statistics_updated_at": _datetime_response(
                summary.latest_statistics_updated_at
            ),
            "policies": [
                bm25_index_coverage_policy_summary_payload(policy) for policy in summary.policies
            ],
        },
        "rows": [bm25_index_coverage_row_payload(row) for row in matrix.rows],
    }


def _bm25_index_coverage_redirect_url(params: dict[str, object]) -> str:
    clean_params = {
        key: value for key, value in params.items() if value is not None and value != ""
    }
    if not clean_params:
        return "/admin/bm25-index-coverage"
    return f"/admin/bm25-index-coverage?{urlencode(clean_params)}"


def _generation_redirect_url(params: dict[str, object]) -> str:
    clean_params = {
        key: value for key, value in params.items() if value is not None and value != ""
    }
    if not clean_params:
        return "/generation"
    return f"/generation?{urlencode(clean_params)}"


def _generation_templates_redirect_url(params: dict[str, object]) -> str:
    clean_params = {
        key: value for key, value in params.items() if value is not None and value != ""
    }
    if not clean_params:
        return "/admin/generation-templates"
    return f"/admin/generation-templates?{urlencode(clean_params)}"


def embedding_coverage_document_payload(
    document: EmbeddingCoverageDocument,
) -> dict[str, object]:
    return {
        "document_id": document.document_id,
        "file_id": document.file_id,
        "document_title": document.document_title,
        "original_file_name": document.original_file_name,
        "file_ext": document.file_ext,
        "document_group": document.document_group,
        "parse_status": document.parse_status,
        "access_scope": document.access_scope,
        "chunk_count": document.chunk_count,
        "uploaded_at": _datetime_response(document.uploaded_at),
        "complete_profile_count": document.complete_profile_count,
        "attention_profile_count": document.attention_profile_count,
        "missing_profile_count": document.missing_profile_count,
        "profiles": [embedding_coverage_profile_cell_payload(cell) for cell in document.profiles],
    }


def embedding_coverage_profile_summary_payload(
    summary: EmbeddingCoverageProfileSummary,
) -> dict[str, object]:
    return {
        "profile_name": summary.profile_name,
        "model_name": summary.model_name,
        "document_count": summary.document_count,
        "complete_document_count": summary.complete_document_count,
        "partial_document_count": summary.partial_document_count,
        "pending_document_count": summary.pending_document_count,
        "running_document_count": summary.running_document_count,
        "failed_document_count": summary.failed_document_count,
        "missing_document_count": summary.missing_document_count,
        "not_chunked_document_count": summary.not_chunked_document_count,
        "total_chunk_count": summary.total_chunk_count,
        "embedded_chunk_count": summary.embedded_chunk_count,
        "coverage_percent": _percent_value(summary.coverage_percent),
        "coverage_label": _percent_label(summary.coverage_percent),
    }


def embedding_coverage_matrix_payload(
    matrix: EmbeddingCoverageMatrix,
) -> dict[str, object]:
    summary = matrix.summary
    return {
        "summary": {
            "document_count": summary.document_count,
            "profile_count": summary.profile_count,
            "total_chunk_count": summary.total_chunk_count,
            "expected_embedding_count": summary.expected_embedding_count,
            "embedded_chunk_count": summary.embedded_chunk_count,
            "complete_cell_count": summary.complete_cell_count,
            "incomplete_cell_count": summary.incomplete_cell_count,
            "attention_cell_count": summary.attention_cell_count,
            "coverage_percent": _percent_value(summary.coverage_percent),
            "coverage_label": _percent_label(summary.coverage_percent),
            "profiles": [
                embedding_coverage_profile_summary_payload(profile)
                for profile in summary.profile_summaries
            ],
        },
        "documents": [
            embedding_coverage_document_payload(document) for document in matrix.documents
        ],
    }


def multi_policy_ingestion_coverage_policy_summary_payload(
    summary: MultiPolicyIngestionCoveragePolicySummary,
) -> dict[str, object]:
    return {
        "chunk_policy_name": summary.chunk_policy_name,
        "document_count": summary.document_count,
        "chunked_document_count": summary.chunked_document_count,
        "complete_document_count": summary.complete_document_count,
        "attention_document_count": summary.attention_document_count,
        "total_chunk_count": summary.total_chunk_count,
        "expected_embedding_count": summary.expected_embedding_count,
        "embedded_chunk_count": summary.embedded_chunk_count,
        "complete_cell_count": summary.complete_cell_count,
        "attention_cell_count": summary.attention_cell_count,
        "not_chunked_cell_count": summary.not_chunked_cell_count,
        "coverage_percent": _percent_value(summary.coverage_percent),
        "coverage_label": _percent_label(summary.coverage_percent),
    }


def multi_policy_ingestion_coverage_row_payload(
    row: MultiPolicyIngestionCoverageRow,
) -> dict[str, object]:
    return {
        "document_id": row.document_id,
        "file_id": row.file_id,
        "document_title": row.document_title,
        "original_file_name": row.original_file_name,
        "file_ext": row.file_ext,
        "document_group": row.document_group,
        "parse_status": row.parse_status,
        "access_scope": row.access_scope,
        "chunk_policy_name": row.chunk_policy_name,
        "target_token_size": row.target_token_size,
        "overlap_token_size": row.overlap_token_size,
        "split_strategy": row.split_strategy,
        "chunk_count": row.chunk_count,
        "uploaded_at": _datetime_response(row.uploaded_at),
        "complete_profile_count": row.complete_profile_count,
        "attention_profile_count": row.attention_profile_count,
        "missing_profile_count": row.missing_profile_count,
        "profiles": [embedding_coverage_profile_cell_payload(cell) for cell in row.profiles],
    }


def multi_policy_ingestion_coverage_matrix_payload(
    matrix: MultiPolicyIngestionCoverageMatrix,
) -> dict[str, object]:
    summary = matrix.summary
    return {
        "summary": {
            "document_count": summary.document_count,
            "policy_count": summary.policy_count,
            "profile_count": summary.profile_count,
            "document_policy_count": summary.document_policy_count,
            "total_chunk_count": summary.total_chunk_count,
            "expected_embedding_count": summary.expected_embedding_count,
            "embedded_chunk_count": summary.embedded_chunk_count,
            "complete_cell_count": summary.complete_cell_count,
            "incomplete_cell_count": summary.incomplete_cell_count,
            "attention_cell_count": summary.attention_cell_count,
            "coverage_percent": _percent_value(summary.coverage_percent),
            "coverage_label": _percent_label(summary.coverage_percent),
            "policies": [
                multi_policy_ingestion_coverage_policy_summary_payload(policy)
                for policy in summary.policy_summaries
            ],
        },
        "rows": [multi_policy_ingestion_coverage_row_payload(row) for row in matrix.rows],
    }


def multi_policy_ingestion_coverage_chunk_detail_payload(
    chunk: MultiPolicyIngestionCoverageChunkDetail,
) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "chunk_seq": chunk.chunk_seq,
        "chunk_type": chunk.chunk_type,
        "token_count": chunk.token_count,
        "char_count": chunk.char_count,
        "chunk_preview": chunk.chunk_preview,
        "heading_path": list(chunk.heading_path),
        "page_no": chunk.page_no,
        "slide_no": chunk.slide_no,
        "sheet_name": chunk.sheet_name,
        "cell_range": chunk.cell_range,
        "source_char_start": chunk.source_char_start,
        "source_char_end": chunk.source_char_end,
        "job_id": chunk.job_id,
        "job_status": chunk.job_status,
        "attempts": chunk.attempts,
        "max_attempts": chunk.max_attempts,
        "error_code": chunk.error_code,
        "error_message": chunk.error_message,
        "job_updated_at": _datetime_response(chunk.job_updated_at),
        "job_started_at": _datetime_response(chunk.job_started_at),
        "job_finished_at": _datetime_response(chunk.job_finished_at),
        "vector_present": chunk.vector_present,
        "vector_elapsed_ms": chunk.vector_elapsed_ms,
        "vector_created_at": _datetime_response(chunk.vector_created_at),
    }


def multi_policy_ingestion_coverage_detail_payload(
    detail: MultiPolicyIngestionCoverageDetail,
) -> dict[str, object]:
    return {
        "document": {
            "document_id": detail.document_id,
            "file_id": detail.file_id,
            "document_title": detail.document_title,
            "original_file_name": detail.original_file_name,
            "file_ext": detail.file_ext,
            "document_group": detail.document_group,
            "parse_status": detail.parse_status,
            "access_scope": detail.access_scope,
            "uploaded_at": _datetime_response(detail.uploaded_at),
        },
        "chunk_policy": {
            "chunk_policy_name": detail.chunk_policy_name,
            "target_token_size": detail.target_token_size,
            "overlap_token_size": detail.overlap_token_size,
            "split_strategy": detail.split_strategy,
        },
        "profile": embedding_coverage_profile_cell_payload(detail.profile),
        "chunks": [
            multi_policy_ingestion_coverage_chunk_detail_payload(chunk) for chunk in detail.chunks
        ],
    }


def missing_embedding_job_reconcile_result_payload(
    result: MissingEmbeddingJobReconcileResult,
) -> dict[str, object]:
    return {
        "document_id": result.document_id,
        "chunk_policy_name": result.chunk_policy_name,
        "profile_name": result.profile_name,
        "chunk_count": result.chunk_count,
        "existing_job_count": result.existing_job_count,
        "missing_job_count": result.missing_job_count,
        "created_job_count": result.created_job_count,
        "created_jobs": [embedding_job_payload(job) for job in result.created_jobs],
    }


def failed_embedding_job_retry_result_payload(
    result: FailedEmbeddingJobRetryResult,
) -> dict[str, object]:
    return {
        "document_id": result.document_id,
        "chunk_policy_name": result.chunk_policy_name,
        "profile_name": result.profile_name,
        "failed_job_count": result.failed_job_count,
        "retryable_failed_job_count": result.retryable_failed_job_count,
        "retried_job_count": result.retried_job_count,
        "retried_jobs": [embedding_job_payload(job) for job in result.retried_jobs],
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
    request_metadata = describe_embedding_provider_route_request_metadata(route.runtime_metadata)
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
        "request_header_names": sorted(request_metadata.request_headers),
        "auth_type": request_metadata.auth_type,
        "auth_token_env": request_metadata.auth_token_env,
        "auth_header_name": request_metadata.auth_header_name,
        "created_at": _datetime_response(route.created_at),
        "updated_at": _datetime_response(route.updated_at),
    }


def embedding_provider_route_portable_payload(
    route: EmbeddingProviderRouteRecord | EmbeddingProviderRouteInput,
) -> dict[str, object]:
    return {
        "profile_name": route.profile_name,
        "provider_name": route.provider_name,
        "provider_mode": route.provider_mode,
        "provider_base_url": route.provider_base_url,
        "timeout_seconds": route.timeout_seconds,
        "priority": route.priority,
        "is_active": route.is_active,
        "health_check_enabled": route.health_check_enabled,
        "runtime_metadata": route.runtime_metadata or {},
    }


def embedding_provider_route_export_payload(
    routes: list[EmbeddingProviderRouteRecord],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "exported_at": _datetime_response(datetime.now(UTC)),
        "route_count": len(routes),
        "routes": [embedding_provider_route_portable_payload(route) for route in routes],
    }


def embedding_provider_preset_payload(
    preset: EmbeddingProviderPreset,
) -> dict[str, object]:
    return {
        "preset_name": preset.preset_name,
        "provider_name": preset.provider_name,
        "backend": preset.backend,
        "model_key": preset.model_key,
        "provider_model_id": preset.provider_model_id,
        "profile_names": list(preset.profile_names),
        "default_host": preset.default_host,
        "default_port": preset.default_port,
        "default_base_url": preset.default_base_url,
    }


def embedding_provider_preset_route_plan_payload(
    plan: EmbeddingProviderPresetRoutePlan,
) -> dict[str, object]:
    return {
        "preset_name": plan.preset_name,
        "profile_name": plan.profile_name,
        "provider_name": plan.provider_name,
        "provider_mode": plan.provider_mode,
        "provider_base_url": plan.provider_base_url,
        "provider_port": plan.provider_port,
        "timeout_seconds": plan.timeout_seconds,
        "priority": plan.priority,
        "is_active": plan.is_active,
        "health_check_enabled": plan.health_check_enabled,
        "runtime_metadata": plan.runtime_metadata,
    }


def embedding_provider_launch_plan_payload(
    plan: EmbeddingProviderLaunchPlan,
) -> dict[str, object]:
    return {
        "preset_name": plan.preset_name,
        "provider_name": plan.provider_name,
        "backend": plan.backend,
        "model_key": plan.model_key,
        "profile_names": list(plan.profile_names),
        "provider_model_id": plan.provider_model_id,
        "host": plan.host,
        "port": plan.port,
        "base_url": plan.base_url,
        "device": plan.device,
        "models_dir": plan.models_dir,
        "command": list(plan.command),
        "environment": plan.environment,
        "shell_command": plan.shell_command,
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
        "recovery_action": embedding_provider_route_readiness_recovery_action(item),
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


def embedding_provider_route_readiness_recovery_action(
    item: EmbeddingProviderRouteReadinessItem,
) -> str:
    if item.ready:
        return "ready_for_worker"
    if item.status == "inactive":
        return "activate_route"
    if item.status == "needs_contract":
        return "run_preflight"
    if item.status == "contract_failed":
        return "review_contract_snapshot"
    if item.status == "health_not_ready":
        return "check_provider_health"
    return "run_preflight"


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
    unacknowledged_alert_count: int = 0,
) -> dict[str, object]:
    failed_schedule_count = sum(
        1 for schedule in schedules if schedule.last_status in {"failed", "error"}
    )
    overall_status, overall_status_reason = embedding_provider_route_operations_status(
        readiness=readiness,
        due_schedule_count=len(due_schedules),
        failed_schedule_count=failed_schedule_count,
        latest_run=latest_run,
        unacknowledged_alert_count=unacknowledged_alert_count,
    )
    return {
        "overall_status": overall_status,
        "overall_status_reason": overall_status_reason,
        "route_count": readiness.route_count,
        "active_route_count": readiness.active_count,
        "ready_route_count": readiness.ready_count,
        "blocked_route_count": readiness.blocked_count,
        "needs_preflight_count": readiness.needs_preflight_count,
        "unacknowledged_alert_count": unacknowledged_alert_count,
        "schedule_count": len(schedules),
        "enabled_schedule_count": sum(1 for schedule in schedules if schedule.is_enabled),
        "due_schedule_count": len(due_schedules),
        "failed_schedule_count": failed_schedule_count,
        "latest_preflight_run": (
            embedding_provider_preflight_run_payload(latest_run) if latest_run is not None else None
        ),
    }


def embedding_provider_route_operations_status(
    *,
    readiness: EmbeddingProviderRouteReadinessSummary,
    due_schedule_count: int,
    failed_schedule_count: int,
    latest_run: EmbeddingProviderPreflightRunRecord | None,
    unacknowledged_alert_count: int,
) -> tuple[str, str]:
    if readiness.active_count == 0:
        return "blocked", "no_active_routes"
    if readiness.blocked_count > 0:
        return "blocked", "blocked_routes"
    if unacknowledged_alert_count > 0:
        return "attention", "unacknowledged_alerts"
    if failed_schedule_count > 0:
        return "attention", "failed_schedules"
    if latest_run is not None and latest_run.status in {"failed", "error"}:
        return "attention", "latest_preflight_failed"
    if due_schedule_count > 0:
        return "attention", "due_schedules"
    return "ready", "ready"


def read_provider_operations_playbook_markdown() -> str:
    return PROVIDER_OPERATIONS_PLAYBOOK_PATH.read_text(encoding="utf-8")


def read_operations_runbook_markdown() -> str:
    return OPERATIONS_RUNBOOK_PATH.read_text(encoding="utf-8")


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


def embedding_batch_run_retention_settings_payload(
    retention_settings: EmbeddingBatchRunRetentionSettings,
) -> dict[str, object]:
    return {
        "enabled": retention_settings.enabled,
        "retention_days": retention_settings.retention_days,
        "cleanup_batch_size": retention_settings.cleanup_batch_size,
    }


def embedding_batch_run_retention_settings_input_from_request(
    payload: EmbeddingBatchRunRetentionSettingsRequest,
) -> EmbeddingBatchRunRetentionSettingsInput:
    return EmbeddingBatchRunRetentionSettingsInput(
        enabled=payload.enabled,
        retention_days=payload.retention_days,
        cleanup_batch_size=payload.cleanup_batch_size,
    )


def embedding_batch_run_cleanup_result_payload(
    result: EmbeddingBatchRunCleanupResult,
) -> dict[str, object]:
    return {
        "enabled": result.enabled,
        "dry_run": result.dry_run,
        "retention_days": result.retention_days,
        "cleanup_batch_size": result.cleanup_batch_size,
        "expired_count": result.expired_count,
        "deleted_count": result.deleted_count,
        "expired_batch_run_count": result.expired_batch_run_count,
        "deleted_batch_run_count": result.deleted_batch_run_count,
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


PROVIDER_ROUTE_CHANGE_EVENT_BY_ACTION = {
    "created": "embedding_provider_route_created",
    "updated": "embedding_provider_route_updated",
    "activation_changed": "embedding_provider_route_activation_changed",
}

PROVIDER_ROUTE_CHANGE_MESSAGE_BY_ACTION = {
    "created": "created",
    "updated": "updated",
    "activation_changed": "activation changed",
}


def embedding_provider_route_audit_snapshot(
    route: EmbeddingProviderRouteRecord,
) -> dict[str, object]:
    request_metadata = describe_embedding_provider_route_request_metadata(route.runtime_metadata)
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
        "request_header_names": sorted(request_metadata.request_headers),
        "auth_type": request_metadata.auth_type,
        "auth_header_name": request_metadata.auth_header_name,
        "runtime_metadata_keys": sorted(route.runtime_metadata.keys()),
    }


def embedding_provider_route_changed_fields(
    previous_route: EmbeddingProviderRouteRecord | None,
    route: EmbeddingProviderRouteRecord,
) -> list[str]:
    current = embedding_provider_route_audit_snapshot(route)
    if previous_route is None:
        return sorted(field for field in current if field != "route_id")
    previous = embedding_provider_route_audit_snapshot(previous_route)
    return sorted(
        field
        for field, value in current.items()
        if field != "route_id" and previous.get(field) != value
    )


def log_embedding_provider_route_change(
    database_url: str,
    *,
    action: str,
    route: EmbeddingProviderRouteRecord,
    previous_route: EmbeddingProviderRouteRecord | None = None,
    request_path: str | None = None,
) -> int | None:
    event_type = PROVIDER_ROUTE_CHANGE_EVENT_BY_ACTION[action]
    current = embedding_provider_route_audit_snapshot(route)
    previous = (
        embedding_provider_route_audit_snapshot(previous_route)
        if previous_route is not None
        else None
    )
    changed_fields = embedding_provider_route_changed_fields(previous_route, route)
    return log_event(
        database_url,
        level="INFO",
        event_type=event_type,
        source="embedding_provider_routes",
        message=(
            f"Embedding provider route {route.provider_name} "
            f"{PROVIDER_ROUTE_CHANGE_MESSAGE_BY_ACTION[action]}."
        ),
        detail={
            "action": action,
            "route_id": route.route_id,
            "profile_name": route.profile_name,
            "provider_name": route.provider_name,
            "previous_profile_name": previous_route.profile_name if previous_route else None,
            "previous_provider_name": previous_route.provider_name if previous_route else None,
            "changed_fields": changed_fields,
            "current": current,
            "previous": previous,
            "changed_by": "operator",
        },
        request_path=request_path,
        correlation_id=f"embedding-provider-route:{route.route_id}:change",
    )


def find_existing_embedding_provider_route_for_input(
    database_url: str,
    route_input: EmbeddingProviderRouteInput,
) -> EmbeddingProviderRouteRecord | None:
    existing_routes = list_embedding_provider_routes(
        database_url,
        profile_name=route_input.profile_name,
    )
    return next(
        (route for route in existing_routes if route.provider_name == route_input.provider_name),
        None,
    )


def upsert_embedding_provider_route_with_audit(
    database_url: str,
    route_input: EmbeddingProviderRouteInput,
    *,
    request_path: str,
) -> EmbeddingProviderRouteRecord:
    validated_input = validate_embedding_provider_route_input(route_input)
    previous_route = find_existing_embedding_provider_route_for_input(
        database_url,
        validated_input,
    )
    route = upsert_embedding_provider_route(database_url, validated_input)
    log_embedding_provider_route_change(
        database_url,
        action="updated" if previous_route else "created",
        route=route,
        previous_route=previous_route,
        request_path=request_path,
    )
    return route


def update_embedding_provider_route_with_audit(
    database_url: str,
    route_id: int,
    route_input: EmbeddingProviderRouteInput,
    *,
    request_path: str,
) -> EmbeddingProviderRouteRecord | None:
    previous_route = get_embedding_provider_route(database_url, route_id)
    if previous_route is None:
        return None
    route = update_embedding_provider_route(database_url, route_id, route_input)
    if route is not None:
        log_embedding_provider_route_change(
            database_url,
            action="updated",
            route=route,
            previous_route=previous_route,
            request_path=request_path,
        )
    return route


def set_embedding_provider_route_active_with_audit(
    database_url: str,
    route_id: int,
    is_active: bool,
    *,
    request_path: str,
) -> EmbeddingProviderRouteRecord | None:
    previous_route = get_embedding_provider_route(database_url, route_id)
    if previous_route is None:
        return None
    route = set_embedding_provider_route_active(database_url, route_id, is_active)
    if route is not None and previous_route.is_active != route.is_active:
        log_embedding_provider_route_change(
            database_url,
            action="activation_changed",
            route=route,
            previous_route=previous_route,
            request_path=request_path,
        )
    return route


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


def embedding_provider_route_runtime_metadata_from_form(
    *,
    request_headers_json: str | None,
    auth_type: str,
    auth_token_env: str | None,
    auth_header_name: str | None,
) -> dict[str, object]:
    runtime_metadata: dict[str, object] = {}
    headers_text = (request_headers_json or "").strip()
    if headers_text and headers_text != "{}":
        try:
            request_headers = json.loads(headers_text)
        except json.JSONDecodeError as exc:
            raise InvalidEmbeddingProviderRouteError(
                "request_headers_json must be a JSON object"
            ) from exc
        if not isinstance(request_headers, dict):
            raise InvalidEmbeddingProviderRouteError("request_headers_json must be a JSON object")
        runtime_metadata["request_headers"] = request_headers

    selected_auth_type = auth_type.strip().lower() if auth_type else AUTH_TYPE_NONE
    if selected_auth_type == AUTH_TYPE_BEARER:
        runtime_metadata["auth"] = {
            "type": AUTH_TYPE_BEARER,
            "token_env": auth_token_env or "",
        }
    elif selected_auth_type == AUTH_TYPE_API_KEY:
        runtime_metadata["auth"] = {
            "type": AUTH_TYPE_API_KEY,
            "key_env": auth_token_env or "",
            "header_name": auth_header_name or "X-API-Key",
        }
    elif selected_auth_type != AUTH_TYPE_NONE:
        raise InvalidEmbeddingProviderRouteError(f"Unsupported auth type: {selected_auth_type}")

    try:
        return normalize_embedding_provider_route_metadata(runtime_metadata)
    except InvalidEmbeddingProviderRouteAuthError as exc:
        raise InvalidEmbeddingProviderRouteError(str(exc)) from exc


def embedding_provider_route_import_inputs_from_request(
    payload: EmbeddingProviderRouteImportRequest,
) -> list[EmbeddingProviderRouteInput]:
    return [
        validate_embedding_provider_route_input(embedding_provider_route_input_from_request(route))
        for route in payload.routes
    ]


def embedding_provider_preset_route_plans_from_request(
    payload: EmbeddingProviderRoutePresetRegistrationRequest,
) -> tuple[EmbeddingProviderPresetRoutePlan, ...]:
    preset = get_embedding_provider_preset(payload.preset_name)
    return build_embedding_provider_preset_route_plans(
        preset,
        host=payload.host,
        port=payload.port,
        base_url=payload.base_url,
        provider_name=payload.provider_name,
        timeout_seconds=payload.timeout_seconds,
        priority=payload.priority,
        is_active=payload.is_active,
        health_check_enabled=payload.health_check_enabled,
        runtime_metadata=payload.runtime_metadata,
        metadata_source="preset_registration_ui",
    )


def embedding_provider_launch_plan_from_request(
    payload: EmbeddingProviderLaunchPlanRequest,
    settings: Settings,
) -> EmbeddingProviderLaunchPlan:
    preset = get_embedding_provider_preset(payload.preset_name)
    return build_embedding_provider_launch_plan(
        preset,
        python_bin=payload.python_bin,
        host=payload.host,
        port=payload.port,
        device=payload.device,
        models_dir=payload.models_dir or str(settings.embedding_models_dir),
        provider_model_id=payload.provider_model_id,
        reload=payload.reload,
    )


def run_registered_embedding_provider_route_preflight(
    database_url: str,
    routes: list[EmbeddingProviderRouteRecord],
) -> dict[str, object]:
    started_at = datetime.now(UTC)
    started_perf = perf_counter()
    sample_set = get_default_embedding_provider_contract_sample_set(database_url)
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
                database_url,
                contract.health,
            )
            log_embedding_provider_route_health_alert(database_url, contract.health)
        contract_snapshot = record_embedding_provider_route_contract_snapshot(
            database_url,
            contract,
        )
        log_embedding_provider_route_contract_alert(database_url, contract)
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

    passed_count = sum(1 for result in results if result["contract"]["passed"])
    response_content: dict[str, object] = {
        "route_count": len(routes),
        "passed_count": passed_count,
        "failed_count": len(routes) - passed_count,
        "sample_set": embedding_provider_contract_sample_set_payload(sample_set),
        "results": results,
        "trigger_source": "preset_registration",
    }
    completed_at = datetime.now(UTC)
    elapsed_ms = int((perf_counter() - started_perf) * 1000)
    preflight_run = record_embedding_provider_preflight_run(
        database_url,
        EmbeddingProviderPreflightRunInput(
            trigger_source="manual_api",
            status="succeeded" if response_content["failed_count"] == 0 else "failed",
            result=response_content,
            profile_name=None,
            active_only=False,
            elapsed_ms=elapsed_ms,
            started_at=started_at,
            completed_at=completed_at,
        ),
    )
    response_content["preflight_run"] = embedding_provider_preflight_run_payload(preflight_run)
    return response_content


def vector_search_result_payload(result: VectorSearchResult) -> dict[str, object]:
    payload = {
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
    if hasattr(result, "search_profile_name"):
        payload["search_profile_name"] = result.search_profile_name
    if hasattr(result, "retrieval_strategy"):
        payload["retrieval_strategy"] = result.retrieval_strategy
    if hasattr(result, "score_components"):
        payload["score_components"] = result.score_components
    if hasattr(result, "matched_term_count"):
        payload["matched_term_count"] = result.matched_term_count
    if hasattr(result, "document_length"):
        payload["document_length"] = result.document_length
    return payload


def search_compare_profile_payload(profile: SearchCompareProfileResult) -> dict[str, object]:
    return {
        "profile_name": profile.profile_name,
        "status": profile.status,
        "error_code": profile.error_code,
        "error_message": profile.error_message,
        "elapsed_ms": profile.elapsed_ms,
        "query_runtime_metadata": profile.query_runtime_metadata,
        "results": [
            {
                **vector_search_result_payload(result.vector_result),
                "search_log_result_id": result.search_log_result_id,
            }
            for result in profile.results
        ],
    }


def search_compare_profile_status_counts(
    profiles: tuple[SearchCompareProfileResult, ...],
) -> dict[str, int]:
    profile_status_counts: dict[str, int] = {"succeeded": 0, "failed": 0}
    for profile in profiles:
        profile_status_counts[profile.status] = profile_status_counts.get(profile.status, 0) + 1
    return profile_status_counts


def search_compare_readiness_profile_payload(
    profile: SearchCompareReadinessProfile,
    *,
    coverage_url: str | None = None,
) -> dict[str, object]:
    return {
        "profile_name": profile.profile_name,
        "chunk_policy_name": profile.chunk_policy_name,
        "chunk_count": profile.chunk_count,
        "job_count": profile.job_count,
        "pending_count": profile.pending_count,
        "running_count": profile.running_count,
        "failed_count": profile.failed_count,
        "succeeded_job_count": profile.succeeded_job_count,
        "skipped_count": profile.skipped_count,
        "embedded_chunk_count": profile.embedded_chunk_count,
        "missing_embedding_count": profile.missing_embedding_count,
        "coverage_percent": _percent_value(profile.coverage_percent),
        "coverage_label": _percent_label(profile.coverage_percent),
        "status": profile.status,
        "ready": profile.ready,
        "latest_job_updated_at": _datetime_response(profile.latest_job_updated_at),
        "latest_embedding_at": _datetime_response(profile.latest_embedding_at),
        "average_embedding_elapsed_ms": (
            str(profile.average_embedding_elapsed_ms)
            if profile.average_embedding_elapsed_ms is not None
            else None
        ),
        "coverage_url": coverage_url,
    }


def search_compare_readiness_coverage_url(
    readiness: SearchCompareReadinessResult,
    profile: SearchCompareReadinessProfile,
) -> str | None:
    if profile.chunk_policy_name is None:
        return None
    query_params = {
        "chunk_policy_name": profile.chunk_policy_name,
        "profile_name": profile.profile_name,
    }
    if readiness.document_group:
        query_params["document_group"] = readiness.document_group
    return f"/admin/multi-policy-ingestion-coverage?{urlencode(query_params)}"


def search_compare_readiness_payload(
    readiness: SearchCompareReadinessResult,
) -> dict[str, object]:
    return {
        "actor_user_id": readiness.actor_user_id,
        "requested_search_scope": readiness.requested_search_scope,
        "effective_search_scope": readiness.effective_search_scope,
        "document_group": readiness.document_group,
        "file_type": readiness.file_type,
        "chunk_policy_names": list(readiness.chunk_policy_names),
        "profile_count": readiness.profile_count,
        "policy_count": readiness.policy_count,
        "expected_embedding_count": readiness.expected_embedding_count,
        "embedded_chunk_count": readiness.embedded_chunk_count,
        "attention_count": readiness.attention_count,
        "coverage_percent": _percent_value(readiness.coverage_percent),
        "coverage_label": _percent_label(readiness.coverage_percent),
        "ready": readiness.ready,
        "profiles": [
            search_compare_readiness_profile_payload(
                profile,
                coverage_url=search_compare_readiness_coverage_url(readiness, profile),
            )
            for profile in readiness.profiles
        ],
    }


def search_compare_coverage_reconcile_payload(
    result: SearchCompareCoverageReconcileResult,
) -> dict[str, object]:
    return {
        "actor_user_id": result.actor_user_id,
        "requested_search_scope": result.requested_search_scope,
        "effective_search_scope": result.effective_search_scope,
        "profile_name": result.profile_name,
        "chunk_policy_name": result.chunk_policy_name,
        "document_group": result.document_group,
        "file_type": result.file_type,
        "chunk_count": result.chunk_count,
        "existing_job_count": result.existing_job_count,
        "missing_job_count": result.missing_job_count,
        "created_job_count": result.created_job_count,
        "failed_job_count": result.failed_job_count,
        "retryable_failed_job_count": result.retryable_failed_job_count,
        "retried_job_count": result.retried_job_count,
        "created_jobs": [embedding_job_payload(job) for job in result.created_jobs],
        "retried_jobs": [embedding_job_payload(job) for job in result.retried_jobs],
    }


def search_compare_payload(result: SearchCompareResult) -> dict[str, object]:
    profile_status_counts = search_compare_profile_status_counts(result.profiles)
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
        "profile_status_counts": profile_status_counts,
        "profile_failure_count": profile_status_counts.get("failed", 0),
        "retrieval_confidence": retrieval_confidence_assessment_payload(
            getattr(result, "confidence_assessment", None)
        ),
        "profiles": [search_compare_profile_payload(profile) for profile in result.profiles],
    }


def search_chunk_policy_compare_run_payload(
    chunk_policy_name: str,
    result: SearchCompareResult,
) -> dict[str, object]:
    result_items = [item for profile in result.profiles for item in profile.results]
    unique_chunk_ids = sorted({item.vector_result.chunk_id for item in result_items})
    top_result = max(
        result_items,
        key=lambda item: item.vector_result.score,
        default=None,
    )
    profile_status_counts = search_compare_profile_status_counts(result.profiles)
    return {
        "chunk_policy_name": chunk_policy_name,
        "search_log_id": result.search_log_id,
        "search_log_url": f"/search/logs?search_log_id={result.search_log_id}",
        "result_count": len(result_items),
        "unique_chunk_count": len(unique_chunk_ids),
        "unique_chunk_ids": unique_chunk_ids,
        "top_score": top_result.vector_result.score if top_result is not None else None,
        "top_result": (
            vector_search_result_payload(top_result.vector_result)
            if top_result is not None
            else None
        ),
        "profile_status_counts": profile_status_counts,
        "profile_failure_count": profile_status_counts.get("failed", 0),
        "total_elapsed_ms": result.total_elapsed_ms,
        "search_result": search_compare_payload(result),
    }


def search_result_context_chunk_payload(chunk: SearchResultContextChunk) -> dict[str, object]:
    return {
        "position": chunk.position,
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "chunk_seq": chunk.chunk_seq,
        "chunk_text": chunk.chunk_text,
        "chunk_preview": chunk.chunk_preview,
        "content_hash": chunk.content_hash,
        "chunk_policy_name": chunk.chunk_policy_name,
        "artifact_id": chunk.artifact_id,
        "block_id": chunk.block_id,
        "chunk_type": chunk.chunk_type,
        "heading_path": list(chunk.heading_path),
        "source_anchor": chunk.source_anchor,
        "page_no": chunk.page_no,
        "slide_no": chunk.slide_no,
        "sheet_name": chunk.sheet_name,
        "cell_range": chunk.cell_range,
        "source_char_start": chunk.source_char_start,
        "source_char_end": chunk.source_char_end,
        "token_count": chunk.token_count,
        "char_count": chunk.char_count,
        "prev_chunk_id": chunk.prev_chunk_id,
        "next_chunk_id": chunk.next_chunk_id,
        "metadata": chunk.metadata,
    }


def search_result_source_block_payload(
    block: SearchResultSourceBlock | None,
) -> dict[str, object] | None:
    if block is None:
        return None
    return {
        "block_id": block.block_id,
        "artifact_id": block.artifact_id,
        "document_id": block.document_id,
        "parent_block_id": block.parent_block_id,
        "block_seq": block.block_seq,
        "block_type": block.block_type,
        "content_preview": block.content_preview,
        "content_markdown_preview": block.content_markdown_preview,
        "heading_path": list(block.heading_path),
        "source_anchor": block.source_anchor,
        "page_no": block.page_no,
        "slide_no": block.slide_no,
        "sheet_name": block.sheet_name,
        "cell_range": block.cell_range,
        "char_start": block.char_start,
        "char_end": block.char_end,
        "token_count": block.token_count,
        "metadata": block.metadata,
        "created_at": _datetime_response(block.created_at),
    }


def search_result_source_artifact_payload(
    artifact: SearchResultSourceArtifact | None,
) -> dict[str, object] | None:
    if artifact is None:
        return None
    return {
        "artifact_id": artifact.artifact_id,
        "extraction_run_id": artifact.extraction_run_id,
        "file_id": artifact.file_id,
        "document_id": artifact.document_id,
        "artifact_type": artifact.artifact_type,
        "content_preview": artifact.content_preview,
        "content_length": artifact.content_length,
        "storage_path": artifact.storage_path,
        "content_hash": artifact.content_hash,
        "size_bytes": artifact.size_bytes,
        "language": artifact.language,
        "metadata": artifact.metadata,
        "created_at": _datetime_response(artifact.created_at),
    }


def search_result_source_context_payload(
    context: SearchResultSourceContext,
) -> dict[str, object]:
    positions = {chunk.position for chunk in context.chunks}
    current_chunk = next(
        (chunk for chunk in context.chunks if chunk.position == "current"),
        None,
    )
    return {
        "search_result": {
            "search_log_result_id": context.search_result.search_log_result_id,
            "search_log_id": context.search_result.search_log_id,
            "profile_name": context.search_result.profile_name,
            "rank": context.search_result.rank,
            "chunk_id": context.search_result.chunk_id,
            "distance": context.search_result.distance,
            "score": context.search_result.score,
            "profile_elapsed_ms": context.search_result.profile_elapsed_ms,
            "created_at": _datetime_response(context.search_result.created_at),
        },
        "document": {
            "document_id": context.document.document_id,
            "file_id": context.document.file_id,
            "document_title": context.document.document_title,
            "document_group": context.document.document_group,
            "document_status": context.document.document_status,
            "original_file_name": context.document.original_file_name,
            "file_ext": context.document.file_ext,
            "storage_path": context.document.storage_path,
        },
        "chunks": [search_result_context_chunk_payload(chunk) for chunk in context.chunks],
        "source_block": search_result_source_block_payload(context.source_block),
        "source_artifact": search_result_source_artifact_payload(context.source_artifact),
        "trace_summary": {
            "has_previous_chunk": "previous" in positions,
            "has_next_chunk": "next" in positions,
            "has_source_block": context.source_block is not None,
            "has_source_artifact": context.source_artifact is not None,
            "context_chunk_count": len(context.chunks),
            "current_source_anchor": current_chunk.source_anchor if current_chunk else {},
        },
    }


def retrieval_context_result_reference_payload(
    result: RetrievalContextResultReference,
) -> dict[str, object]:
    return {
        "search_log_result_id": result.search_log_result_id,
        "profile_name": result.profile_name,
        "search_profile_name": result.search_profile_name,
        "retrieval_strategy": result.retrieval_strategy,
        "rank": result.rank,
        "chunk_id": result.chunk_id,
        "distance": result.distance,
        "score": result.score,
        "score_components": result.score_components,
        "profile_elapsed_ms": result.profile_elapsed_ms,
        "created_at": _datetime_response(result.created_at),
    }


def retrieval_context_chunk_entry_payload(
    chunk: RetrievalContextChunkEntry,
) -> dict[str, object]:
    return {
        "position": chunk.position,
        "chunk_id": chunk.chunk_id,
        "chunk_seq": chunk.chunk_seq,
        "chunk_text": chunk.chunk_text,
        "chunk_preview": chunk.chunk_preview,
        "char_count": chunk.char_count,
        "token_count": chunk.token_count,
        "source_anchor": chunk.source_anchor,
    }


def retrieval_context_candidate_payload(
    candidate: RetrievalContextCandidate,
) -> dict[str, object]:
    return {
        "included": candidate.included,
        "exclusion_reason": candidate.exclusion_reason,
        "citation": {
            "citation_key": candidate.citation.citation_key,
            "chunk_id": candidate.citation.chunk_id,
            "document_id": candidate.citation.document_id,
            "file_id": candidate.citation.file_id,
            "document_title": candidate.citation.document_title,
            "original_file_name": candidate.citation.original_file_name,
            "file_ext": candidate.citation.file_ext,
            "document_group": candidate.citation.document_group,
            "chunk_policy_name": candidate.citation.chunk_policy_name,
            "chunk_seq": candidate.citation.chunk_seq,
            "heading_path": list(candidate.citation.heading_path),
            "page_no": candidate.citation.page_no,
            "slide_no": candidate.citation.slide_no,
            "sheet_name": candidate.citation.sheet_name,
            "cell_range": candidate.citation.cell_range,
            "artifact_id": candidate.citation.artifact_id,
            "block_id": candidate.citation.block_id,
            "source_anchor": candidate.citation.source_anchor,
            "source_label": candidate.citation.source_label,
        },
        "primary_result": retrieval_context_result_reference_payload(candidate.primary_result),
        "supporting_results": [
            retrieval_context_result_reference_payload(result)
            for result in candidate.supporting_results
        ],
        "chunks": [retrieval_context_chunk_entry_payload(chunk) for chunk in candidate.chunks],
        "context_text": candidate.context_text,
        "context_char_count": candidate.context_char_count,
        "original_context_char_count": candidate.original_context_char_count,
        "truncated": candidate.truncated,
    }


def retrieval_context_package_payload(
    package: RetrievalContextPackage,
) -> dict[str, object]:
    search_log = package.search_log.search_log
    return {
        "package_key": package.package_key,
        "generated_at": _datetime_response(package.generated_at),
        "search_log": {
            "search_log_id": search_log.search_log_id,
            "query_text": search_log.query_text,
            "normalized_query_text": search_log.normalized_query_text,
            "actor_user_id": search_log.actor_user_id,
            "actor_login_id": package.search_log.actor_login_id,
            "actor_display_name": package.search_log.actor_display_name,
            "requested_search_scope": search_log.requested_search_scope,
            "effective_search_scope": search_log.effective_search_scope,
            "permission_filter_metadata": search_log.permission_filter_metadata,
            "document_group": search_log.document_group,
            "file_type": search_log.file_type,
            "chunk_policy_name": search_log.chunk_policy_name,
            "strategy_name": search_log.strategy_name,
            "top_k": search_log.top_k,
            "similarity_metric": search_log.similarity_metric,
            "profiles": list(search_log.profiles),
            "query_runtime_metadata": search_log.query_runtime_metadata,
            "total_elapsed_ms": search_log.total_elapsed_ms,
            "created_at": _datetime_response(search_log.created_at),
        },
        "retrieval_confidence": retrieval_confidence_assessment_payload(
            package.confidence_assessment
        ),
        "summary": {
            "candidate_result_count": package.summary.candidate_result_count,
            "unique_candidate_count": package.summary.unique_candidate_count,
            "included_count": package.summary.included_count,
            "excluded_count": package.summary.excluded_count,
            "duplicate_supporting_result_count": (
                package.summary.duplicate_supporting_result_count
            ),
            "max_context_chars": package.summary.max_context_chars,
            "used_context_chars": package.summary.used_context_chars,
            "remaining_context_chars": package.summary.remaining_context_chars,
            "truncated_count": package.summary.truncated_count,
            "source_context_missing_count": package.summary.source_context_missing_count,
            "include_neighbors": package.summary.include_neighbors,
            "max_items": package.summary.max_items,
        },
        "generation_context_text": package.generation_context_text,
        "included_candidates": [
            retrieval_context_candidate_payload(candidate)
            for candidate in package.included_candidates
        ],
        "excluded_candidates": [
            retrieval_context_candidate_payload(candidate)
            for candidate in package.excluded_candidates
        ],
        "candidates": [
            retrieval_context_candidate_payload(candidate) for candidate in package.candidates
        ],
    }


def citation_readiness_issue_payload(
    issue: CitationReadinessIssue,
) -> dict[str, object]:
    return {
        "code": issue.code,
        "severity": issue.severity,
        "message": issue.message,
    }


def citation_readiness_candidate_payload(
    candidate: CitationReadinessCandidate,
) -> dict[str, object]:
    return {
        "citation_key": candidate.citation_key,
        "search_log_result_id": candidate.search_log_result_id,
        "chunk_id": candidate.chunk_id,
        "document_id": candidate.document_id,
        "document_title": candidate.document_title,
        "original_file_name": candidate.original_file_name,
        "source_label": candidate.source_label,
        "included": candidate.included,
        "status": candidate.status,
        "has_document_identity": candidate.has_document_identity,
        "has_chunk_identity": candidate.has_chunk_identity,
        "has_source_anchor": candidate.has_source_anchor,
        "has_location_hint": candidate.has_location_hint,
        "has_lineage_reference": candidate.has_lineage_reference,
        "has_generation_text": candidate.has_generation_text,
        "issue_count": candidate.issue_count,
        "issues": [citation_readiness_issue_payload(issue) for issue in candidate.issues],
    }


def citation_readiness_report_payload(
    report: CitationReadinessReport,
) -> dict[str, object]:
    return {
        "package_key": report.package.package_key,
        "search_log_id": report.package.search_log.search_log.search_log_id,
        "query_text": report.package.search_log.search_log.query_text,
        "generated_at": _datetime_response(report.package.generated_at),
        "retrieval_confidence": retrieval_confidence_assessment_payload(
            report.package.confidence_assessment
        ),
        "summary": {
            "status": report.summary.status,
            "total_candidate_count": report.summary.total_candidate_count,
            "included_candidate_count": report.summary.included_candidate_count,
            "excluded_candidate_count": report.summary.excluded_candidate_count,
            "ready_count": report.summary.ready_count,
            "warning_count": report.summary.warning_count,
            "failed_count": report.summary.failed_count,
            "source_anchor_ready_count": report.summary.source_anchor_ready_count,
            "source_anchor_coverage_percent": _percent_value(
                report.summary.source_anchor_coverage_percent
            ),
            "source_anchor_coverage_label": _percent_label(
                report.summary.source_anchor_coverage_percent
            ),
            "citation_ready_percent": _percent_value(report.summary.citation_ready_percent),
            "citation_ready_label": _percent_label(report.summary.citation_ready_percent),
            "issue_count": report.summary.issue_count,
        },
        "candidates": [
            citation_readiness_candidate_payload(candidate) for candidate in report.candidates
        ],
    }


def _is_sensitive_generation_runtime_key(key: str) -> bool:
    normalized = key.strip().lower()
    if normalized.endswith("_env"):
        return False
    return normalized in {
        "api_key",
        "authorization",
        "bearer_token",
        "password",
        "secret",
        "token",
    }


def redacted_generation_runtime_options(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): (
                "<redacted>"
                if _is_sensitive_generation_runtime_key(str(key))
                else redacted_generation_runtime_options(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [redacted_generation_runtime_options(child) for child in value]
    return value


def generation_provider_runtime_config_payload(
    provider: GenerationProviderConfigRecord,
    settings: Settings,
) -> dict[str, object]:
    try:
        runtime_config = generation_provider_runtime_config_from_record(provider)
    except InvalidGenerationProviderError as exc:
        return {
            "valid": False,
            "error_message": str(exc),
            "mode": provider.provider_mode,
            "remote_base_url": provider.provider_base_url,
            "model_id": provider.model_id,
        }

    api_key_env = generation_provider_api_key_env(provider)
    api_key_configured = (
        bool(lookup_generation_provider_api_key(provider, settings))
        if api_key_env is not None
        else None
    )
    extra_body = provider.runtime_options.get("extra_body", {})
    thinking_disabled = None
    if isinstance(extra_body, dict):
        chat_template_kwargs = extra_body.get("chat_template_kwargs", {})
        if isinstance(chat_template_kwargs, dict) and isinstance(
            chat_template_kwargs.get("enable_thinking"),
            bool,
        ):
            thinking_disabled = not chat_template_kwargs["enable_thinking"]
    return {
        "valid": True,
        "error_message": None,
        "mode": runtime_config.mode,
        "remote_base_url": runtime_config.remote_base_url,
        "remote_timeout_seconds": runtime_config.remote_timeout_seconds,
        "model_id": runtime_config.model_id,
        "max_tokens": runtime_config.max_tokens,
        "temperature": runtime_config.temperature,
        "top_p": runtime_config.top_p,
        "api_key_env": api_key_env if isinstance(api_key_env, str) else None,
        "api_key_configured": api_key_configured,
        "remote_header_names": sorted(runtime_config.remote_headers),
        "extra_body": redacted_generation_runtime_options(extra_body),
        "thinking_disabled": thinking_disabled,
        "secret_policy": "secret values are provided through environment variables only",
    }


def generation_provider_api_key_env(provider: GenerationProviderConfigRecord) -> str | None:
    api_key_env = provider.runtime_options.get("api_key_env")
    if not isinstance(api_key_env, str):
        return None
    normalized = api_key_env.strip()
    return normalized or None


def lookup_generation_provider_api_key(
    provider: GenerationProviderConfigRecord,
    settings: Settings,
) -> str | None:
    api_key_env = generation_provider_api_key_env(provider)
    if api_key_env is None:
        return None
    if api_key_env == DGX_VLLM_GENERATION_API_KEY_ENV:
        return settings.remote_generation_provider_api_key
    return os.getenv(api_key_env)


def resolve_generation_provider_api_key(
    provider: GenerationProviderConfigRecord,
    settings: Settings,
) -> str | None:
    api_key_env = generation_provider_api_key_env(provider)
    if api_key_env is None:
        return None
    api_key = lookup_generation_provider_api_key(provider, settings)
    if not api_key:
        raise InvalidGenerationRunError(
            f"Generation provider API key environment variable is not set: {api_key_env}"
        )
    return api_key


def generation_provider_config_collection_payload(
    providers: tuple[GenerationProviderConfigRecord, ...],
    settings: Settings,
    *,
    include_inactive: bool,
) -> dict[str, object]:
    provider_payloads = [
        {
            **generation_provider_config_payload(provider),
            "runtime_config": generation_provider_runtime_config_payload(provider, settings),
        }
        for provider in providers
    ]
    default_provider = next((provider for provider in providers if provider.is_default), None)
    return {
        "summary": {
            "provider_count": len(providers),
            "active_provider_count": sum(1 for provider in providers if provider.is_active),
            "default_provider_name": default_provider.provider_name if default_provider else None,
            "include_inactive": include_inactive,
        },
        "default_provider": (
            {
                **generation_provider_config_payload(default_provider),
                "runtime_config": generation_provider_runtime_config_payload(
                    default_provider,
                    settings,
                ),
            }
            if default_provider
            else None
        ),
        "providers": provider_payloads,
    }


def generation_provider_config_payload(
    provider: GenerationProviderConfigRecord,
) -> dict[str, object]:
    return {
        "provider_config_id": provider.provider_config_id,
        "provider_name": provider.provider_name,
        "provider_mode": provider.provider_mode,
        "provider_base_url": provider.provider_base_url,
        "model_id": provider.model_id,
        "is_default": provider.is_default,
        "is_active": provider.is_active,
        "request_timeout_seconds": provider.request_timeout_seconds,
        "max_tokens": provider.max_tokens,
        "temperature": provider.temperature,
        "top_p": provider.top_p,
        "runtime_options": redacted_generation_runtime_options(provider.runtime_options),
        "created_by": provider.created_by,
        "created_by_user_id": provider.created_by_user_id,
        "created_at": _datetime_response(provider.created_at),
        "updated_at": _datetime_response(provider.updated_at),
    }


def generation_run_payload(run: GenerationRunRecord) -> dict[str, object]:
    template_completeness = assess_generation_template_completeness(run)
    docx_export_readiness = assess_generation_docx_export_readiness(
        run,
        template_completeness,
    )
    template = _generation_run_export_template(run)
    docx_export_evidence = generation_docx_export_evidence_from_run(
        run,
        template=template,
        readiness=docx_export_readiness,
    )
    return {
        "generation_run_id": run.generation_run_id,
        "search_log_id": run.search_log_id,
        "retrieval_package_key": run.retrieval_package_key,
        "generation_template_id": run.generation_template_id,
        "provider_config_id": run.provider_config_id,
        "provider_name": run.provider_name,
        "provider_mode": run.provider_mode,
        "model_id": run.model_id,
        "prompt_version": run.prompt_version,
        "prompt_hash": run.prompt_hash,
        "context_hash": run.context_hash,
        "status": run.status,
        "guardrail_status": run.guardrail_status,
        "retrieval_confidence_status": run.retrieval_confidence_status,
        "citation_readiness_status": run.citation_readiness_status,
        "query_text": run.query_text,
        "answer_text": run.answer_text,
        "finish_reason": run.finish_reason,
        "input_token_count": run.input_token_count,
        "output_token_count": run.output_token_count,
        "total_token_count": run.total_token_count,
        "elapsed_ms": run.elapsed_ms,
        "request_metadata": run.request_metadata,
        "response_metadata": run.response_metadata,
        "guardrail_metadata": run.guardrail_metadata,
        "template_completeness": generation_template_completeness_payload(template_completeness),
        "docx_export_readiness": generation_docx_export_readiness_payload(
            docx_export_readiness,
        ),
        "docx_export_evidence": generation_docx_export_evidence_payload(
            docx_export_evidence,
        ),
        "error_message": run.error_message,
        "created_by": run.created_by,
        "created_by_user_id": run.created_by_user_id,
        "started_at": _datetime_response(run.started_at),
        "finished_at": _datetime_response(run.finished_at),
        "created_at": _datetime_response(run.created_at),
        "updated_at": _datetime_response(run.updated_at),
    }


def generation_run_history_item_payload(item: GenerationRunHistoryItem) -> dict[str, object]:
    run = item.run
    return {
        "generation_run_id": run.generation_run_id,
        "search_log_id": run.search_log_id,
        "query_text": run.query_text,
        "provider_name": run.provider_name,
        "provider_mode": run.provider_mode,
        "model_id": run.model_id,
        "status": run.status,
        "guardrail_status": run.guardrail_status,
        "answer_quality_status": item.answer_quality_status,
        "answer_quality_reason_codes": list(item.answer_quality_reason_codes),
        "citation_coverage_percent": item.citation_coverage_percent,
        "expected_citation_count": item.expected_citation_count,
        "cited_citation_count": item.cited_citation_count,
        "missing_citation_count": item.missing_citation_count,
        "unrecognized_citation_count": item.unrecognized_citation_count,
        "retrieval_confidence_status": run.retrieval_confidence_status,
        "citation_readiness_status": run.citation_readiness_status,
        "finish_reason": run.finish_reason,
        "total_token_count": run.total_token_count,
        "elapsed_ms": run.elapsed_ms,
        "created_by": run.created_by,
        "created_at": _datetime_response(run.created_at),
        "updated_at": _datetime_response(run.updated_at),
    }


def generation_run_history_payload(history: GenerationRunHistory) -> dict[str, object]:
    return {
        "filters": {
            "limit": history.filters.limit,
            "answer_quality_status": history.filters.answer_quality_status,
            "provider_mode": history.filters.provider_mode,
            "run_status": history.filters.run_status,
        },
        "summary": {
            "run_count": history.summary.run_count,
            "passed_count": history.summary.passed_count,
            "warning_count": history.summary.warning_count,
            "failed_count": history.summary.failed_count,
            "not_evaluated_count": history.summary.not_evaluated_count,
            "not_available_count": history.summary.not_available_count,
        },
        "runs": [generation_run_history_item_payload(item) for item in history.runs],
    }


def generation_run_citation_payload(
    citation: GenerationRunCitationRecord,
) -> dict[str, object]:
    return {
        "generation_run_citation_id": citation.generation_run_citation_id,
        "generation_run_id": citation.generation_run_id,
        "citation_key": citation.citation_key,
        "citation_index": citation.citation_index,
        "search_log_result_id": citation.search_log_result_id,
        "chunk_id": citation.chunk_id,
        "document_id": citation.document_id,
        "file_id": citation.file_id,
        "source_label": citation.source_label,
        "source_anchor": citation.source_anchor,
        "citation_payload": citation.citation_payload,
        "was_cited": citation.was_cited,
        "created_at": _datetime_response(citation.created_at),
    }


def generation_prompt_package_payload(
    prompt_package: GenerationPromptPackage,
) -> dict[str, object]:
    return {
        "prompt_contract_version": prompt_package.prompt_contract_version,
        "prompt_version": prompt_package.prompt_version,
        "response_language": prompt_package.response_language,
        "generation_template_id": prompt_package.generation_template_id,
        "template_key": prompt_package.template_key,
        "template_name": prompt_package.template_name,
        "template_version": prompt_package.template_version,
        "document_type": prompt_package.document_type,
        "output_format": prompt_package.output_format,
        "generation_template": prompt_package.template_snapshot,
        "query_text": prompt_package.query_text,
        "retrieval_package_key": prompt_package.retrieval_package_key,
        "search_log_id": prompt_package.search_log_id,
        "messages": prompt_package.openai_messages,
        "citation_keys": list(prompt_package.citation_keys),
        "context_text": prompt_package.context_text,
        "prompt_hash": prompt_package.prompt_hash,
        "context_hash": prompt_package.context_hash,
        "blocked": prompt_package.blocked,
        "block_reason": prompt_package.block_reason,
    }


def generation_template_payload(template: GenerationTemplateRecord) -> dict[str, object]:
    return {
        "generation_template_id": template.generation_template_id,
        "template_key": template.template_key,
        "template_family": template.template_family,
        "template_name": template.template_name,
        "template_version": template.template_version,
        "document_type": template.document_type,
        "language": template.language,
        "output_format": template.output_format,
        "section_schema": [dict(section) for section in template.section_schema],
        "system_instruction": template.system_instruction,
        "user_instruction_suffix": template.user_instruction_suffix,
        "style_guidance": template.style_guidance,
        "citation_policy": template.citation_policy,
        "is_default": template.is_default,
        "is_active": template.is_active,
        "clone_source_template_id": template.clone_source_template_id,
        "change_note": template.change_note,
        "created_by": template.created_by,
        "created_by_user_id": template.created_by_user_id,
        "created_at": _datetime_response(template.created_at),
        "updated_at": _datetime_response(template.updated_at),
    }


def generation_template_collection_payload(
    templates: tuple[GenerationTemplateRecord, ...],
    *,
    include_inactive: bool,
) -> dict[str, object]:
    default_template = next(
        (template for template in templates if template.is_default and template.is_active),
        None,
    )
    return {
        "summary": {
            "template_count": len(templates),
            "active_template_count": sum(1 for template in templates if template.is_active),
            "default_template_key": (
                default_template.template_key if default_template is not None else None
            ),
            "default_template_version": (
                default_template.template_version if default_template is not None else None
            ),
            "default_template_name": (
                default_template.template_name if default_template is not None else None
            ),
            "default_template_family": (
                default_template.template_family if default_template is not None else None
            ),
            "applied_template_label": (
                (
                    f"{default_template.template_name} "
                    f"({default_template.template_key} / {default_template.template_version})"
                )
                if default_template is not None
                else None
            ),
            "family_count": len({template.template_family for template in templates}),
            "document_types": sorted(
                {template.document_type for template in templates if template.is_active}
            ),
            "languages": sorted(
                {template.language for template in templates if template.is_active}
            ),
        },
        "include_inactive": include_inactive,
        "default_template": (
            generation_template_payload(default_template) if default_template is not None else None
        ),
        "templates": [generation_template_payload(template) for template in templates],
    }


def generation_template_input_from_request(
    payload: GenerationTemplateManagementRequest,
) -> GenerationTemplateInput:
    return GenerationTemplateInput(
        template_key=payload.template_key,
        template_family=payload.template_family,
        template_name=payload.template_name,
        template_version=payload.template_version,
        document_type=payload.document_type,
        language=payload.language,
        output_format=payload.output_format,
        section_schema=payload.section_schema,
        system_instruction=payload.system_instruction,
        user_instruction_suffix=payload.user_instruction_suffix,
        style_guidance=payload.style_guidance,
        citation_policy=payload.citation_policy,
        is_default=payload.is_default,
        is_active=payload.is_active,
        clone_source_template_id=payload.clone_source_template_id,
        change_note=payload.change_note,
        created_by=payload.created_by,
        created_by_user_id=payload.created_by_user_id,
    )


def _generation_template_json_field(
    raw_value: str,
    field_name: str,
    *,
    default_json: str,
) -> object:
    value = raw_value.strip() or default_json
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise InvalidGenerationTemplateError(f"{field_name} must be valid JSON") from exc


def _generation_run_export_template(run: GenerationRunRecord) -> dict[str, object]:
    response_template = run.response_metadata.get("template")
    if isinstance(response_template, dict):
        return dict(response_template)
    request_template = run.request_metadata.get("generation_template")
    if isinstance(request_template, dict):
        return {
            "template_key": request_template.get("template_key")
            or run.request_metadata.get("template_key"),
            "template_name": request_template.get("template_name"),
            "template_version": request_template.get("template_version")
            or run.request_metadata.get("template_version"),
            "document_type": request_template.get("document_type")
            or run.request_metadata.get("document_type"),
            "output_format": request_template.get("output_format")
            or run.request_metadata.get("output_format"),
        }
    return {
        "template_key": run.request_metadata.get("template_key") or "-",
        "template_version": run.request_metadata.get("template_version") or "-",
        "document_type": run.request_metadata.get("document_type") or "-",
        "output_format": run.request_metadata.get("output_format") or "-",
    }


def _generation_run_export_datetime(value: datetime | None) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S") if value is not None else "-"


def _generation_run_markdown_export(
    run: GenerationRunRecord,
    citations: tuple[GenerationRunCitationRecord, ...],
) -> str:
    template = _generation_run_export_template(run)
    answer_quality = run.response_metadata.get("answer_quality")
    answer_quality_status = (
        answer_quality.get("status")
        if isinstance(answer_quality, dict)
        else run.guardrail_metadata.get("answer_quality_status")
    )
    template_label = " ".join(
        str(part)
        for part in (
            template.get("template_name"),
            f"({template.get('template_key')})" if template.get("template_key") else None,
            template.get("template_version"),
        )
        if part
    )
    lines = [
        f"# Generation Run #{run.generation_run_id}",
        "",
        "## Metadata",
        f"- Search Log: #{run.search_log_id}",
        f"- Query: {run.query_text}",
        f"- Status: {run.status}",
        f"- Guardrail: {run.guardrail_status}",
        f"- Retrieval Confidence: {run.retrieval_confidence_status}",
        f"- Citation Readiness: {run.citation_readiness_status}",
        f"- Answer Quality: {answer_quality_status or '-'}",
        f"- Provider: {run.provider_name} ({run.provider_mode})",
        f"- Model: {run.model_id}",
        f"- Template: {template_label or '-'}",
        f"- Document Type: {template.get('document_type') or '-'}",
        f"- Output Format: {template.get('output_format') or '-'}",
        f"- Prompt Version: {run.prompt_version}",
        f"- Prompt Hash: {run.prompt_hash or '-'}",
        f"- Context Hash: {run.context_hash or '-'}",
        f"- Retrieval Package: {run.retrieval_package_key}",
        f"- Started At: {_generation_run_export_datetime(run.started_at)}",
        f"- Finished At: {_generation_run_export_datetime(run.finished_at)}",
        "",
        "## Answer",
        "",
        run.answer_text or "",
        "",
        "## Citations",
        "",
    ]
    if citations:
        for citation in citations:
            payload = citation.citation_payload
            source_label = citation.source_label or str(
                payload.get("document_title") or payload.get("original_file_name") or "-"
            )
            anchor = (
                json.dumps(citation.source_anchor, ensure_ascii=False, sort_keys=True)
                if citation.source_anchor
                else "-"
            )
            lines.extend(
                (
                    f"- [{citation.citation_key}] {source_label}",
                    f"  - Used In Answer: {'yes' if citation.was_cited else 'no'}",
                    f"  - Search Result ID: {citation.search_log_result_id or '-'}",
                    f"  - Chunk ID: {citation.chunk_id or payload.get('chunk_id') or '-'}",
                    f"  - Document ID: {citation.document_id or payload.get('document_id') or '-'}",
                    f"  - File ID: {citation.file_id or payload.get('file_id') or '-'}",
                    f"  - Source Anchor: `{anchor}`",
                )
            )
    else:
        lines.append("- No citation trace was stored.")
    lines.extend(("", "## Raw Runtime Metadata", ""))
    lines.append("```json")
    lines.append(
        json.dumps(
            {
                "request_metadata": run.request_metadata,
                "response_metadata": run.response_metadata,
                "guardrail_metadata": run.guardrail_metadata,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=str,
        )
    )
    lines.append("```")
    return "\n".join(lines).rstrip() + "\n"


def generation_execution_report_payload(
    report: GenerationExecutionReport,
) -> dict[str, object]:
    return {
        "provider": generation_provider_config_payload(report.provider),
        "prompt_package": generation_prompt_package_payload(report.prompt_package),
        "run": generation_run_payload(report.run),
        "citations": [generation_run_citation_payload(citation) for citation in report.citations],
    }


def direct_generation_result_payload(
    result: DirectGenerationResult,
) -> dict[str, object]:
    search_log_id = result.search_result.search_log_id
    generation_run_id = result.generation_report.run.generation_run_id
    return {
        "mode": "direct_query",
        "search_log_id": search_log_id,
        "generation_run_id": generation_run_id,
        "links": {
            "search_log": f"/search/logs?search_log_id={search_log_id}",
            "retrieval_context": f"/search/context?search_log_id={search_log_id}",
            "generation_run": f"/generation/runs/{generation_run_id}",
        },
        "search": search_compare_payload(result.search_result),
        "retrieval_context": retrieval_context_package_payload(result.retrieval_package),
        "generation": generation_execution_report_payload(result.generation_report),
    }


def document_summary_result_payload(
    result: DocumentSummaryResult,
) -> dict[str, object]:
    search_log_id = result.search_log_id
    generation_run_id = result.generation_report.run.generation_run_id
    return {
        "mode": "document_summary",
        "document": document_inventory_item_payload(result.document),
        "source_chunk_count": len(result.source_chunk_ids),
        "source_chunk_ids": list(result.source_chunk_ids),
        "search_log_id": search_log_id,
        "generation_run_id": generation_run_id,
        "links": {
            "document": f"/documents/{result.document.document_id}",
            "search_log": f"/search/logs?search_log_id={search_log_id}",
            "retrieval_context": f"/search/context?search_log_id={search_log_id}",
            "generation_run": f"/generation/runs/{generation_run_id}",
        },
        "retrieval_context": retrieval_context_package_payload(result.retrieval_package),
        "generation": generation_execution_report_payload(result.generation_report),
    }


def document_summary_history_item_payload(
    item: DocumentSummaryHistoryItem,
) -> dict[str, object]:
    run = item.run
    return {
        "generation_run_id": run.generation_run_id,
        "search_log_id": run.search_log_id,
        "document_id": item.document_id,
        "file_id": item.file_id,
        "document_title": item.document_title,
        "original_file_name": item.original_file_name,
        "document_label": item.document_label,
        "document_group": item.document_group,
        "file_type": item.file_type,
        "template_key": item.template_key,
        "template_name": item.template_name,
        "summary_instruction": item.summary_instruction,
        "source_chunk_count": item.source_chunk_count,
        "provider_name": run.provider_name,
        "provider_mode": run.provider_mode,
        "model_id": run.model_id,
        "status": run.status,
        "guardrail_status": run.guardrail_status,
        "retrieval_confidence_status": run.retrieval_confidence_status,
        "citation_readiness_status": run.citation_readiness_status,
        "total_token_count": run.total_token_count,
        "elapsed_ms": run.elapsed_ms,
        "created_at": _datetime_response(run.created_at),
        "created_at_label": _datetime_label(run.created_at),
        "links": {
            "document": f"/documents/{item.document_id}" if item.document_id else None,
            "generation_run": f"/generation/runs/{run.generation_run_id}",
            "search_log": f"/search/logs?search_log_id={run.search_log_id}",
            "retrieval_context": f"/search/context?search_log_id={run.search_log_id}",
        },
    }


def document_summary_history_payload(history: DocumentSummaryHistory) -> dict[str, object]:
    return {
        "filters": {
            "limit": history.filters.limit,
            "run_status": history.filters.run_status,
            "generation_template_key": history.filters.generation_template_key,
        },
        "summary": {
            "run_count": history.summary.run_count,
            "succeeded_count": history.summary.succeeded_count,
            "failed_count": history.summary.failed_count,
            "no_answer_count": history.summary.no_answer_count,
            "latest_created_at": _datetime_response(history.summary.latest_created_at),
            "latest_created_at_label": _datetime_label(history.summary.latest_created_at),
        },
        "runs": [document_summary_history_item_payload(item) for item in history.runs],
    }


def search_experiment_profile_execution_payload(
    summary: SearchExperimentProfileExecutionSummary,
) -> dict[str, object]:
    return {
        "profile_name": summary.profile_name,
        "raw_result_count": summary.raw_result_count,
        "retained_result_count": summary.retained_result_count,
        "excluded_by_threshold_count": summary.excluded_by_threshold_count,
        "top_score": summary.top_score,
        "average_score": summary.average_score,
        "elapsed_ms": summary.elapsed_ms,
    }


def search_experiment_run_record_payload(run: SearchExperimentRunRecord) -> dict[str, object]:
    return {
        "experiment_run_id": run.experiment_run_id,
        "run_name": run.run_name,
        "query_text": run.query_text,
        "normalized_query_text": run.normalized_query_text,
        "actor_user_id": run.actor_user_id,
        "requested_search_scope": run.requested_search_scope,
        "effective_search_scope": run.effective_search_scope,
        "document_group": run.document_group,
        "file_type": run.file_type,
        "chunk_policy_name": run.chunk_policy_name,
        "strategy_name": run.strategy_name,
        "similarity_metric": run.similarity_metric,
        "top_k": run.top_k,
        "score_threshold": run.score_threshold,
        "profile_names": list(run.profile_names),
        "status": run.status,
        "total_profile_count": run.total_profile_count,
        "completed_profile_count": run.completed_profile_count,
        "result_count": run.result_count,
        "failure_count": run.failure_count,
        "total_elapsed_ms": run.total_elapsed_ms,
        "runtime_metadata": run.runtime_metadata,
        "error_message": run.error_message,
        "created_by": run.created_by,
        "created_by_user_id": run.created_by_user_id,
        "started_at": _datetime_response(run.started_at),
        "started_at_label": _datetime_label(run.started_at),
        "finished_at": _datetime_response(run.finished_at),
        "finished_at_label": _datetime_label(run.finished_at),
        "created_at": _datetime_response(run.created_at),
        "created_at_label": _datetime_label(run.created_at),
        "updated_at": _datetime_response(run.updated_at),
        "updated_at_label": _datetime_label(run.updated_at),
    }


def search_experiment_profile_run_payload(
    profile: SearchExperimentProfileRunRecord,
) -> dict[str, object]:
    return {
        "experiment_profile_run_id": profile.experiment_profile_run_id,
        "experiment_run_id": profile.experiment_run_id,
        "profile_name": profile.profile_name,
        "search_log_id": profile.search_log_id,
        "status": profile.status,
        "result_count": profile.result_count,
        "top_score": profile.top_score,
        "average_score": profile.average_score,
        "elapsed_ms": profile.elapsed_ms,
        "runtime_metadata": profile.runtime_metadata,
        "error_message": profile.error_message,
        "started_at": _datetime_response(profile.started_at),
        "started_at_label": _datetime_label(profile.started_at),
        "finished_at": _datetime_response(profile.finished_at),
        "finished_at_label": _datetime_label(profile.finished_at),
        "created_at": _datetime_response(profile.created_at),
        "created_at_label": _datetime_label(profile.created_at),
        "updated_at": _datetime_response(profile.updated_at),
        "updated_at_label": _datetime_label(profile.updated_at),
    }


def search_experiment_detail_payload(
    detail: SearchExperimentRunDetail,
) -> dict[str, object]:
    return {
        "experiment_run": search_experiment_run_record_payload(detail.run),
        "profiles": [search_experiment_profile_run_payload(profile) for profile in detail.profiles],
    }


def golden_search_experiment_batch_summary_payload(
    summary: GoldenSearchExperimentBatchSummary,
) -> dict[str, object]:
    return {
        "batch_key": summary.batch_key,
        "question_set_id": summary.question_set_id,
        "question_set_name": summary.question_set_name,
        "batch_prefix": summary.batch_prefix,
        "strategy_name": summary.strategy_name,
        "top_k": summary.top_k,
        "score_threshold": summary.score_threshold,
        "chunk_policy_name": summary.chunk_policy_name,
        "profile_names": list(summary.profile_names),
        "status": summary.status,
        "question_count": summary.question_count,
        "succeeded_count": summary.succeeded_count,
        "failed_count": summary.failed_count,
        "running_count": summary.running_count,
        "total_result_count": summary.total_result_count,
        "average_result_count": summary.average_result_count,
        "total_elapsed_ms": summary.total_elapsed_ms,
        "average_elapsed_ms": summary.average_elapsed_ms,
        "first_experiment_run_id": summary.first_experiment_run_id,
        "last_experiment_run_id": summary.last_experiment_run_id,
        "first_created_at": _datetime_response(summary.first_created_at),
        "first_created_at_label": _datetime_label(summary.first_created_at),
        "last_updated_at": _datetime_response(summary.last_updated_at),
        "last_updated_at_label": _datetime_label(summary.last_updated_at),
    }


def golden_search_experiment_batch_question_payload(
    question: GoldenSearchExperimentBatchQuestionSummary,
) -> dict[str, object]:
    return {
        "question_id": question.question_id,
        "question_text": question.question_text,
        "experiment_run": search_experiment_run_record_payload(question.experiment_run),
    }


def golden_search_experiment_batch_detail_payload(
    detail: GoldenSearchExperimentBatchDetail,
) -> dict[str, object]:
    return {
        "summary": golden_search_experiment_batch_summary_payload(detail.summary),
        "questions": [
            golden_search_experiment_batch_question_payload(question)
            for question in detail.questions
        ],
    }


def golden_search_experiment_batch_metric_summary_payload(
    metric_summary: GoldenSearchExperimentBatchMetricSummary,
) -> dict[str, object]:
    return {
        "batch": golden_search_experiment_batch_summary_payload(metric_summary.summary),
        "overall": {
            "question_count": metric_summary.overall.question_count,
            "recall_question_count": metric_summary.overall.recall_question_count,
            "ndcg_question_count": metric_summary.overall.ndcg_question_count,
            "no_answer_question_count": metric_summary.overall.no_answer_question_count,
            "hidden_violation_count": metric_summary.overall.hidden_violation_count,
            "mean_recall_at_k": metric_summary.overall.mean_recall_at_k,
            "mean_reciprocal_rank": metric_summary.overall.mean_reciprocal_rank,
            "mean_ndcg": metric_summary.overall.mean_ndcg,
            "no_answer_success_rate": metric_summary.overall.no_answer_success_rate,
        },
        "profiles": [
            {
                "profile_name": profile.profile_name,
                "question_count": profile.question_count,
                "recall_question_count": profile.recall_question_count,
                "ndcg_question_count": profile.ndcg_question_count,
                "no_answer_question_count": profile.no_answer_question_count,
                "hidden_violation_count": profile.hidden_violation_count,
                "mean_recall_at_k": profile.mean_recall_at_k,
                "mean_reciprocal_rank": profile.mean_reciprocal_rank,
                "mean_ndcg": profile.mean_ndcg,
                "no_answer_success_rate": profile.no_answer_success_rate,
                "total_result_count": profile.total_result_count,
                "average_result_count": profile.average_result_count,
                "average_elapsed_ms": profile.average_elapsed_ms,
            }
            for profile in metric_summary.profiles
        ],
        "questions": [
            golden_search_experiment_batch_question_metric_payload(question_metric)
            for question_metric in metric_summary.questions
        ],
    }


def golden_search_experiment_batch_question_metric_payload(
    question_metric: GoldenSearchExperimentBatchQuestionMetricSummary,
) -> dict[str, object]:
    metric = question_metric.metric
    return {
        "question_id": question_metric.question_id,
        "question_text": question_metric.question_text,
        "profile_name": question_metric.profile_name,
        "experiment_run_id": question_metric.experiment_run_id,
        "search_log_id": question_metric.search_log_id,
        "top_k": question_metric.top_k,
        "result_count": question_metric.result_count,
        "elapsed_ms": question_metric.elapsed_ms,
        "visible_expected_count": metric.visible_expected_count,
        "retrieved_count": metric.retrieved_count,
        "matched_visible_count": metric.matched_visible_count,
        "hidden_violation_count": metric.hidden_violation_count,
        "matched_chunk_ids": list(metric.matched_chunk_ids),
        "hidden_violation_chunk_ids": list(metric.hidden_violation_chunk_ids),
        "recall_at_k": metric.recall_at_k,
        "reciprocal_rank": metric.reciprocal_rank,
        "dcg": metric.dcg,
        "ideal_dcg": metric.ideal_dcg,
        "ndcg": metric.ndcg,
        "no_answer_success": metric.no_answer_success,
    }


def golden_batch_metric_snapshot_record_payload(
    snapshot: GoldenBatchMetricSnapshotRecord,
) -> dict[str, object]:
    return {
        "snapshot_id": snapshot.snapshot_id,
        "batch_key": snapshot.batch_key,
        "question_set_id": snapshot.question_set_id,
        "question_set_name": snapshot.question_set_name,
        "batch_prefix": snapshot.batch_prefix,
        "strategy_name": snapshot.strategy_name,
        "top_k": snapshot.top_k,
        "score_threshold": snapshot.score_threshold,
        "chunk_policy_name": snapshot.chunk_policy_name,
        "profile_names": list(snapshot.profile_names),
        "batch_status": snapshot.batch_status,
        "batch_question_count": snapshot.batch_question_count,
        "batch_succeeded_count": snapshot.batch_succeeded_count,
        "batch_failed_count": snapshot.batch_failed_count,
        "batch_running_count": snapshot.batch_running_count,
        "total_result_count": snapshot.total_result_count,
        "average_result_count": snapshot.average_result_count,
        "total_elapsed_ms": snapshot.total_elapsed_ms,
        "average_elapsed_ms": snapshot.average_elapsed_ms,
        "evaluated_row_count": snapshot.evaluated_row_count,
        "recall_question_count": snapshot.recall_question_count,
        "ndcg_question_count": snapshot.ndcg_question_count,
        "no_answer_question_count": snapshot.no_answer_question_count,
        "hidden_violation_count": snapshot.hidden_violation_count,
        "mean_recall_at_k": snapshot.mean_recall_at_k,
        "mean_reciprocal_rank": snapshot.mean_reciprocal_rank,
        "mean_ndcg": snapshot.mean_ndcg,
        "no_answer_success_rate": snapshot.no_answer_success_rate,
        "source_first_experiment_run_id": snapshot.source_first_experiment_run_id,
        "source_last_experiment_run_id": snapshot.source_last_experiment_run_id,
        "source_first_created_at": _datetime_response(snapshot.source_first_created_at),
        "source_first_created_at_label": _datetime_label(snapshot.source_first_created_at),
        "source_last_updated_at": _datetime_response(snapshot.source_last_updated_at),
        "source_last_updated_at_label": _datetime_label(snapshot.source_last_updated_at),
        "metric_payload": snapshot.metric_payload,
        "created_by": snapshot.created_by,
        "created_by_user_id": snapshot.created_by_user_id,
        "created_at": _datetime_response(snapshot.created_at),
        "created_at_label": _datetime_label(snapshot.created_at),
    }


def golden_batch_profile_metric_snapshot_payload(
    profile: GoldenBatchProfileMetricSnapshotRecord,
) -> dict[str, object]:
    return {
        "snapshot_profile_metric_id": profile.snapshot_profile_metric_id,
        "snapshot_id": profile.snapshot_id,
        "profile_name": profile.profile_name,
        "question_count": profile.question_count,
        "recall_question_count": profile.recall_question_count,
        "ndcg_question_count": profile.ndcg_question_count,
        "no_answer_question_count": profile.no_answer_question_count,
        "hidden_violation_count": profile.hidden_violation_count,
        "mean_recall_at_k": profile.mean_recall_at_k,
        "mean_reciprocal_rank": profile.mean_reciprocal_rank,
        "mean_ndcg": profile.mean_ndcg,
        "no_answer_success_rate": profile.no_answer_success_rate,
        "total_result_count": profile.total_result_count,
        "average_result_count": profile.average_result_count,
        "average_elapsed_ms": profile.average_elapsed_ms,
    }


def golden_batch_question_metric_snapshot_payload(
    question: GoldenBatchQuestionMetricSnapshotRecord,
) -> dict[str, object]:
    return {
        "snapshot_question_metric_id": question.snapshot_question_metric_id,
        "snapshot_id": question.snapshot_id,
        "question_id": question.question_id,
        "question_text": question.question_text,
        "profile_name": question.profile_name,
        "experiment_run_id": question.experiment_run_id,
        "search_log_id": question.search_log_id,
        "top_k": question.top_k,
        "result_count": question.result_count,
        "elapsed_ms": question.elapsed_ms,
        "visible_expected_count": question.visible_expected_count,
        "retrieved_count": question.retrieved_count,
        "matched_visible_count": question.matched_visible_count,
        "hidden_violation_count": question.hidden_violation_count,
        "matched_chunk_ids": list(question.matched_chunk_ids),
        "hidden_violation_chunk_ids": list(question.hidden_violation_chunk_ids),
        "recall_at_k": question.recall_at_k,
        "reciprocal_rank": question.reciprocal_rank,
        "dcg": question.dcg,
        "ideal_dcg": question.ideal_dcg,
        "ndcg": question.ndcg,
        "no_answer_success": question.no_answer_success,
    }


def golden_batch_metric_snapshot_detail_payload(
    detail: GoldenBatchMetricSnapshotDetail,
) -> dict[str, object]:
    return {
        "snapshot": golden_batch_metric_snapshot_record_payload(detail.snapshot),
        "profiles": [
            golden_batch_profile_metric_snapshot_payload(profile) for profile in detail.profiles
        ],
        "questions": [
            golden_batch_question_metric_snapshot_payload(question) for question in detail.questions
        ],
    }


def golden_batch_metric_snapshot_overall_comparison_payload(
    comparison: GoldenBatchMetricSnapshotComparison,
) -> dict[str, object]:
    overall = comparison.overall
    return {
        "evaluated_row_count_delta": overall.evaluated_row_count_delta,
        "total_result_count_delta": overall.total_result_count_delta,
        "average_result_count_delta": overall.average_result_count_delta,
        "total_elapsed_ms_delta": overall.total_elapsed_ms_delta,
        "average_elapsed_ms_delta": overall.average_elapsed_ms_delta,
        "hidden_violation_count_delta": overall.hidden_violation_count_delta,
        "mean_recall_at_k_delta": overall.mean_recall_at_k_delta,
        "mean_reciprocal_rank_delta": overall.mean_reciprocal_rank_delta,
        "mean_ndcg_delta": overall.mean_ndcg_delta,
        "no_answer_success_rate_delta": overall.no_answer_success_rate_delta,
    }


def golden_batch_profile_metric_snapshot_comparison_payload(
    profile: GoldenBatchProfileMetricSnapshotComparison,
) -> dict[str, object]:
    return {
        "profile_name": profile.profile_name,
        "comparison_status": profile.comparison_status,
        "base": (
            golden_batch_profile_metric_snapshot_payload(profile.base)
            if profile.base is not None
            else None
        ),
        "target": (
            golden_batch_profile_metric_snapshot_payload(profile.target)
            if profile.target is not None
            else None
        ),
        "question_count_delta": profile.question_count_delta,
        "hidden_violation_count_delta": profile.hidden_violation_count_delta,
        "mean_recall_at_k_delta": profile.mean_recall_at_k_delta,
        "mean_reciprocal_rank_delta": profile.mean_reciprocal_rank_delta,
        "mean_ndcg_delta": profile.mean_ndcg_delta,
        "no_answer_success_rate_delta": profile.no_answer_success_rate_delta,
        "average_result_count_delta": profile.average_result_count_delta,
        "average_elapsed_ms_delta": profile.average_elapsed_ms_delta,
    }


def golden_batch_question_metric_snapshot_comparison_payload(
    question: GoldenBatchQuestionMetricSnapshotComparison,
) -> dict[str, object]:
    return {
        "question_id": question.question_id,
        "question_text": question.question_text,
        "profile_name": question.profile_name,
        "comparison_status": question.comparison_status,
        "base": (
            golden_batch_question_metric_snapshot_payload(question.base)
            if question.base is not None
            else None
        ),
        "target": (
            golden_batch_question_metric_snapshot_payload(question.target)
            if question.target is not None
            else None
        ),
        "result_count_delta": question.result_count_delta,
        "elapsed_ms_delta": question.elapsed_ms_delta,
        "matched_visible_count_delta": question.matched_visible_count_delta,
        "hidden_violation_count_delta": question.hidden_violation_count_delta,
        "recall_at_k_delta": question.recall_at_k_delta,
        "reciprocal_rank_delta": question.reciprocal_rank_delta,
        "ndcg_delta": question.ndcg_delta,
    }


def golden_batch_metric_snapshot_comparison_payload(
    comparison: GoldenBatchMetricSnapshotComparison,
) -> dict[str, object]:
    return {
        "base": golden_batch_metric_snapshot_detail_payload(comparison.base),
        "target": golden_batch_metric_snapshot_detail_payload(comparison.target),
        "overall": golden_batch_metric_snapshot_overall_comparison_payload(comparison),
        "profiles": [
            golden_batch_profile_metric_snapshot_comparison_payload(profile)
            for profile in comparison.profiles
        ],
        "questions": [
            golden_batch_question_metric_snapshot_comparison_payload(question)
            for question in comparison.questions
        ],
        "compatibility_warnings": list(comparison.compatibility_warnings),
    }


def golden_batch_metric_snapshot_trend_point_payload(
    point: GoldenBatchMetricSnapshotTrendPoint,
) -> dict[str, object]:
    return {
        "sequence_number": point.sequence_number,
        "previous_snapshot_id": point.previous_snapshot_id,
        "snapshot": golden_batch_metric_snapshot_record_payload(point.snapshot),
        "evaluated_row_count_delta": point.evaluated_row_count_delta,
        "total_result_count_delta": point.total_result_count_delta,
        "average_result_count_delta": point.average_result_count_delta,
        "total_elapsed_ms_delta": point.total_elapsed_ms_delta,
        "average_elapsed_ms_delta": point.average_elapsed_ms_delta,
        "hidden_violation_count_delta": point.hidden_violation_count_delta,
        "mean_recall_at_k_delta": point.mean_recall_at_k_delta,
        "mean_reciprocal_rank_delta": point.mean_reciprocal_rank_delta,
        "mean_ndcg_delta": point.mean_ndcg_delta,
        "no_answer_success_rate_delta": point.no_answer_success_rate_delta,
    }


def golden_batch_metric_snapshot_trend_payload(
    trend: GoldenBatchMetricSnapshotTrend,
) -> dict[str, object]:
    return {
        "batch_key": trend.batch_key,
        "snapshot_count": len(trend.points),
        "first_snapshot": (
            golden_batch_metric_snapshot_record_payload(trend.first_snapshot)
            if trend.first_snapshot is not None
            else None
        ),
        "latest_snapshot": (
            golden_batch_metric_snapshot_record_payload(trend.latest_snapshot)
            if trend.latest_snapshot is not None
            else None
        ),
        "points": [
            golden_batch_metric_snapshot_trend_point_payload(point) for point in trend.points
        ],
    }


def golden_batch_metric_snapshot_compare_ui_rows(
    comparison: GoldenBatchMetricSnapshotComparison,
) -> list[dict[str, str]]:
    base = comparison.base.snapshot
    target = comparison.target.snapshot
    overall = comparison.overall
    return [
        {
            "label_key": "search_experiments.evaluated_rows",
            "base": f"{base.evaluated_row_count}",
            "target": f"{target.evaluated_row_count}",
            "delta": _signed_int_label(overall.evaluated_row_count_delta),
            "delta_class": _delta_text_class(overall.evaluated_row_count_delta),
        },
        {
            "label_key": "search_experiments.recall_at_k",
            "base": _percent_value_label(base.mean_recall_at_k),
            "target": _percent_value_label(target.mean_recall_at_k),
            "delta": _percent_delta_label(overall.mean_recall_at_k_delta),
            "delta_class": _delta_text_class(
                overall.mean_recall_at_k_delta,
                higher_is_better=True,
            ),
        },
        {
            "label_key": "search_experiments.mrr",
            "base": _percent_value_label(base.mean_reciprocal_rank),
            "target": _percent_value_label(target.mean_reciprocal_rank),
            "delta": _percent_delta_label(overall.mean_reciprocal_rank_delta),
            "delta_class": _delta_text_class(
                overall.mean_reciprocal_rank_delta,
                higher_is_better=True,
            ),
        },
        {
            "label_key": "search_experiments.ndcg",
            "base": _percent_value_label(base.mean_ndcg),
            "target": _percent_value_label(target.mean_ndcg),
            "delta": _percent_delta_label(overall.mean_ndcg_delta),
            "delta_class": _delta_text_class(
                overall.mean_ndcg_delta,
                higher_is_better=True,
            ),
        },
        {
            "label_key": "search_experiments.hidden_violations",
            "base": f"{base.hidden_violation_count}",
            "target": f"{target.hidden_violation_count}",
            "delta": _signed_int_label(overall.hidden_violation_count_delta),
            "delta_class": _delta_text_class(
                overall.hidden_violation_count_delta,
                higher_is_better=False,
            ),
        },
        {
            "label_key": "search_experiments.avg_results",
            "base": f"{base.average_result_count:.2f}",
            "target": f"{target.average_result_count:.2f}",
            "delta": _signed_float_label(overall.average_result_count_delta),
            "delta_class": _delta_text_class(overall.average_result_count_delta),
        },
        {
            "label_key": "search_experiments.avg_elapsed",
            "base": _ms_value_label(base.average_elapsed_ms),
            "target": _ms_value_label(target.average_elapsed_ms),
            "delta": _ms_delta_label(overall.average_elapsed_ms_delta),
            "delta_class": _delta_text_class(
                overall.average_elapsed_ms_delta,
                higher_is_better=False,
            ),
        },
    ]


def golden_batch_metric_snapshot_trend_ui_rows(
    trend: GoldenBatchMetricSnapshotTrend,
) -> list[dict[str, str]]:
    return [
        {
            "snapshot_id": f"#{point.snapshot.snapshot_id}",
            "created_at": _datetime_label(point.snapshot.created_at),
            "recall_at_k": _percent_value_label(point.snapshot.mean_recall_at_k),
            "recall_delta": _percent_delta_label(point.mean_recall_at_k_delta),
            "recall_delta_class": _delta_text_class(
                point.mean_recall_at_k_delta,
                higher_is_better=True,
            ),
            "mrr": _percent_value_label(point.snapshot.mean_reciprocal_rank),
            "mrr_delta": _percent_delta_label(point.mean_reciprocal_rank_delta),
            "mrr_delta_class": _delta_text_class(
                point.mean_reciprocal_rank_delta,
                higher_is_better=True,
            ),
            "ndcg": _percent_value_label(point.snapshot.mean_ndcg),
            "ndcg_delta": _percent_delta_label(point.mean_ndcg_delta),
            "ndcg_delta_class": _delta_text_class(
                point.mean_ndcg_delta,
                higher_is_better=True,
            ),
            "hidden": f"{point.snapshot.hidden_violation_count}",
            "hidden_delta": _signed_int_label(point.hidden_violation_count_delta),
            "hidden_delta_class": _delta_text_class(
                point.hidden_violation_count_delta,
                higher_is_better=False,
            ),
            "average_elapsed": _ms_value_label(point.snapshot.average_elapsed_ms),
            "average_elapsed_delta": _ms_delta_label(point.average_elapsed_ms_delta),
            "average_elapsed_delta_class": _delta_text_class(
                point.average_elapsed_ms_delta,
                higher_is_better=False,
            ),
        }
        for point in trend.points
    ]


def golden_batch_metric_snapshot_trend_summary_ui_rows(
    trend: GoldenBatchMetricSnapshotTrend,
) -> list[dict[str, str]]:
    if trend.first_snapshot is None or trend.latest_snapshot is None:
        return []
    overall = _compare_metric_snapshot_records_for_ui(
        trend.first_snapshot,
        trend.latest_snapshot,
    )
    return [
        {
            "label_key": "search_experiments.recall_at_k",
            "first": _percent_value_label(trend.first_snapshot.mean_recall_at_k),
            "latest": _percent_value_label(trend.latest_snapshot.mean_recall_at_k),
            "delta": _percent_delta_label(overall["mean_recall_at_k_delta"]),
            "delta_class": _delta_text_class(
                overall["mean_recall_at_k_delta"],
                higher_is_better=True,
            ),
        },
        {
            "label_key": "search_experiments.mrr",
            "first": _percent_value_label(trend.first_snapshot.mean_reciprocal_rank),
            "latest": _percent_value_label(trend.latest_snapshot.mean_reciprocal_rank),
            "delta": _percent_delta_label(overall["mean_reciprocal_rank_delta"]),
            "delta_class": _delta_text_class(
                overall["mean_reciprocal_rank_delta"],
                higher_is_better=True,
            ),
        },
        {
            "label_key": "search_experiments.ndcg",
            "first": _percent_value_label(trend.first_snapshot.mean_ndcg),
            "latest": _percent_value_label(trend.latest_snapshot.mean_ndcg),
            "delta": _percent_delta_label(overall["mean_ndcg_delta"]),
            "delta_class": _delta_text_class(
                overall["mean_ndcg_delta"],
                higher_is_better=True,
            ),
        },
        {
            "label_key": "search_experiments.hidden_violations",
            "first": f"{trend.first_snapshot.hidden_violation_count}",
            "latest": f"{trend.latest_snapshot.hidden_violation_count}",
            "delta": _signed_int_label(overall["hidden_violation_count_delta"]),
            "delta_class": _delta_text_class(
                overall["hidden_violation_count_delta"],
                higher_is_better=False,
            ),
        },
        {
            "label_key": "search_experiments.avg_elapsed",
            "first": _ms_value_label(trend.first_snapshot.average_elapsed_ms),
            "latest": _ms_value_label(trend.latest_snapshot.average_elapsed_ms),
            "delta": _ms_delta_label(overall["average_elapsed_ms_delta"]),
            "delta_class": _delta_text_class(
                overall["average_elapsed_ms_delta"],
                higher_is_better=False,
            ),
        },
    ]


def _compare_metric_snapshot_records_for_ui(
    base: GoldenBatchMetricSnapshotRecord,
    target: GoldenBatchMetricSnapshotRecord,
) -> dict[str, float | int | None]:
    return {
        "mean_recall_at_k_delta": _optional_numeric_delta(
            base.mean_recall_at_k,
            target.mean_recall_at_k,
        ),
        "mean_reciprocal_rank_delta": _optional_numeric_delta(
            base.mean_reciprocal_rank,
            target.mean_reciprocal_rank,
        ),
        "mean_ndcg_delta": _optional_numeric_delta(base.mean_ndcg, target.mean_ndcg),
        "hidden_violation_count_delta": target.hidden_violation_count - base.hidden_violation_count,
        "average_elapsed_ms_delta": _optional_numeric_delta(
            base.average_elapsed_ms,
            target.average_elapsed_ms,
        ),
    }


def _percent_value_label(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def _percent_delta_label(value: float | int | None) -> str:
    return "-" if value is None else f"{float(value) * 100:+.2f}pp"


def _ms_value_label(value: float | int | None) -> str:
    return "-" if value is None else f"{float(value):.1f} ms"


def _ms_delta_label(value: float | int | None) -> str:
    return "-" if value is None else f"{float(value):+.1f} ms"


def _signed_int_label(value: int | None) -> str:
    return "-" if value is None else f"{value:+d}"


def _signed_float_label(value: float | int | None) -> str:
    return "-" if value is None else f"{float(value):+.2f}"


def _optional_numeric_delta(
    base_value: float | int | None,
    target_value: float | int | None,
) -> float | None:
    if base_value is None or target_value is None:
        return None
    return float(target_value) - float(base_value)


def _delta_text_class(
    value: float | int | None,
    *,
    higher_is_better: bool | None = None,
) -> str:
    if value is None or abs(float(value)) < 0.0000001:
        return "text-secondary"
    if higher_is_better is None:
        return "text-body"
    improved = float(value) > 0 if higher_is_better else float(value) < 0
    return "text-success" if improved else "text-danger"


def search_experiment_execution_payload(
    report: SearchExperimentExecutionReport,
) -> dict[str, object]:
    run = report.run
    return {
        "experiment_run": search_experiment_run_record_payload(run),
        "strategy": {
            "strategy_name": report.strategy_selection.strategy.strategy_name,
            "display_name": report.strategy_selection.strategy.display_name,
            "mode": report.strategy_selection.strategy.mode,
            "similarity_metric": report.strategy_selection.strategy.similarity_metric,
            "top_k": report.strategy_selection.top_k,
            "score_threshold": report.strategy_selection.score_threshold,
            "runtime_parameters": report.strategy_selection.runtime_parameters,
        },
        "profile_summaries": [
            search_experiment_profile_execution_payload(summary)
            for summary in report.profile_summaries
        ],
        "search_result": search_compare_payload(report.search_result),
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


def _valid_bm25_tokenizer_name_or_none(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return validate_bm25_tokenizer_name(value.strip())
    except ValueError:
        return None


def search_log_bm25_tokenizer_name(query_runtime_metadata: dict[str, object]) -> str | None:
    direct_tokenizer_name = _valid_bm25_tokenizer_name_or_none(
        query_runtime_metadata.get("bm25_tokenizer_name")
    )
    if direct_tokenizer_name is not None:
        return direct_tokenizer_name

    profile_keyword_searches = query_runtime_metadata.get("profile_keyword_searches")
    if isinstance(profile_keyword_searches, dict):
        profile_metadata = profile_keyword_searches.get(BM25_SEARCH_PROFILE_NAME)
        if isinstance(profile_metadata, dict):
            profile_tokenizer_name = _valid_bm25_tokenizer_name_or_none(
                profile_metadata.get("tokenizer_name")
            )
            if profile_tokenizer_name is not None:
                return profile_tokenizer_name
    return None


def search_log_reranked_vector_profile_name(
    query_runtime_metadata: dict[str, object],
) -> str | None:
    profile_reranked_searches = query_runtime_metadata.get("profile_reranked_searches")
    if not isinstance(profile_reranked_searches, dict):
        return None
    profile_metadata = profile_reranked_searches.get(RERANKED_SEARCH_PROFILE_NAME)
    if not isinstance(profile_metadata, dict):
        return None
    value = profile_metadata.get("reranked_vector_profile_name") or profile_metadata.get(
        "source_vector_profile_name"
    )
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


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
        "bm25_tokenizer_name": search_log_bm25_tokenizer_name(search_log.query_runtime_metadata),
        "reranked_vector_profile_name": search_log_reranked_vector_profile_name(
            search_log.query_runtime_metadata
        ),
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

    raw_bm25_tokenizer_name = (
        query_params.get("bm25_tokenizer_name") or DEFAULT_BM25_TOKENIZER_NAME
    ).strip()
    try:
        bm25_tokenizer_name = validate_bm25_tokenizer_name(raw_bm25_tokenizer_name)
    except ValueError:
        bm25_tokenizer_name = DEFAULT_BM25_TOKENIZER_NAME

    return {
        "replay_search_log_id": (query_params.get("replay_search_log_id") or "").strip(),
        "query_text": (query_params.get("query_text") or "").strip(),
        "actor_user_id": actor_user_id,
        "requested_search_scope": scope,
        "top_k": top_k,
        "document_group": (query_params.get("document_group") or "").strip(),
        "file_type": (query_params.get("file_type") or "").strip(),
        "chunk_policy_name": (query_params.get("chunk_policy_name") or "").strip(),
        "bm25_tokenizer_name": bm25_tokenizer_name,
        "hybrid_vector_profile_name": (
            query_params.get("hybrid_vector_profile_name") or ""
        ).strip(),
        "reranked_vector_profile_name": (
            query_params.get("reranked_vector_profile_name") or ""
        ).strip(),
        "profiles": profiles,
        "profile_selection_explicit": bool(query_params.getlist("profiles")),
        "require_real_provider": (query_params.get("require_real_provider") or "").lower()
        in {"1", "true", "yes", "on"},
    }


def search_reranker_runtime_control_payload(settings: object) -> dict[str, object]:
    raw_mode = str(getattr(settings, "reranker_provider_mode", "mock") or "mock").strip().lower()
    raw_base_url = getattr(settings, "remote_reranker_provider_url", None)
    raw_timeout = getattr(settings, "remote_reranker_provider_timeout_seconds", 60.0)
    try:
        config = reranker_runtime_config_from_settings(settings)
        return {
            "status": "configured",
            "validation_error": "",
            "mode": config.mode,
            "remote_base_url": config.remote_base_url or "",
            "timeout_seconds": config.remote_timeout_seconds,
            "reranker_profile_name": DEFAULT_RERANKER_PROFILE_NAME,
            "reranker_model_id": DEFAULT_RERANKER_MODEL_ID,
        }
    except (InvalidRerankerError, TypeError, ValueError) as exc:
        try:
            timeout_seconds: float | None = float(raw_timeout)
        except (TypeError, ValueError):
            timeout_seconds = None
        return {
            "status": "invalid",
            "validation_error": str(exc),
            "mode": raw_mode or "mock",
            "remote_base_url": str(raw_base_url).strip().rstrip("/") if raw_base_url else "",
            "timeout_seconds": timeout_seconds,
            "reranker_profile_name": DEFAULT_RERANKER_PROFILE_NAME,
            "reranker_model_id": DEFAULT_RERANKER_MODEL_ID,
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


def search_runtime_failure_payload(
    failure: SearchRuntimeFailureRecord,
) -> dict[str, object]:
    return {
        "search_log_id": failure.search_log_id,
        "query_text": failure.query_text,
        "actor_user_id": failure.actor_user_id,
        "actor_login_id": failure.actor_login_id,
        "actor_display_name": failure.actor_display_name,
        "requested_search_scope": failure.requested_search_scope,
        "effective_search_scope": failure.effective_search_scope,
        "document_group": failure.document_group,
        "file_type": failure.file_type,
        "chunk_policy_name": failure.chunk_policy_name,
        "top_k": failure.top_k,
        "profile_name": failure.profile_name,
        "error_code": failure.error_code,
        "error_message": failure.error_message,
        "elapsed_ms": failure.elapsed_ms,
        "created_at": _datetime_response(failure.created_at),
        "search_log_url": f"/search/logs?search_log_id={failure.search_log_id}",
    }


def search_latency_outlier_payload(
    outlier: SearchLatencyOutlierRecord,
) -> dict[str, object]:
    return {
        "search_log_id": outlier.search_log_id,
        "query_text": outlier.query_text,
        "actor_user_id": outlier.actor_user_id,
        "actor_login_id": outlier.actor_login_id,
        "actor_display_name": outlier.actor_display_name,
        "requested_search_scope": outlier.requested_search_scope,
        "effective_search_scope": outlier.effective_search_scope,
        "document_group": outlier.document_group,
        "file_type": outlier.file_type,
        "chunk_policy_name": outlier.chunk_policy_name,
        "top_k": outlier.top_k,
        "profiles": list(outlier.profiles),
        "profile_count": len(outlier.profiles),
        "total_elapsed_ms": outlier.total_elapsed_ms,
        "succeeded_profile_count": outlier.succeeded_profile_count,
        "failed_profile_count": outlier.failed_profile_count,
        "created_at": _datetime_response(outlier.created_at),
        "search_log_url": f"/search/logs?search_log_id={outlier.search_log_id}",
    }


def search_no_result_payload(record: SearchNoResultRecord) -> dict[str, object]:
    return {
        "search_log_id": record.search_log_id,
        "query_text": record.query_text,
        "actor_user_id": record.actor_user_id,
        "actor_login_id": record.actor_login_id,
        "actor_display_name": record.actor_display_name,
        "requested_search_scope": record.requested_search_scope,
        "effective_search_scope": record.effective_search_scope,
        "document_group": record.document_group,
        "file_type": record.file_type,
        "chunk_policy_name": record.chunk_policy_name,
        "top_k": record.top_k,
        "profiles": list(record.profiles),
        "profile_count": len(record.profiles),
        "total_elapsed_ms": record.total_elapsed_ms,
        "failed_profile_count": record.failed_profile_count,
        "created_at": _datetime_response(record.created_at),
        "created_at_label": _datetime_label(record.created_at),
        "search_log_url": f"/search/logs?search_log_id={record.search_log_id}",
    }


def search_duplicate_fingerprint_payload(
    record: SearchDuplicateFingerprintRecord,
) -> dict[str, object]:
    return {
        "condition_fingerprint": record.condition_fingerprint,
        "duplicate_count": record.duplicate_count,
        "latest_search_log_id": record.latest_search_log_id,
        "first_search_log_id": record.first_search_log_id,
        "query_text": record.query_text,
        "actor_user_id": record.actor_user_id,
        "actor_login_id": record.actor_login_id,
        "actor_display_name": record.actor_display_name,
        "requested_search_scope": record.requested_search_scope,
        "effective_search_scope": record.effective_search_scope,
        "document_group": record.document_group,
        "file_type": record.file_type,
        "chunk_policy_name": record.chunk_policy_name,
        "top_k": record.top_k,
        "similarity_metric": record.similarity_metric,
        "profiles": list(record.profiles),
        "profile_count": len(record.profiles),
        "zero_result_count": record.zero_result_count,
        "runtime_failure_count": record.runtime_failure_count,
        "average_total_elapsed_ms": record.average_total_elapsed_ms,
        "first_created_at": _datetime_response(record.first_created_at),
        "first_created_at_label": _datetime_label(record.first_created_at),
        "latest_created_at": _datetime_response(record.latest_created_at),
        "latest_created_at_label": _datetime_label(record.latest_created_at),
        "latest_search_log_url": f"/search/logs?search_log_id={record.latest_search_log_id}",
    }


def search_operations_summary_payload(
    summary: SearchOperationsSummaryRecord,
) -> dict[str, object]:
    return {
        "lookback_hours": summary.lookback_hours,
        "min_total_elapsed_ms": summary.min_total_elapsed_ms,
        "search_count": summary.search_count,
        "result_row_count": summary.result_row_count,
        "no_result_count": summary.no_result_count,
        "runtime_failure_count": summary.runtime_failure_count,
        "latency_outlier_count": summary.latency_outlier_count,
        "real_provider_required_count": summary.real_provider_required_count,
        "mock_fallback_allowed_count": summary.mock_fallback_allowed_count,
        "duplicate_fingerprint_count": summary.duplicate_fingerprint_count,
        "max_duplicate_count": summary.max_duplicate_count,
        "average_total_elapsed_ms": summary.average_total_elapsed_ms,
        "latest_search_at": _datetime_response(summary.latest_search_at),
        "latest_search_at_label": _datetime_label(summary.latest_search_at),
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


def golden_search_experiment_batch_input_from_request(
    payload: GoldenSearchExperimentBatchRequest,
) -> GoldenSearchExperimentBatchInput:
    return GoldenSearchExperimentBatchInput(
        question_set_id=payload.question_set_id,
        run_name_prefix=payload.run_name_prefix,
        profiles=tuple(payload.profiles) if payload.profiles is not None else None,
        strategy_name=payload.strategy_name,
        top_k=payload.top_k,
        score_threshold=payload.score_threshold,
        chunk_policy_name=payload.chunk_policy_name,
        runtime_metadata=dict(payload.runtime_metadata),
        created_by=payload.created_by,
        created_by_user_id=payload.created_by_user_id,
        allow_mock_fallback=payload.allow_mock_fallback,
    )


def golden_search_experiment_batch_payload(
    report: GoldenSearchExperimentBatchReport,
) -> dict[str, object]:
    batch_key = (
        golden_search_experiment_batch_key_from_run(report.question_reports[0].experiment.run)
        if report.question_reports
        else None
    )
    return {
        "batch_key": batch_key,
        "question_set": golden_question_set_payload(report.question_set),
        "question_count": len(report.question_reports),
        "total_elapsed_ms": report.total_elapsed_ms,
        "experiment_run_ids_by_question": {
            str(question_id): experiment_run_id
            for question_id, experiment_run_id in report.experiment_run_ids_by_question.items()
        },
        "questions": [
            {
                "question": golden_question_payload(item.question),
                "experiment": search_experiment_execution_payload(item.experiment),
            }
            for item in report.question_reports
        ],
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


def dashboard_file_type_summary_payload(
    summary: DashboardFileTypeSummary,
) -> dict[str, object]:
    return {
        "file_type": summary.file_type,
        "file_count": summary.file_count,
        "document_count": summary.document_count,
        "total_file_size_bytes": summary.total_file_size_bytes,
        "total_file_size_label": _byte_count_label(summary.total_file_size_bytes),
    }


def dashboard_document_group_summary_payload(
    summary: DashboardDocumentGroupSummary,
) -> dict[str, object]:
    return {
        "document_group": summary.document_group,
        "file_count": summary.file_count,
        "document_count": summary.document_count,
        "chunk_count": summary.chunk_count,
    }


def dashboard_chunk_policy_summary_payload(
    summary: DashboardChunkPolicySummary,
) -> dict[str, object]:
    return {
        "chunk_policy_name": summary.chunk_policy_name,
        "chunk_count": summary.chunk_count,
        "average_token_count": summary.average_token_count,
    }


def dashboard_core_metrics_payload(
    metrics: DashboardCoreMetrics,
) -> dict[str, object]:
    return {
        "file_count": metrics.file_count,
        "document_count": metrics.document_count,
        "chunk_count": metrics.chunk_count,
        "embedding_job_count": metrics.embedding_job_count,
        "search_log_count": metrics.search_log_count,
        "total_file_size_bytes": metrics.total_file_size_bytes,
        "total_file_size_label": _byte_count_label(metrics.total_file_size_bytes),
        "average_file_size_bytes": metrics.average_file_size_bytes,
        "average_file_size_label": _byte_count_label(metrics.average_file_size_bytes),
        "duplicate_checksum_count": metrics.duplicate_checksum_count,
        "average_chunk_token_count": metrics.average_chunk_token_count,
        "file_types": [
            dashboard_file_type_summary_payload(summary) for summary in metrics.file_types
        ],
        "document_groups": [
            dashboard_document_group_summary_payload(summary) for summary in metrics.document_groups
        ],
        "chunk_policies": [
            dashboard_chunk_policy_summary_payload(summary) for summary in metrics.chunk_policies
        ],
    }


def dashboard_health_signal_payload(signal: DashboardHealthSignal) -> dict[str, object]:
    return {
        "code": signal.code,
        "severity": signal.severity,
        "count": signal.count,
        "threshold": signal.threshold,
        "action_url": signal.action_url,
    }


def dashboard_health_threshold_settings_payload(
    settings: DashboardHealthThresholdSettings,
) -> dict[str, object]:
    return {"thresholds": dict(settings.thresholds)}


def dashboard_health_threshold_ui_rows(
    settings: DashboardHealthThresholdSettings,
) -> list[dict[str, object]]:
    return [
        {
            "code": code,
            "severity": severity,
            "value": settings.thresholds.get(code, 1),
        }
        for code, severity in DASHBOARD_HEALTH_THRESHOLD_UI_ROWS
    ]


def dashboard_operational_health_payload(
    health: DashboardOperationalHealth,
    threshold_settings: DashboardHealthThresholdSettings | None = None,
) -> dict[str, object]:
    payload = {
        "status": health.status,
        "signal_count": health.signal_count,
        "critical_count": health.critical_count,
        "warning_count": health.warning_count,
        "signals": [dashboard_health_signal_payload(signal) for signal in health.signals],
    }
    if threshold_settings is not None:
        payload["thresholds"] = dict(threshold_settings.thresholds)
    return payload


def dashboard_snapshot_export_payload(
    database_url: str,
    *,
    lookback_hours: int = DEFAULT_DASHBOARD_THROUGHPUT_LOOKBACK_HOURS,
) -> dict[str, object]:
    core_metrics = get_dashboard_core_metrics(database_url)
    pipeline_queue = get_pipeline_queue_summary(database_url)
    throughput_latency = get_dashboard_throughput_latency_snapshot(
        database_url,
        lookback_hours=lookback_hours,
    )
    recent_failures = get_dashboard_recent_failures(database_url, limit=10)
    embedding_backlog = get_embedding_job_backlog_summary(database_url)
    evaluations = get_evaluation_dashboard_summary(database_url, recent_limit=10)
    threshold_settings = load_dashboard_health_threshold_settings(database_url)
    operational_health = summarize_dashboard_operational_health(
        pipeline_queue=pipeline_queue,
        embedding_backlog=embedding_backlog,
        recent_failures=recent_failures,
        thresholds=threshold_settings.thresholds,
    )
    return {
        "version": 1,
        "exported_at": _datetime_response(datetime.now(UTC)),
        "lookback_hours": throughput_latency.lookback_hours,
        "operational_health": dashboard_operational_health_payload(
            operational_health,
            threshold_settings=threshold_settings,
        ),
        "core_metrics": dashboard_core_metrics_payload(core_metrics),
        "pipeline_queue": pipeline_queue_summary_payload(pipeline_queue),
        "throughput_latency": dashboard_throughput_latency_snapshot_payload(throughput_latency),
        "recent_failures": dashboard_failure_summary_payload(recent_failures),
        "embedding_backlog": embedding_job_backlog_summary_payload(embedding_backlog),
        "evaluations": evaluation_dashboard_summary_payload(evaluations),
    }


def dashboard_snapshot_summary_csv(snapshot: dict[str, object]) -> str:
    core_metrics = dict(snapshot["core_metrics"])
    health = dict(snapshot["operational_health"])
    pipeline_queue = dict(snapshot["pipeline_queue"])
    embedding_backlog = dict(snapshot["embedding_backlog"])
    throughput_latency = dict(snapshot["throughput_latency"])
    throughput_embedding = dict(throughput_latency["embedding"])
    throughput_search = dict(throughput_latency["search"])
    recent_failures = dict(snapshot["recent_failures"])
    evaluations = dict(snapshot["evaluations"])
    fieldnames = [
        "exported_at",
        "lookback_hours",
        "health_status",
        "health_critical_count",
        "health_warning_count",
        "documents",
        "chunks",
        "embedding_jobs",
        "search_logs",
        "pipeline_claimable",
        "pipeline_attention",
        "embedding_claimable",
        "embedding_attention",
        "embedding_jobs_per_second",
        "search_average_latency_ms",
        "recent_failure_count",
        "evaluation_run_count",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(
        {
            "exported_at": snapshot["exported_at"],
            "lookback_hours": snapshot["lookback_hours"],
            "health_status": health["status"],
            "health_critical_count": health["critical_count"],
            "health_warning_count": health["warning_count"],
            "documents": core_metrics["document_count"],
            "chunks": core_metrics["chunk_count"],
            "embedding_jobs": core_metrics["embedding_job_count"],
            "search_logs": core_metrics["search_log_count"],
            "pipeline_claimable": pipeline_queue["claimable_count"],
            "pipeline_attention": pipeline_queue["attention_count"],
            "embedding_claimable": embedding_backlog["claimable_count"],
            "embedding_attention": embedding_backlog["attention_count"],
            "embedding_jobs_per_second": throughput_embedding["throughput_per_second"],
            "search_average_latency_ms": throughput_search["average_total_elapsed_ms"],
            "recent_failure_count": recent_failures["total_count"],
            "evaluation_run_count": evaluations["evaluation_run_count"],
        }
    )
    return output.getvalue()


def dashboard_pipeline_stage_latency_payload(
    stage: DashboardPipelineStageLatency,
) -> dict[str, object]:
    return {
        "stage": stage.stage,
        "completed_count": stage.completed_count,
        "succeeded_count": stage.succeeded_count,
        "failed_count": stage.failed_count,
        "canceled_count": stage.canceled_count,
        "average_duration_ms": stage.average_duration_ms,
        "average_duration_label": _duration_ms_label(stage.average_duration_ms),
    }


def dashboard_pipeline_throughput_payload(
    pipeline: DashboardPipelineThroughput,
) -> dict[str, object]:
    return {
        "completed_count": pipeline.completed_count,
        "succeeded_count": pipeline.succeeded_count,
        "failed_count": pipeline.failed_count,
        "canceled_count": pipeline.canceled_count,
        "skipped_count": pipeline.skipped_count,
        "average_duration_ms": pipeline.average_duration_ms,
        "average_duration_label": _duration_ms_label(pipeline.average_duration_ms),
        "latest_finished_at": _datetime_response(pipeline.latest_finished_at),
        "latest_finished_label": _datetime_label(pipeline.latest_finished_at),
        "stages": [dashboard_pipeline_stage_latency_payload(stage) for stage in pipeline.stages],
    }


def dashboard_embedding_profile_throughput_payload(
    profile: DashboardEmbeddingProfileThroughput,
) -> dict[str, object]:
    return {
        "profile_name": profile.profile_name,
        "completed_job_count": profile.completed_job_count,
        "succeeded_job_count": profile.succeeded_job_count,
        "failed_job_count": profile.failed_job_count,
        "skipped_job_count": profile.skipped_job_count,
        "average_job_duration_ms": profile.average_job_duration_ms,
        "average_job_duration_label": _duration_ms_label(profile.average_job_duration_ms),
        "batch_run_count": profile.batch_run_count,
        "processed_count": profile.processed_count,
        "succeeded_count": profile.succeeded_count,
        "failed_count": profile.failed_count,
        "deferred_count": profile.deferred_count,
        "average_batch_elapsed_ms": profile.average_batch_elapsed_ms,
        "average_batch_elapsed_label": _duration_ms_label(profile.average_batch_elapsed_ms),
        "throughput_per_second": profile.throughput_per_second,
        "success_rate_percent": profile.success_rate_percent,
        "success_rate_label": _percent_label(profile.success_rate_percent),
    }


def dashboard_embedding_throughput_payload(
    embedding: DashboardEmbeddingThroughput,
) -> dict[str, object]:
    return {
        "completed_job_count": embedding.completed_job_count,
        "succeeded_job_count": embedding.succeeded_job_count,
        "failed_job_count": embedding.failed_job_count,
        "skipped_job_count": embedding.skipped_job_count,
        "average_job_duration_ms": embedding.average_job_duration_ms,
        "average_job_duration_label": _duration_ms_label(embedding.average_job_duration_ms),
        "batch_run_count": embedding.batch_run_count,
        "processed_count": embedding.processed_count,
        "succeeded_count": embedding.succeeded_count,
        "failed_count": embedding.failed_count,
        "deferred_count": embedding.deferred_count,
        "average_batch_elapsed_ms": embedding.average_batch_elapsed_ms,
        "average_batch_elapsed_label": _duration_ms_label(embedding.average_batch_elapsed_ms),
        "throughput_per_second": embedding.throughput_per_second,
        "latest_completed_at": _datetime_response(embedding.latest_completed_at),
        "latest_completed_label": _datetime_label(embedding.latest_completed_at),
        "profiles": [
            dashboard_embedding_profile_throughput_payload(profile)
            for profile in embedding.profiles
        ],
    }


def dashboard_search_profile_latency_payload(
    profile: DashboardSearchProfileLatency,
) -> dict[str, object]:
    return {
        "profile_name": profile.profile_name,
        "search_log_count": profile.search_log_count,
        "result_count": profile.result_count,
        "average_profile_elapsed_ms": profile.average_profile_elapsed_ms,
        "average_profile_elapsed_label": _duration_ms_label(profile.average_profile_elapsed_ms),
    }


def dashboard_search_latency_payload(
    search: DashboardSearchLatency,
) -> dict[str, object]:
    return {
        "search_log_count": search.search_log_count,
        "result_count": search.result_count,
        "average_total_elapsed_ms": search.average_total_elapsed_ms,
        "average_total_elapsed_label": _duration_ms_label(search.average_total_elapsed_ms),
        "average_profile_elapsed_ms": search.average_profile_elapsed_ms,
        "average_profile_elapsed_label": _duration_ms_label(search.average_profile_elapsed_ms),
        "latest_search_at": _datetime_response(search.latest_search_at),
        "latest_search_label": _datetime_label(search.latest_search_at),
        "profiles": [
            dashboard_search_profile_latency_payload(profile) for profile in search.profiles
        ],
    }


def dashboard_throughput_latency_snapshot_payload(
    snapshot: DashboardThroughputLatencySnapshot,
) -> dict[str, object]:
    return {
        "lookback_hours": snapshot.lookback_hours,
        "pipeline": dashboard_pipeline_throughput_payload(snapshot.pipeline),
        "embedding": dashboard_embedding_throughput_payload(snapshot.embedding),
        "search": dashboard_search_latency_payload(snapshot.search),
    }


def dashboard_failure_record_payload(record: DashboardFailureRecord) -> dict[str, object]:
    return {
        "source": record.source,
        "severity": record.severity,
        "title": record.title,
        "message": record.message,
        "occurred_at": _datetime_response(record.occurred_at),
        "occurred_at_label": _datetime_label(record.occurred_at),
        "status": record.status,
        "action_url": record.action_url,
        "reference_id": record.reference_id,
        "metadata": record.metadata,
    }


def dashboard_failure_summary_payload(
    summary: DashboardFailureSummary,
) -> dict[str, object]:
    return {
        "total_count": summary.total_count,
        "pipeline_failure_count": summary.pipeline_failure_count,
        "embedding_failure_count": summary.embedding_failure_count,
        "parsing_failure_count": summary.parsing_failure_count,
        "app_error_count": summary.app_error_count,
        "provider_alert_count": summary.provider_alert_count,
        "failures": [dashboard_failure_record_payload(record) for record in summary.failures],
    }


def dashboard_failure_detail_payload(detail: DashboardFailureDetail) -> dict[str, object]:
    return {
        "source": detail.source,
        "reference_id": detail.reference_id,
        "title": detail.title,
        "severity": detail.severity,
        "status": detail.status,
        "message": detail.message,
        "occurred_at": _datetime_response(detail.occurred_at),
        "occurred_at_label": _datetime_label(detail.occurred_at),
        "action_url": detail.action_url,
        "summary": _json_safe_dashboard_display_value(detail.summary),
        "context": _json_safe_dashboard_display_value(detail.context),
        "raw": _json_safe_dashboard_raw_value(detail.raw),
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
        "latest_pipeline_progress_percent": _percent_value(item.latest_pipeline_progress_percent),
        "latest_pipeline_progress_label": _percent_label(item.latest_pipeline_progress_percent),
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
        "artifact_id": chunk.artifact_id,
        "block_id": chunk.block_id,
        "chunk_type": chunk.chunk_type,
        "content_markdown": chunk.content_markdown,
        "parser_name": chunk.parser_name,
        "parser_version": chunk.parser_version,
        "heading_path": list(chunk.heading_path),
        "source_anchor": chunk.source_anchor,
        "page_no": chunk.page_no,
        "slide_no": chunk.slide_no,
        "sheet_name": chunk.sheet_name,
        "cell_range": chunk.cell_range,
        "source_char_start": chunk.source_char_start,
        "source_char_end": chunk.source_char_end,
        "token_count": chunk.token_count,
        "char_count": chunk.char_count,
        "prev_chunk_id": chunk.prev_chunk_id,
        "next_chunk_id": chunk.next_chunk_id,
        "metadata": chunk.metadata,
    }


def _chunks_for_source_trace(
    chunks: list[ChunkRecord],
    *,
    artifact_id: int,
    block_ids: set[int],
) -> list[ChunkRecord]:
    return [
        chunk
        for chunk in chunks
        if chunk.artifact_id == artifact_id
        or (chunk.block_id is not None and chunk.block_id in block_ids)
    ]


def chunk_source_trace_preview_payload(
    artifact: ExtractionArtifactRecord | None,
    blocks: list[DocumentBlockRecord],
    chunks: list[ChunkRecord],
) -> dict[str, object]:
    block_ids = {block.block_id for block in blocks}
    chunks_by_block_id: dict[int, list[ChunkRecord]] = {block_id: [] for block_id in block_ids}
    unlinked_chunks: list[ChunkRecord] = []
    for chunk in chunks:
        if chunk.block_id is not None and chunk.block_id in chunks_by_block_id:
            chunks_by_block_id[chunk.block_id].append(chunk)
        else:
            unlinked_chunks.append(chunk)

    traced_block_count = sum(1 for block in blocks if chunks_by_block_id.get(block.block_id))
    policy_names = sorted({chunk.chunk_policy_name for chunk in chunks})
    return {
        "selected_artifact_id": artifact.artifact_id if artifact is not None else None,
        "selected_artifact": (
            extraction_artifact_payload(artifact) if artifact is not None else None
        ),
        "summary": {
            "block_count": len(blocks),
            "chunk_count": len(chunks),
            "traced_block_count": traced_block_count,
            "unlinked_chunk_count": len(unlinked_chunks),
            "chunk_policy_names": policy_names,
        },
        "block_traces": [
            {
                "block": document_block_payload(block),
                "chunk_count": len(chunks_by_block_id[block.block_id]),
                "chunks": [chunk_payload(chunk) for chunk in chunks_by_block_id[block.block_id]],
            }
            for block in blocks
        ],
        "unlinked_chunks": [chunk_payload(chunk) for chunk in unlinked_chunks],
    }


def _text_preview(value: str | None, *, limit: int = 600) -> str | None:
    if value is None:
        return None
    return value if len(value) <= limit else f"{value[:limit]}..."


def extraction_run_payload(run: ExtractionRunRecord) -> dict[str, object]:
    return {
        "extraction_run_id": run.extraction_run_id,
        "file_id": run.file_id,
        "document_id": run.document_id,
        "extraction_profile_name": run.extraction_profile_name,
        "status": run.status,
        "provider_mode": run.provider_mode,
        "extractor_name": run.extractor_name,
        "extractor_version": run.extractor_version,
        "started_at": _datetime_response(run.started_at),
        "finished_at": _datetime_response(run.finished_at),
        "elapsed_ms": run.elapsed_ms,
        "warning_count": run.warning_count,
        "error_count": run.error_count,
        "error_code": run.error_code,
        "error_message": run.error_message,
        "runtime_metadata": run.runtime_metadata,
        "created_at": _datetime_response(run.created_at),
        "updated_at": _datetime_response(run.updated_at),
    }


def _resolve_extraction_rerun_profile_name(
    file_record: FileMetadataRecord,
    requested_profile_name: str | None,
) -> str:
    if requested_profile_name is not None:
        profile_name = requested_profile_name.strip()
        if not profile_name:
            raise InvalidIngestionArtifactError("extraction_profile_name must not be blank")
        return profile_name

    profile_name = select_local_extraction_profile_name(file_record.file_ext)
    if profile_name is None:
        file_type = file_record.file_ext or "(none)"
        raise InvalidIngestionArtifactError(
            f"No local extraction profile supports file type: {file_type}"
        )
    return profile_name


def build_extraction_rerun_request(
    *,
    document: DocumentInventoryItem,
    file_record: FileMetadataRecord,
    payload: ExtractionRerunRequest,
) -> ExtractionRuntimeRequest:
    requested_by = (payload.requested_by or "").strip() or "extraction-rerun-api"
    profile_name = _resolve_extraction_rerun_profile_name(
        file_record,
        payload.extraction_profile_name,
    )
    options = dict(payload.options)
    options["rerun_request"] = {
        "source": "extraction_rerun_api",
        "requested_by": requested_by,
        "document_id": document.document_id,
        "file_id": file_record.file_id,
    }
    return ExtractionRuntimeRequest(
        file_id=file_record.file_id,
        document_id=document.document_id,
        storage_path=file_record.storage_path,
        extraction_profile_name=profile_name,
        mime_type=file_record.mime_type,
        detected_file_type=file_record.file_ext.lstrip(".") or None,
        options=options,
        trace_id=f"extraction-rerun-{document.document_id}-{datetime.now(UTC).isoformat()}",
    )


def extraction_rerun_response_payload(
    *,
    document: DocumentInventoryItem,
    file_record: FileMetadataRecord,
    request: ExtractionRuntimeRequest,
    persisted: PersistedExtractionRuntimeResult,
) -> dict[str, object]:
    return {
        "document": document_inventory_item_payload(document),
        "source_file": {
            "file_id": file_record.file_id,
            "original_file_name": file_record.original_file_name,
            "file_ext": file_record.file_ext,
            "mime_type": file_record.mime_type,
            "storage_path": file_record.storage_path,
        },
        "extraction_request": {
            "extraction_profile_name": request.extraction_profile_name,
            "provider_mode": "local",
            "detected_file_type": request.detected_file_type,
            "mime_type": request.mime_type,
            "trace_id": request.trace_id,
            "options": request.options,
        },
        "run": extraction_run_payload(persisted.run),
        "artifacts": [extraction_artifact_payload(artifact) for artifact in persisted.artifacts],
        "blocks": [document_block_payload(block) for block in persisted.blocks],
        "artifact_count": len(persisted.artifacts),
        "block_count": len(persisted.blocks),
    }


def extraction_rerun_feedback_payload(
    *,
    run_id: int | None = None,
    status_value: str | None = None,
    artifact_count: int | None = None,
    block_count: int | None = None,
    error_message: str | None = None,
) -> dict[str, object] | None:
    if error_message:
        return {
            "ok": False,
            "status": "failed",
            "run_id": run_id,
            "artifact_count": artifact_count or 0,
            "block_count": block_count or 0,
            "error_message": error_message,
        }
    normalized_status = (status_value or "").strip()
    if run_id is None or not normalized_status:
        return None
    return {
        "ok": normalized_status == "succeeded",
        "status": normalized_status,
        "run_id": run_id,
        "artifact_count": artifact_count or 0,
        "block_count": block_count or 0,
        "error_message": None,
    }


def document_artifacts_redirect_url(
    document_id: int,
    params: dict[str, object | None],
) -> str:
    query = urlencode(
        [
            (key, str(value))
            for key, value in params.items()
            if value is not None and str(value) != ""
        ]
    )
    suffix = f"?{query}" if query else ""
    return f"/documents/{document_id}/artifacts{suffix}"


def extraction_artifact_payload(artifact: ExtractionArtifactRecord) -> dict[str, object]:
    content_text = artifact.content_text or ""
    return {
        "artifact_id": artifact.artifact_id,
        "extraction_run_id": artifact.extraction_run_id,
        "file_id": artifact.file_id,
        "document_id": artifact.document_id,
        "artifact_type": artifact.artifact_type,
        "content_preview": _text_preview(artifact.content_text),
        "content_length": len(content_text) if artifact.content_text is not None else None,
        "storage_path": artifact.storage_path,
        "content_hash": artifact.content_hash,
        "size_bytes": artifact.size_bytes,
        "language": artifact.language,
        "metadata": artifact.metadata,
        "created_at": _datetime_response(artifact.created_at),
    }


def extraction_artifact_preview_payload(
    artifact: ExtractionArtifactRecord,
) -> dict[str, object]:
    content_text = artifact.content_text
    return {
        **extraction_artifact_payload(artifact),
        "content_text": content_text,
        "content_lines": len(content_text.splitlines()) if content_text is not None else None,
    }


def document_block_payload(block: DocumentBlockRecord) -> dict[str, object]:
    return {
        "block_id": block.block_id,
        "artifact_id": block.artifact_id,
        "document_id": block.document_id,
        "parent_block_id": block.parent_block_id,
        "block_seq": block.block_seq,
        "block_type": block.block_type,
        "content_preview": _text_preview(block.content_text, limit=300),
        "content_markdown_preview": _text_preview(block.content_markdown, limit=300),
        "heading_path": list(block.heading_path),
        "source_anchor": block.source_anchor,
        "page_no": block.page_no,
        "slide_no": block.slide_no,
        "sheet_name": block.sheet_name,
        "cell_range": block.cell_range,
        "char_start": block.char_start,
        "char_end": block.char_end,
        "token_count": block.token_count,
        "metadata": block.metadata,
        "created_at": _datetime_response(block.created_at),
    }


def document_block_summary_payload(
    blocks: list[DocumentBlockRecord],
) -> dict[str, object]:
    block_type_counts: dict[str, int] = {}
    source_anchor_count = 0
    page_numbers: set[int] = set()
    slide_numbers: set[int] = set()
    sheet_names: set[str] = set()

    for block in blocks:
        block_type_counts[block.block_type] = block_type_counts.get(block.block_type, 0) + 1
        if block.source_anchor:
            source_anchor_count += 1
        if block.page_no is not None:
            page_numbers.add(block.page_no)
        if block.slide_no is not None:
            slide_numbers.add(block.slide_no)
        if block.sheet_name:
            sheet_names.add(block.sheet_name)

    return {
        "block_count": len(blocks),
        "block_type_counts": dict(sorted(block_type_counts.items())),
        "source_anchor_count": source_anchor_count,
        "page_count": len(page_numbers),
        "slide_count": len(slide_numbers),
        "sheet_count": len(sheet_names),
        "sheet_names": sorted(sheet_names),
    }


EXTRACTION_QUALITY_MIN_TEXT_LENGTH = 80
EXTRACTION_QUALITY_MIN_SOURCE_ANCHOR_COVERAGE_PERCENT = 80.0


def _extraction_quality_issue(
    *,
    code: str,
    severity: str,
    message: str,
    metric: object | None = None,
) -> dict[str, object]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "metric": metric,
    }


def extraction_quality_check_payload(
    artifact: ExtractionArtifactRecord | None,
    blocks: list[DocumentBlockRecord],
    extraction_runs: list[ExtractionRunRecord],
) -> dict[str, object]:
    block_summary = document_block_summary_payload(blocks)
    block_count = int(block_summary["block_count"])
    source_anchor_count = int(block_summary["source_anchor_count"])
    source_anchor_coverage_percent = (
        round(source_anchor_count / block_count * 100, 2) if block_count else None
    )
    block_type_counts = block_summary["block_type_counts"]
    issues: list[dict[str, object]] = []

    if artifact is None:
        issues.append(
            _extraction_quality_issue(
                code="no_artifact_selected",
                severity="info",
                message="No extraction artifact is selected.",
            )
        )
        return {
            "status": "not_available",
            "content_length": None,
            "content_lines": None,
            "block_count": block_count,
            "source_anchor_count": source_anchor_count,
            "source_anchor_coverage_percent": source_anchor_coverage_percent,
            "issue_count": len(issues),
            "warning_count": 0,
            "failed_count": 0,
            "issues": issues,
        }

    content_text = artifact.content_text or ""
    stripped_text = content_text.strip()
    content_length = len(content_text) if artifact.content_text is not None else None
    content_lines = len(content_text.splitlines()) if artifact.content_text is not None else None

    if not stripped_text:
        issues.append(
            _extraction_quality_issue(
                code="missing_content_text",
                severity="failed",
                message="The selected artifact has no extracted text.",
                metric=content_length,
            )
        )
    elif len(stripped_text) < EXTRACTION_QUALITY_MIN_TEXT_LENGTH:
        issues.append(
            _extraction_quality_issue(
                code="short_content_text",
                severity="warning",
                message=(
                    "The selected artifact text is shorter than the baseline review threshold."
                ),
                metric=len(stripped_text),
            )
        )

    if block_count == 0:
        issues.append(
            _extraction_quality_issue(
                code="missing_blocks",
                severity="failed",
                message="No document blocks were created from the selected artifact.",
                metric=block_count,
            )
        )
    else:
        if source_anchor_count == 0:
            issues.append(
                _extraction_quality_issue(
                    code="missing_source_anchors",
                    severity="warning",
                    message="Document blocks have no source anchors.",
                    metric=source_anchor_count,
                )
            )
        elif (
            source_anchor_coverage_percent is not None
            and source_anchor_coverage_percent
            < EXTRACTION_QUALITY_MIN_SOURCE_ANCHOR_COVERAGE_PERCENT
        ):
            issues.append(
                _extraction_quality_issue(
                    code="low_source_anchor_coverage",
                    severity="warning",
                    message="Source anchor coverage is below the baseline review threshold.",
                    metric=source_anchor_coverage_percent,
                )
            )
        if "heading" not in block_type_counts:
            issues.append(
                _extraction_quality_issue(
                    code="missing_heading_blocks",
                    severity="warning",
                    message="No heading blocks were detected in the selected artifact.",
                )
            )

    selected_run = next(
        (run for run in extraction_runs if run.extraction_run_id == artifact.extraction_run_id),
        None,
    )
    if selected_run is not None:
        if selected_run.status == "failed" or selected_run.error_count > 0:
            issues.append(
                _extraction_quality_issue(
                    code="extraction_run_errors",
                    severity="failed",
                    message="The extraction run has errors.",
                    metric=selected_run.error_count,
                )
            )
        elif selected_run.warning_count > 0:
            issues.append(
                _extraction_quality_issue(
                    code="extraction_run_warnings",
                    severity="warning",
                    message="The extraction run completed with warnings.",
                    metric=selected_run.warning_count,
                )
            )

    warning_count = sum(1 for issue in issues if issue["severity"] == "warning")
    failed_count = sum(1 for issue in issues if issue["severity"] == "failed")
    status_value = "failed" if failed_count else "warning" if warning_count else "passed"

    return {
        "status": status_value,
        "content_length": content_length,
        "content_lines": content_lines,
        "block_count": block_count,
        "source_anchor_count": source_anchor_count,
        "source_anchor_coverage_percent": source_anchor_coverage_percent,
        "issue_count": len(issues),
        "warning_count": warning_count,
        "failed_count": failed_count,
        "issues": issues,
    }


def extraction_quality_snapshot_payload(
    snapshot: ExtractionQualitySnapshotRecord,
) -> dict[str, object]:
    return {
        "snapshot_id": snapshot.snapshot_id,
        "document_id": snapshot.document_id,
        "file_id": snapshot.file_id,
        "artifact_id": snapshot.artifact_id,
        "extraction_run_id": snapshot.extraction_run_id,
        "artifact_type": snapshot.artifact_type,
        "extraction_profile_name": snapshot.extraction_profile_name,
        "extractor_name": snapshot.extractor_name,
        "extractor_version": snapshot.extractor_version,
        "status": snapshot.status,
        "content_length": snapshot.content_length,
        "content_lines": snapshot.content_lines,
        "block_count": snapshot.block_count,
        "source_anchor_count": snapshot.source_anchor_count,
        "source_anchor_coverage_percent": snapshot.source_anchor_coverage_percent,
        "source_anchor_coverage_label": _percent_label(
            snapshot.source_anchor_coverage_percent,
        ),
        "issue_count": snapshot.issue_count,
        "warning_count": snapshot.warning_count,
        "failed_count": snapshot.failed_count,
        "block_summary": snapshot.block_summary,
        "quality_payload": snapshot.quality_payload,
        "created_by": snapshot.created_by,
        "created_by_user_id": snapshot.created_by_user_id,
        "created_at": _datetime_response(snapshot.created_at),
        "created_at_label": _datetime_label(snapshot.created_at),
    }


def extraction_quality_snapshot_summary_payload(
    summary: ExtractionQualitySnapshotSummary,
) -> dict[str, object]:
    return {
        "document_id": summary.document_id,
        "artifact_id": summary.artifact_id,
        "snapshot_count": summary.snapshot_count,
        "passed_count": summary.passed_count,
        "warning_count": summary.warning_count,
        "failed_count": summary.failed_count,
        "latest_snapshot": (
            extraction_quality_snapshot_payload(summary.latest_snapshot)
            if summary.latest_snapshot is not None
            else None
        ),
    }


def extraction_quality_snapshot_input_from_context(
    *,
    document: DocumentInventoryItem,
    artifact: ExtractionArtifactRecord,
    blocks: list[DocumentBlockRecord],
    extraction_runs: list[ExtractionRunRecord],
    created_by: str | None,
    created_by_user_id: int | None,
) -> ExtractionQualitySnapshotInput:
    block_summary = document_block_summary_payload(blocks)
    quality_check = extraction_quality_check_payload(artifact, blocks, extraction_runs)
    selected_run = next(
        (run for run in extraction_runs if run.extraction_run_id == artifact.extraction_run_id),
        None,
    )
    return ExtractionQualitySnapshotInput(
        document_id=document.document_id,
        file_id=artifact.file_id,
        artifact_id=artifact.artifact_id,
        extraction_run_id=artifact.extraction_run_id,
        artifact_type=artifact.artifact_type,
        extraction_profile_name=(
            selected_run.extraction_profile_name if selected_run is not None else None
        ),
        extractor_name=selected_run.extractor_name if selected_run is not None else None,
        extractor_version=selected_run.extractor_version if selected_run is not None else None,
        status=str(quality_check["status"]),
        content_length=(
            int(quality_check["content_length"])
            if quality_check["content_length"] is not None
            else None
        ),
        content_lines=(
            int(quality_check["content_lines"])
            if quality_check["content_lines"] is not None
            else None
        ),
        block_count=int(quality_check["block_count"]),
        source_anchor_count=int(quality_check["source_anchor_count"]),
        source_anchor_coverage_percent=quality_check["source_anchor_coverage_percent"],
        issue_count=int(quality_check["issue_count"]),
        warning_count=int(quality_check["warning_count"]),
        failed_count=int(quality_check["failed_count"]),
        block_summary=block_summary,
        quality_payload=quality_check,
        created_by=created_by,
        created_by_user_id=created_by_user_id,
    )


EXTRACTION_ARTIFACT_EXPORT_FORMATS = {
    "markdown",
    "blocks_json",
    "metadata_json",
    "quality_json",
    "bundle_json",
}


def _extraction_artifact_export_filename(
    *,
    document_id: int,
    artifact_id: int,
    export_format: str,
) -> str:
    extension = "md" if export_format == "markdown" else "json"
    return f"document-{document_id}-artifact-{artifact_id}-{export_format}.{extension}"


def _attachment_headers(filename: str) -> dict[str, str]:
    return {"Content-Disposition": f'attachment; filename="{filename}"'}


def extraction_artifact_export_payload(
    *,
    document: DocumentInventoryItem,
    artifact: ExtractionArtifactRecord,
    blocks: list[DocumentBlockRecord],
    extraction_runs: list[ExtractionRunRecord],
    export_format: str,
) -> dict[str, object]:
    artifact_payload = extraction_artifact_preview_payload(artifact)
    block_payloads = [document_block_payload(block) for block in blocks]
    block_summary = document_block_summary_payload(blocks)
    quality_check = extraction_quality_check_payload(artifact, blocks, extraction_runs)

    if export_format == "blocks_json":
        return {
            "document": document_inventory_item_payload(document),
            "selected_artifact_id": artifact.artifact_id,
            "selected_artifact": extraction_artifact_payload(artifact),
            "block_summary": block_summary,
            "blocks": block_payloads,
        }
    if export_format == "metadata_json":
        return {
            "document": document_inventory_item_payload(document),
            "selected_artifact_id": artifact.artifact_id,
            "selected_artifact": extraction_artifact_payload(artifact),
            "metadata": artifact.metadata,
        }
    if export_format == "quality_json":
        return {
            "document": document_inventory_item_payload(document),
            "selected_artifact_id": artifact.artifact_id,
            "selected_artifact": extraction_artifact_payload(artifact),
            "block_summary": block_summary,
            "quality_check": quality_check,
        }
    if export_format == "bundle_json":
        return {
            "document": document_inventory_item_payload(document),
            "extraction_runs": [extraction_run_payload(run) for run in extraction_runs],
            "selected_artifact_id": artifact.artifact_id,
            "selected_artifact": artifact_payload,
            "block_summary": block_summary,
            "quality_check": quality_check,
            "blocks": block_payloads,
        }
    raise ValueError(f"Unsupported extraction artifact export format: {export_format}")


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
        "readiness_percent_label": _percent_label(
            summary.readiness_percent * 100 if summary.readiness_percent is not None else None
        ),
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

    @app.get("/api/dashboard/core-metrics")
    def api_get_dashboard_core_metrics() -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        metrics = get_dashboard_core_metrics(settings.database_url)
        return JSONResponse(content={"core_metrics": dashboard_core_metrics_payload(metrics)})

    @app.get("/api/dashboard/operational-health")
    def api_get_dashboard_operational_health() -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        pipeline_queue = get_pipeline_queue_summary(settings.database_url)
        embedding_backlog = get_embedding_job_backlog_summary(settings.database_url)
        recent_failures = get_dashboard_recent_failures(settings.database_url, limit=10)
        threshold_settings = load_dashboard_health_threshold_settings(settings.database_url)
        health = summarize_dashboard_operational_health(
            pipeline_queue=pipeline_queue,
            embedding_backlog=embedding_backlog,
            recent_failures=recent_failures,
            thresholds=threshold_settings.thresholds,
        )
        return JSONResponse(
            content={
                "operational_health": dashboard_operational_health_payload(
                    health,
                    threshold_settings=threshold_settings,
                )
            }
        )

    @app.get("/api/dashboard/health-thresholds")
    def api_get_dashboard_health_thresholds() -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        threshold_settings = load_dashboard_health_threshold_settings(settings.database_url)
        return JSONResponse(
            content={"settings": dashboard_health_threshold_settings_payload(threshold_settings)}
        )

    @app.put("/api/dashboard/health-thresholds")
    def api_update_dashboard_health_thresholds(
        payload: DashboardHealthThresholdSettingsRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            threshold_settings = update_dashboard_health_threshold_settings(
                settings.database_url,
                DashboardHealthThresholdSettingsInput(thresholds=dict(payload.thresholds)),
            )
        except InvalidDashboardHealthThresholdSettingsError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={"settings": dashboard_health_threshold_settings_payload(threshold_settings)}
        )

    @app.post("/api/dashboard/health-thresholds/reset")
    def api_reset_dashboard_health_thresholds() -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        threshold_settings = reset_dashboard_health_threshold_settings(settings.database_url)
        return JSONResponse(
            content={"settings": dashboard_health_threshold_settings_payload(threshold_settings)}
        )

    @app.get("/admin/dashboard-settings", response_class=HTMLResponse)
    def dashboard_settings_page(request: Request) -> HTMLResponse:
        error_message = None
        threshold_settings = DashboardHealthThresholdSettings(
            thresholds=dict(DEFAULT_DASHBOARD_HEALTH_THRESHOLDS)
        )
        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                threshold_settings = load_dashboard_health_threshold_settings(settings.database_url)
            except Exception as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "dashboard_settings.html",
            template_context(
                request,
                database_configured=bool(settings.database_url),
                error_message=error_message,
                threshold_settings=dashboard_health_threshold_settings_payload(threshold_settings),
                threshold_rows=dashboard_health_threshold_ui_rows(threshold_settings),
            ),
        )

    @app.get("/api/dashboard/export")
    def api_export_dashboard_snapshot(
        format: str = "json",
        lookback_hours: int = DEFAULT_DASHBOARD_THROUGHPUT_LOOKBACK_HOURS,
    ) -> Response:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        normalized_format = format.strip().lower()
        if normalized_format not in {"csv", "json"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="format must be json or csv.",
            )

        try:
            snapshot = dashboard_snapshot_export_payload(
                settings.database_url,
                lookback_hours=lookback_hours,
            )
        except InvalidDashboardThroughputError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        filename_base = f"dashboard-snapshot-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        if normalized_format == "csv":
            return Response(
                content=dashboard_snapshot_summary_csv(snapshot),
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": (f'attachment; filename="{filename_base}.csv"')},
            )

        return JSONResponse(
            content=snapshot,
            headers={"Content-Disposition": f'attachment; filename="{filename_base}.json"'},
        )

    @app.get("/api/dashboard/pipeline-queue")
    def api_get_dashboard_pipeline_queue() -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        summary = get_pipeline_queue_summary(settings.database_url)
        return JSONResponse(content={"pipeline_queue": pipeline_queue_summary_payload(summary)})

    @app.get("/api/dashboard/throughput-latency")
    def api_get_dashboard_throughput_latency(lookback_hours: int = 24) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            snapshot = get_dashboard_throughput_latency_snapshot(
                settings.database_url,
                lookback_hours=lookback_hours,
            )
        except InvalidDashboardThroughputError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={"throughput_latency": dashboard_throughput_latency_snapshot_payload(snapshot)}
        )

    @app.get("/api/dashboard/embedding-backlog")
    def api_get_dashboard_embedding_backlog() -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        summary = get_embedding_job_backlog_summary(settings.database_url)
        return JSONResponse(content={"backlog": embedding_job_backlog_summary_payload(summary)})

    @app.get("/api/dashboard/recent-failures")
    def api_get_dashboard_recent_failures(limit: int = 10) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            summary = get_dashboard_recent_failures(settings.database_url, limit=limit)
        except InvalidDashboardFailureError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content={"recent_failures": dashboard_failure_summary_payload(summary)})

    @app.get("/api/dashboard/recent-failures/{source}/{reference_id}")
    def api_get_dashboard_failure_detail(source: str, reference_id: str) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            detail = get_dashboard_failure_detail(
                settings.database_url,
                source=source,
                reference_id=reference_id,
            )
        except InvalidDashboardFailureError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if detail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dashboard failure detail not found.",
            )

        return JSONResponse(content={"failure_detail": dashboard_failure_detail_payload(detail)})

    @app.get("/api/admin/go-live-readiness")
    def api_get_go_live_readiness() -> JSONResponse:
        report = build_go_live_readiness_report(settings)
        return JSONResponse(content={"go_live_readiness": go_live_readiness_report_payload(report)})

    @app.get("/api/admin/foreground-worker-runtime")
    def api_get_foreground_worker_runtime() -> JSONResponse:
        report = build_foreground_worker_runtime_report(BASE_DIR.parent)
        return JSONResponse(
            content={"foreground_worker_runtime": foreground_worker_runtime_report_payload(report)}
        )

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

    @app.get("/api/documents/{document_id}/ingestion-artifacts")
    def api_get_document_ingestion_artifacts(
        document_id: int,
        artifact_id: int | None = None,
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
            extraction_runs = list_document_extraction_runs(settings.database_url, document_id)
            extraction_artifacts = list_document_extraction_artifacts(
                settings.database_url,
                document_id,
            )
            artifact_ids = {artifact.artifact_id for artifact in extraction_artifacts}
            selected_artifact_id = (
                artifact_id
                if artifact_id is not None
                else (extraction_artifacts[0].artifact_id if extraction_artifacts else None)
            )
            if selected_artifact_id is not None and selected_artifact_id not in artifact_ids:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Extraction artifact not found for document.",
                )
            document_blocks = (
                list_document_blocks(
                    settings.database_url,
                    document_id,
                    artifact_id=selected_artifact_id,
                )
                if selected_artifact_id is not None
                else []
            )
        except InvalidDocumentInventoryError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except InvalidIngestionArtifactError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "document": document_inventory_item_payload(document),
                "extraction_runs": [extraction_run_payload(run) for run in extraction_runs],
                "artifacts": [
                    extraction_artifact_payload(artifact) for artifact in extraction_artifacts
                ],
                "selected_artifact_id": selected_artifact_id,
                "blocks": [document_block_payload(block) for block in document_blocks],
            },
        )

    @app.get("/api/documents/{document_id}/extraction-preview")
    def api_get_document_extraction_preview(
        document_id: int,
        artifact_id: int | None = None,
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
            extraction_runs = list_document_extraction_runs(settings.database_url, document_id)
            extraction_artifacts = list_document_extraction_artifacts(
                settings.database_url,
                document_id,
            )
            selected_artifact_id = (
                artifact_id
                if artifact_id is not None
                else (extraction_artifacts[0].artifact_id if extraction_artifacts else None)
            )
            selected_artifact = (
                next(
                    (
                        artifact
                        for artifact in extraction_artifacts
                        if artifact.artifact_id == selected_artifact_id
                    ),
                    None,
                )
                if selected_artifact_id is not None
                else None
            )
            if selected_artifact_id is not None and selected_artifact is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Extraction artifact not found for document.",
                )
            document_blocks = (
                list_document_blocks(
                    settings.database_url,
                    document_id,
                    artifact_id=selected_artifact_id,
                )
                if selected_artifact_id is not None
                else []
            )
        except InvalidDocumentInventoryError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except InvalidIngestionArtifactError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "document": document_inventory_item_payload(document),
                "extraction_runs": [extraction_run_payload(run) for run in extraction_runs],
                "artifacts": [
                    extraction_artifact_payload(artifact) for artifact in extraction_artifacts
                ],
                "selected_artifact_id": selected_artifact_id,
                "selected_artifact": (
                    extraction_artifact_preview_payload(selected_artifact)
                    if selected_artifact is not None
                    else None
                ),
                "blocks": [document_block_payload(block) for block in document_blocks],
                "block_summary": document_block_summary_payload(document_blocks),
            },
        )

    @app.get("/api/documents/{document_id}/chunk-source-trace")
    def api_get_document_chunk_source_trace(
        document_id: int,
        artifact_id: int | None = None,
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
            extraction_artifacts = list_document_extraction_artifacts(
                settings.database_url,
                document_id,
            )
            selected_artifact_id = (
                artifact_id
                if artifact_id is not None
                else (extraction_artifacts[0].artifact_id if extraction_artifacts else None)
            )
            selected_artifact = (
                next(
                    (
                        artifact
                        for artifact in extraction_artifacts
                        if artifact.artifact_id == selected_artifact_id
                    ),
                    None,
                )
                if selected_artifact_id is not None
                else None
            )
            if selected_artifact_id is not None and selected_artifact is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Extraction artifact not found for document.",
                )
            document_blocks = (
                list_document_blocks(
                    settings.database_url,
                    document_id,
                    artifact_id=selected_artifact_id,
                )
                if selected_artifact_id is not None
                else []
            )
            document_chunks = (
                list_document_chunks(
                    settings.database_url,
                    document_id,
                    chunk_policy_name=chunk_policy_name,
                )
                if selected_artifact_id is not None
                else []
            )
            trace_chunks = (
                _chunks_for_source_trace(
                    document_chunks,
                    artifact_id=selected_artifact_id,
                    block_ids={block.block_id for block in document_blocks},
                )
                if selected_artifact_id is not None
                else []
            )
        except InvalidDocumentInventoryError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except InvalidIngestionArtifactError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except InvalidChunkError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "document": document_inventory_item_payload(document),
                "trace": chunk_source_trace_preview_payload(
                    selected_artifact,
                    document_blocks,
                    trace_chunks,
                ),
            },
        )

    @app.get("/api/documents/{document_id}/extraction-quality")
    def api_get_document_extraction_quality(
        document_id: int,
        artifact_id: int | None = None,
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
            extraction_runs = list_document_extraction_runs(settings.database_url, document_id)
            extraction_artifacts = list_document_extraction_artifacts(
                settings.database_url,
                document_id,
            )
            selected_artifact_id = (
                artifact_id
                if artifact_id is not None
                else (extraction_artifacts[0].artifact_id if extraction_artifacts else None)
            )
            selected_artifact = (
                next(
                    (
                        artifact
                        for artifact in extraction_artifacts
                        if artifact.artifact_id == selected_artifact_id
                    ),
                    None,
                )
                if selected_artifact_id is not None
                else None
            )
            if selected_artifact_id is not None and selected_artifact is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Extraction artifact not found for document.",
                )
            document_blocks = (
                list_document_blocks(
                    settings.database_url,
                    document_id,
                    artifact_id=selected_artifact_id,
                )
                if selected_artifact_id is not None
                else []
            )
        except InvalidDocumentInventoryError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except InvalidIngestionArtifactError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "document": document_inventory_item_payload(document),
                "selected_artifact_id": selected_artifact_id,
                "selected_artifact": (
                    extraction_artifact_payload(selected_artifact)
                    if selected_artifact is not None
                    else None
                ),
                "block_summary": document_block_summary_payload(document_blocks),
                "quality_check": extraction_quality_check_payload(
                    selected_artifact,
                    document_blocks,
                    extraction_runs,
                ),
            },
        )

    @app.get("/api/documents/{document_id}/extraction-quality-snapshots")
    def api_list_document_extraction_quality_snapshots(
        document_id: int,
        artifact_id: int | None = None,
        limit: int = Query(default=20, ge=1, le=100),
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
            snapshots = list_extraction_quality_snapshots(
                settings.database_url,
                document_id,
                artifact_id=artifact_id,
                limit=limit,
            )
        except InvalidDocumentInventoryError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except InvalidIngestionArtifactError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "document": document_inventory_item_payload(document),
                "snapshot_count": len(snapshots),
                "snapshots": [
                    extraction_quality_snapshot_payload(snapshot) for snapshot in snapshots
                ],
            },
        )

    @app.get("/api/documents/{document_id}/extraction-quality-summary")
    def api_get_document_extraction_quality_snapshot_summary(
        document_id: int,
        artifact_id: int | None = None,
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
            summary = get_extraction_quality_snapshot_summary(
                settings.database_url,
                document_id,
                artifact_id=artifact_id,
            )
        except InvalidDocumentInventoryError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except InvalidIngestionArtifactError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "document": document_inventory_item_payload(document),
                "summary": extraction_quality_snapshot_summary_payload(summary),
            },
        )

    @app.post(
        "/api/documents/{document_id}/extraction-quality-snapshots",
        status_code=status.HTTP_201_CREATED,
    )
    def api_create_document_extraction_quality_snapshot(
        document_id: int,
        payload: ExtractionQualitySnapshotCreateRequest,
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
            extraction_runs = list_document_extraction_runs(settings.database_url, document_id)
            extraction_artifacts = list_document_extraction_artifacts(
                settings.database_url,
                document_id,
            )
            selected_artifact_id = (
                payload.artifact_id
                if payload.artifact_id is not None
                else (extraction_artifacts[0].artifact_id if extraction_artifacts else None)
            )
            selected_artifact = (
                next(
                    (
                        artifact
                        for artifact in extraction_artifacts
                        if artifact.artifact_id == selected_artifact_id
                    ),
                    None,
                )
                if selected_artifact_id is not None
                else None
            )
            if selected_artifact_id is None or selected_artifact is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Extraction artifact not found for document.",
                )
            document_blocks = list_document_blocks(
                settings.database_url,
                document_id,
                artifact_id=selected_artifact_id,
            )
            snapshot = create_extraction_quality_snapshot(
                settings.database_url,
                extraction_quality_snapshot_input_from_context(
                    document=document,
                    artifact=selected_artifact,
                    blocks=document_blocks,
                    extraction_runs=extraction_runs,
                    created_by=payload.created_by,
                    created_by_user_id=payload.created_by_user_id,
                ),
            )
        except InvalidDocumentInventoryError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except InvalidIngestionArtifactError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "document": document_inventory_item_payload(document),
                "snapshot": extraction_quality_snapshot_payload(snapshot),
            },
        )

    @app.get("/api/documents/{document_id}/extraction-export")
    def api_export_document_extraction_artifact(
        document_id: int,
        artifact_id: int | None = None,
        export_format: str = Query("markdown", alias="format"),
    ) -> Response:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        normalized_format = export_format.strip().lower()
        if normalized_format not in EXTRACTION_ARTIFACT_EXPORT_FORMATS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "format must be markdown, blocks_json, metadata_json, "
                    "quality_json, or bundle_json."
                ),
            )

        try:
            document = get_document_inventory_item(settings.database_url, document_id)
            if document is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Document not found.",
                )
            extraction_runs = list_document_extraction_runs(settings.database_url, document_id)
            extraction_artifacts = list_document_extraction_artifacts(
                settings.database_url,
                document_id,
            )
            selected_artifact_id = (
                artifact_id
                if artifact_id is not None
                else (extraction_artifacts[0].artifact_id if extraction_artifacts else None)
            )
            selected_artifact = (
                next(
                    (
                        artifact
                        for artifact in extraction_artifacts
                        if artifact.artifact_id == selected_artifact_id
                    ),
                    None,
                )
                if selected_artifact_id is not None
                else None
            )
            if selected_artifact_id is None or selected_artifact is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Extraction artifact not found for document.",
                )
            document_blocks = list_document_blocks(
                settings.database_url,
                document_id,
                artifact_id=selected_artifact_id,
            )
        except InvalidDocumentInventoryError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except InvalidIngestionArtifactError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        filename = _extraction_artifact_export_filename(
            document_id=document_id,
            artifact_id=selected_artifact_id,
            export_format=normalized_format,
        )
        headers = _attachment_headers(filename)
        if normalized_format == "markdown":
            return Response(
                content=selected_artifact.content_text or "",
                media_type="text/markdown; charset=utf-8",
                headers=headers,
            )

        return JSONResponse(
            content=extraction_artifact_export_payload(
                document=document,
                artifact=selected_artifact,
                blocks=document_blocks,
                extraction_runs=extraction_runs,
                export_format=normalized_format,
            ),
            headers=headers,
        )

    @app.post(
        "/api/documents/{document_id}/extraction-rerun",
        status_code=status.HTTP_201_CREATED,
    )
    def api_rerun_document_extraction(
        document_id: int,
        payload: ExtractionRerunRequest,
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
            file_record = get_file_metadata(settings.database_url, document.file_id)
            if file_record is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="File metadata not found for document.",
                )
            extraction_request = build_extraction_rerun_request(
                document=document,
                file_record=file_record,
                payload=payload,
            )
            runtime_result = run_local_extraction(extraction_request)
            persisted = persist_extraction_runtime_result(
                settings.database_url,
                extraction_request,
                runtime_result,
            )
        except InvalidDocumentInventoryError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except (InvalidFileMetadataError, InvalidIngestionArtifactError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=extraction_rerun_response_payload(
                document=document,
                file_record=file_record,
                request=extraction_request,
                persisted=persisted,
            ),
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

    @app.get("/api/admin/reranker-provider/status")
    def api_remote_reranker_provider_operations_status(
        request_smoke: bool = False,
    ) -> JSONResponse:
        operations_status = get_remote_reranker_operations_status(
            settings,
            request_smoke=request_smoke,
        )
        return JSONResponse(
            status_code=operations_status.status_code,
            content=operations_status.payload,
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

    @app.get("/api/admin/embedding-provider-routes/presets")
    def api_list_embedding_provider_route_presets() -> JSONResponse:
        presets = list_embedding_provider_presets()
        return JSONResponse(
            content={
                "preset_count": len(presets),
                "presets": [embedding_provider_preset_payload(preset) for preset in presets],
            }
        )

    @app.post("/api/admin/embedding-provider-routes/presets/launch-plan")
    def api_build_embedding_provider_launch_plan(
        payload: EmbeddingProviderLaunchPlanRequest,
    ) -> JSONResponse:
        try:
            plan = embedding_provider_launch_plan_from_request(payload, settings)
        except InvalidEmbeddingProviderPresetError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse(content={"plan": embedding_provider_launch_plan_payload(plan)})

    @app.post("/api/admin/embedding-provider-routes/presets/register")
    def api_register_embedding_provider_route_preset(
        payload: EmbeddingProviderRoutePresetRegistrationRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            plans = embedding_provider_preset_route_plans_from_request(payload)
            routes = [
                upsert_embedding_provider_route_with_audit(
                    settings.database_url,
                    plan.to_route_input(),
                    request_path="/api/admin/embedding-provider-routes/presets/register",
                )
                for plan in plans
            ]
            preflight = (
                run_registered_embedding_provider_route_preflight(
                    settings.database_url,
                    routes,
                )
                if payload.run_preflight
                else None
            )
        except (
            InvalidEmbeddingProviderPresetError,
            InvalidEmbeddingProviderRouteError,
            InvalidEmbeddingProviderContractSampleSetError,
            InvalidEmbeddingProviderRouteHealthSnapshotError,
            InvalidEmbeddingProviderRouteContractSnapshotError,
            InvalidEmbeddingProviderPreflightRunError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "registered_count": len(routes),
                "plans": [embedding_provider_preset_route_plan_payload(plan) for plan in plans],
                "routes": [embedding_provider_route_payload(route) for route in routes],
                "preflight": preflight,
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

    @app.get("/api/admin/embedding-provider-routes/export")
    def api_export_embedding_provider_routes(
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
        return JSONResponse(content=embedding_provider_route_export_payload(routes))

    @app.post("/api/admin/embedding-provider-routes/import")
    def api_import_embedding_provider_routes(
        payload: EmbeddingProviderRouteImportRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            route_inputs = embedding_provider_route_import_inputs_from_request(payload)
            imported_routes = (
                [
                    upsert_embedding_provider_route_with_audit(
                        settings.database_url,
                        route_input,
                        request_path="/api/admin/embedding-provider-routes/import",
                    )
                    for route_input in route_inputs
                ]
                if not payload.dry_run
                else []
            )
        except InvalidEmbeddingProviderRouteError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse(
            content={
                "dry_run": payload.dry_run,
                "route_count": len(route_inputs),
                "imported_count": len(imported_routes),
                "routes": (
                    [embedding_provider_route_payload(route) for route in imported_routes]
                    if imported_routes
                    else [
                        embedding_provider_route_portable_payload(route_input)
                        for route_input in route_inputs
                    ]
                ),
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

    @app.get("/api/admin/embedding-provider-routes/model-availability")
    def api_embedding_provider_model_availability() -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            matrix = get_provider_model_availability_matrix(
                settings.database_url,
                models_dir=settings.embedding_models_dir,
            )
        except (
            InvalidEmbeddingProviderRouteError,
            InvalidEmbeddingProviderRouteHealthSnapshotError,
            InvalidEmbeddingProviderRouteContractSnapshotError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content=provider_model_availability_matrix_payload(matrix))

    @app.get("/api/admin/embedding-provider-routes/model-availability/{profile_name}")
    def api_embedding_provider_model_availability_drilldown(
        profile_name: str,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            drilldown = get_provider_model_availability_drilldown(
                settings.database_url,
                models_dir=settings.embedding_models_dir,
                profile_name=profile_name,
            )
        except (
            InvalidEmbeddingProviderRouteError,
            InvalidEmbeddingProviderRouteHealthSnapshotError,
            InvalidEmbeddingProviderRouteContractSnapshotError,
            InvalidEmbeddingProviderPreflightRunError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if drilldown is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown provider model availability profile: {profile_name}",
            )

        return JSONResponse(content=provider_model_availability_drilldown_payload(drilldown))

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
            unacknowledged_alert_count = count_provider_route_alert_logs(
                settings.database_url,
                acknowledged=False,
            )
        except (
            InvalidEmbeddingProviderRouteError,
            InvalidEmbeddingProviderRouteHealthSnapshotError,
            InvalidEmbeddingProviderRouteContractSnapshotError,
            InvalidEmbeddingProviderPreflightScheduleError,
            InvalidEmbeddingProviderPreflightRunError,
            InvalidAdminLogError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "operations_summary": embedding_provider_route_operations_summary_payload(
                    readiness=readiness,
                    schedules=schedules,
                    due_schedules=due_schedules,
                    latest_run=latest_runs[0] if latest_runs else None,
                    unacknowledged_alert_count=unacknowledged_alert_count,
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

    @app.get("/api/admin/embedding-provider-routes/change-logs")
    def api_list_embedding_provider_route_change_logs(
        profile_name: str | None = None,
        provider_name: str | None = None,
        route_id: int | None = None,
        limit: int = 20,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            changes = list_provider_route_change_logs(
                settings.database_url,
                profile_name=profile_name,
                provider_name=provider_name,
                route_id=route_id,
                limit=limit,
            )
        except InvalidAdminLogError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "change_count": len(changes),
                "changes": [admin_log_payload(change) for change in changes],
            }
        )

    @app.get("/api/admin/embedding-provider-routes/change-logs/{log_id}")
    def api_get_embedding_provider_route_change_log(log_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            change = get_provider_route_change_log(settings.database_url, log_id)
        except InvalidAdminLogError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if change is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Provider route change log not found.",
            )

        return JSONResponse(content=provider_route_change_diff_payload(change))

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

    @app.get("/api/admin/embedding-provider-routes/preflight-runs/{run_id}")
    def api_get_embedding_provider_preflight_run(run_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            run = get_embedding_provider_preflight_run(settings.database_url, run_id)
        except InvalidEmbeddingProviderPreflightRunError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Provider preflight run not found.",
            )
        return JSONResponse(content={"run": embedding_provider_preflight_run_payload(run)})

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
            route = upsert_embedding_provider_route_with_audit(
                settings.database_url,
                embedding_provider_route_input_from_request(payload),
                request_path="/api/admin/embedding-provider-routes",
            )
        except InvalidEmbeddingProviderRouteError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content={"route": embedding_provider_route_payload(route)})

    @app.put("/api/admin/embedding-provider-routes/{route_id}")
    def api_update_embedding_provider_route(
        route_id: int,
        payload: EmbeddingProviderRouteRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            route = update_embedding_provider_route_with_audit(
                settings.database_url,
                route_id,
                embedding_provider_route_input_from_request(payload),
                request_path=f"/api/admin/embedding-provider-routes/{route_id}",
            )
        except InvalidEmbeddingProviderRouteError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if route is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Embedding provider route not found.",
            )

        return JSONResponse(content={"route": embedding_provider_route_payload(route)})

    @app.patch("/api/admin/embedding-provider-routes/{route_id}/activation")
    def api_update_embedding_provider_route_activation(
        route_id: int,
        payload: EmbeddingProviderRouteActivationRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            route = set_embedding_provider_route_active_with_audit(
                settings.database_url,
                route_id,
                payload.is_active,
                request_path=f"/api/admin/embedding-provider-routes/{route_id}/activation",
            )
        except InvalidEmbeddingProviderRouteError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if route is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Embedding provider route not found.",
            )

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

    @app.get("/api/admin/embedding-jobs/backlog-summary")
    def api_get_embedding_job_backlog_summary() -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        summary = get_embedding_job_backlog_summary(settings.database_url)
        return JSONResponse(content={"backlog": embedding_job_backlog_summary_payload(summary)})

    @app.get("/api/admin/embedding-coverage")
    def api_get_embedding_coverage_matrix(
        parse_status: str | None = None,
        document_group: str | None = None,
        profile_name: str | None = None,
        limit: int = 100,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            matrix = get_embedding_coverage_matrix(
                settings.database_url,
                parse_status=parse_status,
                document_group=document_group,
                profile_name=profile_name,
                limit=limit,
            )
        except InvalidEmbeddingCoverageError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content=embedding_coverage_matrix_payload(matrix))

    @app.get("/api/admin/bm25-index-coverage")
    def api_get_bm25_index_coverage_matrix(
        parse_status: str | None = None,
        document_group: str | None = None,
        chunk_policy_name: str | None = None,
        tokenizer_name: str = DEFAULT_BM25_TOKENIZER_NAME,
        limit: int = 100,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            matrix = get_bm25_index_coverage_matrix(
                settings.database_url,
                parse_status=parse_status,
                document_group=document_group,
                chunk_policy_name=chunk_policy_name,
                tokenizer_name=tokenizer_name,
                limit=limit,
            )
        except InvalidBM25IndexCoverageError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content=bm25_index_coverage_matrix_payload(matrix))

    @app.post("/api/admin/bm25-index-coverage/backfill")
    def api_backfill_bm25_index(
        payload: BM25IndexBackfillRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            report = refresh_bm25_keyword_indexes(
                settings.database_url,
                options=BM25IndexRefreshOptions(
                    chunk_policy_names=tuple(payload.chunk_policy_names or ()),
                    tokenizer_name=payload.tokenizer_name,
                    continue_on_error=payload.continue_on_error,
                ),
            )
        except InvalidBM25KeywordIndexError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content={"backfill": bm25_index_refresh_report_payload(report)})

    @app.get("/api/admin/multi-policy-ingestion-coverage")
    def api_get_multi_policy_ingestion_coverage_matrix(
        parse_status: str | None = None,
        document_group: str | None = None,
        profile_name: str | None = None,
        chunk_policy_name: str | None = None,
        limit: int = 100,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            matrix = get_multi_policy_ingestion_coverage_matrix(
                settings.database_url,
                parse_status=parse_status,
                document_group=document_group,
                profile_name=profile_name,
                chunk_policy_name=chunk_policy_name,
                limit=limit,
            )
        except InvalidEmbeddingCoverageError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content=multi_policy_ingestion_coverage_matrix_payload(matrix))

    @app.get("/api/admin/multi-policy-ingestion-coverage/detail")
    def api_get_multi_policy_ingestion_coverage_detail(
        document_id: int,
        chunk_policy_name: str,
        profile_name: str,
        chunk_limit: int = 100,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            detail = get_multi_policy_ingestion_coverage_detail(
                settings.database_url,
                document_id=document_id,
                chunk_policy_name=chunk_policy_name,
                profile_name=profile_name,
                chunk_limit=chunk_limit,
            )
        except InvalidEmbeddingCoverageError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if detail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Multi-policy ingestion coverage detail not found.",
            )

        return JSONResponse(content=multi_policy_ingestion_coverage_detail_payload(detail))

    @app.post("/api/admin/multi-policy-ingestion-coverage/reconcile-missing-jobs")
    def api_reconcile_multi_policy_missing_embedding_jobs(
        payload: MissingEmbeddingJobReconcileRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            result = reconcile_missing_embedding_jobs_for_document_policy_profile(
                settings.database_url,
                document_id=payload.document_id,
                chunk_policy_name=payload.chunk_policy_name,
                profile_name=payload.profile_name,
                max_jobs=payload.max_jobs,
            )
        except InvalidEmbeddingJobError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content=missing_embedding_job_reconcile_result_payload(result))

    @app.post("/api/admin/multi-policy-ingestion-coverage/retry-failed-jobs")
    def api_retry_multi_policy_failed_embedding_jobs(
        payload: FailedEmbeddingJobRetryRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            result = retry_failed_embedding_jobs_for_document_policy_profile(
                settings.database_url,
                document_id=payload.document_id,
                chunk_policy_name=payload.chunk_policy_name,
                profile_name=payload.profile_name,
                max_jobs=payload.max_jobs,
            )
        except InvalidEmbeddingJobError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content=failed_embedding_job_retry_result_payload(result))

    @app.post("/admin/multi-policy-ingestion-coverage/reconcile-missing-jobs")
    def multi_policy_missing_embedding_jobs_reconcile_action(
        document_id: int = Form(...),
        detail_chunk_policy_name: str = Form(...),
        detail_profile_name: str = Form(...),
        parse_status: str = Form(""),
        document_group: str = Form(""),
        profile_name: str = Form(""),
        chunk_policy_name: str = Form(""),
        limit: int = Form(100),
        lang: str = Form(""),
    ) -> RedirectResponse:
        redirect_params: dict[str, object] = {
            "detail_document_id": document_id,
            "detail_chunk_policy_name": detail_chunk_policy_name,
            "detail_profile_name": detail_profile_name,
            "limit": limit,
        }
        for key, value in {
            "parse_status": parse_status,
            "document_group": document_group,
            "profile_name": profile_name,
            "chunk_policy_name": chunk_policy_name,
            "lang": lang,
        }.items():
            if value:
                redirect_params[key] = value

        if not settings.database_url:
            redirect_params["reconcile_error"] = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                result = reconcile_missing_embedding_jobs_for_document_policy_profile(
                    settings.database_url,
                    document_id=document_id,
                    chunk_policy_name=detail_chunk_policy_name,
                    profile_name=detail_profile_name,
                )
                redirect_params["reconcile_created"] = result.created_job_count
                redirect_params["reconcile_missing"] = result.missing_job_count
            except InvalidEmbeddingJobError as exc:
                redirect_params["reconcile_error"] = str(exc)

        return RedirectResponse(
            url=(
                "/admin/multi-policy-ingestion-coverage?"
                f"{urlencode(redirect_params)}#coverage-detail"
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.post("/admin/multi-policy-ingestion-coverage/retry-failed-jobs")
    def multi_policy_failed_embedding_jobs_retry_action(
        document_id: int = Form(...),
        detail_chunk_policy_name: str = Form(...),
        detail_profile_name: str = Form(...),
        parse_status: str = Form(""),
        document_group: str = Form(""),
        profile_name: str = Form(""),
        chunk_policy_name: str = Form(""),
        limit: int = Form(100),
        lang: str = Form(""),
    ) -> RedirectResponse:
        redirect_params: dict[str, object] = {
            "detail_document_id": document_id,
            "detail_chunk_policy_name": detail_chunk_policy_name,
            "detail_profile_name": detail_profile_name,
            "limit": limit,
        }
        for key, value in {
            "parse_status": parse_status,
            "document_group": document_group,
            "profile_name": profile_name,
            "chunk_policy_name": chunk_policy_name,
            "lang": lang,
        }.items():
            if value:
                redirect_params[key] = value

        if not settings.database_url:
            redirect_params["retry_error"] = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                result = retry_failed_embedding_jobs_for_document_policy_profile(
                    settings.database_url,
                    document_id=document_id,
                    chunk_policy_name=detail_chunk_policy_name,
                    profile_name=detail_profile_name,
                )
                redirect_params["retry_retried"] = result.retried_job_count
                redirect_params["retry_failed"] = result.retryable_failed_job_count
            except InvalidEmbeddingJobError as exc:
                redirect_params["retry_error"] = str(exc)

        return RedirectResponse(
            url=(
                "/admin/multi-policy-ingestion-coverage?"
                f"{urlencode(redirect_params)}#coverage-detail"
            ),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.get("/api/admin/embedding-jobs/stale-leases")
    def api_list_stale_embedding_job_leases(
        profile_name: str | None = None,
        reclaimable_only: bool = False,
        limit: int = 100,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            jobs = list_stale_embedding_jobs(
                settings.database_url,
                profile_name=profile_name,
                reclaimable_only=reclaimable_only,
                limit=limit,
            )
        except InvalidEmbeddingJobError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "stale_job_count": len(jobs),
                "jobs": [embedding_job_payload(job) for job in jobs],
            },
        )

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

    @app.post("/api/admin/embedding-jobs/retry-failed")
    def api_retry_failed_embedding_jobs(
        payload: EmbeddingFailedJobBulkRetryRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            retry_result = retry_failed_embedding_jobs_for_scope(
                settings.database_url,
                profile_name=payload.profile_name,
                limit=payload.limit,
            )
        except InvalidEmbeddingJobError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content=retry_result)

    @app.post("/api/admin/embedding-jobs/{job_id}/release-stale-lease")
    def api_release_stale_embedding_job_lease(job_id: int) -> JSONResponse:
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
            released = release_stale_embedding_job_lease(settings.database_url, job_id)
        except InvalidEmbeddingJobError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        if released is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Embedding job does not have a reclaimable stale lease.",
            )
        return JSONResponse(content={"job": embedding_job_payload(released)})

    @app.get("/api/admin/embedding-batch-runs")
    def api_list_embedding_batch_runs(
        worker_name: str | None = None,
        profile_name: str | None = None,
        stopped_reason: str | None = None,
        limit: int = 50,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            batch_runs = list_embedding_worker_batch_runs(
                settings.database_url,
                worker_name=worker_name,
                profile_name=profile_name,
                stopped_reason=stopped_reason,
                limit=limit,
            )
        except InvalidEmbeddingWorkerBatchRunError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "batch_run_count": len(batch_runs),
                "summary": embedding_worker_batch_run_summary(batch_runs),
                "batch_runs": [
                    embedding_worker_batch_run_payload(batch_run) for batch_run in batch_runs
                ],
            }
        )

    @app.get("/api/admin/embedding-batch-runs/throughput-summary")
    def api_get_embedding_batch_run_throughput_summary(
        worker_name: str | None = None,
        profile_name: str | None = None,
        stopped_reason: str | None = None,
        limit: int = 50,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            batch_runs = list_embedding_worker_batch_runs(
                settings.database_url,
                worker_name=worker_name,
                profile_name=profile_name,
                stopped_reason=stopped_reason,
                limit=limit,
            )
        except InvalidEmbeddingWorkerBatchRunError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "batch_run_count": len(batch_runs),
                "throughput": embedding_worker_batch_run_throughput_summary(batch_runs),
            }
        )

    @app.get("/api/admin/embedding-batch-runs/retention-settings")
    def api_get_embedding_batch_run_retention_settings() -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        retention_settings = load_embedding_batch_run_retention_settings(settings.database_url)
        return JSONResponse(
            content={
                "settings": embedding_batch_run_retention_settings_payload(retention_settings),
            }
        )

    @app.put("/api/admin/embedding-batch-runs/retention-settings")
    def api_update_embedding_batch_run_retention_settings(
        payload: EmbeddingBatchRunRetentionSettingsRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            retention_settings = update_embedding_batch_run_retention_settings(
                settings.database_url,
                embedding_batch_run_retention_settings_input_from_request(payload),
            )
        except InvalidEmbeddingBatchRunRetentionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "settings": embedding_batch_run_retention_settings_payload(retention_settings),
            }
        )

    @app.post("/api/admin/embedding-batch-runs/cleanup")
    def api_cleanup_embedding_batch_run_records(
        payload: EmbeddingBatchRunCleanupRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            result = cleanup_expired_embedding_batch_run_records(
                settings.database_url,
                dry_run=payload.dry_run,
            )
        except InvalidEmbeddingBatchRunRetentionError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content={"cleanup": embedding_batch_run_cleanup_result_payload(result)})

    @app.get("/api/admin/embedding-batch-runs/{batch_run_id}")
    def api_get_embedding_batch_run(batch_run_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            batch_run = get_embedding_worker_batch_run(settings.database_url, batch_run_id)
        except InvalidEmbeddingWorkerBatchRunError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if batch_run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Embedding batch run not found.",
            )

        return JSONResponse(content={"batch_run": embedding_worker_batch_run_payload(batch_run)})

    @app.post("/api/admin/embedding-batch-runs/{batch_run_id}/retry-failed")
    def api_retry_failed_embedding_batch_run_jobs(batch_run_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            batch_run = get_embedding_worker_batch_run(settings.database_url, batch_run_id)
            if batch_run is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Embedding batch run not found.",
                )
            retry_result = retry_failed_embedding_worker_batch_run_jobs(
                settings.database_url,
                batch_run,
            )
        except (
            InvalidEmbeddingWorkerBatchRunError,
            InvalidEmbeddingJobError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content=retry_result)

    @app.get("/api/admin/dgx-ingestion-benchmarks")
    def api_list_dgx_ingestion_benchmarks(
        provider: str | None = None,
        profile_name: str | None = None,
        passed: bool | None = None,
        limit: int = 50,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            benchmark_runs = list_dgx_ingestion_benchmark_runs(
                settings.database_url,
                provider=provider or None,
                profile_name=profile_name or None,
                passed=passed,
                limit=limit,
            )
        except InvalidDgxIngestionBenchmarkError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "benchmark_run_count": len(benchmark_runs),
                "summary": dgx_ingestion_benchmark_summary(benchmark_runs),
                "benchmark_runs": [
                    dgx_ingestion_benchmark_run_payload(run) for run in benchmark_runs
                ],
            }
        )

    @app.get("/api/admin/dgx-ingestion-benchmarks/trend-summary")
    def api_get_dgx_ingestion_benchmark_trend_summary(
        provider: str | None = None,
        profile_name: str | None = None,
        limit: int = 50,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            benchmark_runs = list_dgx_ingestion_benchmark_runs(
                settings.database_url,
                provider=provider or None,
                profile_name=profile_name or None,
                limit=limit,
            )
            details = [
                detail
                for run in benchmark_runs
                if (
                    detail := get_dgx_ingestion_benchmark_detail(
                        settings.database_url,
                        run.benchmark_run_id,
                    )
                )
                is not None
            ]
        except InvalidDgxIngestionBenchmarkError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "trend": dgx_ingestion_benchmark_trend_summary_payload(details),
            }
        )

    @app.get("/api/admin/dgx-ingestion-benchmarks/compare")
    def api_compare_dgx_ingestion_benchmarks(
        left_run_id: int,
        right_run_id: int,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        if left_run_id == right_run_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Choose two different DGX ingestion benchmark runs.",
            )

        try:
            left_detail = get_dgx_ingestion_benchmark_detail(
                settings.database_url,
                left_run_id,
            )
            right_detail = get_dgx_ingestion_benchmark_detail(
                settings.database_url,
                right_run_id,
            )
        except InvalidDgxIngestionBenchmarkError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if left_detail is None or right_detail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="DGX ingestion benchmark run not found.",
            )

        return JSONResponse(
            content={
                "comparison": dgx_ingestion_benchmark_compare_payload(
                    left_detail,
                    right_detail,
                )
            }
        )

    @app.get("/api/admin/dgx-ingestion-benchmarks/{benchmark_run_id}")
    def api_get_dgx_ingestion_benchmark(benchmark_run_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            detail = get_dgx_ingestion_benchmark_detail(
                settings.database_url,
                benchmark_run_id,
            )
        except InvalidDgxIngestionBenchmarkError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if detail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="DGX ingestion benchmark run not found.",
            )

        return JSONResponse(content={"benchmark": dgx_ingestion_benchmark_detail_payload(detail)})

    def retry_search_log_profile_response_payload(
        search_log_id: int,
        raw_profile_name: str,
    ) -> dict[str, object]:
        try:
            source_log = get_search_log(settings.database_url or "", search_log_id)
        except InvalidSearchLogError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if source_log is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Search log not found.",
            )
        if source_log.actor_user_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Search log actor_user_id is required for retry.",
            )

        profile_name = raw_profile_name.strip()
        if not profile_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="profile_name is required.",
            )
        if profile_name not in source_log.profiles:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="profile_name is not included in the source search log.",
            )

        try:
            result = run_search_compare(
                settings.database_url or "",
                SearchCompareInput(
                    query_text=source_log.query_text,
                    actor_user_id=source_log.actor_user_id,
                    requested_search_scope=(
                        source_log.requested_search_scope
                        or source_log.effective_search_scope
                        or "company"
                    ),
                    top_k=source_log.top_k,
                    profiles=(profile_name,),
                    chunk_policy_name=source_log.chunk_policy_name,
                    document_group=source_log.document_group,
                    file_type=source_log.file_type,
                    bm25_tokenizer_name=(
                        search_log_bm25_tokenizer_name(source_log.query_runtime_metadata)
                        or DEFAULT_BM25_TOKENIZER_NAME
                    ),
                    reranked_vector_profile_name=search_log_reranked_vector_profile_name(
                        source_log.query_runtime_metadata
                    ),
                ),
                fallback_runtime_config=embedding_provider_runtime_config_from_settings(settings),
                fallback_reranker_runtime_config=reranker_runtime_config_from_settings(settings),
            )
        except (
            InvalidEmbeddingProviderError,
            InvalidQueryEmbeddingError,
            InvalidRerankerError,
            InvalidSearchCompareError,
            InvalidPermissionError,
            InvalidVectorSearchError,
            InvalidSearchLogError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return {
            "source_search_log_id": source_log.search_log_id,
            "retry_profile_name": profile_name,
            "retry_search_log_url": f"/search/logs?search_log_id={result.search_log_id}",
            "search_result": search_compare_payload(result),
        }

    @app.post("/api/search/compare/readiness")
    def api_search_compare_readiness(payload: SearchCompareReadinessRequest) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            readiness = get_search_compare_readiness(
                settings.database_url,
                SearchCompareReadinessInput(
                    actor_user_id=payload.actor_user_id,
                    requested_search_scope=payload.requested_search_scope,
                    profiles=tuple(payload.profiles) if payload.profiles is not None else None,
                    chunk_policy_name=payload.chunk_policy_name,
                    chunk_policy_names=(
                        tuple(payload.chunk_policy_names)
                        if payload.chunk_policy_names is not None
                        else None
                    ),
                    document_group=payload.document_group,
                    file_type=payload.file_type,
                ),
            )
        except (InvalidSearchCompareError, InvalidPermissionError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content=search_compare_readiness_payload(readiness))

    @app.post("/api/search/compare/readiness/reconcile-coverage")
    def api_search_compare_readiness_reconcile_coverage(
        payload: SearchCompareReadinessCoverageReconcileRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            result = reconcile_search_compare_policy_coverage(
                settings.database_url,
                SearchCompareCoverageReconcileInput(
                    actor_user_id=payload.actor_user_id,
                    requested_search_scope=payload.requested_search_scope,
                    profile_name=payload.profile_name,
                    chunk_policy_name=payload.chunk_policy_name,
                    document_group=payload.document_group,
                    file_type=payload.file_type,
                    max_jobs=payload.max_jobs,
                ),
            )
        except (InvalidSearchCompareError, InvalidPermissionError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content=search_compare_coverage_reconcile_payload(result))

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
                    bm25_tokenizer_name=(
                        payload.bm25_tokenizer_name or DEFAULT_BM25_TOKENIZER_NAME
                    ),
                    hybrid_vector_profile_name=payload.hybrid_vector_profile_name,
                    reranked_vector_profile_name=payload.reranked_vector_profile_name,
                    allow_mock_fallback=payload.allow_mock_fallback,
                ),
                fallback_runtime_config=embedding_provider_runtime_config_from_settings(settings),
                fallback_reranker_runtime_config=reranker_runtime_config_from_settings(settings),
            )
        except (
            InvalidEmbeddingProviderError,
            InvalidQueryEmbeddingError,
            InvalidRerankerError,
            InvalidSearchCompareError,
            InvalidPermissionError,
            InvalidVectorSearchError,
            InvalidSearchLogError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content=search_compare_payload(result))

    @app.post("/api/search/compare/chunk-policies")
    def api_search_compare_chunk_policies(
        payload: SearchChunkPolicyCompareRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        chunk_policy_names = [name.strip() for name in payload.chunk_policy_names]
        if any(not name for name in chunk_policy_names):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="chunk_policy_names must not contain blank values.",
            )
        if len(set(chunk_policy_names)) != len(chunk_policy_names):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="chunk_policy_names must be unique.",
            )

        known_policy_names = {
            policy.chunk_policy_name
            for policy in list_chunk_policy_summaries(settings.database_url)
        }
        unknown_policy_names = [
            name for name in chunk_policy_names if name not in known_policy_names
        ]
        if unknown_policy_names:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Unknown chunk_policy_names: " f"{', '.join(sorted(unknown_policy_names))}"
                ),
            )

        started_at = perf_counter()
        try:
            runs = [
                search_chunk_policy_compare_run_payload(
                    chunk_policy_name,
                    run_search_compare(
                        settings.database_url,
                        SearchCompareInput(
                            query_text=payload.query_text,
                            actor_user_id=payload.actor_user_id,
                            requested_search_scope=payload.requested_search_scope,
                            top_k=payload.top_k,
                            profiles=(
                                tuple(payload.profiles) if payload.profiles is not None else None
                            ),
                            chunk_policy_name=chunk_policy_name,
                            document_group=payload.document_group,
                            file_type=payload.file_type,
                            bm25_tokenizer_name=(
                                payload.bm25_tokenizer_name or DEFAULT_BM25_TOKENIZER_NAME
                            ),
                            hybrid_vector_profile_name=payload.hybrid_vector_profile_name,
                            reranked_vector_profile_name=payload.reranked_vector_profile_name,
                            allow_mock_fallback=payload.allow_mock_fallback,
                        ),
                        fallback_runtime_config=(
                            embedding_provider_runtime_config_from_settings(settings)
                        ),
                        fallback_reranker_runtime_config=(
                            reranker_runtime_config_from_settings(settings)
                        ),
                    ),
                )
                for chunk_policy_name in chunk_policy_names
            ]
        except (
            InvalidEmbeddingProviderError,
            InvalidQueryEmbeddingError,
            InvalidRerankerError,
            InvalidSearchCompareError,
            InvalidPermissionError,
            InvalidVectorSearchError,
            InvalidSearchLogError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        chunk_id_sets = [set(run["unique_chunk_ids"]) for run in runs]
        shared_chunk_ids = sorted(set.intersection(*chunk_id_sets) if chunk_id_sets else set())
        return JSONResponse(
            content={
                "query_text": payload.query_text,
                "actor_user_id": payload.actor_user_id,
                "requested_search_scope": payload.requested_search_scope,
                "top_k": payload.top_k,
                "profiles": payload.profiles or [],
                "document_group": payload.document_group,
                "file_type": payload.file_type,
                "bm25_tokenizer_name": payload.bm25_tokenizer_name or DEFAULT_BM25_TOKENIZER_NAME,
                "hybrid_vector_profile_name": payload.hybrid_vector_profile_name,
                "reranked_vector_profile_name": payload.reranked_vector_profile_name,
                "policy_count": len(runs),
                "shared_chunk_count": len(shared_chunk_ids),
                "shared_chunk_ids": shared_chunk_ids,
                "total_elapsed_ms": max(0, int((perf_counter() - started_at) * 1000)),
                "runs": runs,
            },
        )

    @app.get("/api/search/results/{search_log_result_id}/source-context")
    def api_get_search_result_source_context(search_log_result_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            context = get_search_result_source_context(
                settings.database_url,
                search_log_result_id,
            )
        except InvalidSearchResultContextError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if context is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Search result source context not found.",
            )

        return JSONResponse(content=search_result_source_context_payload(context))

    @app.get("/api/search/logs/{search_log_id}/retrieval-context")
    def api_get_search_log_retrieval_context(
        search_log_id: int,
        max_context_chars: int = Query(default=DEFAULT_CONTEXT_CHAR_BUDGET, ge=500, le=50000),
        include_neighbors: bool = True,
        max_items: int = Query(default=DEFAULT_CONTEXT_MAX_ITEMS, ge=1, le=100),
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            package = build_retrieval_context_package(
                settings.database_url,
                RetrievalContextInput(
                    search_log_id=search_log_id,
                    max_context_chars=max_context_chars,
                    include_neighbors=include_neighbors,
                    max_items=max_items,
                ),
            )
        except InvalidRetrievalContextError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if package is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Search log retrieval context not found.",
            )

        return JSONResponse(content=retrieval_context_package_payload(package))

    @app.get("/api/search/logs/{search_log_id}/citation-readiness")
    def api_get_search_log_citation_readiness(
        search_log_id: int,
        max_context_chars: int = Query(default=DEFAULT_CONTEXT_CHAR_BUDGET, ge=500, le=50000),
        include_neighbors: bool = True,
        max_items: int = Query(default=DEFAULT_CONTEXT_MAX_ITEMS, ge=1, le=100),
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            report = build_citation_readiness_report(
                settings.database_url,
                CitationReadinessInput(
                    search_log_id=search_log_id,
                    max_context_chars=max_context_chars,
                    include_neighbors=include_neighbors,
                    max_items=max_items,
                ),
            )
        except InvalidRetrievalContextError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if report is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Search log citation readiness not found.",
            )

        return JSONResponse(content=citation_readiness_report_payload(report))

    @app.get("/api/generation/templates")
    def api_list_generation_templates(include_inactive: bool = False) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        templates = list_generation_templates(
            settings.database_url,
            include_inactive=include_inactive,
        )
        return JSONResponse(
            content={
                "templates": [generation_template_payload(template) for template in templates],
                "default_template_key": next(
                    (
                        template.template_key
                        for template in templates
                        if template.is_default and template.is_active
                    ),
                    None,
                ),
            }
        )

    @app.get("/api/admin/generation-templates")
    def api_admin_list_generation_templates(include_inactive: bool = True) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        templates = list_generation_templates(
            settings.database_url,
            include_inactive=include_inactive,
        )
        return JSONResponse(
            content=generation_template_collection_payload(
                templates,
                include_inactive=include_inactive,
            )
        )

    @app.get("/api/admin/generation-templates/{template_key}")
    def api_admin_get_generation_template(template_key: str) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            template = get_generation_template_by_key(
                settings.database_url,
                template_key,
                include_inactive=True,
            )
        except InvalidGenerationTemplateError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if template is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Generation template not found.",
            )
        return JSONResponse(content={"template": generation_template_payload(template)})

    @app.post("/api/admin/generation-templates")
    def api_admin_create_generation_template(
        payload: GenerationTemplateManagementRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            template = upsert_generation_template(
                settings.database_url,
                generation_template_input_from_request(payload),
            )
        except InvalidGenerationTemplateError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"template": generation_template_payload(template)},
        )

    @app.put("/api/admin/generation-templates/{template_key}")
    def api_admin_update_generation_template(
        template_key: str,
        payload: GenerationTemplateManagementRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        if payload.template_key.strip().lower() != template_key.strip().lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="template_key path and payload must match.",
            )

        try:
            template = upsert_generation_template(
                settings.database_url,
                generation_template_input_from_request(payload),
            )
        except InvalidGenerationTemplateError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse(content={"template": generation_template_payload(template)})

    @app.patch("/api/admin/generation-templates/{template_key}/active")
    def api_admin_update_generation_template_active(
        template_key: str,
        payload: GenerationTemplateActiveRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            template = set_generation_template_active(
                settings.database_url,
                template_key,
                is_active=payload.is_active,
            )
        except InvalidGenerationTemplateError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if template is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Generation template not found.",
            )
        return JSONResponse(content={"template": generation_template_payload(template)})

    @app.post("/api/admin/generation-templates/{template_key}/default")
    def api_admin_set_generation_template_default(template_key: str) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            template = set_generation_template_default(settings.database_url, template_key)
        except InvalidGenerationTemplateError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if template is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Generation template not found.",
            )
        return JSONResponse(content={"template": generation_template_payload(template)})

    @app.post("/api/admin/generation-templates/{template_key}/clone")
    def api_admin_clone_generation_template_version(
        template_key: str,
        payload: GenerationTemplateCloneRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            template = clone_generation_template_version(
                settings.database_url,
                GenerationTemplateCloneInput(
                    source_template_key=template_key,
                    target_template_key=payload.target_template_key,
                    target_template_version=payload.target_template_version,
                    target_template_name=payload.target_template_name,
                    make_default=payload.make_default,
                    is_active=payload.is_active,
                    change_note=payload.change_note,
                    created_by=payload.created_by,
                    created_by_user_id=payload.created_by_user_id,
                ),
            )
        except InvalidGenerationTemplateError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if template is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source generation template not found.",
            )
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"template": generation_template_payload(template)},
        )

    @app.post("/api/admin/generation-templates/{template_key}/rollback")
    def api_admin_rollback_generation_template_version(template_key: str) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            template = rollback_generation_template_version(settings.database_url, template_key)
        except InvalidGenerationTemplateError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if template is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Generation template rollback target not found.",
            )
        return JSONResponse(content={"template": generation_template_payload(template)})

    @app.post("/api/generation/direct-runs")
    def api_create_direct_generation_run(payload: DirectGenerationRequest) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        api_key: str | None = None
        if payload.provider_mode.strip().lower() == (
            GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE
        ):
            try:
                provider = get_generation_provider_config_for_mode(
                    settings.database_url,
                    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
                )
                if provider is None:
                    raise InvalidGenerationRunError(
                        "active remote_openai_compatible generation provider config was not found"
                    )
                api_key = resolve_generation_provider_api_key(provider, settings)
            except (InvalidGenerationRunError, InvalidGenerationProviderError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(exc),
                ) from exc

        try:
            result = run_direct_generation_query(
                settings.database_url,
                DirectGenerationInput(
                    query_text=payload.query_text,
                    actor_user_id=payload.actor_user_id,
                    requested_search_scope=payload.requested_search_scope,
                    provider_mode=payload.provider_mode,
                    generation_template_key=payload.generation_template_key,
                    top_k=payload.top_k,
                    profiles=tuple(payload.profiles) if payload.profiles is not None else None,
                    chunk_policy_name=payload.chunk_policy_name,
                    document_group=payload.document_group,
                    file_type=payload.file_type,
                    bm25_tokenizer_name=(
                        payload.bm25_tokenizer_name or DEFAULT_BM25_TOKENIZER_NAME
                    ),
                    hybrid_vector_profile_name=payload.hybrid_vector_profile_name,
                    reranked_vector_profile_name=payload.reranked_vector_profile_name,
                    allow_mock_fallback=payload.allow_mock_fallback,
                    max_context_chars=payload.max_context_chars,
                    include_neighbors=payload.include_neighbors,
                    max_items=payload.max_items,
                ),
                fallback_runtime_config=embedding_provider_runtime_config_from_settings(settings),
                fallback_reranker_runtime_config=reranker_runtime_config_from_settings(settings),
                api_key=api_key,
            )
        except (
            InvalidDirectGenerationError,
            InvalidEmbeddingProviderError,
            InvalidGenerationRunError,
            InvalidPermissionError,
            InvalidQueryEmbeddingError,
            InvalidRerankerError,
            InvalidRetrievalContextError,
            InvalidSearchCompareError,
            InvalidSearchLogError,
            InvalidVectorSearchError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=direct_generation_result_payload(result),
        )

    @app.post("/api/documents/{document_id}/summary-runs")
    def api_create_document_summary_run(
        document_id: int,
        payload: DocumentSummaryRunRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        api_key: str | None = None
        if payload.provider_mode.strip().lower() == (
            GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE
        ):
            try:
                provider = get_generation_provider_config_for_mode(
                    settings.database_url,
                    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
                )
                if provider is None:
                    raise InvalidGenerationRunError(
                        "active remote_openai_compatible generation provider config was not found"
                    )
                api_key = resolve_generation_provider_api_key(provider, settings)
            except (InvalidGenerationRunError, InvalidGenerationProviderError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(exc),
                ) from exc

        try:
            result = run_document_summary_generation(
                settings.database_url,
                DocumentSummaryInput(
                    document_id=document_id,
                    actor_user_id=payload.actor_user_id,
                    summary_instruction=payload.summary_instruction,
                    provider_mode=payload.provider_mode,
                    generation_template_key=payload.generation_template_key,
                    max_chunks=payload.max_chunks,
                    max_context_chars=payload.max_context_chars,
                    include_neighbors=payload.include_neighbors,
                    chunk_policy_name=payload.chunk_policy_name,
                ),
                api_key=api_key,
            )
        except (
            InvalidDocumentSummaryError,
            InvalidGenerationRunError,
            InvalidRetrievalContextError,
            InvalidSearchLogError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=document_summary_result_payload(result),
        )

    @app.get("/api/generation/document-summaries")
    def api_list_document_summary_runs(
        limit: int = Query(default=DEFAULT_DOCUMENT_SUMMARY_HISTORY_LIMIT),
        run_status: str = Query(default=GENERATION_RUN_HISTORY_FILTER_ALL),
        generation_template_key: str = Query(default=GENERATION_RUN_HISTORY_FILTER_ALL),
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            history = list_document_summary_history(
                settings.database_url,
                history_filter=DocumentSummaryHistoryFilter(
                    limit=limit,
                    run_status=run_status,
                    generation_template_key=generation_template_key,
                ),
            )
        except InvalidDocumentSummaryError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse(content=document_summary_history_payload(history))

    @app.post("/api/search/logs/{search_log_id}/generation-runs/mock")
    def api_create_mock_generation_run(
        search_log_id: int,
        max_context_chars: int = Query(default=DEFAULT_CONTEXT_CHAR_BUDGET, ge=500, le=50000),
        include_neighbors: bool = True,
        max_items: int = Query(default=DEFAULT_CONTEXT_MAX_ITEMS, ge=1, le=100),
        generation_template_key: str | None = Query(default=None),
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            package = build_retrieval_context_package(
                settings.database_url,
                RetrievalContextInput(
                    search_log_id=search_log_id,
                    max_context_chars=max_context_chars,
                    include_neighbors=include_neighbors,
                    max_items=max_items,
                ),
            )
        except InvalidRetrievalContextError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if package is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Search log retrieval context not found.",
            )

        try:
            report = execute_mock_generation_run(
                settings.database_url,
                package,
                generation_template_key=generation_template_key,
                created_by="api_mock_generation",
            )
        except InvalidGenerationRunError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=generation_execution_report_payload(report),
        )

    @app.post("/api/search/logs/{search_log_id}/generation-runs/remote")
    def api_create_remote_generation_run(
        search_log_id: int,
        max_context_chars: int = Query(default=DEFAULT_CONTEXT_CHAR_BUDGET, ge=500, le=50000),
        include_neighbors: bool = True,
        max_items: int = Query(default=DEFAULT_CONTEXT_MAX_ITEMS, ge=1, le=100),
        generation_template_key: str | None = Query(default=None),
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            package = build_retrieval_context_package(
                settings.database_url,
                RetrievalContextInput(
                    search_log_id=search_log_id,
                    max_context_chars=max_context_chars,
                    include_neighbors=include_neighbors,
                    max_items=max_items,
                ),
            )
        except InvalidRetrievalContextError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if package is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Search log retrieval context not found.",
            )

        try:
            provider = get_generation_provider_config_for_mode(
                settings.database_url,
                GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
            )
            if provider is None:
                raise InvalidGenerationRunError(
                    "active remote_openai_compatible generation provider config was not found"
                )
            api_key = resolve_generation_provider_api_key(provider, settings)
            report = execute_remote_generation_run(
                settings.database_url,
                package,
                generation_template_key=generation_template_key,
                api_key=api_key,
                created_by="api_remote_generation",
            )
        except (InvalidGenerationRunError, InvalidGenerationProviderError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=generation_execution_report_payload(report),
        )

    @app.get("/api/search/logs/{search_log_id}/generation-prompt/preview")
    def api_preview_generation_prompt(
        search_log_id: int,
        max_context_chars: int = Query(default=DEFAULT_CONTEXT_CHAR_BUDGET, ge=500, le=50000),
        include_neighbors: bool = True,
        max_items: int = Query(default=DEFAULT_CONTEXT_MAX_ITEMS, ge=1, le=100),
        response_language: str = Query(default="ko", min_length=1, max_length=16),
        generation_template_key: str | None = Query(default=None),
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            package = build_retrieval_context_package(
                settings.database_url,
                RetrievalContextInput(
                    search_log_id=search_log_id,
                    max_context_chars=max_context_chars,
                    include_neighbors=include_neighbors,
                    max_items=max_items,
                ),
            )
            if package is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Search log retrieval context not found.",
                )
            generation_template = (
                get_generation_template_by_key(settings.database_url, generation_template_key)
                if generation_template_key
                else get_default_generation_template(settings.database_url)
            )
            if generation_template_key and generation_template is None:
                raise InvalidGenerationRunError("active generation template was not found")
            prompt_package = build_generation_prompt_package(
                package,
                response_language=response_language,
                generation_template=generation_template,
            )
        except InvalidGenerationRunError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except InvalidRetrievalContextError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except InvalidGenerationPromptError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "retrieval_context": retrieval_context_package_payload(package),
                "prompt_package": generation_prompt_package_payload(prompt_package),
            }
        )

    @app.get("/api/generation/runs")
    def api_list_generation_runs(
        limit: int = Query(default=DEFAULT_GENERATION_RUN_HISTORY_LIMIT),
        answer_quality_status: str = Query(default=GENERATION_RUN_HISTORY_FILTER_ALL),
        provider_mode: str = Query(default=GENERATION_RUN_HISTORY_FILTER_ALL),
        run_status: str = Query(default=GENERATION_RUN_HISTORY_FILTER_ALL),
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            history = list_generation_run_history(
                settings.database_url,
                history_filter=GenerationRunHistoryFilter(
                    limit=limit,
                    answer_quality_status=answer_quality_status,
                    provider_mode=provider_mode,
                    run_status=run_status,
                ),
            )
        except InvalidGenerationRunError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse(content=generation_run_history_payload(history))

    @app.get("/api/generation/runs/{generation_run_id}")
    def api_get_generation_run(generation_run_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            run = get_generation_run(settings.database_url, generation_run_id)
        except InvalidGenerationRunError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Generation run not found.",
            )
        citations = list_generation_run_citations(
            settings.database_url,
            generation_run_id,
        )
        return JSONResponse(
            content={
                "run": generation_run_payload(run),
                "citations": [generation_run_citation_payload(citation) for citation in citations],
            }
        )

    @app.get("/api/generation/runs/{generation_run_id}/template-completeness")
    def api_get_generation_run_template_completeness(generation_run_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            run = get_generation_run(settings.database_url, generation_run_id)
        except InvalidGenerationRunError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Generation run not found.",
            )

        assessment = assess_generation_template_completeness(run)
        return JSONResponse(
            content={
                "generation_run_id": generation_run_id,
                "template_completeness": generation_template_completeness_payload(assessment),
            }
        )

    @app.get("/api/generation/runs/{generation_run_id}/export/markdown")
    def api_export_generation_run_markdown(generation_run_id: int) -> Response:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            run = get_generation_run(settings.database_url, generation_run_id)
        except InvalidGenerationRunError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Generation run not found.",
            )
        citations = list_generation_run_citations(
            settings.database_url,
            generation_run_id,
        )
        markdown = _generation_run_markdown_export(run, citations)
        filename = f"generation-run-{generation_run_id}.md"
        return Response(
            content=markdown,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/generation/runs/{generation_run_id}/export/docx")
    def api_export_generation_run_docx(generation_run_id: int) -> Response:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            run = get_generation_run(settings.database_url, generation_run_id)
        except InvalidGenerationRunError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Generation run not found.",
            )
        citations = list_generation_run_citations(
            settings.database_url,
            generation_run_id,
        )
        markdown = _generation_run_markdown_export(run, citations)
        template = _generation_run_export_template(run)
        template_completeness = assess_generation_template_completeness(run)
        docx_export_readiness = assess_generation_docx_export_readiness(
            run,
            template_completeness,
        )
        docx_export_evidence = generation_docx_export_evidence_from_run(
            run,
            template=template,
            readiness=docx_export_readiness,
        )
        docx_bytes = markdown_to_docx_bytes(
            markdown,
            title=f"Generation Run #{generation_run_id}",
            document_type=str(template.get("document_type") or ""),
            export_evidence=docx_export_evidence,
        )
        filename = f"generation-run-{generation_run_id}.docx"
        return Response(
            content=docx_bytes,
            media_type=GENERATION_DOCX_MEDIA_TYPE,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-NeX-PCX-Export-Readiness": docx_export_readiness.status,
                "X-NeX-PCX-Export-Readiness-Reasons": (
                    ",".join(docx_export_readiness.reason_codes) or "-"
                ),
                "X-NeX-PCX-Export-Evidence": (
                    f"generation_run_id={docx_export_evidence.generation_run_id};"
                    f"search_log_id={docx_export_evidence.search_log_id};"
                    f"readiness={docx_export_evidence.export_readiness_status}"
                ),
            },
        )

    @app.get("/api/admin/generation-provider-metrics/snapshot")
    def api_get_generation_provider_metrics_snapshot(
        limit: int = Query(default=DEFAULT_GENERATION_PROVIDER_METRIC_SNAPSHOT_LIMIT),
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        try:
            snapshot = get_generation_provider_metric_snapshot(
                settings.database_url,
                limit=limit,
            )
        except InvalidGenerationProviderMetricSnapshotError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse(content=generation_provider_metric_snapshot_payload(snapshot))

    @app.get("/api/admin/generation-provider-configs")
    def api_list_generation_provider_runtime_configs(
        include_inactive: bool = True,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        providers = list_generation_provider_configs(
            settings.database_url,
            include_inactive=include_inactive,
        )
        return JSONResponse(
            content=generation_provider_config_collection_payload(
                providers,
                settings,
                include_inactive=include_inactive,
            )
        )

    @app.get("/api/admin/generation-provider-configs/default")
    def api_get_default_generation_provider_runtime_config() -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        provider = get_default_generation_provider_config(settings.database_url)
        if provider is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Default generation provider config not found.",
            )
        return JSONResponse(
            content={
                "provider": generation_provider_config_payload(provider),
                "runtime_config": generation_provider_runtime_config_payload(provider, settings),
            }
        )

    @app.post("/api/admin/generation-provider-configs/seed-dgx-vllm")
    def api_seed_dgx_vllm_generation_provider_runtime_config(
        payload: GenerationProviderDgxVllmSeedRequest | None = None,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        seed_payload = payload or GenerationProviderDgxVllmSeedRequest()
        try:
            provider = seed_dgx_vllm_generation_provider_config(
                settings.database_url,
                provider_name=seed_payload.provider_name,
                provider_base_url=seed_payload.provider_base_url,
                model_id=seed_payload.model_id,
                api_key_env=seed_payload.api_key_env,
                request_timeout_seconds=seed_payload.request_timeout_seconds,
                max_tokens=seed_payload.max_tokens,
                temperature=seed_payload.temperature,
                top_p=seed_payload.top_p,
                is_default=seed_payload.is_default,
                is_active=seed_payload.is_active,
                thinking_disabled=seed_payload.thinking_disabled,
                created_by=seed_payload.created_by,
                created_by_user_id=seed_payload.created_by_user_id,
            )
        except InvalidGenerationRunError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "provider": generation_provider_config_payload(provider),
                "runtime_config": generation_provider_runtime_config_payload(provider, settings),
                "seed": {
                    "provider_name": provider.provider_name,
                    "is_default": provider.is_default,
                    "api_key_env": provider.runtime_options.get("api_key_env"),
                    "secret_persisted": False,
                },
            },
        )

    @app.post("/api/search/logs/{search_log_id}/retry-profile")
    def api_retry_search_log_profile(
        search_log_id: int,
        payload: SearchProfileRetryRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        return JSONResponse(
            content=retry_search_log_profile_response_payload(
                search_log_id,
                payload.profile_name,
            )
        )

    @app.post("/api/search/experiments/run")
    def api_run_search_experiment(payload: SearchExperimentRunRequest) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            report = execute_search_experiment(
                settings.database_url,
                SearchExperimentExecutionInput(
                    run_name=payload.run_name,
                    query_text=payload.query_text,
                    actor_user_id=payload.actor_user_id,
                    requested_search_scope=payload.requested_search_scope,
                    profiles=tuple(payload.profiles) if payload.profiles is not None else None,
                    strategy_name=payload.strategy_name,
                    top_k=payload.top_k,
                    score_threshold=payload.score_threshold,
                    chunk_policy_name=payload.chunk_policy_name,
                    document_group=payload.document_group,
                    file_type=payload.file_type,
                    runtime_metadata=payload.runtime_metadata,
                    created_by=payload.created_by,
                    created_by_user_id=payload.created_by_user_id,
                    allow_mock_fallback=payload.allow_mock_fallback,
                ),
                fallback_runtime_config=embedding_provider_runtime_config_from_settings(settings),
                fallback_reranker_runtime_config=reranker_runtime_config_from_settings(settings),
            )
        except (
            InvalidEmbeddingProviderError,
            InvalidQueryEmbeddingError,
            InvalidRerankerError,
            InvalidSearchExperimentExecutionError,
            InvalidSearchExperimentError,
            InvalidSearchCompareError,
            InvalidPermissionError,
            InvalidVectorSearchError,
            InvalidSearchLogError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content=search_experiment_execution_payload(report))

    @app.get("/api/search/experiments")
    def api_list_search_experiments(
        limit: int = Query(default=50, ge=1, le=500),
        status_filter: str | None = Query(default=None, alias="status"),
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            runs = list_search_experiment_runs(
                settings.database_url,
                status=status_filter,
                limit=limit,
            )
        except InvalidSearchExperimentError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "experiments": [search_experiment_run_record_payload(run) for run in runs],
            }
        )

    @app.get("/api/search/experiments/golden-question-batches")
    def api_list_golden_search_experiment_batches(
        limit: int = Query(default=20, ge=1, le=100),
        question_set_id: int | None = Query(default=None, ge=1),
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            summaries = list_golden_search_experiment_batch_summaries(
                settings.database_url,
                question_set_id=question_set_id,
                limit=limit,
            )
        except InvalidSearchExperimentError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "batches": [
                    golden_search_experiment_batch_summary_payload(summary) for summary in summaries
                ],
            }
        )

    @app.get("/api/search/experiments/golden-question-batches/{batch_key}/metric-snapshots")
    def api_list_golden_search_experiment_batch_metric_snapshots(
        batch_key: str,
        limit: int = Query(default=10, ge=1, le=100),
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            snapshots = list_golden_batch_metric_snapshots(
                settings.database_url,
                batch_key,
                limit=limit,
            )
        except (InvalidGoldenBatchMetricSnapshotError, InvalidSearchExperimentError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "snapshots": [
                    golden_batch_metric_snapshot_record_payload(snapshot) for snapshot in snapshots
                ],
            }
        )

    @app.get("/api/search/experiments/golden-question-batches/{batch_key}/metric-snapshots/trend")
    def api_get_golden_search_experiment_batch_metric_snapshot_trend(
        batch_key: str,
        limit: int = Query(default=20, ge=1, le=100),
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            trend = get_golden_batch_metric_snapshot_trend(
                settings.database_url,
                batch_key,
                limit=limit,
            )
        except (InvalidGoldenBatchMetricSnapshotError, InvalidSearchExperimentError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content={"trend": golden_batch_metric_snapshot_trend_payload(trend)})

    @app.post("/api/search/experiments/golden-question-batches/{batch_key}/metric-snapshots")
    def api_record_golden_search_experiment_batch_metric_snapshot(
        batch_key: str,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            snapshot = record_golden_batch_metric_snapshot(
                settings.database_url,
                batch_key,
                created_by="api",
            )
        except (InvalidGoldenBatchMetricSnapshotError, InvalidSearchExperimentError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if snapshot is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Golden search experiment batch not found.",
            )

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=golden_batch_metric_snapshot_detail_payload(snapshot),
        )

    @app.get("/api/search/experiments/golden-question-batches/{batch_key}/metrics")
    def api_get_golden_search_experiment_batch_metrics(batch_key: str) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            metric_summary = get_golden_search_experiment_batch_metric_summary(
                settings.database_url,
                batch_key,
            )
        except InvalidSearchExperimentError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if metric_summary is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Golden search experiment batch not found.",
            )

        content = golden_search_experiment_batch_metric_summary_payload(metric_summary)
        try:
            latest_snapshot = get_latest_golden_batch_metric_snapshot(
                settings.database_url,
                batch_key,
            )
        except InvalidGoldenBatchMetricSnapshotError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        content["latest_snapshot"] = (
            golden_batch_metric_snapshot_record_payload(latest_snapshot)
            if latest_snapshot is not None
            else None
        )
        return JSONResponse(content=content)

    @app.get("/api/search/experiments/golden-question-batches/{batch_key}")
    def api_get_golden_search_experiment_batch_detail(batch_key: str) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            detail = get_golden_search_experiment_batch_detail(settings.database_url, batch_key)
        except InvalidSearchExperimentError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if detail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Golden search experiment batch not found.",
            )

        return JSONResponse(content=golden_search_experiment_batch_detail_payload(detail))

    @app.get("/api/search/experiments/golden-question-batch-metric-snapshots/compare")
    def api_compare_golden_search_experiment_batch_metric_snapshots(
        base_snapshot_id: int = Query(..., ge=1),
        target_snapshot_id: int = Query(..., ge=1),
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            comparison = compare_golden_batch_metric_snapshots(
                settings.database_url,
                base_snapshot_id=base_snapshot_id,
                target_snapshot_id=target_snapshot_id,
            )
        except InvalidGoldenBatchMetricSnapshotError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if comparison is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Golden batch metric snapshot comparison target not found.",
            )

        return JSONResponse(
            content={"comparison": golden_batch_metric_snapshot_comparison_payload(comparison)}
        )

    @app.get("/api/search/experiments/golden-question-batch-metric-snapshots/{snapshot_id}")
    def api_get_golden_search_experiment_batch_metric_snapshot(
        snapshot_id: int,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            snapshot = get_golden_batch_metric_snapshot_detail(
                settings.database_url,
                snapshot_id,
            )
        except InvalidGoldenBatchMetricSnapshotError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if snapshot is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Golden batch metric snapshot not found.",
            )

        return JSONResponse(content=golden_batch_metric_snapshot_detail_payload(snapshot))

    @app.get("/api/search/experiments/{experiment_run_id}")
    def api_get_search_experiment_detail(experiment_run_id: int) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            detail = get_search_experiment_run_detail(
                settings.database_url,
                experiment_run_id,
            )
        except InvalidSearchExperimentError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if detail is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Search experiment run not found.",
            )

        return JSONResponse(content=search_experiment_detail_payload(detail))

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
                    bm25_tokenizer_name=(
                        payload.bm25_tokenizer_name or DEFAULT_BM25_TOKENIZER_NAME
                    ),
                    hybrid_vector_profile_name=payload.hybrid_vector_profile_name,
                    reranked_vector_profile_name=payload.reranked_vector_profile_name,
                    allow_mock_fallback=payload.allow_mock_fallback,
                ),
                fallback_runtime_config=embedding_provider_runtime_config_from_settings(settings),
                fallback_reranker_runtime_config=reranker_runtime_config_from_settings(settings),
            )
        except (
            InvalidEmbeddingProviderError,
            InvalidQueryEmbeddingError,
            InvalidRerankerError,
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
        provider_mode_filter: str | None = None,
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
                provider_mode_filter=provider_mode_filter,
                limit=limit,
            )
            logs = filter_search_logs_by_fingerprint(logs, fingerprint)
        except InvalidSearchLogError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(content={"logs": [search_log_record_payload(log) for log in logs]})

    @app.get("/api/search/logs/runtime-failures")
    def api_list_search_runtime_failures(
        profile_name: str | None = None,
        limit: int = 20,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            failures = list_search_runtime_failures(
                settings.database_url,
                profile_name=profile_name,
                limit=limit,
            )
        except InvalidSearchLogError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "failures": [search_runtime_failure_payload(failure) for failure in failures],
            }
        )

    @app.get("/api/search/logs/latency-outliers")
    def api_list_search_latency_outliers(
        min_total_elapsed_ms: int = 1000,
        limit: int = 20,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            outliers = list_search_latency_outliers(
                settings.database_url,
                min_total_elapsed_ms=min_total_elapsed_ms,
                limit=limit,
            )
        except InvalidSearchLogError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "min_total_elapsed_ms": min_total_elapsed_ms,
                "outliers": [search_latency_outlier_payload(outlier) for outlier in outliers],
            }
        )

    @app.get("/api/search/logs/no-results")
    def api_list_search_no_result_logs(limit: int = 20) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            records = list_search_no_result_logs(settings.database_url, limit=limit)
        except InvalidSearchLogError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "records": [search_no_result_payload(record) for record in records],
            }
        )

    @app.get("/api/search/logs/duplicate-fingerprints")
    def api_list_search_duplicate_fingerprints(
        min_count: int = 2,
        limit: int = 20,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            records = list_search_duplicate_fingerprints(
                settings.database_url,
                min_count=min_count,
                limit=limit,
            )
        except InvalidSearchLogError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "min_count": min_count,
                "records": [search_duplicate_fingerprint_payload(record) for record in records],
            }
        )

    @app.get("/api/search/logs/operations-summary")
    def api_get_search_operations_summary(
        lookback_hours: int = 24,
        min_total_elapsed_ms: int = 1000,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            summary = get_search_operations_summary(
                settings.database_url,
                lookback_hours=lookback_hours,
                min_total_elapsed_ms=min_total_elapsed_ms,
            )
        except InvalidSearchLogError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return JSONResponse(
            content={
                "operations_summary": search_operations_summary_payload(summary),
            }
        )

    @app.post("/api/search/logs/runtime-failures/retry")
    def api_retry_search_runtime_failures(
        payload: SearchRuntimeFailureRetryRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )
        if not payload.failures:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="failures must not be empty.",
            )
        if len(payload.failures) > 20:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="failures must contain at most 20 items.",
            )

        results: list[dict[str, object]] = []
        for failure in payload.failures:
            try:
                retry_result = retry_search_log_profile_response_payload(
                    failure.search_log_id,
                    failure.profile_name,
                )
            except HTTPException as exc:
                results.append(
                    {
                        "source_search_log_id": failure.search_log_id,
                        "retry_profile_name": failure.profile_name,
                        "status": "failed",
                        "detail": exc.detail,
                    }
                )
                continue

            search_result = retry_result["search_result"]
            retry_search_log_id = (
                search_result["search_log_id"] if isinstance(search_result, dict) else None
            )
            results.append(
                {
                    "source_search_log_id": retry_result["source_search_log_id"],
                    "retry_profile_name": retry_result["retry_profile_name"],
                    "status": "succeeded",
                    "retry_search_log_id": retry_search_log_id,
                    "retry_search_log_url": retry_result["retry_search_log_url"],
                    "search_result": search_result,
                }
            )

        retried_count = sum(1 for result in results if result["status"] == "succeeded")
        failed_count = len(results) - retried_count
        return JSONResponse(
            content={
                "requested_count": len(payload.failures),
                "retried_count": retried_count,
                "failed_count": failed_count,
                "results": results,
            }
        )

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
                fallback_runtime_config=embedding_provider_runtime_config_from_settings(settings),
                fallback_reranker_runtime_config=reranker_runtime_config_from_settings(settings),
            )
        except (
            InvalidEmbeddingProviderError,
            InvalidQueryEmbeddingError,
            InvalidRerankerError,
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

    @app.post("/api/search/experiments/golden-question-set/run")
    def api_run_golden_search_experiment_batch(
        payload: GoldenSearchExperimentBatchRequest,
    ) -> JSONResponse:
        if not settings.database_url:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NEX_PCX_DATABASE_URL is not configured.",
            )

        try:
            report = execute_golden_search_experiment_batch(
                settings.database_url,
                golden_search_experiment_batch_input_from_request(payload),
                fallback_runtime_config=embedding_provider_runtime_config_from_settings(settings),
                fallback_reranker_runtime_config=reranker_runtime_config_from_settings(settings),
            )
        except (
            InvalidEmbeddingProviderError,
            InvalidQueryEmbeddingError,
            InvalidRerankerError,
            InvalidGoldenSearchExperimentError,
            InvalidGoldenQuestionError,
            InvalidSearchExperimentExecutionError,
            InvalidSearchExperimentError,
            InvalidSearchCompareError,
            InvalidPermissionError,
            InvalidVectorSearchError,
            InvalidSearchLogError,
        ) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if report is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Golden question set not found.",
            )

        batch_content = golden_search_experiment_batch_payload(report)
        snapshot = None
        batch_key = batch_content.get("batch_key")
        if isinstance(batch_key, str) and batch_key:
            try:
                snapshot = record_golden_batch_metric_snapshot(
                    settings.database_url,
                    batch_key,
                    created_by=payload.created_by or "golden-search-experiment-batch",
                    created_by_user_id=payload.created_by_user_id,
                )
            except (InvalidGoldenBatchMetricSnapshotError, InvalidSearchExperimentError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(exc),
                ) from exc
        batch_content["metric_snapshot"] = (
            golden_batch_metric_snapshot_record_payload(snapshot.snapshot)
            if snapshot is not None
            else None
        )

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"batch": batch_content},
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
    def dashboard(
        request: Request,
        lookback_hours: int = DEFAULT_DASHBOARD_THROUGHPUT_LOOKBACK_HOURS,
        refresh_seconds: int = 0,
    ) -> HTMLResponse:
        core_metrics: DashboardCoreMetrics | None = None
        pipeline_queue: PipelineQueueSummary | None = None
        recent_failures: DashboardFailureSummary | None = None
        evaluation_dashboard: EvaluationDashboardSummary | None = None
        embedding_backlog: EmbeddingJobBacklogSummary | None = None
        throughput_latency: DashboardThroughputLatencySnapshot | None = None
        operational_health: DashboardOperationalHealth | None = None
        threshold_settings = DashboardHealthThresholdSettings(
            thresholds=dict(DEFAULT_DASHBOARD_HEALTH_THRESHOLDS)
        )
        rendered_at = datetime.now(UTC)
        selected_lookback_hours = DEFAULT_DASHBOARD_THROUGHPUT_LOOKBACK_HOURS
        selected_refresh_seconds = 0
        error_message = None
        try:
            selected_lookback_hours = validate_lookback_hours(lookback_hours)
        except InvalidDashboardThroughputError as exc:
            error_message = str(exc)
        try:
            selected_refresh_seconds = validate_dashboard_refresh_seconds(
                refresh_seconds,
            )
        except ValueError as exc:
            if error_message is None:
                error_message = str(exc)

        if not settings.database_url:
            error_message = error_message or "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                core_metrics = get_dashboard_core_metrics(settings.database_url)
            except Exception as exc:
                error_message = str(exc)
            try:
                pipeline_queue = get_pipeline_queue_summary(settings.database_url)
            except InvalidPipelineJobError as exc:
                if error_message is None:
                    error_message = str(exc)
            except Exception as exc:
                if error_message is None:
                    error_message = str(exc)
            try:
                throughput_latency = get_dashboard_throughput_latency_snapshot(
                    settings.database_url,
                    lookback_hours=selected_lookback_hours,
                )
            except InvalidDashboardThroughputError as exc:
                if error_message is None:
                    error_message = str(exc)
            except Exception as exc:
                if error_message is None:
                    error_message = str(exc)
            try:
                recent_failures = get_dashboard_recent_failures(
                    settings.database_url,
                    limit=8,
                )
            except InvalidDashboardFailureError as exc:
                if error_message is None:
                    error_message = str(exc)
            except Exception as exc:
                if error_message is None:
                    error_message = str(exc)
            try:
                evaluation_dashboard = get_evaluation_dashboard_summary(
                    settings.database_url,
                    recent_limit=5,
                )
            except InvalidEvaluationDashboardError as exc:
                if error_message is None:
                    error_message = str(exc)
            except Exception as exc:
                if error_message is None:
                    error_message = str(exc)
            try:
                embedding_backlog = get_embedding_job_backlog_summary(settings.database_url)
            except InvalidEmbeddingJobError as exc:
                if error_message is None:
                    error_message = str(exc)
            except Exception as exc:
                if error_message is None:
                    error_message = str(exc)
            try:
                threshold_settings = load_dashboard_health_threshold_settings(settings.database_url)
            except Exception as exc:
                if error_message is None:
                    error_message = str(exc)

        operational_health = summarize_dashboard_operational_health(
            pipeline_queue=pipeline_queue,
            embedding_backlog=embedding_backlog,
            recent_failures=recent_failures,
            thresholds=threshold_settings.thresholds,
        )

        return TEMPLATES.TemplateResponse(
            request,
            "dashboard.html",
            template_context(
                request,
                database_configured=bool(settings.database_url),
                core_metrics=(
                    dashboard_core_metrics_payload(core_metrics)
                    if core_metrics is not None
                    else None
                ),
                operational_health=dashboard_operational_health_payload(
                    operational_health,
                    threshold_settings=threshold_settings,
                ),
                pipeline_queue=(
                    pipeline_queue_summary_payload(pipeline_queue)
                    if pipeline_queue is not None
                    else None
                ),
                recent_failures=(
                    dashboard_failure_summary_payload(recent_failures)
                    if recent_failures is not None
                    else None
                ),
                throughput_latency=(
                    dashboard_throughput_latency_snapshot_payload(throughput_latency)
                    if throughput_latency is not None
                    else None
                ),
                selected_lookback_hours=selected_lookback_hours,
                dashboard_time_window_options=dashboard_time_window_options(
                    request,
                    selected_lookback_hours=selected_lookback_hours,
                ),
                dashboard_rendered_at=_datetime_response(rendered_at),
                dashboard_rendered_at_label=_datetime_label(rendered_at),
                selected_refresh_seconds=selected_refresh_seconds,
                dashboard_refresh_interval_options=dashboard_refresh_interval_options(
                    request,
                    selected_refresh_seconds=selected_refresh_seconds,
                ),
                dashboard_refresh_now_url=dashboard_query_url(request, {}),
                evaluation_dashboard=evaluation_dashboard,
                embedding_backlog=embedding_backlog,
                error_message=error_message,
            ),
        )

    @app.get("/admin/go-live-readiness", response_class=HTMLResponse)
    def go_live_readiness_page(request: Request) -> HTMLResponse:
        report = build_go_live_readiness_report(settings)
        return TEMPLATES.TemplateResponse(
            request,
            "go_live_readiness.html",
            template_context(
                request,
                go_live_readiness=go_live_readiness_report_payload(report),
            ),
        )

    @app.get("/admin/operations-runbook", response_class=HTMLResponse)
    def operations_runbook_page(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request,
            "operations_runbook.html",
            template_context(
                request,
                runbook_markdown=read_operations_runbook_markdown(),
                runbook_source="docs/operations_runbook.md",
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

    @app.get("/documents/{document_id}/artifacts", response_class=HTMLResponse)
    def document_ingestion_artifacts_page(
        request: Request,
        document_id: int,
        artifact_id: int | None = None,
        rerun_run_id: int | None = None,
        rerun_status: str | None = None,
        rerun_artifact_count: int | None = None,
        rerun_block_count: int | None = None,
        rerun_error: str | None = None,
    ) -> HTMLResponse:
        document: DocumentInventoryItem | None = None
        extraction_runs: list[ExtractionRunRecord] = []
        extraction_artifacts: list[ExtractionArtifactRecord] = []
        document_blocks: list[DocumentBlockRecord] = []
        selected_artifact: ExtractionArtifactRecord | None = None
        selected_artifact_id: int | None = None
        extraction_quality_snapshot_summary: ExtractionQualitySnapshotSummary | None = None
        source_trace_chunks: list[ChunkRecord] = []
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                document = get_document_inventory_item(settings.database_url, document_id)
                if document is None:
                    error_message = f"Document not found: {document_id}"
                else:
                    extraction_runs = list_document_extraction_runs(
                        settings.database_url,
                        document_id,
                    )
                    extraction_artifacts = list_document_extraction_artifacts(
                        settings.database_url,
                        document_id,
                    )
                    artifact_ids = {artifact.artifact_id for artifact in extraction_artifacts}
                    selected_artifact_id = (
                        artifact_id
                        if artifact_id is not None
                        else (extraction_artifacts[0].artifact_id if extraction_artifacts else None)
                    )
                    if (
                        selected_artifact_id is not None
                        and selected_artifact_id not in artifact_ids
                    ):
                        error_message = (
                            f"Extraction artifact not found for document: {selected_artifact_id}"
                        )
                        selected_artifact_id = None
                    elif selected_artifact_id is not None:
                        selected_artifact = next(
                            (
                                artifact
                                for artifact in extraction_artifacts
                                if artifact.artifact_id == selected_artifact_id
                            ),
                            None,
                        )
                        document_blocks = list_document_blocks(
                            settings.database_url,
                            document_id,
                            artifact_id=selected_artifact_id,
                        )
                        document_chunks = list_document_chunks(
                            settings.database_url,
                            document_id,
                        )
                        source_trace_chunks = _chunks_for_source_trace(
                            document_chunks,
                            artifact_id=selected_artifact_id,
                            block_ids={block.block_id for block in document_blocks},
                        )
                    extraction_quality_snapshot_summary = get_extraction_quality_snapshot_summary(
                        settings.database_url,
                        document_id,
                        artifact_id=selected_artifact_id,
                    )
            except (
                InvalidDocumentInventoryError,
                InvalidIngestionArtifactError,
                InvalidChunkError,
            ) as exc:
                error_message = str(exc)
            except Exception as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "document_artifacts.html",
            template_context(
                request,
                database_configured=bool(settings.database_url),
                document=document,
                extraction_runs=extraction_runs,
                extraction_artifacts=extraction_artifacts,
                document_blocks=document_blocks,
                document_block_summary=document_block_summary_payload(document_blocks),
                selected_artifact=selected_artifact,
                selected_artifact_preview=(
                    extraction_artifact_preview_payload(selected_artifact)
                    if selected_artifact is not None
                    else None
                ),
                extraction_quality_check=extraction_quality_check_payload(
                    selected_artifact,
                    document_blocks,
                    extraction_runs,
                ),
                extraction_quality_snapshot_summary=(
                    extraction_quality_snapshot_summary_payload(
                        extraction_quality_snapshot_summary,
                    )
                    if extraction_quality_snapshot_summary is not None
                    else None
                ),
                extraction_rerun_feedback=extraction_rerun_feedback_payload(
                    run_id=rerun_run_id,
                    status_value=rerun_status,
                    artifact_count=rerun_artifact_count,
                    block_count=rerun_block_count,
                    error_message=rerun_error,
                ),
                chunk_source_trace_preview=chunk_source_trace_preview_payload(
                    selected_artifact,
                    document_blocks,
                    source_trace_chunks,
                ),
                selected_artifact_id=selected_artifact_id,
                error_message=error_message,
            ),
        )

    @app.post("/documents/{document_id}/extraction-rerun")
    def submit_document_extraction_rerun(
        document_id: int,
        extraction_profile_name: str | None = Form(None),
        requested_by: str | None = Form("extraction-rerun-ui"),
        selected_artifact_id: int | None = Form(None),
    ) -> RedirectResponse:
        redirect_params: dict[str, object | None] = {
            "artifact_id": selected_artifact_id,
        }

        if not settings.database_url:
            redirect_params["rerun_error"] = "NEX_PCX_DATABASE_URL is not configured."
            return RedirectResponse(
                document_artifacts_redirect_url(document_id, redirect_params),
                status_code=status.HTTP_303_SEE_OTHER,
            )

        try:
            document = get_document_inventory_item(settings.database_url, document_id)
            if document is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Document not found.",
                )
            file_record = get_file_metadata(settings.database_url, document.file_id)
            if file_record is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="File metadata not found for document.",
                )
            extraction_request = build_extraction_rerun_request(
                document=document,
                file_record=file_record,
                payload=ExtractionRerunRequest(
                    extraction_profile_name=extraction_profile_name,
                    requested_by=requested_by,
                    options={"ui_action": "document_artifacts_page"},
                ),
            )
            runtime_result = run_local_extraction(extraction_request)
            persisted = persist_extraction_runtime_result(
                settings.database_url,
                extraction_request,
                runtime_result,
            )
            if persisted.artifacts:
                redirect_params["artifact_id"] = persisted.artifacts[0].artifact_id
            redirect_params.update(
                {
                    "rerun_run_id": persisted.run.extraction_run_id,
                    "rerun_status": persisted.run.status,
                    "rerun_artifact_count": len(persisted.artifacts),
                    "rerun_block_count": len(persisted.blocks),
                }
            )
        except HTTPException as exc:
            redirect_params["rerun_error"] = str(exc.detail)
        except (
            InvalidDocumentInventoryError,
            InvalidFileMetadataError,
            InvalidIngestionArtifactError,
        ) as exc:
            redirect_params["rerun_error"] = str(exc)

        return RedirectResponse(
            document_artifacts_redirect_url(document_id, redirect_params),
            status_code=status.HTTP_303_SEE_OTHER,
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
        embedding_profile_options: list[str] = []
        chunk_policy_options: list[ChunkPolicySummaryRecord] = []
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                actor_options = list_search_actor_options(settings.database_url)
                embedding_profile_options = [
                    profile.profile_name
                    for profile in list_active_embedding_profiles(settings.database_url)
                ]
                profile_options = list(embedding_profile_options)
                if BM25_SEARCH_PROFILE_NAME not in profile_options:
                    profile_options.append(BM25_SEARCH_PROFILE_NAME)
                if HYBRID_SEARCH_PROFILE_NAME not in profile_options:
                    profile_options.append(HYBRID_SEARCH_PROFILE_NAME)
                if RERANKED_SEARCH_PROFILE_NAME not in profile_options:
                    profile_options.append(RERANKED_SEARCH_PROFILE_NAME)
                chunk_policy_options = list_chunk_policy_summaries(settings.database_url)
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
                embedding_profile_options=embedding_profile_options,
                default_actor_id=default_actor_id,
                search_prefill=search_compare_prefill_payload(request, default_actor_id),
                search_scope_options=SEARCH_COMPARE_SCOPE_OPTIONS,
                search_file_type_options=SEARCH_COMPARE_FILE_TYPES,
                chunk_policy_options=chunk_policy_options,
                bm25_tokenizer_options=[asdict(tokenizer) for tokenizer in list_bm25_tokenizers()],
                search_reranker_runtime=search_reranker_runtime_control_payload(settings),
                error_message=error_message,
                database_configured=bool(settings.database_url),
            ),
        )

    @app.get("/search/context", response_class=HTMLResponse)
    def retrieval_context_page(
        request: Request,
        search_log_id: int | None = Query(default=None, ge=1),
        max_context_chars: int = Query(default=DEFAULT_CONTEXT_CHAR_BUDGET, ge=500, le=50000),
        include_neighbors: bool = True,
        max_items: int = Query(default=DEFAULT_CONTEXT_MAX_ITEMS, ge=1, le=100),
    ) -> HTMLResponse:
        latest_logs: list[SearchLogListItem] = []
        package: RetrievalContextPackage | None = None
        package_payload: dict[str, object] | None = None
        package_json = ""
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                latest_logs = list_search_logs(settings.database_url, limit=12)
                if search_log_id is not None:
                    package = build_retrieval_context_package(
                        settings.database_url,
                        RetrievalContextInput(
                            search_log_id=search_log_id,
                            max_context_chars=max_context_chars,
                            include_neighbors=include_neighbors,
                            max_items=max_items,
                        ),
                    )
                    if package is None:
                        error_message = "Search log retrieval context not found."
                    else:
                        package_payload = retrieval_context_package_payload(package)
                        package_json = json.dumps(
                            package_payload,
                            ensure_ascii=False,
                            indent=2,
                        )
            except (InvalidRetrievalContextError, InvalidSearchLogError) as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "retrieval_context.html",
            template_context(
                request,
                database_configured=bool(settings.database_url),
                latest_logs=latest_logs,
                selected_search_log_id=search_log_id,
                max_context_chars=max_context_chars,
                include_neighbors=include_neighbors,
                max_items=max_items,
                package=package,
                package_payload=package_payload,
                package_json=package_json,
                error_message=error_message,
            ),
        )

    @app.get("/search/citation-readiness", response_class=HTMLResponse)
    def citation_readiness_page(
        request: Request,
        search_log_id: int | None = Query(default=None, ge=1),
        max_context_chars: int = Query(default=DEFAULT_CONTEXT_CHAR_BUDGET, ge=500, le=50000),
        include_neighbors: bool = True,
        max_items: int = Query(default=DEFAULT_CONTEXT_MAX_ITEMS, ge=1, le=100),
    ) -> HTMLResponse:
        latest_logs: list[SearchLogListItem] = []
        report: CitationReadinessReport | None = None
        report_payload: dict[str, object] | None = None
        report_json = ""
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                latest_logs = list_search_logs(settings.database_url, limit=12)
                if search_log_id is not None:
                    report = build_citation_readiness_report(
                        settings.database_url,
                        CitationReadinessInput(
                            search_log_id=search_log_id,
                            max_context_chars=max_context_chars,
                            include_neighbors=include_neighbors,
                            max_items=max_items,
                        ),
                    )
                    if report is None:
                        error_message = "Search log citation readiness not found."
                    else:
                        report_payload = citation_readiness_report_payload(report)
                        report_json = json.dumps(
                            report_payload,
                            ensure_ascii=False,
                            indent=2,
                        )
            except (InvalidRetrievalContextError, InvalidSearchLogError) as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "citation_readiness.html",
            template_context(
                request,
                database_configured=bool(settings.database_url),
                latest_logs=latest_logs,
                selected_search_log_id=search_log_id,
                max_context_chars=max_context_chars,
                include_neighbors=include_neighbors,
                max_items=max_items,
                report=report,
                report_payload=report_payload,
                report_json=report_json,
                error_message=error_message,
            ),
        )

    @app.get("/generation", response_class=HTMLResponse)
    def generation_page(
        request: Request,
        search_log_id: int | None = Query(default=None, ge=1),
        generation_run_id: int | None = Query(default=None, ge=1),
        max_context_chars: int = Query(default=DEFAULT_CONTEXT_CHAR_BUDGET, ge=500, le=50000),
        include_neighbors: bool = True,
        max_items: int = Query(default=DEFAULT_CONTEXT_MAX_ITEMS, ge=1, le=100),
        generation_template_key: str | None = Query(default=None),
        generation_status: str | None = None,
        generation_error: str | None = None,
    ) -> HTMLResponse:
        latest_logs: list[SearchLogListItem] = []
        package: RetrievalContextPackage | None = None
        prompt_preview: GenerationPromptPackage | None = None
        selected_run: GenerationRunRecord | None = None
        selected_citations: tuple[GenerationRunCitationRecord, ...] = ()
        selected_template_completeness: GenerationTemplateCompletenessAssessment | None = None
        selected_docx_export_readiness: dict[str, object] | None = None
        default_generation_provider: dict[str, object] | None = None
        default_generation_runtime: dict[str, object] | None = None
        remote_generation_available = False
        actor_options: list[dict[str, object]] = []
        profile_options: list[str] = []
        chunk_policy_options: list[ChunkPolicySummaryRecord] = []
        document_summary_options: list[DocumentInventoryItem] = []
        bm25_tokenizer_options = [asdict(tokenizer) for tokenizer in list_bm25_tokenizers()]
        generation_template_options: tuple[GenerationTemplateRecord, ...] = ()
        summary_generation_template_options: tuple[GenerationTemplateRecord, ...] = ()
        selected_generation_template_key = ""
        default_summary_template_key = DEFAULT_DOCUMENT_SUMMARY_TEMPLATE_KEY
        error_message = generation_error

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                latest_logs = list_search_logs(settings.database_url, limit=12)
                actor_options = list_search_actor_options(settings.database_url)
                document_summary_options = [
                    document
                    for document in list_document_inventory(settings.database_url, limit=50)
                    if document.chunk_count > 0
                ]
                profile_options = [
                    profile.profile_name
                    for profile in list_active_embedding_profiles(settings.database_url)
                ]
                if BM25_SEARCH_PROFILE_NAME not in profile_options:
                    profile_options.append(BM25_SEARCH_PROFILE_NAME)
                if HYBRID_SEARCH_PROFILE_NAME not in profile_options:
                    profile_options.append(HYBRID_SEARCH_PROFILE_NAME)
                if RERANKED_SEARCH_PROFILE_NAME not in profile_options:
                    profile_options.append(RERANKED_SEARCH_PROFILE_NAME)
                chunk_policy_options = list_chunk_policy_summaries(settings.database_url)
                generation_template_options = list_generation_templates(settings.database_url)
                summary_generation_template_options = tuple(
                    template
                    for template in generation_template_options
                    if template.document_type == "summary"
                )
                if not summary_generation_template_options:
                    summary_generation_template_options = generation_template_options
                default_generation_template = next(
                    (template for template in generation_template_options if template.is_default),
                    None,
                )
                requested_generation_template_key = (
                    generation_template_key.strip() if generation_template_key else ""
                )
                selected_generation_template = (
                    get_generation_template_by_key(
                        settings.database_url,
                        requested_generation_template_key,
                    )
                    if requested_generation_template_key
                    else default_generation_template
                )
                if requested_generation_template_key and selected_generation_template is None:
                    error_message = "active generation template was not found"
                    selected_generation_template = default_generation_template
                selected_generation_template_key = (
                    selected_generation_template.template_key
                    if selected_generation_template is not None
                    else ""
                )
                default_summary_template_key = (
                    DEFAULT_DOCUMENT_SUMMARY_TEMPLATE_KEY
                    if any(
                        template.template_key == DEFAULT_DOCUMENT_SUMMARY_TEMPLATE_KEY
                        for template in summary_generation_template_options
                    )
                    else (
                        summary_generation_template_options[0].template_key
                        if summary_generation_template_options
                        else selected_generation_template_key
                    )
                )
                provider = get_generation_provider_config_for_mode(
                    settings.database_url,
                    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
                ) or get_default_generation_provider_config(settings.database_url)
                if provider is not None:
                    default_generation_provider = generation_provider_config_payload(provider)
                    default_generation_runtime = generation_provider_runtime_config_payload(
                        provider,
                        settings,
                    )
                    remote_generation_available = (
                        default_generation_runtime.get("valid") is True
                        and default_generation_runtime.get("mode") == "remote_openai_compatible"
                        and default_generation_runtime.get("api_key_configured") is not False
                    )
                if generation_run_id is not None:
                    selected_run = get_generation_run(settings.database_url, generation_run_id)
                    if selected_run is None:
                        error_message = "Generation run not found."
                    else:
                        selected_template_completeness = assess_generation_template_completeness(
                            selected_run
                        )
                        selected_docx_export_readiness = generation_docx_export_readiness_payload(
                            assess_generation_docx_export_readiness(
                                selected_run,
                                selected_template_completeness,
                            )
                        )
                        selected_citations = list_generation_run_citations(
                            settings.database_url,
                            selected_run.generation_run_id,
                        )
                        if search_log_id is None:
                            search_log_id = selected_run.search_log_id
                if search_log_id is not None:
                    package = build_retrieval_context_package(
                        settings.database_url,
                        RetrievalContextInput(
                            search_log_id=search_log_id,
                            max_context_chars=max_context_chars,
                            include_neighbors=include_neighbors,
                            max_items=max_items,
                        ),
                    )
                    if package is None and error_message is None:
                        error_message = "Search log retrieval context not found."
                    elif package is not None:
                        prompt_preview = build_generation_prompt_package(
                            package,
                            generation_template=selected_generation_template,
                        )
            except (
                InvalidChunkPolicyManagementError,
                InvalidEmbeddingJobError,
                InvalidGenerationRunError,
                InvalidGenerationPromptError,
                InvalidRetrievalContextError,
                InvalidSearchLogError,
            ) as exc:
                error_message = str(exc)

        direct_default_provider_mode = (
            GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE
            if remote_generation_available
            else GENERATION_PROVIDER_MODE_MOCK
        )
        default_actor_id = actor_options[0]["user_id"] if actor_options else ""
        default_profile_name = (
            RERANKED_SEARCH_PROFILE_NAME
            if RERANKED_SEARCH_PROFILE_NAME in profile_options
            else (profile_options[0] if profile_options else BM25_SEARCH_PROFILE_NAME)
        )

        return TEMPLATES.TemplateResponse(
            request,
            "generation.html",
            template_context(
                request,
                database_configured=bool(settings.database_url),
                latest_logs=latest_logs,
                selected_search_log_id=search_log_id,
                selected_generation_run_id=generation_run_id,
                max_context_chars=max_context_chars,
                include_neighbors=include_neighbors,
                max_items=max_items,
                package=package,
                prompt_preview=prompt_preview,
                selected_run=selected_run,
                selected_citations=selected_citations,
                selected_template_completeness=selected_template_completeness,
                selected_docx_export_readiness=selected_docx_export_readiness,
                default_generation_provider=default_generation_provider,
                default_generation_runtime=default_generation_runtime,
                remote_generation_available=remote_generation_available,
                actor_options=actor_options,
                default_actor_id=default_actor_id,
                search_scope_options=SEARCH_COMPARE_SCOPE_OPTIONS,
                direct_generation_provider_modes=(
                    GENERATION_PROVIDER_MODE_MOCK,
                    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
                ),
                direct_default_provider_mode=direct_default_provider_mode,
                profile_options=profile_options,
                default_profile_name=default_profile_name,
                search_file_type_options=SEARCH_COMPARE_FILE_TYPES,
                chunk_policy_options=chunk_policy_options,
                document_summary_options=document_summary_options,
                bm25_tokenizer_options=bm25_tokenizer_options,
                default_bm25_tokenizer_name=DEFAULT_BM25_TOKENIZER_NAME,
                generation_template_options=generation_template_options,
                summary_generation_template_options=summary_generation_template_options,
                selected_generation_template_key=selected_generation_template_key,
                default_summary_template_key=default_summary_template_key,
                default_document_summary_max_chunks=DEFAULT_DOCUMENT_SUMMARY_MAX_CHUNKS,
                generation_status=generation_status or "",
                error_message=error_message,
            ),
        )

    @app.post("/generation/direct-runs")
    def generation_direct_run_page(
        direct_query_text: str = Form(...),
        direct_actor_user_id: int = Form(...),
        direct_requested_search_scope: str = Form("company"),
        direct_provider_mode: str = Form(GENERATION_PROVIDER_MODE_MOCK),
        direct_generation_template_key: str = Form(""),
        direct_top_k: int = Form(5),
        direct_profile_name: str = Form(BM25_SEARCH_PROFILE_NAME),
        direct_chunk_policy_name: str = Form(""),
        direct_document_group: str = Form(""),
        direct_file_type: str = Form(""),
        direct_bm25_tokenizer_name: str = Form(DEFAULT_BM25_TOKENIZER_NAME),
        direct_max_context_chars: int = Form(DEFAULT_CONTEXT_CHAR_BUDGET),
        direct_include_neighbors: bool = Form(False),
        direct_max_items: int = Form(DEFAULT_CONTEXT_MAX_ITEMS),
    ) -> RedirectResponse:
        redirect_params: dict[str, object] = {
            "max_context_chars": direct_max_context_chars,
            "include_neighbors": str(direct_include_neighbors).lower(),
            "max_items": direct_max_items,
        }
        if direct_generation_template_key.strip():
            redirect_params["generation_template_key"] = direct_generation_template_key.strip()
        if not settings.database_url:
            redirect_params["generation_error"] = "NEX_PCX_DATABASE_URL is not configured."
            return RedirectResponse(
                _generation_redirect_url(redirect_params),
                status_code=status.HTTP_303_SEE_OTHER,
            )

        api_key: str | None = None
        try:
            provider_mode = direct_provider_mode.strip().lower()
            if provider_mode == GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE:
                provider = get_generation_provider_config_for_mode(
                    settings.database_url,
                    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
                )
                if provider is None:
                    raise InvalidGenerationRunError(
                        "active remote_openai_compatible generation provider config was not found"
                    )
                api_key = resolve_generation_provider_api_key(provider, settings)
            profile_name = direct_profile_name.strip()
            result = run_direct_generation_query(
                settings.database_url,
                DirectGenerationInput(
                    query_text=direct_query_text,
                    actor_user_id=direct_actor_user_id,
                    requested_search_scope=direct_requested_search_scope,
                    provider_mode=provider_mode,
                    generation_template_key=direct_generation_template_key.strip() or None,
                    top_k=direct_top_k,
                    profiles=(profile_name,) if profile_name else None,
                    chunk_policy_name=direct_chunk_policy_name.strip() or None,
                    document_group=direct_document_group.strip() or None,
                    file_type=direct_file_type.strip() or None,
                    bm25_tokenizer_name=(direct_bm25_tokenizer_name or DEFAULT_BM25_TOKENIZER_NAME),
                    allow_mock_fallback=True,
                    max_context_chars=direct_max_context_chars,
                    include_neighbors=direct_include_neighbors,
                    max_items=direct_max_items,
                ),
                fallback_runtime_config=embedding_provider_runtime_config_from_settings(settings),
                fallback_reranker_runtime_config=reranker_runtime_config_from_settings(settings),
                api_key=api_key,
            )
            redirect_params["search_log_id"] = result.search_result.search_log_id
            redirect_params["generation_run_id"] = result.generation_report.run.generation_run_id
            redirect_params["generation_status"] = "direct_created"
        except (
            InvalidDirectGenerationError,
            InvalidEmbeddingProviderError,
            InvalidGenerationProviderError,
            InvalidGenerationRunError,
            InvalidPermissionError,
            InvalidQueryEmbeddingError,
            InvalidRerankerError,
            InvalidRetrievalContextError,
            InvalidSearchCompareError,
            InvalidSearchLogError,
            InvalidVectorSearchError,
        ) as exc:
            redirect_params["generation_error"] = str(exc)

        return RedirectResponse(
            _generation_redirect_url(redirect_params),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.post("/generation/document-summaries")
    def generation_document_summary_page(
        summary_document_id: int = Form(...),
        summary_actor_user_id: int = Form(...),
        summary_instruction: str = Form(""),
        summary_provider_mode: str = Form(GENERATION_PROVIDER_MODE_MOCK),
        summary_generation_template_key: str = Form(DEFAULT_DOCUMENT_SUMMARY_TEMPLATE_KEY),
        summary_max_chunks: int = Form(DEFAULT_DOCUMENT_SUMMARY_MAX_CHUNKS),
        summary_max_context_chars: int = Form(DEFAULT_CONTEXT_CHAR_BUDGET),
        summary_include_neighbors: bool = Form(False),
        summary_chunk_policy_name: str = Form(""),
    ) -> RedirectResponse:
        redirect_params: dict[str, object] = {
            "max_context_chars": summary_max_context_chars,
            "include_neighbors": str(summary_include_neighbors).lower(),
            "max_items": summary_max_chunks,
        }
        if summary_generation_template_key.strip():
            redirect_params["generation_template_key"] = summary_generation_template_key.strip()
        if not settings.database_url:
            redirect_params["generation_error"] = "NEX_PCX_DATABASE_URL is not configured."
            return RedirectResponse(
                _generation_redirect_url(redirect_params),
                status_code=status.HTTP_303_SEE_OTHER,
            )

        api_key: str | None = None
        try:
            provider_mode = summary_provider_mode.strip().lower()
            if provider_mode == GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE:
                provider = get_generation_provider_config_for_mode(
                    settings.database_url,
                    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
                )
                if provider is None:
                    raise InvalidGenerationRunError(
                        "active remote_openai_compatible generation provider config was not found"
                    )
                api_key = resolve_generation_provider_api_key(provider, settings)
            result = run_document_summary_generation(
                settings.database_url,
                DocumentSummaryInput(
                    document_id=summary_document_id,
                    actor_user_id=summary_actor_user_id,
                    summary_instruction=summary_instruction,
                    provider_mode=provider_mode,
                    generation_template_key=summary_generation_template_key.strip() or None,
                    max_chunks=summary_max_chunks,
                    max_context_chars=summary_max_context_chars,
                    include_neighbors=summary_include_neighbors,
                    chunk_policy_name=summary_chunk_policy_name.strip() or None,
                ),
                api_key=api_key,
            )
            redirect_params["search_log_id"] = result.search_log_id
            redirect_params["generation_run_id"] = result.generation_report.run.generation_run_id
            redirect_params["generation_status"] = "document_summary_created"
        except (
            InvalidDocumentSummaryError,
            InvalidGenerationProviderError,
            InvalidGenerationRunError,
            InvalidRetrievalContextError,
            InvalidSearchLogError,
        ) as exc:
            redirect_params["generation_error"] = str(exc)

        return RedirectResponse(
            _generation_redirect_url(redirect_params),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.post("/generation/runs/mock")
    def generation_mock_run_page(
        search_log_id: int = Form(...),
        max_context_chars: int = Form(DEFAULT_CONTEXT_CHAR_BUDGET),
        include_neighbors: bool = Form(False),
        max_items: int = Form(DEFAULT_CONTEXT_MAX_ITEMS),
        generation_template_key: str = Form(""),
    ) -> RedirectResponse:
        redirect_params: dict[str, object] = {
            "search_log_id": search_log_id,
            "max_context_chars": max_context_chars,
            "include_neighbors": str(include_neighbors).lower(),
            "max_items": max_items,
        }
        if generation_template_key.strip():
            redirect_params["generation_template_key"] = generation_template_key.strip()
        if not settings.database_url:
            redirect_params["generation_error"] = "NEX_PCX_DATABASE_URL is not configured."
            return RedirectResponse(
                _generation_redirect_url(redirect_params),
                status_code=status.HTTP_303_SEE_OTHER,
            )

        try:
            package = build_retrieval_context_package(
                settings.database_url,
                RetrievalContextInput(
                    search_log_id=search_log_id,
                    max_context_chars=max_context_chars,
                    include_neighbors=include_neighbors,
                    max_items=max_items,
                ),
            )
            if package is None:
                redirect_params["generation_error"] = "Search log retrieval context not found."
            else:
                report = execute_mock_generation_run(
                    settings.database_url,
                    package,
                    generation_template_key=generation_template_key.strip() or None,
                    created_by="generation_ui_mock",
                )
                redirect_params["generation_run_id"] = report.run.generation_run_id
                redirect_params["generation_status"] = "created"
        except (
            InvalidGenerationRunError,
            InvalidRetrievalContextError,
            InvalidSearchLogError,
        ) as exc:
            redirect_params["generation_error"] = str(exc)

        return RedirectResponse(
            _generation_redirect_url(redirect_params),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.post("/generation/runs/remote")
    def generation_remote_run_page(
        search_log_id: int = Form(...),
        max_context_chars: int = Form(DEFAULT_CONTEXT_CHAR_BUDGET),
        include_neighbors: bool = Form(False),
        max_items: int = Form(DEFAULT_CONTEXT_MAX_ITEMS),
        generation_template_key: str = Form(""),
    ) -> RedirectResponse:
        redirect_params: dict[str, object] = {
            "search_log_id": search_log_id,
            "max_context_chars": max_context_chars,
            "include_neighbors": str(include_neighbors).lower(),
            "max_items": max_items,
        }
        if generation_template_key.strip():
            redirect_params["generation_template_key"] = generation_template_key.strip()
        if not settings.database_url:
            redirect_params["generation_error"] = "NEX_PCX_DATABASE_URL is not configured."
            return RedirectResponse(
                _generation_redirect_url(redirect_params),
                status_code=status.HTTP_303_SEE_OTHER,
            )

        try:
            package = build_retrieval_context_package(
                settings.database_url,
                RetrievalContextInput(
                    search_log_id=search_log_id,
                    max_context_chars=max_context_chars,
                    include_neighbors=include_neighbors,
                    max_items=max_items,
                ),
            )
            if package is None:
                redirect_params["generation_error"] = "Search log retrieval context not found."
            else:
                provider = get_generation_provider_config_for_mode(
                    settings.database_url,
                    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
                )
                if provider is None:
                    raise InvalidGenerationRunError(
                        "active remote_openai_compatible generation provider config was not found"
                    )
                api_key = resolve_generation_provider_api_key(provider, settings)
                report = execute_remote_generation_run(
                    settings.database_url,
                    package,
                    generation_template_key=generation_template_key.strip() or None,
                    api_key=api_key,
                    created_by="generation_ui_remote",
                )
                redirect_params["generation_run_id"] = report.run.generation_run_id
                redirect_params["generation_status"] = "remote_created"
        except (
            InvalidGenerationProviderError,
            InvalidGenerationRunError,
            InvalidRetrievalContextError,
            InvalidSearchLogError,
        ) as exc:
            redirect_params["generation_error"] = str(exc)

        return RedirectResponse(
            _generation_redirect_url(redirect_params),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.get("/generation/runs", response_class=HTMLResponse)
    def generation_run_history_page(
        request: Request,
        limit: int = Query(default=DEFAULT_GENERATION_RUN_HISTORY_LIMIT),
        answer_quality_status: str = Query(default=GENERATION_RUN_HISTORY_FILTER_ALL),
        provider_mode: str = Query(default=GENERATION_RUN_HISTORY_FILTER_ALL),
        run_status: str = Query(default=GENERATION_RUN_HISTORY_FILTER_ALL),
    ) -> HTMLResponse:
        history: GenerationRunHistory | None = None
        history_json = ""
        error_message: str | None = None

        history_filter = GenerationRunHistoryFilter(
            limit=limit,
            answer_quality_status=answer_quality_status,
            provider_mode=provider_mode,
            run_status=run_status,
        )
        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                history = list_generation_run_history(
                    settings.database_url,
                    history_filter=history_filter,
                )
                history_json = json.dumps(
                    generation_run_history_payload(history),
                    ensure_ascii=False,
                    indent=2,
                )
                history_filter = history.filters
            except InvalidGenerationRunError as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "generation_run_history.html",
            template_context(
                request,
                database_configured=bool(settings.database_url),
                history=history,
                history_json=history_json,
                selected_limit=history_filter.limit,
                selected_answer_quality_status=history_filter.answer_quality_status,
                selected_provider_mode=history_filter.provider_mode,
                selected_run_status=history_filter.run_status,
                max_limit=MAX_GENERATION_RUN_HISTORY_LIMIT,
                answer_quality_status_options=(
                    GENERATION_RUN_HISTORY_FILTER_ALL,
                    "passed",
                    "warning",
                    "failed",
                    "not_evaluated",
                    GENERATION_ANSWER_QUALITY_NOT_AVAILABLE,
                ),
                provider_mode_options=(
                    GENERATION_RUN_HISTORY_FILTER_ALL,
                    "mock",
                    "remote_openai_compatible",
                ),
                run_status_options=(
                    GENERATION_RUN_HISTORY_FILTER_ALL,
                    "succeeded",
                    "failed",
                    "no_answer",
                    "blocked",
                    "running",
                    "pending",
                    "canceled",
                ),
                error_message=error_message,
            ),
        )

    @app.get("/generation/document-summaries", response_class=HTMLResponse)
    def document_summary_history_page(
        request: Request,
        limit: int = Query(default=DEFAULT_DOCUMENT_SUMMARY_HISTORY_LIMIT),
        run_status: str = Query(default=GENERATION_RUN_HISTORY_FILTER_ALL),
        generation_template_key: str = Query(default=GENERATION_RUN_HISTORY_FILTER_ALL),
    ) -> HTMLResponse:
        history: DocumentSummaryHistory | None = None
        history_json = ""
        summary_template_options: tuple[GenerationTemplateRecord, ...] = ()
        error_message: str | None = None

        history_filter = DocumentSummaryHistoryFilter(
            limit=limit,
            run_status=run_status,
            generation_template_key=generation_template_key,
        )
        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                summary_template_options = tuple(
                    template
                    for template in list_generation_templates(settings.database_url)
                    if template.document_type == "summary"
                )
                history = list_document_summary_history(
                    settings.database_url,
                    history_filter=history_filter,
                )
                history_json = json.dumps(
                    document_summary_history_payload(history),
                    ensure_ascii=False,
                    indent=2,
                )
                history_filter = history.filters
            except (InvalidDocumentSummaryError, InvalidGenerationTemplateError) as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "document_summary_history.html",
            template_context(
                request,
                database_configured=bool(settings.database_url),
                history=history,
                history_json=history_json,
                selected_limit=history_filter.limit,
                selected_run_status=history_filter.run_status,
                selected_generation_template_key=history_filter.generation_template_key,
                max_limit=MAX_DOCUMENT_SUMMARY_HISTORY_LIMIT,
                summary_template_options=summary_template_options,
                run_status_options=(
                    GENERATION_RUN_HISTORY_FILTER_ALL,
                    "succeeded",
                    "failed",
                    "no_answer",
                    "blocked",
                    "running",
                    "pending",
                    "canceled",
                ),
                error_message=error_message,
            ),
        )

    @app.get("/generation/runs/{generation_run_id}", response_class=HTMLResponse)
    def generation_run_detail_page(
        request: Request,
        generation_run_id: int,
    ) -> HTMLResponse:
        selected_run: GenerationRunRecord | None = None
        selected_citations: tuple[GenerationRunCitationRecord, ...] = ()
        selected_template_completeness: GenerationTemplateCompletenessAssessment | None = None
        selected_docx_export_readiness: dict[str, object] | None = None
        error_message: str | None = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                selected_run = get_generation_run(settings.database_url, generation_run_id)
                if selected_run is None:
                    error_message = "Generation run not found."
                else:
                    selected_template_completeness = assess_generation_template_completeness(
                        selected_run
                    )
                    selected_docx_export_readiness = generation_docx_export_readiness_payload(
                        assess_generation_docx_export_readiness(
                            selected_run,
                            selected_template_completeness,
                        )
                    )
                    selected_citations = list_generation_run_citations(
                        settings.database_url,
                        selected_run.generation_run_id,
                    )
            except InvalidGenerationRunError as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "generation_run_detail.html",
            template_context(
                request,
                database_configured=bool(settings.database_url),
                selected_run=selected_run,
                selected_citations=selected_citations,
                selected_template_completeness=selected_template_completeness,
                selected_docx_export_readiness=selected_docx_export_readiness,
                error_message=error_message,
            ),
        )

    @app.get("/admin/generation-templates", response_class=HTMLResponse)
    def generation_templates_page(
        request: Request,
        include_inactive: bool = True,
        template_key: str | None = None,
        new_template: bool = False,
        saved_template: str | None = None,
        cloned_template: str | None = None,
        rolled_back_template: str | None = None,
        template_error: str | None = None,
    ) -> HTMLResponse:
        templates: tuple[GenerationTemplateRecord, ...] = ()
        collection_payload: dict[str, object] | None = None
        collection_json = ""
        selected_template: GenerationTemplateRecord | None = None
        selected_template_payload: dict[str, object] | None = None
        clone_target_template_key = ""
        clone_target_template_version = "v2"
        clone_target_template_name = ""
        section_schema_json = "[]"
        style_guidance_json = "{}"
        citation_policy_json = "{}"
        error_message = template_error

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                templates = list_generation_templates(
                    settings.database_url,
                    include_inactive=include_inactive,
                )
                collection_payload = generation_template_collection_payload(
                    templates,
                    include_inactive=include_inactive,
                )
                collection_json = json.dumps(collection_payload, ensure_ascii=False, indent=2)
                if template_key and not new_template:
                    selected_template = get_generation_template_by_key(
                        settings.database_url,
                        template_key,
                        include_inactive=True,
                    )
                    if selected_template is None:
                        error_message = "Generation template not found."
                if selected_template is None and templates and not new_template:
                    selected_template = next(
                        (template for template in templates if template.is_default),
                        templates[0],
                    )
                if selected_template is not None:
                    selected_template_payload = generation_template_payload(selected_template)
                    section_schema_json = json.dumps(
                        selected_template_payload["section_schema"],
                        ensure_ascii=False,
                        indent=2,
                    )
                    style_guidance_json = json.dumps(
                        selected_template_payload["style_guidance"],
                        ensure_ascii=False,
                        indent=2,
                    )
                    citation_policy_json = json.dumps(
                        selected_template_payload["citation_policy"],
                        ensure_ascii=False,
                        indent=2,
                    )
                    clone_target_template_key = suggest_generation_template_clone_key(
                        selected_template
                    )
                    clone_target_template_version = suggest_generation_template_next_version(
                        selected_template
                    )
                    clone_target_template_name = (
                        f"{selected_template.template_name} {clone_target_template_version}"
                    )
            except InvalidGenerationTemplateError as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "generation_templates.html",
            template_context(
                request,
                database_configured=bool(settings.database_url),
                templates=templates,
                collection_payload=collection_payload,
                collection_json=collection_json,
                include_inactive=include_inactive,
                new_template=new_template,
                selected_template=selected_template,
                selected_template_payload=selected_template_payload,
                section_schema_json=section_schema_json,
                style_guidance_json=style_guidance_json,
                citation_policy_json=citation_policy_json,
                clone_target_template_key=clone_target_template_key,
                clone_target_template_version=clone_target_template_version,
                clone_target_template_name=clone_target_template_name,
                saved_template=saved_template,
                cloned_template=cloned_template,
                rolled_back_template=rolled_back_template,
                error_message=error_message,
                document_type_options=tuple(sorted(GENERATION_TEMPLATE_DOCUMENT_TYPES)),
                template_language_options=tuple(sorted(GENERATION_TEMPLATE_LANGUAGES)),
                output_format_options=(GENERATION_TEMPLATE_OUTPUT_FORMAT_MARKDOWN,),
            ),
        )

    @app.post("/admin/generation-templates/upsert")
    def generation_templates_upsert_action(
        template_key: str = Form(...),
        template_name: str = Form(...),
        template_version: str = Form("v1"),
        template_family: str = Form(""),
        document_type: str = Form("grounded_answer"),
        language: str = Form("ko"),
        output_format: str = Form("markdown"),
        section_schema: str = Form("[]"),
        system_instruction: str = Form(...),
        user_instruction_suffix: str = Form(""),
        style_guidance: str = Form("{}"),
        citation_policy: str = Form("{}"),
        change_note: str = Form(""),
        is_default: bool = Form(False),
        is_active: bool = Form(False),
    ) -> RedirectResponse:
        query_params: dict[str, object]
        if not settings.database_url:
            query_params = {"template_error": "NEX_PCX_DATABASE_URL is not configured."}
        else:
            try:
                template = upsert_generation_template(
                    settings.database_url,
                    GenerationTemplateInput(
                        template_key=template_key,
                        template_family=template_family or None,
                        template_name=template_name,
                        template_version=template_version,
                        document_type=document_type,
                        language=language,
                        output_format=output_format,
                        section_schema=_generation_template_json_field(
                            section_schema,
                            "section_schema",
                            default_json="[]",
                        ),
                        system_instruction=system_instruction,
                        user_instruction_suffix=user_instruction_suffix,
                        style_guidance=_generation_template_json_field(
                            style_guidance,
                            "style_guidance",
                            default_json="{}",
                        ),
                        citation_policy=_generation_template_json_field(
                            citation_policy,
                            "citation_policy",
                            default_json="{}",
                        ),
                        change_note=change_note,
                        is_default=is_default,
                        is_active=is_active,
                        created_by="generation-template-ui",
                    ),
                )
                query_params = {
                    "template_key": template.template_key,
                    "saved_template": template.template_key,
                    "include_inactive": "true",
                }
            except InvalidGenerationTemplateError as exc:
                query_params = {
                    "template_key": template_key,
                    "template_error": str(exc),
                    "include_inactive": "true",
                }

        return RedirectResponse(
            _generation_templates_redirect_url(query_params),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.post("/admin/generation-templates/{template_key}/clone")
    def generation_templates_clone_action(
        template_key: str,
        target_template_key: str = Form(...),
        target_template_version: str = Form(...),
        target_template_name: str = Form(""),
        change_note: str = Form(""),
        make_default: bool = Form(False),
        is_active: bool = Form(False),
        include_inactive: bool = Form(True),
    ) -> RedirectResponse:
        query_params: dict[str, object]
        if not settings.database_url:
            query_params = {"template_error": "NEX_PCX_DATABASE_URL is not configured."}
        else:
            try:
                template = clone_generation_template_version(
                    settings.database_url,
                    GenerationTemplateCloneInput(
                        source_template_key=template_key,
                        target_template_key=target_template_key,
                        target_template_version=target_template_version,
                        target_template_name=target_template_name or None,
                        make_default=make_default,
                        is_active=is_active,
                        change_note=change_note,
                        created_by="generation-template-ui",
                    ),
                )
                if template is None:
                    query_params = {
                        "template_key": template_key,
                        "template_error": "Generation template not found.",
                    }
                else:
                    query_params = {
                        "template_key": template.template_key,
                        "cloned_template": template.template_key,
                    }
            except InvalidGenerationTemplateError as exc:
                query_params = {
                    "template_key": template_key,
                    "template_error": str(exc),
                }
        query_params["include_inactive"] = str(include_inactive).lower()
        return RedirectResponse(
            _generation_templates_redirect_url(query_params),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.post("/admin/generation-templates/{template_key}/active")
    def generation_templates_active_action(
        template_key: str,
        is_active: bool = Form(False),
        include_inactive: bool = Form(True),
    ) -> RedirectResponse:
        query_params: dict[str, object]
        if not settings.database_url:
            query_params = {"template_error": "NEX_PCX_DATABASE_URL is not configured."}
        else:
            try:
                template = set_generation_template_active(
                    settings.database_url,
                    template_key,
                    is_active=is_active,
                )
                if template is None:
                    query_params = {"template_error": "Generation template not found."}
                else:
                    query_params = {
                        "template_key": template.template_key,
                        "saved_template": template.template_key,
                    }
            except InvalidGenerationTemplateError as exc:
                query_params = {"template_key": template_key, "template_error": str(exc)}
        query_params["include_inactive"] = str(include_inactive).lower()
        return RedirectResponse(
            _generation_templates_redirect_url(query_params),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.post("/admin/generation-templates/{template_key}/default")
    def generation_templates_default_action(
        template_key: str,
        include_inactive: bool = Form(True),
    ) -> RedirectResponse:
        query_params: dict[str, object]
        if not settings.database_url:
            query_params = {"template_error": "NEX_PCX_DATABASE_URL is not configured."}
        else:
            try:
                template = set_generation_template_default(settings.database_url, template_key)
                if template is None:
                    query_params = {"template_error": "Generation template not found."}
                else:
                    query_params = {
                        "template_key": template.template_key,
                        "saved_template": template.template_key,
                    }
            except InvalidGenerationTemplateError as exc:
                query_params = {"template_key": template_key, "template_error": str(exc)}
        query_params["include_inactive"] = str(include_inactive).lower()
        return RedirectResponse(
            _generation_templates_redirect_url(query_params),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.post("/admin/generation-templates/{template_key}/rollback")
    def generation_templates_rollback_action(
        template_key: str,
        include_inactive: bool = Form(True),
    ) -> RedirectResponse:
        query_params: dict[str, object]
        if not settings.database_url:
            query_params = {"template_error": "NEX_PCX_DATABASE_URL is not configured."}
        else:
            try:
                template = rollback_generation_template_version(settings.database_url, template_key)
                if template is None:
                    query_params = {"template_error": "Generation template not found."}
                else:
                    query_params = {
                        "template_key": template.template_key,
                        "rolled_back_template": template.template_key,
                    }
            except InvalidGenerationTemplateError as exc:
                query_params = {"template_key": template_key, "template_error": str(exc)}
        query_params["include_inactive"] = str(include_inactive).lower()
        return RedirectResponse(
            _generation_templates_redirect_url(query_params),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.get("/admin/generation-provider-metrics", response_class=HTMLResponse)
    def generation_provider_metrics_page(
        request: Request,
        limit: int = DEFAULT_GENERATION_PROVIDER_METRIC_SNAPSHOT_LIMIT,
    ) -> HTMLResponse:
        snapshot = None
        snapshot_json = ""
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                snapshot = get_generation_provider_metric_snapshot(
                    settings.database_url,
                    limit=limit,
                )
                snapshot_json = json.dumps(
                    generation_provider_metric_snapshot_payload(snapshot),
                    ensure_ascii=False,
                    indent=2,
                )
            except InvalidGenerationProviderMetricSnapshotError as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "generation_provider_metrics.html",
            template_context(
                request,
                database_configured=bool(settings.database_url),
                snapshot=snapshot,
                snapshot_json=snapshot_json,
                selected_limit=limit,
                max_limit=MAX_GENERATION_PROVIDER_METRIC_SNAPSHOT_LIMIT,
                error_message=error_message,
            ),
        )

    @app.get("/admin/generation-provider-configs", response_class=HTMLResponse)
    def generation_provider_configs_page(
        request: Request,
        include_inactive: bool = True,
        seeded_provider: str | None = None,
        seed_error: str | None = None,
    ) -> HTMLResponse:
        providers: tuple[GenerationProviderConfigRecord, ...] = ()
        collection_payload: dict[str, object] | None = None
        collection_json = ""
        error_message = seed_error

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                providers = list_generation_provider_configs(
                    settings.database_url,
                    include_inactive=include_inactive,
                )
                collection_payload = generation_provider_config_collection_payload(
                    providers,
                    settings,
                    include_inactive=include_inactive,
                )
                collection_json = json.dumps(collection_payload, ensure_ascii=False, indent=2)
            except InvalidGenerationRunError as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "generation_provider_configs.html",
            template_context(
                request,
                database_configured=bool(settings.database_url),
                providers=providers,
                collection_payload=collection_payload,
                collection_json=collection_json,
                include_inactive=include_inactive,
                seeded_provider=seeded_provider,
                error_message=error_message,
                dgx_defaults={
                    "provider_name": DGX_VLLM_GENERATION_PROVIDER_NAME,
                    "provider_base_url": DGX_VLLM_GENERATION_BASE_URL,
                    "model_id": DGX_VLLM_GENERATION_MODEL_ID,
                    "api_key_env": DGX_VLLM_GENERATION_API_KEY_ENV,
                    "request_timeout_seconds": DGX_VLLM_GENERATION_TIMEOUT_SECONDS,
                    "max_tokens": DGX_VLLM_GENERATION_MAX_TOKENS,
                    "temperature": DGX_VLLM_GENERATION_TEMPERATURE,
                    "top_p": DGX_VLLM_GENERATION_TOP_P,
                },
            ),
        )

    @app.post("/admin/generation-provider-configs/seed-dgx-vllm")
    def generation_provider_configs_seed_dgx_vllm_action(
        provider_name: str = Form(DGX_VLLM_GENERATION_PROVIDER_NAME),
        provider_base_url: str = Form(DGX_VLLM_GENERATION_BASE_URL),
        model_id: str = Form(DGX_VLLM_GENERATION_MODEL_ID),
        api_key_env: str = Form(DGX_VLLM_GENERATION_API_KEY_ENV),
        request_timeout_seconds: int = Form(DGX_VLLM_GENERATION_TIMEOUT_SECONDS),
        max_tokens: int = Form(DGX_VLLM_GENERATION_MAX_TOKENS),
        temperature: float = Form(DGX_VLLM_GENERATION_TEMPERATURE),
        top_p: float = Form(DGX_VLLM_GENERATION_TOP_P),
        is_default: bool = Form(False),
        is_active: bool = Form(True),
        thinking_disabled: bool = Form(True),
    ) -> RedirectResponse:
        query_params: dict[str, object]
        if not settings.database_url:
            query_params = {"seed_error": "NEX_PCX_DATABASE_URL is not configured."}
        else:
            try:
                provider = seed_dgx_vllm_generation_provider_config(
                    settings.database_url,
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
                    created_by="generation-provider-config-ui",
                )
                query_params = {"seeded_provider": provider.provider_name}
            except InvalidGenerationRunError as exc:
                query_params = {"seed_error": str(exc)}

        return RedirectResponse(
            url=f"/admin/generation-provider-configs?{urlencode(query_params)}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.get("/search/experiments", response_class=HTMLResponse)
    def search_experiments_page(
        request: Request,
        experiment_run_id: int | None = None,
        golden_batch_key: str | None = None,
        base_metric_snapshot_id: int | None = None,
        target_metric_snapshot_id: int | None = None,
        status_filter: str | None = None,
        limit: int = 50,
    ) -> HTMLResponse:
        runs: list[SearchExperimentRunRecord] = []
        selected_detail: SearchExperimentRunDetail | None = None
        golden_batches: list[GoldenSearchExperimentBatchSummary] = []
        selected_golden_batch_detail: GoldenSearchExperimentBatchDetail | None = None
        selected_golden_batch_metric_summary: GoldenSearchExperimentBatchMetricSummary | None = None
        selected_golden_batch_metric_snapshots: list[GoldenBatchMetricSnapshotRecord] = []
        selected_golden_batch_metric_snapshot_comparison: (
            GoldenBatchMetricSnapshotComparison | None
        ) = None
        selected_golden_batch_metric_snapshot_trend: GoldenBatchMetricSnapshotTrend | None = None
        selected_base_metric_snapshot_id = base_metric_snapshot_id
        selected_target_metric_snapshot_id = target_metric_snapshot_id
        selected_golden_batch_metric_snapshot_compare_rows: list[dict[str, str]] = []
        selected_golden_batch_metric_snapshot_trend_rows: list[dict[str, str]] = []
        selected_golden_batch_metric_snapshot_trend_summary_rows: list[dict[str, str]] = []
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                golden_batches = list_golden_search_experiment_batch_summaries(
                    settings.database_url,
                    limit=10,
                )
                selected_golden_batch_key = golden_batch_key
                if selected_golden_batch_key is None and golden_batches:
                    selected_golden_batch_key = golden_batches[0].batch_key
                if selected_golden_batch_key:
                    selected_golden_batch_detail = get_golden_search_experiment_batch_detail(
                        settings.database_url,
                        selected_golden_batch_key,
                    )
                    if selected_golden_batch_detail is None:
                        error_message = "Golden search experiment batch not found."
                    else:
                        selected_golden_batch_metric_summary = (
                            get_golden_search_experiment_batch_metric_summary(
                                settings.database_url,
                                selected_golden_batch_key,
                            )
                        )
                        selected_golden_batch_metric_snapshots = list_golden_batch_metric_snapshots(
                            settings.database_url,
                            selected_golden_batch_key,
                            limit=5,
                        )
                        selected_golden_batch_metric_snapshot_trend = (
                            get_golden_batch_metric_snapshot_trend(
                                settings.database_url,
                                selected_golden_batch_key,
                                limit=10,
                            )
                        )
                        selected_golden_batch_metric_snapshot_trend_rows = (
                            golden_batch_metric_snapshot_trend_ui_rows(
                                selected_golden_batch_metric_snapshot_trend
                            )
                        )
                        selected_golden_batch_metric_snapshot_trend_summary_rows = (
                            golden_batch_metric_snapshot_trend_summary_ui_rows(
                                selected_golden_batch_metric_snapshot_trend
                            )
                        )
                        if len(selected_golden_batch_metric_snapshots) >= 2:
                            if selected_target_metric_snapshot_id is None:
                                selected_target_metric_snapshot_id = (
                                    selected_golden_batch_metric_snapshots[0].snapshot_id
                                )
                            if selected_base_metric_snapshot_id is None:
                                selected_base_metric_snapshot_id = (
                                    selected_golden_batch_metric_snapshots[1].snapshot_id
                                )
                            if (
                                selected_base_metric_snapshot_id is not None
                                and selected_target_metric_snapshot_id is not None
                                and selected_base_metric_snapshot_id
                                != selected_target_metric_snapshot_id
                            ):
                                selected_golden_batch_metric_snapshot_comparison = (
                                    compare_golden_batch_metric_snapshots(
                                        settings.database_url,
                                        base_snapshot_id=selected_base_metric_snapshot_id,
                                        target_snapshot_id=selected_target_metric_snapshot_id,
                                    )
                                )
                                if selected_golden_batch_metric_snapshot_comparison is None:
                                    error_message = (
                                        "Golden batch metric snapshot comparison target "
                                        "not found."
                                    )
                                else:
                                    selected_golden_batch_metric_snapshot_compare_rows = (
                                        golden_batch_metric_snapshot_compare_ui_rows(
                                            selected_golden_batch_metric_snapshot_comparison
                                        )
                                    )
                runs = list_search_experiment_runs(
                    settings.database_url,
                    status=status_filter.strip() if status_filter else None,
                    limit=limit,
                )
                selected_run_id = experiment_run_id
                if selected_run_id is None and runs:
                    selected_run_id = runs[0].experiment_run_id
                if selected_run_id is not None:
                    selected_detail = get_search_experiment_run_detail(
                        settings.database_url,
                        selected_run_id,
                    )
                    if selected_detail is None:
                        error_message = f"Search experiment run not found: {selected_run_id}"
            except (InvalidSearchExperimentError, ValueError) as exc:
                error_message = str(exc)
            except Exception as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "search_experiments.html",
            template_context(
                request,
                database_configured=bool(settings.database_url),
                experiments=runs,
                selected_detail=selected_detail,
                golden_batches=golden_batches,
                selected_golden_batch_detail=selected_golden_batch_detail,
                selected_golden_batch_metric_summary=selected_golden_batch_metric_summary,
                selected_golden_batch_metric_snapshots=selected_golden_batch_metric_snapshots,
                selected_golden_batch_metric_snapshot_comparison=(
                    selected_golden_batch_metric_snapshot_comparison
                ),
                selected_golden_batch_metric_snapshot_compare_rows=(
                    selected_golden_batch_metric_snapshot_compare_rows
                ),
                selected_golden_batch_metric_snapshot_trend=(
                    selected_golden_batch_metric_snapshot_trend
                ),
                selected_golden_batch_metric_snapshot_trend_rows=(
                    selected_golden_batch_metric_snapshot_trend_rows
                ),
                selected_golden_batch_metric_snapshot_trend_summary_rows=(
                    selected_golden_batch_metric_snapshot_trend_summary_rows
                ),
                selected_base_metric_snapshot_id=selected_base_metric_snapshot_id,
                selected_target_metric_snapshot_id=selected_target_metric_snapshot_id,
                selected_golden_batch_key=(
                    selected_golden_batch_detail.summary.batch_key
                    if selected_golden_batch_detail
                    else golden_batch_key
                ),
                selected_experiment_run_id=(
                    selected_detail.run.experiment_run_id if selected_detail else experiment_run_id
                ),
                selected_status_filter=status_filter or "",
                selected_limit=limit,
                status_options=tuple(sorted(SEARCH_EXPERIMENT_RUN_STATUSES)),
                error_message=error_message,
            ),
        )

    @app.get("/search/logs", response_class=HTMLResponse)
    def search_logs_page(
        request: Request,
        actor_user_id: str | None = None,
        requested_search_scope: str | None = None,
        document_group: str | None = None,
        provider_mode_filter: str | None = None,
        fingerprint: str | None = None,
        search_log_id: int | None = None,
        compare_search_log_id: int | None = None,
        operations_lookback_hours: int = 24,
        operations_min_total_elapsed_ms: int = 1000,
        limit: int = 50,
    ) -> HTMLResponse:
        actor_options: list[dict[str, object]] = []
        question_sets: list[GoldenQuestionSetRecord] = []
        logs: list[SearchLogListItem] = []
        runtime_failures: list[SearchRuntimeFailureRecord] = []
        latency_outliers: list[SearchLatencyOutlierRecord] = []
        no_result_logs: list[SearchNoResultRecord] = []
        duplicate_fingerprints: list[SearchDuplicateFingerprintRecord] = []
        operations_summary: SearchOperationsSummaryRecord | None = None
        selected_log: SearchLogDetailRecord | None = None
        selected_log_comparison: dict[str, object] | None = None
        retention_settings = SearchLogRetentionSettings()
        comparison_error_message = None
        error_message = None
        actor_user_id_value: int | None = None
        scope_value = requested_search_scope.strip() if requested_search_scope else None
        document_group_value = document_group.strip() if document_group else None
        provider_mode_filter_value = provider_mode_filter.strip() if provider_mode_filter else None
        fingerprint_value = normalize_search_fingerprint(fingerprint)
        if scope_value == "":
            scope_value = None
        if document_group_value == "":
            document_group_value = None
        if provider_mode_filter_value == "":
            provider_mode_filter_value = None

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
                runtime_failures = list_search_runtime_failures(
                    settings.database_url,
                    limit=8,
                )
                latency_outliers = list_search_latency_outliers(
                    settings.database_url,
                    min_total_elapsed_ms=1000,
                    limit=8,
                )
                no_result_logs = list_search_no_result_logs(
                    settings.database_url,
                    limit=8,
                )
                operations_summary = get_search_operations_summary(
                    settings.database_url,
                    lookback_hours=operations_lookback_hours,
                    min_total_elapsed_ms=operations_min_total_elapsed_ms,
                )
                duplicate_fingerprints = list_search_duplicate_fingerprints(
                    settings.database_url,
                    min_count=2,
                    limit=8,
                )
                logs = list_search_logs(
                    settings.database_url,
                    actor_user_id=actor_user_id_value,
                    requested_search_scope=scope_value,
                    document_group=document_group_value,
                    provider_mode_filter=provider_mode_filter_value,
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
                runtime_failures=runtime_failures,
                latency_outliers=latency_outliers,
                no_result_logs=no_result_logs,
                duplicate_fingerprints=duplicate_fingerprints,
                operations_summary=operations_summary,
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
                selected_provider_mode_filter=provider_mode_filter_value or "",
                selected_fingerprint=fingerprint_value or "",
                selected_search_log_id=search_log_id,
                selected_compare_search_log_id=compare_search_log_id or "",
                selected_operations_lookback_hours=operations_lookback_hours,
                selected_operations_min_total_elapsed_ms=operations_min_total_elapsed_ms,
                operations_lookback_options=(
                    {"hours": 1, "label": "1h"},
                    {"hours": 24, "label": "24h"},
                    {"hours": 168, "label": "7d"},
                    {"hours": 720, "label": "30d"},
                ),
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

    @app.get("/admin/embedding-coverage", response_class=HTMLResponse)
    def embedding_coverage_page(
        request: Request,
        parse_status: str | None = None,
        document_group: str | None = None,
        profile_name: str | None = None,
        limit: int = 100,
    ) -> HTMLResponse:
        matrix: EmbeddingCoverageMatrix | None = None
        profiles: list[EmbeddingProfileRecord] = []
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                profiles = list_active_embedding_profiles(settings.database_url)
                matrix = get_embedding_coverage_matrix(
                    settings.database_url,
                    parse_status=parse_status,
                    document_group=document_group,
                    profile_name=profile_name,
                    limit=limit,
                )
            except InvalidEmbeddingCoverageError as exc:
                error_message = str(exc)
            except InvalidEmbeddingJobError as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "embedding_coverage.html",
            template_context(
                request,
                matrix=matrix,
                profiles=profiles,
                selected_parse_status=parse_status or "",
                selected_document_group=document_group or "",
                selected_profile_name=profile_name or "",
                selected_limit=limit,
                error_message=error_message,
                database_configured=bool(settings.database_url),
            ),
        )

    @app.get("/admin/bm25-index-coverage", response_class=HTMLResponse)
    def bm25_index_coverage_page(
        request: Request,
        parse_status: str | None = None,
        document_group: str | None = None,
        chunk_policy_name: str | None = None,
        tokenizer_name: str = DEFAULT_BM25_TOKENIZER_NAME,
        limit: int = 100,
        backfill_status: str | None = None,
        backfill_tokenizer_name: str | None = None,
        backfill_policy_count: int | None = None,
        backfill_succeeded_count: int | None = None,
        backfill_failed_count: int | None = None,
        backfill_error: str | None = None,
    ) -> HTMLResponse:
        matrix: BM25IndexCoverageMatrix | None = None
        policies: list[ChunkPolicySummaryRecord] = []
        error_message = None
        bm25_tokenizers = list_bm25_tokenizers()
        selected_tokenizer_available = next(
            (
                tokenizer.available
                for tokenizer in bm25_tokenizers
                if tokenizer.tokenizer_name == tokenizer_name
            ),
            True,
        )

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                policies = list_chunk_policy_summaries(settings.database_url)
                matrix = get_bm25_index_coverage_matrix(
                    settings.database_url,
                    parse_status=parse_status,
                    document_group=document_group,
                    chunk_policy_name=chunk_policy_name,
                    tokenizer_name=tokenizer_name,
                    limit=limit,
                )
            except (
                InvalidBM25IndexCoverageError,
                InvalidChunkPolicyManagementError,
            ) as exc:
                error_message = str(exc)

        backfill_feedback = None
        if backfill_status:
            backfill_feedback = {
                "status": backfill_status,
                "tokenizer_name": backfill_tokenizer_name or tokenizer_name,
                "policy_count": backfill_policy_count,
                "succeeded_count": backfill_succeeded_count,
                "failed_count": backfill_failed_count,
                "error": backfill_error,
            }

        return TEMPLATES.TemplateResponse(
            request,
            "bm25_index_coverage.html",
            template_context(
                request,
                matrix=matrix,
                policies=policies,
                bm25_tokenizer_options=[asdict(tokenizer) for tokenizer in bm25_tokenizers],
                selected_tokenizer_available=selected_tokenizer_available,
                selected_parse_status=parse_status or "",
                selected_document_group=document_group or "",
                selected_chunk_policy_name=chunk_policy_name or "",
                selected_tokenizer_name=tokenizer_name,
                selected_limit=limit,
                backfill_feedback=backfill_feedback,
                error_message=error_message,
                database_configured=bool(settings.database_url),
            ),
        )

    @app.post("/admin/bm25-index-coverage/backfill")
    def bm25_index_coverage_backfill_page(
        parse_status: str | None = Form(default=None),
        document_group: str | None = Form(default=None),
        chunk_policy_name: str | None = Form(default=None),
        tokenizer_name: str = Form(default=DEFAULT_BM25_TOKENIZER_NAME),
        limit: int = Form(default=100),
    ) -> RedirectResponse:
        redirect_params: dict[str, object] = {
            "parse_status": parse_status,
            "document_group": document_group,
            "chunk_policy_name": chunk_policy_name,
            "tokenizer_name": tokenizer_name,
            "limit": limit,
        }
        if not settings.database_url:
            redirect_params.update(
                {
                    "backfill_status": "failed",
                    "backfill_tokenizer_name": tokenizer_name,
                    "backfill_error": "NEX_PCX_DATABASE_URL is not configured.",
                }
            )
            return RedirectResponse(
                _bm25_index_coverage_redirect_url(redirect_params),
                status_code=status.HTTP_303_SEE_OTHER,
            )

        try:
            report = refresh_bm25_keyword_indexes(
                settings.database_url,
                options=BM25IndexRefreshOptions(
                    chunk_policy_names=(chunk_policy_name,) if chunk_policy_name else (),
                    tokenizer_name=tokenizer_name,
                    continue_on_error=True,
                ),
            )
        except InvalidBM25KeywordIndexError as exc:
            redirect_params.update(
                {
                    "backfill_status": "failed",
                    "backfill_tokenizer_name": tokenizer_name,
                    "backfill_error": str(exc),
                }
            )
        else:
            redirect_params.update(
                {
                    "backfill_status": report.status,
                    "backfill_tokenizer_name": report.tokenizer_name,
                    "backfill_policy_count": report.policy_count,
                    "backfill_succeeded_count": report.succeeded_count,
                    "backfill_failed_count": report.failed_count,
                }
            )
        return RedirectResponse(
            _bm25_index_coverage_redirect_url(redirect_params),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.get("/admin/multi-policy-ingestion-coverage", response_class=HTMLResponse)
    def multi_policy_ingestion_coverage_page(
        request: Request,
        parse_status: str | None = None,
        document_group: str | None = None,
        profile_name: str | None = None,
        chunk_policy_name: str | None = None,
        detail_document_id: int | None = None,
        detail_chunk_policy_name: str | None = None,
        detail_profile_name: str | None = None,
        reconcile_created: int | None = None,
        reconcile_missing: int | None = None,
        reconcile_error: str | None = None,
        retry_retried: int | None = None,
        retry_failed: int | None = None,
        retry_error: str | None = None,
        limit: int = 100,
    ) -> HTMLResponse:
        matrix: MultiPolicyIngestionCoverageMatrix | None = None
        selected_detail: MultiPolicyIngestionCoverageDetail | None = None
        profiles: list[EmbeddingProfileRecord] = []
        policies: list[ChunkPolicySummaryRecord] = []
        error_message = None
        detail_error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                profiles = list_active_embedding_profiles(settings.database_url)
                policies = list_chunk_policy_summaries(settings.database_url)
                matrix = get_multi_policy_ingestion_coverage_matrix(
                    settings.database_url,
                    parse_status=parse_status,
                    document_group=document_group,
                    profile_name=profile_name,
                    chunk_policy_name=chunk_policy_name,
                    limit=limit,
                )
                detail_requested = (
                    detail_document_id is not None
                    or detail_chunk_policy_name is not None
                    or detail_profile_name is not None
                )
                if detail_requested:
                    if (
                        detail_document_id is None
                        or detail_chunk_policy_name is None
                        or detail_profile_name is None
                    ):
                        detail_error_message = (
                            "detail_document_id, detail_chunk_policy_name, "
                            "and detail_profile_name are required together."
                        )
                    else:
                        selected_detail = get_multi_policy_ingestion_coverage_detail(
                            settings.database_url,
                            document_id=detail_document_id,
                            chunk_policy_name=detail_chunk_policy_name,
                            profile_name=detail_profile_name,
                        )
                        if selected_detail is None:
                            detail_error_message = (
                                "Multi-policy ingestion coverage detail not found."
                            )
            except (InvalidEmbeddingCoverageError, InvalidEmbeddingJobError) as exc:
                error_message = str(exc)

        filter_query = urlencode(
            {
                key: value
                for key, value in {
                    "parse_status": parse_status or "",
                    "document_group": document_group or "",
                    "profile_name": profile_name or "",
                    "chunk_policy_name": chunk_policy_name or "",
                    "limit": str(limit),
                }.items()
                if value
            }
        )

        return TEMPLATES.TemplateResponse(
            request,
            "multi_policy_ingestion_coverage.html",
            template_context(
                request,
                matrix=matrix,
                selected_detail=selected_detail,
                profiles=profiles,
                policies=policies,
                selected_parse_status=parse_status or "",
                selected_document_group=document_group or "",
                selected_profile_name=profile_name or "",
                selected_chunk_policy_name=chunk_policy_name or "",
                selected_detail_document_id=detail_document_id,
                selected_detail_chunk_policy_name=detail_chunk_policy_name or "",
                selected_detail_profile_name=detail_profile_name or "",
                reconcile_created=reconcile_created,
                reconcile_missing=reconcile_missing,
                reconcile_error=reconcile_error or "",
                retry_retried=retry_retried,
                retry_failed=retry_failed,
                retry_error=retry_error or "",
                selected_limit=limit,
                error_message=error_message,
                detail_error_message=detail_error_message,
                selected_filter_query=filter_query,
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
        backlog_summary: EmbeddingJobBacklogSummary | None = None
        stale_jobs: list[EmbeddingJobRecord] = []
        failed_retryable_count = 0
        failed_exhausted_count = 0
        profiles: list[EmbeddingProfileRecord] = []
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                backlog_summary = get_embedding_job_backlog_summary(settings.database_url)
                if profile_name is None:
                    failed_retryable_count = backlog_summary.retryable_failed_count
                    failed_exhausted_count = backlog_summary.exhausted_failed_count
                else:
                    for profile_summary in backlog_summary.profile_summaries:
                        if profile_summary.profile_name == profile_name:
                            failed_retryable_count = profile_summary.retryable_failed_count
                            failed_exhausted_count = profile_summary.exhausted_failed_count
                            break
                stale_jobs = list_stale_embedding_jobs(
                    settings.database_url,
                    profile_name=profile_name,
                    limit=20,
                )
                profiles = list_active_embedding_profiles(settings.database_url)
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
                backlog_summary=backlog_summary,
                stale_jobs=stale_jobs,
                failed_retryable_count=failed_retryable_count,
                failed_exhausted_count=failed_exhausted_count,
                profiles=profiles,
                selected_status=status_filter or "",
                selected_profile_name=profile_name or "",
                selected_job_id=job_id,
                error_message=error_message,
                database_configured=bool(settings.database_url),
            ),
        )

    @app.get("/admin/embedding-batch-runs", response_class=HTMLResponse)
    def embedding_batch_runs_page(
        request: Request,
        worker_name: str | None = None,
        profile_name: str | None = None,
        stopped_reason: str | None = None,
        limit: int = 50,
        batch_run_id: int | None = None,
    ) -> HTMLResponse:
        batch_runs: list[EmbeddingWorkerBatchRunRecord] = []
        selected_batch_run = None
        profiles: list[EmbeddingProfileRecord] = []
        retention_settings = EmbeddingBatchRunRetentionSettings()
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                batch_runs = list_embedding_worker_batch_runs(
                    settings.database_url,
                    worker_name=worker_name,
                    profile_name=profile_name,
                    stopped_reason=stopped_reason,
                    limit=limit,
                )
                profiles = list_active_embedding_profiles(settings.database_url)
                retention_settings = load_embedding_batch_run_retention_settings(
                    settings.database_url
                )
                if batch_run_id is not None:
                    selected_batch_run = get_embedding_worker_batch_run(
                        settings.database_url,
                        batch_run_id,
                    )
                    if selected_batch_run is None:
                        error_message = f"Embedding batch run not found: {batch_run_id}"
            except (InvalidEmbeddingWorkerBatchRunError, InvalidEmbeddingJobError) as exc:
                error_message = str(exc)

        selected_payload = (
            embedding_worker_batch_run_payload(selected_batch_run)
            if selected_batch_run is not None
            else None
        )

        return TEMPLATES.TemplateResponse(
            request,
            "embedding_batch_runs.html",
            template_context(
                request,
                batch_runs=batch_runs,
                selected_batch_run=selected_batch_run,
                selected_batch_run_payload=selected_payload,
                selected_batch_run_json=(
                    json.dumps(selected_payload, ensure_ascii=False, indent=2)
                    if selected_payload is not None
                    else ""
                ),
                batch_run_summary=embedding_worker_batch_run_summary(batch_runs),
                throughput_summary=embedding_worker_batch_run_throughput_summary(batch_runs),
                retention_settings=retention_settings,
                profiles=profiles,
                selected_worker_name=worker_name or "",
                selected_profile_name=profile_name or "",
                selected_stopped_reason=stopped_reason or "",
                selected_limit=limit,
                selected_batch_run_id=batch_run_id,
                error_message=error_message,
                database_configured=bool(settings.database_url),
            ),
        )

    @app.get("/admin/dgx-ingestion-benchmarks", response_class=HTMLResponse)
    def dgx_ingestion_benchmarks_page(
        request: Request,
        provider: str | None = None,
        profile_name: str | None = None,
        passed: str | None = None,
        limit: int = 50,
        benchmark_run_id: int | None = None,
    ) -> HTMLResponse:
        benchmark_runs: list[DgxIngestionBenchmarkRunRecord] = []
        selected_detail: DgxIngestionBenchmarkDetail | None = None
        error_message = None
        selected_passed = (passed or "").strip().lower()
        passed_filter: bool | None = None
        if selected_passed:
            if selected_passed in {"true", "1", "yes"}:
                passed_filter = True
                selected_passed = "true"
            elif selected_passed in {"false", "0", "no"}:
                passed_filter = False
                selected_passed = "false"
            else:
                error_message = f"Unsupported passed filter: {passed}"

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        elif error_message is None:
            try:
                benchmark_runs = list_dgx_ingestion_benchmark_runs(
                    settings.database_url,
                    provider=provider or None,
                    profile_name=profile_name or None,
                    passed=passed_filter,
                    limit=limit,
                )
                selected_run_id = (
                    benchmark_run_id
                    if benchmark_run_id is not None
                    else (benchmark_runs[0].benchmark_run_id if benchmark_runs else None)
                )
                if selected_run_id is not None:
                    selected_detail = get_dgx_ingestion_benchmark_detail(
                        settings.database_url,
                        selected_run_id,
                    )
                    if selected_detail is None:
                        error_message = f"DGX ingestion benchmark run not found: {selected_run_id}"
            except InvalidDgxIngestionBenchmarkError as exc:
                error_message = str(exc)

        selected_payload = (
            dgx_ingestion_benchmark_detail_payload(selected_detail)
            if selected_detail is not None
            else None
        )

        return TEMPLATES.TemplateResponse(
            request,
            "dgx_ingestion_benchmarks.html",
            template_context(
                request,
                benchmark_runs=benchmark_runs,
                benchmark_summary=dgx_ingestion_benchmark_summary(benchmark_runs),
                selected_detail=selected_detail,
                selected_detail_payload=selected_payload,
                selected_detail_json=(
                    json.dumps(selected_payload, ensure_ascii=False, indent=2)
                    if selected_payload is not None
                    else ""
                ),
                selected_provider=provider or "",
                selected_profile_name=profile_name or "",
                selected_passed=selected_passed,
                selected_limit=limit,
                selected_benchmark_run_id=(
                    selected_detail.run.benchmark_run_id if selected_detail else benchmark_run_id
                ),
                error_message=error_message,
                database_configured=bool(settings.database_url),
            ),
        )

    @app.get("/admin/dgx-ingestion-benchmarks/compare", response_class=HTMLResponse)
    def dgx_ingestion_benchmark_compare_page(
        request: Request,
        left_run_id: int | None = None,
        right_run_id: int | None = None,
        limit: int = 50,
    ) -> HTMLResponse:
        benchmark_runs: list[DgxIngestionBenchmarkRunRecord] = []
        comparison_payload = None
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                benchmark_runs = list_dgx_ingestion_benchmark_runs(
                    settings.database_url,
                    limit=limit,
                )
                if left_run_id is None and right_run_id is None and len(benchmark_runs) >= 2:
                    right_run_id = benchmark_runs[0].benchmark_run_id
                    left_run_id = benchmark_runs[1].benchmark_run_id
                if left_run_id is not None and right_run_id is not None:
                    if left_run_id == right_run_id:
                        error_message = "Choose two different DGX ingestion benchmark runs."
                    else:
                        left_detail = get_dgx_ingestion_benchmark_detail(
                            settings.database_url,
                            left_run_id,
                        )
                        right_detail = get_dgx_ingestion_benchmark_detail(
                            settings.database_url,
                            right_run_id,
                        )
                        if left_detail is None or right_detail is None:
                            error_message = "DGX ingestion benchmark run not found."
                        else:
                            comparison_payload = dgx_ingestion_benchmark_compare_payload(
                                left_detail,
                                right_detail,
                            )
            except InvalidDgxIngestionBenchmarkError as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "dgx_ingestion_benchmark_compare.html",
            template_context(
                request,
                benchmark_runs=benchmark_runs,
                comparison=comparison_payload,
                comparison_json=(
                    json.dumps(comparison_payload, ensure_ascii=False, indent=2)
                    if comparison_payload is not None
                    else ""
                ),
                selected_left_run_id=left_run_id,
                selected_right_run_id=right_run_id,
                selected_limit=limit,
                error_message=error_message,
                database_configured=bool(settings.database_url),
            ),
        )

    @app.get("/admin/dgx-ingestion-benchmarks/trends", response_class=HTMLResponse)
    def dgx_ingestion_benchmark_trends_page(
        request: Request,
        provider: str | None = None,
        profile_name: str | None = None,
        limit: int = 50,
    ) -> HTMLResponse:
        trend_payload = dgx_ingestion_benchmark_trend_summary_payload([])
        error_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                benchmark_runs = list_dgx_ingestion_benchmark_runs(
                    settings.database_url,
                    provider=provider or None,
                    profile_name=profile_name or None,
                    limit=limit,
                )
                details = [
                    detail
                    for run in benchmark_runs
                    if (
                        detail := get_dgx_ingestion_benchmark_detail(
                            settings.database_url,
                            run.benchmark_run_id,
                        )
                    )
                    is not None
                ]
                trend_payload = dgx_ingestion_benchmark_trend_summary_payload(details)
            except InvalidDgxIngestionBenchmarkError as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "dgx_ingestion_benchmark_trends.html",
            template_context(
                request,
                trend=trend_payload,
                trend_json=json.dumps(trend_payload, ensure_ascii=False, indent=2),
                selected_provider=provider or "",
                selected_profile_name=profile_name or "",
                selected_limit=limit,
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

    @app.get("/admin/reranker-provider", response_class=HTMLResponse)
    def reranker_provider_page(
        request: Request,
        request_smoke: bool = Query(False),
    ) -> HTMLResponse:
        operations_status = get_remote_reranker_operations_status(
            settings,
            request_smoke=request_smoke,
        )
        return TEMPLATES.TemplateResponse(
            request,
            "reranker_provider.html",
            template_context(
                request,
                operations_status=operations_status.payload,
                operations_status_code=operations_status.status_code,
                request_smoke=request_smoke,
            ),
        )

    @app.get("/admin/embedding-provider-routes", response_class=HTMLResponse)
    def embedding_provider_routes_page(
        request: Request,
        availability_profile: str | None = Query(None),
    ) -> HTMLResponse:
        routes: list[EmbeddingProviderRouteRecord] = []
        profiles: list[EmbeddingProfileRecord] = []
        model_availability: ProviderModelAvailabilityMatrix | None = None
        model_availability_drilldown: ProviderModelAvailabilityDrilldown | None = None
        selected_availability_profile = availability_profile
        error_message = None
        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                routes = list_embedding_provider_routes(settings.database_url)
                profiles = list_active_embedding_profiles(settings.database_url)
                model_availability = get_provider_model_availability_matrix(
                    settings.database_url,
                    models_dir=settings.embedding_models_dir,
                )
                if selected_availability_profile is None and model_availability.rows:
                    selected_availability_profile = model_availability.rows[0].profile_name
                if selected_availability_profile:
                    model_availability_drilldown = get_provider_model_availability_drilldown(
                        settings.database_url,
                        models_dir=settings.embedding_models_dir,
                        profile_name=selected_availability_profile,
                    )
            except (
                InvalidEmbeddingProviderRouteError,
                InvalidEmbeddingJobError,
                InvalidEmbeddingProviderRouteHealthSnapshotError,
                InvalidEmbeddingProviderRouteContractSnapshotError,
                InvalidEmbeddingProviderPreflightRunError,
            ) as exc:
                error_message = str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "embedding_provider_routes.html",
            template_context(
                request,
                routes=routes,
                profiles=profiles,
                model_availability=model_availability,
                model_availability_drilldown=model_availability_drilldown,
                selected_availability_profile=selected_availability_profile or "",
                provider_presets=list_embedding_provider_presets(),
                embedding_models_dir=str(settings.embedding_models_dir),
                error_message=error_message,
                success_message=None,
                database_configured=bool(settings.database_url),
            ),
        )

    @app.get("/admin/embedding-provider-routes/playbook", response_class=HTMLResponse)
    def embedding_provider_routes_playbook_page(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            request,
            "embedding_provider_route_playbook.html",
            template_context(
                request,
                playbook_markdown=read_provider_operations_playbook_markdown(),
                playbook_source="docs/provider_operations_playbook.md",
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
        request_headers_json: str | None = Form("{}"),
        auth_type: str = Form(AUTH_TYPE_NONE),
        auth_token_env: str | None = Form(None),
        auth_header_name: str | None = Form("X-API-Key"),
    ) -> HTMLResponse:
        routes: list[EmbeddingProviderRouteRecord] = []
        profiles: list[EmbeddingProfileRecord] = []
        model_availability: ProviderModelAvailabilityMatrix | None = None
        model_availability_drilldown: ProviderModelAvailabilityDrilldown | None = None
        selected_availability_profile = profile_name
        error_message = None
        success_message = None

        if not settings.database_url:
            error_message = "NEX_PCX_DATABASE_URL is not configured."
        else:
            try:
                upsert_embedding_provider_route_with_audit(
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
                        runtime_metadata=embedding_provider_route_runtime_metadata_from_form(
                            request_headers_json=request_headers_json,
                            auth_type=auth_type,
                            auth_token_env=auth_token_env,
                            auth_header_name=auth_header_name,
                        ),
                    ),
                    request_path="/admin/embedding-provider-routes",
                )
                success_message = "Embedding provider route saved."
            except InvalidEmbeddingProviderRouteError as exc:
                error_message = str(exc)

            try:
                routes = list_embedding_provider_routes(settings.database_url)
                profiles = list_active_embedding_profiles(settings.database_url)
                model_availability = get_provider_model_availability_matrix(
                    settings.database_url,
                    models_dir=settings.embedding_models_dir,
                )
                if selected_availability_profile:
                    model_availability_drilldown = get_provider_model_availability_drilldown(
                        settings.database_url,
                        models_dir=settings.embedding_models_dir,
                        profile_name=selected_availability_profile,
                    )
            except (
                InvalidEmbeddingProviderRouteError,
                InvalidEmbeddingJobError,
                InvalidEmbeddingProviderRouteHealthSnapshotError,
                InvalidEmbeddingProviderRouteContractSnapshotError,
                InvalidEmbeddingProviderPreflightRunError,
            ) as exc:
                error_message = error_message or str(exc)

        return TEMPLATES.TemplateResponse(
            request,
            "embedding_provider_routes.html",
            template_context(
                request,
                routes=routes,
                profiles=profiles,
                model_availability=model_availability,
                model_availability_drilldown=model_availability_drilldown,
                selected_availability_profile=selected_availability_profile or "",
                provider_presets=list_embedding_provider_presets(),
                embedding_models_dir=str(settings.embedding_models_dir),
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
