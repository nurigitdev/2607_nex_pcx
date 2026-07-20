from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from psycopg.types.json import Json

from app.core.admin_logging import log_event
from app.core.config import Settings
from app.core.dashboard_failures import get_dashboard_recent_failures
from app.core.dashboard_metrics import get_dashboard_core_metrics
from app.core.dashboard_throughput import get_dashboard_throughput_latency_snapshot
from app.core.database import connect
from app.core.embedding_jobs import (
    EmbeddingJobInput,
    create_embedding_job,
    get_embedding_job_backlog_summary,
    mark_embedding_job_failed,
)
from app.core.embedding_worker import EMBEDDING_WORKER_BATCH_STOP_QUEUE_EMPTY
from app.core.embedding_worker_batch_runs import (
    EmbeddingWorkerBatchRunInput,
    record_embedding_worker_batch_run,
)
from app.core.evaluation_dashboard import get_evaluation_dashboard_summary
from app.core.evaluation_metrics import (
    ExpectedTarget,
    QuestionEvaluationInput,
    evaluate_question,
    summarize_question_metrics,
)
from app.core.evaluation_runs import (
    EvaluationResultInput,
    EvaluationRunInput,
    complete_evaluation_run,
    create_evaluation_result,
    create_evaluation_run,
)
from app.core.pipeline_jobs import (
    PipelineJobInput,
    create_pipeline_job,
    get_pipeline_queue_summary,
)
from app.main import create_app

pytestmark = pytest.mark.integration


def _create_dashboard_embedding_backlog_fixture(database_url: str) -> dict[str, object]:
    checksum = f"dashboard-backlog-{uuid4()}"
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO files (
                    original_file_name,
                    stored_file_name,
                    file_ext,
                    file_size_bytes,
                    sha256_checksum,
                    storage_path,
                    document_group,
                    parse_status,
                    parse_error_message
                )
                VALUES (
                    %s,
                    %s,
                    '.md',
                    2048,
                    %s,
                    %s,
                    'slice-151',
                    'failed',
                    'dashboard parse fixture failed'
                )
                RETURNING file_id
                """,
                (
                    f"{checksum}.md",
                    f"{checksum}.stored.md",
                    checksum,
                    f"/tmp/{checksum}.md",
                ),
            )
            file_id = cursor.fetchone()["file_id"]
            cursor.execute(
                """
                INSERT INTO documents (file_id, document_title, document_group)
                VALUES (%s, %s, 'slice-151')
                RETURNING document_id
                """,
                (file_id, f"Dashboard backlog fixture {checksum}"),
            )
            document_id = cursor.fetchone()["document_id"]
            cursor.execute(
                """
                INSERT INTO chunks (
                    document_id,
                    chunk_seq,
                    chunk_text,
                    content_hash,
                    chunk_policy_name,
                    token_count,
                    char_count
                )
                VALUES (%s, 0, %s, %s, 'heading_512_64', 11, %s)
                RETURNING chunk_id
                """,
                (
                    document_id,
                    "Dashboard backlog fixture chunk",
                    f"chunk-{checksum}",
                    len("Dashboard backlog fixture chunk"),
                ),
            )
            chunk_id = cursor.fetchone()["chunk_id"]
            cursor.execute(
                """
                INSERT INTO chunks (
                    document_id,
                    chunk_seq,
                    chunk_text,
                    content_hash,
                    chunk_policy_name,
                    token_count,
                    char_count
                )
                VALUES (%s, 1, %s, %s, 'heading_512_64', 9, %s)
                RETURNING chunk_id
                """,
                (
                    document_id,
                    "Dashboard failed embedding fixture chunk",
                    f"failed-chunk-{checksum}",
                    len("Dashboard failed embedding fixture chunk"),
                ),
            )
            failed_chunk_id = cursor.fetchone()["chunk_id"]

    profile_name = "kure_v1_1024"
    job = create_embedding_job(
        database_url,
        EmbeddingJobInput(chunk_id=chunk_id, profile_name=profile_name),
    )
    failed_job = create_embedding_job(
        database_url,
        EmbeddingJobInput(chunk_id=failed_chunk_id, profile_name=profile_name),
    )
    marked_failed_job = mark_embedding_job_failed(
        database_url,
        failed_job.job.job_id,
        error_code="SLICE_153_EMBEDDING_FAILED",
        error_message="dashboard embedding fixture failed",
    )
    return {
        "file_id": file_id,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "job_id": job.job.job_id,
        "failed_job_id": marked_failed_job.job_id if marked_failed_job else failed_job.job.job_id,
        "profile_name": profile_name,
    }


def _create_dashboard_pipeline_queue_fixture(
    database_url: str,
    *,
    file_id: int,
    document_id: int,
    user_id: int,
) -> dict[str, int]:
    queued = create_pipeline_job(
        database_url,
        PipelineJobInput(
            job_type="document_ingestion",
            file_id=file_id,
            document_id=document_id,
            requested_by_user_id=user_id,
            total_units=4,
        ),
    )
    stale_running = create_pipeline_job(
        database_url,
        PipelineJobInput(
            job_type="parsing",
            file_id=file_id,
            document_id=document_id,
            requested_by_user_id=user_id,
            total_units=6,
        ),
    )
    failed = create_pipeline_job(
        database_url,
        PipelineJobInput(
            job_type="chunking",
            file_id=file_id,
            document_id=document_id,
            requested_by_user_id=user_id,
            total_units=3,
        ),
    )
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE pipeline_jobs
                SET status = 'running',
                    stage = 'parsing',
                    processed_units = 2,
                    progress_percent = 33.33,
                    attempts = 1,
                    lease_owner = 'dashboard-stale-worker',
                    lease_expires_at = now() - interval '5 minutes',
                    heartbeat_at = now() - interval '10 minutes',
                    updated_at = now()
                WHERE job_id = %s
                """,
                (stale_running.job_id,),
            )
            cursor.execute(
                """
                UPDATE pipeline_jobs
                SET status = 'failed',
                    stage = 'chunking',
                    attempts = 1,
                    error_code = 'SLICE_152_FAILED',
                    error_message = 'dashboard pipeline fixture failed',
                    finished_at = now(),
                    updated_at = now()
                WHERE job_id = %s
                """,
                (failed.job_id,),
            )
    return {
        "queued_job_id": queued.job_id,
        "stale_job_id": stale_running.job_id,
        "failed_job_id": failed.job_id,
    }


def _create_dashboard_fixture(database_url: str) -> dict[str, object]:
    set_name = f"slice-048-{uuid4()}"
    backlog_fixture = _create_dashboard_embedding_backlog_fixture(database_url)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT user_id FROM app_users WHERE login_id = 'alice.member'")
            user_id = cursor.fetchone()["user_id"]
            cursor.execute(
                """
                INSERT INTO golden_question_sets (
                    set_name,
                    description,
                    created_by_user_id
                )
                VALUES (%s, 'Slice 048 dashboard fixture', %s)
                RETURNING question_set_id
                """,
                (set_name, user_id),
            )
            question_set_id = cursor.fetchone()["question_set_id"]
            cursor.execute(
                """
                INSERT INTO golden_questions (
                    question_set_id,
                    question_text,
                    question_type,
                    actor_user_id,
                    requested_search_scope,
                    top_k,
                    created_by_user_id
                )
                VALUES (
                    %s,
                    'Which dashboard-only policy should not exist?',
                    'no_answer',
                    %s,
                    'company',
                    5,
                    %s
                )
                RETURNING question_id
                """,
                (question_set_id, user_id, user_id),
            )
            question_id = cursor.fetchone()["question_id"]
            cursor.execute(
                """
                INSERT INTO golden_question_expected_targets (
                    question_id,
                    expected_heading_path,
                    expectation_type,
                    relevance_grade
                )
                VALUES (%s, ARRAY['No Answer']::TEXT[], 'visible', 0)
                """,
                (question_id,),
            )
            cursor.execute(
                """
                INSERT INTO search_logs (
                    query_text,
                    normalized_query_text,
                    document_group,
                    top_k,
                    profiles,
                    query_runtime_metadata,
                    total_elapsed_ms,
                    created_by
                )
                VALUES (
                    'dashboard core metrics fixture',
                    'dashboard core metrics fixture',
                    'slice-151',
                    5,
                    %s,
                    %s,
                    37,
                    'integration-test'
                )
                RETURNING search_log_id
                """,
                (Json(["kure_v1_1024"]), Json({"purpose": "slice-151"})),
            )
            search_log_id = cursor.fetchone()["search_log_id"]
            cursor.execute(
                """
                INSERT INTO search_log_results (
                    search_log_id,
                    profile_name,
                    rank,
                    chunk_id,
                    score,
                    profile_elapsed_ms
                )
                VALUES (%s, 'kure_v1_1024', 1, %s, 0.91, 11)
                """,
                (search_log_id, backlog_fixture["chunk_id"]),
            )

    pipeline_fixture = _create_dashboard_pipeline_queue_fixture(
        database_url,
        file_id=int(backlog_fixture["file_id"]),
        document_id=int(backlog_fixture["document_id"]),
        user_id=int(user_id),
    )
    batch_started_at = datetime.now(UTC) - timedelta(milliseconds=20)
    batch_run = record_embedding_worker_batch_run(
        database_url,
        EmbeddingWorkerBatchRunInput(
            worker_name=f"dashboard-throughput-worker-{uuid4()}",
            profile_name=str(backlog_fixture["profile_name"]),
            provider_source="route",
            provider_mode="mock",
            limit_requested=2,
            result_count=2,
            processed_count=2,
            succeeded_count=1,
            failed_count=1,
            deferred_count=0,
            idle_count=0,
            stopped_reason=EMBEDDING_WORKER_BATCH_STOP_QUEUE_EMPTY,
            job_ids=(
                int(backlog_fixture["job_id"]),
                int(backlog_fixture["failed_job_id"]),
            ),
            runtime_metadata={"purpose": "slice-155-dashboard-throughput"},
            elapsed_ms=20,
            started_at=batch_started_at,
            completed_at=batch_started_at + timedelta(milliseconds=20),
        ),
    )
    app_error_log_id = log_event(
        database_url,
        level="ERROR",
        event_type="dashboard_recent_failure_fixture",
        source="integration-test",
        message="dashboard app log failure",
        detail={"slice": 153},
        correlation_id=f"slice-153-error-{uuid4()}",
    )
    provider_alert_log_id = log_event(
        database_url,
        level="ERROR",
        event_type="embedding_provider_route_health_alert",
        source="integration-test",
        message="dashboard provider alert fixture",
        detail={"slice": 153},
        correlation_id=f"slice-153-alert-{uuid4()}",
    )

    metric = evaluate_question(
        QuestionEvaluationInput(
            question_id=question_id,
            top_k=5,
            expected_targets=(
                ExpectedTarget(expected_heading_path=("No Answer",), relevance_grade=0),
            ),
            ranked_results=(),
        )
    )
    succeeded_run = create_evaluation_run(
        database_url,
        EvaluationRunInput(
            question_set_id=question_set_id,
            run_name=f"slice-048-succeeded-{uuid4()}",
            profile_name="kure_v1_1024",
            top_k=5,
        ),
    )
    create_evaluation_result(
        database_url,
        EvaluationResultInput(evaluation_run_id=succeeded_run.evaluation_run_id, metric=metric),
    )
    completed_run = complete_evaluation_run(
        database_url,
        succeeded_run.evaluation_run_id,
        summarize_question_metrics((metric,)),
    )
    failed_run = create_evaluation_run(
        database_url,
        EvaluationRunInput(
            question_set_id=question_set_id,
            run_name=f"slice-048-failed-{uuid4()}",
            profile_name="kure_v1_1024",
            status="failed",
            top_k=5,
        ),
    )
    return {
        "set_name": set_name,
        "question_set_id": question_set_id,
        "question_id": question_id,
        "succeeded_run_id": completed_run.evaluation_run_id,
        "failed_run_id": failed_run.evaluation_run_id,
        "embedding_file_id": backlog_fixture["file_id"],
        "embedding_job_id": backlog_fixture["job_id"],
        "failed_embedding_job_id": backlog_fixture["failed_job_id"],
        "embedding_profile_name": backlog_fixture["profile_name"],
        "pipeline_queued_job_id": pipeline_fixture["queued_job_id"],
        "pipeline_stale_job_id": pipeline_fixture["stale_job_id"],
        "pipeline_failed_job_id": pipeline_fixture["failed_job_id"],
        "batch_run_id": batch_run.batch_run_id,
        "search_log_id": search_log_id,
        "app_error_log_id": app_error_log_id,
        "provider_alert_log_id": provider_alert_log_id,
    }


def _cleanup_dashboard_fixture(database_url: str, fixture: dict[str, object]) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM golden_question_sets WHERE set_name = %s",
                (fixture["set_name"],),
            )
            cursor.execute(
                "DELETE FROM search_logs WHERE search_log_id = %s",
                (fixture["search_log_id"],),
            )
            cursor.execute(
                "DELETE FROM files WHERE file_id = %s",
                (fixture["embedding_file_id"],),
            )
            cursor.execute(
                "DELETE FROM embedding_worker_batch_runs WHERE batch_run_id = %s",
                (fixture["batch_run_id"],),
            )
            log_ids = [
                log_id
                for log_id in (
                    fixture.get("app_error_log_id"),
                    fixture.get("provider_alert_log_id"),
                )
                if log_id is not None
            ]
            if log_ids:
                cursor.execute("DELETE FROM app_logs WHERE log_id = ANY(%s)", (log_ids,))


def test_evaluation_dashboard_summary_api_and_page(
    migrated_database_url: str,
) -> None:
    fixture = _create_dashboard_fixture(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))
    selected_lookback_hours = 6
    try:
        summary = get_evaluation_dashboard_summary(migrated_database_url, recent_limit=10)
        core_metrics = get_dashboard_core_metrics(migrated_database_url)
        pipeline_queue = get_pipeline_queue_summary(migrated_database_url)
        throughput_latency = get_dashboard_throughput_latency_snapshot(
            migrated_database_url,
            lookback_hours=selected_lookback_hours,
        )
        backlog_summary = get_embedding_job_backlog_summary(migrated_database_url)
        recent_failures = get_dashboard_recent_failures(migrated_database_url, limit=20)
        status_counts = {item.status: item.count for item in summary.status_counts}

        with TestClient(app) as client:
            core_metrics_response = client.get("/api/dashboard/core-metrics")
            operational_health_response = client.get("/api/dashboard/operational-health")
            dashboard_export_json_response = client.get(
                "/api/dashboard/export",
                params={
                    "lookback_hours": selected_lookback_hours,
                    "format": "json",
                },
            )
            dashboard_export_csv_response = client.get(
                "/api/dashboard/export",
                params={
                    "lookback_hours": selected_lookback_hours,
                    "format": "csv",
                },
            )
            pipeline_queue_response = client.get("/api/dashboard/pipeline-queue")
            throughput_latency_response = client.get(
                "/api/dashboard/throughput-latency",
                params={"lookback_hours": selected_lookback_hours},
            )
            api_response = client.get("/api/dashboard/evaluations", params={"recent_limit": 10})
            backlog_api_response = client.get("/api/dashboard/embedding-backlog")
            failures_api_response = client.get(
                "/api/dashboard/recent-failures",
                params={"limit": 20},
            )
            pipeline_detail_response = client.get(
                f"/api/dashboard/recent-failures/pipeline/{fixture['pipeline_failed_job_id']}"
            )
            embedding_detail_response = client.get(
                f"/api/dashboard/recent-failures/embedding/{fixture['failed_embedding_job_id']}"
            )
            parsing_detail_response = client.get(
                f"/api/dashboard/recent-failures/parsing/{fixture['embedding_file_id']}"
            )
            app_log_detail_response = client.get(
                f"/api/dashboard/recent-failures/app_log/{fixture['app_error_log_id']}"
            )
            provider_alert_detail_response = client.get(
                f"/api/dashboard/recent-failures/provider_alert/{fixture['provider_alert_log_id']}"
            )
            bad_response = client.get("/api/dashboard/evaluations", params={"recent_limit": 0})
            bad_failures_response = client.get(
                "/api/dashboard/recent-failures",
                params={"limit": 0},
            )
            bad_throughput_response = client.get(
                "/api/dashboard/throughput-latency",
                params={"lookback_hours": 0},
            )
            bad_dashboard_export_format_response = client.get(
                "/api/dashboard/export",
                params={"format": "xml"},
            )
            bad_dashboard_export_lookback_response = client.get(
                "/api/dashboard/export",
                params={"lookback_hours": 0},
            )
            bad_failure_detail_response = client.get("/api/dashboard/recent-failures/unknown/1")
            missing_failure_detail_response = client.get(
                "/api/dashboard/recent-failures/pipeline/999999999"
            )
            page_response = client.get(
                "/",
                params={
                    "lookback_hours": selected_lookback_hours,
                    "refresh_seconds": 30,
                },
            )
            threshold_settings_response = client.get("/api/dashboard/health-thresholds")
            threshold_update_response = client.put(
                "/api/dashboard/health-thresholds",
                json={"thresholds": {"pipeline_retryable": 5}},
            )
            threshold_restore_response = client.post("/api/dashboard/health-thresholds/reset")
            dashboard_settings_page_response = client.get("/admin/dashboard-settings")
            threshold_bad_code_response = client.put(
                "/api/dashboard/health-thresholds",
                json={"thresholds": {"unknown": 1}},
            )
            threshold_bad_value_response = client.put(
                "/api/dashboard/health-thresholds",
                json={"thresholds": {"pipeline_retryable": 0}},
            )

        api_payload = api_response.json()["evaluations"]
        core_metrics_payload = core_metrics_response.json()["core_metrics"]
        operational_health_payload = operational_health_response.json()["operational_health"]
        dashboard_export_json = dashboard_export_json_response.json()
        pipeline_queue_payload = pipeline_queue_response.json()["pipeline_queue"]
        throughput_latency_payload = throughput_latency_response.json()["throughput_latency"]
        failures_payload = failures_api_response.json()["recent_failures"]
        pipeline_detail = pipeline_detail_response.json()["failure_detail"]
        embedding_detail = embedding_detail_response.json()["failure_detail"]
        parsing_detail = parsing_detail_response.json()["failure_detail"]
        app_log_detail = app_log_detail_response.json()["failure_detail"]
        provider_alert_detail = provider_alert_detail_response.json()["failure_detail"]
        recent_run_ids = {run.evaluation_run_id for run in summary.recent_runs}
        api_recent_run_ids = {run["evaluation_run_id"] for run in api_payload["recent_runs"]}
        failure_sources = {failure.source for failure in recent_failures.failures}
        api_failure_sources = {failure["source"] for failure in failures_payload["failures"]}

        assert core_metrics.document_count >= 1
        assert core_metrics.chunk_count >= 1
        assert core_metrics.search_log_count >= 1
        assert core_metrics_response.status_code == 200
        assert core_metrics_payload["document_count"] == core_metrics.document_count
        assert core_metrics_payload["chunk_count"] == core_metrics.chunk_count
        assert core_metrics_payload["search_log_count"] == core_metrics.search_log_count
        assert core_metrics_payload["total_file_size_bytes"] >= 2048
        assert any(item["file_type"] == ".md" for item in core_metrics_payload["file_types"])
        assert any(
            item["document_group"] == "slice-151"
            for item in core_metrics_payload["document_groups"]
        )
        assert any(
            item["chunk_policy_name"] == "heading_512_64"
            for item in core_metrics_payload["chunk_policies"]
        )
        assert operational_health_response.status_code == 200
        assert operational_health_payload["status"] == "critical"
        assert operational_health_payload["critical_count"] >= 1
        assert operational_health_payload["warning_count"] >= 1
        assert any(
            signal["code"] == "pipeline_stale" for signal in operational_health_payload["signals"]
        )
        assert dashboard_export_json_response.status_code == 200
        assert dashboard_export_json_response.headers["content-disposition"].endswith('.json"')
        assert dashboard_export_json["version"] == 1
        assert dashboard_export_json["lookback_hours"] == selected_lookback_hours
        assert dashboard_export_json["operational_health"]["status"] == "critical"
        assert dashboard_export_json["operational_health"]["thresholds"]["pipeline_retryable"] == 1
        assert dashboard_export_json["operational_health"]["signals"][0]["threshold"] >= 1
        assert dashboard_export_json["core_metrics"]["document_count"] == (
            core_metrics.document_count
        )
        assert dashboard_export_json["pipeline_queue"]["total_count"] == (
            pipeline_queue.total_count
        )
        assert dashboard_export_json["throughput_latency"]["lookback_hours"] == (
            selected_lookback_hours
        )
        assert dashboard_export_json["embedding_backlog"]["total_count"] == (
            backlog_summary.total_count
        )
        assert dashboard_export_json["evaluations"]["active_question_set_count"] == (
            summary.active_question_set_count
        )
        assert dashboard_export_csv_response.status_code == 200
        assert dashboard_export_csv_response.headers["content-type"].startswith("text/csv")
        assert dashboard_export_csv_response.headers["content-disposition"].endswith('.csv"')
        assert "exported_at,lookback_hours,health_status" in (dashboard_export_csv_response.text)
        assert ",6,critical," in dashboard_export_csv_response.text
        assert threshold_settings_response.status_code == 200
        assert (
            threshold_settings_response.json()["settings"]["thresholds"]["pipeline_retryable"] == 1
        )
        assert threshold_update_response.status_code == 200
        assert threshold_update_response.json()["settings"]["thresholds"]["pipeline_retryable"] == 5
        assert threshold_restore_response.status_code == 200
        assert (
            threshold_restore_response.json()["settings"]["thresholds"]["pipeline_retryable"] == 1
        )
        assert dashboard_settings_page_response.status_code == 200
        assert "Dashboard 설정" in dashboard_settings_page_response.text
        assert "/api/dashboard/health-thresholds" in (dashboard_settings_page_response.text)
        assert 'data-threshold-code="pipeline_retryable"' in (dashboard_settings_page_response.text)
        assert 'value="1"' in dashboard_settings_page_response.text
        assert threshold_bad_code_response.status_code == 400
        assert threshold_bad_value_response.status_code == 400
        assert pipeline_queue.queued_count >= 1
        assert pipeline_queue.stale_running_count >= 1
        assert pipeline_queue.failed_count >= 1
        assert pipeline_queue_response.status_code == 200
        assert pipeline_queue_payload["total_count"] == pipeline_queue.total_count
        assert pipeline_queue_payload["queued_count"] == pipeline_queue.queued_count
        assert pipeline_queue_payload["stale_running_count"] >= 1
        assert pipeline_queue_payload["failed_count"] >= 1
        assert pipeline_queue_payload["claimable_count"] >= 3
        assert any(item["stage"] == "parsing" for item in pipeline_queue_payload["stages"])
        assert any(
            item["stage"] == "parsing"
            and item["average_progress_percent"] == "33.33"
            and item["average_progress_label"] == "33.33%"
            for item in pipeline_queue_payload["stages"]
        )
        assert any(
            item["job_type"] == "document_ingestion" for item in pipeline_queue_payload["job_types"]
        )
        assert throughput_latency.pipeline.completed_count >= 1
        assert throughput_latency.embedding.processed_count >= 2
        assert throughput_latency.search.search_log_count >= 1
        assert throughput_latency_response.status_code == 200
        assert throughput_latency_payload["lookback_hours"] == selected_lookback_hours
        assert throughput_latency_payload["pipeline"]["completed_count"] == (
            throughput_latency.pipeline.completed_count
        )
        assert throughput_latency_payload["embedding"]["processed_count"] >= 2
        assert throughput_latency_payload["embedding"]["throughput_per_second"] > 0
        assert throughput_latency_payload["search"]["average_total_elapsed_ms"] is not None
        assert any(
            item["profile_name"] == fixture["embedding_profile_name"]
            and item["processed_count"] >= 2
            for item in throughput_latency_payload["embedding"]["profiles"]
        )
        assert any(
            item["profile_name"] == fixture["embedding_profile_name"]
            and item["result_count"] >= 1
            and item["average_profile_elapsed_ms"] is not None
            for item in throughput_latency_payload["search"]["profiles"]
        )
        assert summary.active_question_set_count >= 1
        assert summary.question_count >= 1
        assert summary.expected_target_count >= 1
        assert status_counts["succeeded"] >= 1
        assert status_counts["failed"] >= 1
        assert fixture["succeeded_run_id"] in recent_run_ids
        assert fixture["failed_run_id"] in recent_run_ids
        assert api_response.status_code == 200
        assert api_payload["active_question_set_count"] == summary.active_question_set_count
        assert fixture["succeeded_run_id"] in api_recent_run_ids
        assert backlog_api_response.status_code == 200
        assert backlog_api_response.json()["backlog"]["total_count"] == backlog_summary.total_count
        assert backlog_api_response.json()["backlog"]["pending_count"] >= 1
        assert failures_api_response.status_code == 200
        assert recent_failures.pipeline_failure_count >= 1
        assert recent_failures.embedding_failure_count >= 1
        assert recent_failures.parsing_failure_count >= 1
        assert recent_failures.app_error_count >= 1
        assert recent_failures.provider_alert_count >= 1
        assert {"pipeline", "embedding", "parsing", "app_log", "provider_alert"}.issubset(
            failure_sources
        )
        assert failures_payload["pipeline_failure_count"] == (
            recent_failures.pipeline_failure_count
        )
        assert failures_payload["embedding_failure_count"] == (
            recent_failures.embedding_failure_count
        )
        assert {"pipeline", "embedding", "parsing", "app_log", "provider_alert"}.issubset(
            api_failure_sources
        )
        assert pipeline_detail_response.status_code == 200
        assert pipeline_detail["source"] == "pipeline"
        assert pipeline_detail["context"]["job_type"] == "chunking"
        assert pipeline_detail["summary"]["error_code"] == "SLICE_152_FAILED"
        assert "job" in pipeline_detail["raw"]
        assert embedding_detail_response.status_code == 200
        assert embedding_detail["source"] == "embedding"
        assert embedding_detail["summary"]["error_code"] == "SLICE_153_EMBEDDING_FAILED"
        assert embedding_detail["summary"]["profile_name"] == fixture["embedding_profile_name"]
        assert parsing_detail_response.status_code == 200
        assert parsing_detail["source"] == "parsing"
        assert parsing_detail["summary"]["parse_error_message"] == (
            "dashboard parse fixture failed"
        )
        assert app_log_detail_response.status_code == 200
        assert app_log_detail["source"] == "app_log"
        assert app_log_detail["summary"]["event_type"] == "dashboard_recent_failure_fixture"
        assert app_log_detail["summary"]["occurred_at"] == app_log_detail["occurred_at_label"]
        assert "T" not in app_log_detail["summary"]["occurred_at"]
        assert "+" not in app_log_detail["summary"]["occurred_at"]
        assert provider_alert_detail_response.status_code == 200
        assert provider_alert_detail["source"] == "provider_alert"
        assert provider_alert_detail["context"]["traceback_present"] is False
        assert bad_response.status_code == 400
        assert bad_failures_response.status_code == 400
        assert bad_throughput_response.status_code == 400
        assert bad_dashboard_export_format_response.status_code == 400
        assert bad_dashboard_export_lookback_response.status_code == 400
        assert bad_failure_detail_response.status_code == 400
        assert missing_failure_detail_response.status_code == 404
        assert page_response.status_code == 200
        assert "Core Metrics" in page_response.text
        assert "slice-151" in page_response.text
        assert "/api/dashboard/core-metrics" in page_response.text
        assert "운영 상태" in page_response.text
        assert "위험" in page_response.text
        assert "Pipeline stale lease" in page_response.text
        assert "/api/dashboard/operational-health" in page_response.text
        assert 'href="/documents"' in page_response.text
        assert 'href="/admin/embedding-jobs"' in page_response.text
        assert 'href="/search/logs"' in page_response.text
        assert "Pipeline Queue 스냅샷" in page_response.text
        assert "/api/dashboard/pipeline-queue" in page_response.text
        assert 'href="/admin/jobs?status=queued"' in page_response.text
        assert 'href="/admin/jobs?status=failed"' in page_response.text
        assert "처리량 / Latency 스냅샷" in page_response.text
        assert "마지막 갱신" in page_response.text
        assert "자동 갱신" in page_response.text
        assert 'data-refresh-seconds="30"' in page_response.text
        assert "스냅샷 내보내기" in page_response.text
        assert "/api/dashboard/export?lookback_hours=6&amp;format=json" in (page_response.text)
        assert "/api/dashboard/export?lookback_hours=6&amp;format=csv" in (page_response.text)
        assert "최근 6시간" in page_response.text
        assert "/?refresh_seconds=30&amp;lookback_hours=1" in page_response.text
        assert "/?lookback_hours=6&amp;refresh_seconds=60" in page_response.text
        assert "/api/dashboard/throughput-latency?lookback_hours=6" in page_response.text
        assert "jobs/sec" in page_response.text
        assert "33.33%" in page_response.text
        assert "0E-20" not in page_response.text
        assert "dashboard-stale-worker" not in page_response.text
        assert "최근 운영 실패" in page_response.text
        assert "/api/dashboard/recent-failures" in page_response.text
        assert 'href="/documents?parse_status=failed"' in page_response.text
        assert "data-failure-detail-button" in page_response.text
        assert "data-failure-detail-panel" in page_response.text
        assert "운영 실패 상세" in page_response.text
        assert "dashboard pipeline fixture failed" in page_response.text
        assert "dashboard embedding fixture failed" in page_response.text
        assert "dashboard parse fixture failed" in page_response.text
        assert "dashboard app log failure" in page_response.text
        assert "dashboard provider alert fixture" in page_response.text
        assert "임베딩 Queue 스냅샷" in page_response.text
        assert fixture["embedding_profile_name"] in page_response.text
        assert "/api/dashboard/embedding-backlog" in page_response.text
        assert 'href="/admin/embedding-jobs?status=pending"' in page_response.text
        assert "골든 평가 스냅샷" in page_response.text
        assert "활성 질문 세트" in page_response.text
        assert 'href="/evaluations?status=succeeded"' in page_response.text
        assert fixture["set_name"] in page_response.text
        assert f"#{fixture['succeeded_run_id']}" in page_response.text
        assert "/api/dashboard/evaluations" in page_response.text
    finally:
        _cleanup_dashboard_fixture(migrated_database_url, fixture)
