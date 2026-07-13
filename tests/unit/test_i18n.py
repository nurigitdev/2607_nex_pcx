import pytest

from app.core.i18n import InvalidLanguageError, get_translator, normalize_language, resolve_language


def test_translator_uses_korean_default_and_english_catalog() -> None:
    korean = get_translator("ko")
    english = get_translator("en")

    assert korean("nav.dashboard") == "대시보드"
    assert english("nav.dashboard") == "Dashboard"
    assert korean("nav.permissions") == "권한"
    assert english("permissions.title") == "Permission Simulation"
    assert korean("upload_permissions.fieldset") == "권한 Metadata"
    assert english("upload_permissions.fieldset") == "Permission Metadata"
    assert korean("document_permissions.title") == "문서 권한 편집"
    assert english("document_permissions.title") == "Document Permission Edit"
    assert korean("search_matrix.title") == "권한 검색 Matrix"
    assert english("search_matrix.title") == "Permission Search Matrix"
    assert korean("search_scope.managed_org") == "관리 조직"
    assert english("search_scope.managed_org") == "Managed Org"
    assert korean("permission_readiness.title") == "권한 Metadata 준비도"
    assert english("permission_readiness.title") == "Permission Metadata Readiness"
    assert korean("nav.design_system") == "디자인 시스템"
    assert english("design_system.title") == "Design System"
    assert korean("nav.embedding_provider") == "Provider 상태"
    assert english("embedding_provider.title") == "Embedding Provider Health"
    assert korean("nav.embedding_batches") == "Batch 실행"
    assert english("embedding_batch_runs.title") == "Embedding Batch Run History"
    assert korean("embedding_batch_runs.stop_queue_empty") == "Queue Empty"
    assert english("embedding_batch_runs.no_run_selected") == (
        "Select a batch run to review detail."
    )
    assert korean("embedding_batch_runs.retry_failed_jobs") == "실패 Job 재시도"
    assert english("embedding_batch_runs.retry_complete") == (
        "Retry requested: {retried_count} retried, {skipped_count} skipped"
    )
    assert korean("embedding_backlog.title") == "임베딩 Queue Backlog"
    assert english("embedding_backlog.help") == (
        "Summarizes queue, failure, and stale lease state by profile."
    )
    assert korean("dashboard.core_metrics") == "Core Metrics"
    assert english("dashboard.core_metrics") == "Core Metrics"
    assert korean("dashboard.pipeline_queue_snapshot") == "Pipeline Queue 스냅샷"
    assert english("dashboard.pipeline_queue_snapshot") == "Pipeline Queue Snapshot"
    assert korean("dashboard.recent_failures") == "최근 운영 실패"
    assert english("dashboard.recent_failures") == "Recent Operational Failures"
    assert korean("dashboard.failure_source_provider_alert") == "Provider Alert"
    assert english("dashboard.failure_source_embedding") == "Embedding"
    assert korean("dashboard.failure_detail_title") == "운영 실패 상세"
    assert english("dashboard.failure_detail_open_target") == "Open Target"
    assert korean("dashboard.embedding_queue_snapshot") == "임베딩 Queue 스냅샷"
    assert english("dashboard.embedding_queue_snapshot") == "Embedding Queue Snapshot"
    assert korean("nav.embedding_routes") == "Provider 라우팅"
    assert english("embedding_routes.title") == "Embedding Provider Routing"
    assert korean("embedding_routes.playbook") == "운영 Playbook"
    assert english("embedding_routes.playbook_title") == "Provider Operations Playbook"
    assert korean("embedding_routes.operations_summary") == "Provider 운영 요약"
    assert korean("embedding_routes.operations_status") == "운영 상태"
    assert english("embedding_routes.operations_status_blocked") == "Action Needed"
    assert korean("embedding_routes.operations_reason_unacknowledged_alerts") == (
        "미확인 alert 있음"
    )
    assert english("embedding_routes.operations_summary_failed") == (
        "Operations summary failed to load"
    )
    assert korean("embedding_routes.shortcut_run_profile_preflight") == ("Profile Preflight 실행")
    assert english("embedding_routes.shortcut_profile_preflight_complete") == (
        "Profile preflight complete"
    )
    assert korean("embedding_routes.route_saved") == "Route 저장됨"
    assert english("embedding_routes.route_activation_failed") == ("Route status update failed")
    assert korean("embedding_routes.readiness") == "Provider Route Readiness"
    assert english("embedding_routes.readiness_failed") == "Readiness failed to load"
    assert korean("embedding_routes.status_needs_contract") == "Needs Contract"
    assert korean("embedding_routes.recovery_run_preflight") == "Preflight 실행"
    assert english("embedding_routes.recovery_check_provider_health") == "Check provider health"
    assert korean("embedding_routes.health_summary") == "Provider Route Health"
    assert korean("embedding_routes.health_loaded") == "Health 확인 완료"
    assert korean("embedding_routes.run_health_check") == "확인"
    assert korean("embedding_routes.health_history") == "Health Snapshot 이력"
    assert english("embedding_routes.status_mismatch") == "Mismatch"
    assert english("embedding_routes.manual_snapshot_saved") == "Snapshot saved"
    assert english("embedding_routes.no_health_snapshots") == "No health snapshots saved."
    assert korean("embedding_routes.alerts") == "Provider Route Alert"
    assert english("embedding_routes.acknowledge") == "Acknowledge"
    assert korean("embedding_routes.acknowledgement_note_placeholder") == "확인 메모"
    assert english("embedding_routes.acknowledged_by") == "Acknowledged By"
    assert korean("embedding_routes.sample_sets") == "Contract Sample Sets"
    assert english("embedding_routes.save_sample_set") == "Save Sample Set"
    assert korean("embedding_routes.preflight_run_history") == "Preflight 실행 이력"
    assert english("embedding_routes.preflight_run_history") == "Preflight Run History"
    assert korean("embedding_routes.preflight_run_detail") == "Preflight 실행 상세"
    assert english("embedding_routes.preflight_run_detail_action") == "Detail"
    assert korean("embedding_routes.no_preflight_runs") == (
        "저장된 preflight 실행 이력이 없습니다."
    )
    assert english("embedding_routes.preflight_runs_failed") == (
        "Preflight run history failed to load"
    )
    assert korean("embedding_routes.retention_settings") == "운영 데이터 보존 설정"
    assert english("embedding_routes.preview_cleanup") == "Preview Cleanup"
    assert korean("embedding_routes.cleanup_complete") == "Cleanup 완료"
    assert english("embedding_routes.preflight_runs") == "Preflight Runs"
    assert korean("embedding_routes.preflight_schedules") == "Preflight 스케줄"
    assert english("embedding_routes.save_schedule") == "Save Schedule"
    assert korean("embedding_routes.schedule_saved") == "스케줄 저장됨"
    assert korean("embedding_routes.preflight_due_operations") == "Due 스케줄 실행"
    assert english("embedding_routes.run_due_schedules") == "Run Due Schedules"
    assert korean("search_reproducibility.title") == "검색 재현성 Metadata"
    assert english("search_reproducibility.title") == "Search Reproducibility Metadata"
    assert korean("search_reproducibility.fingerprint") == "Fingerprint"
    assert english("search_reproducibility.fingerprint") == "Fingerprint"
    assert korean("search_replay.action") == "동일 조건으로 재실행"
    assert english("search_replay.action") == "Replay Same Conditions"
    assert korean("search_export.action") == "JSON Export"
    assert english("search_export.action") == "JSON Export"
    assert korean("search_export.csv_action") == "CSV Export"
    assert english("search_export.csv_action") == "CSV Export"
    assert korean("search_export.report_action") == "Report Export"
    assert english("search_export.report_action") == "Report Export"
    assert korean("search_log_compare.title") == "검색 로그 비교"
    assert english("search_log_compare.title") == "Search Log Compare"
    assert korean("search_log_compare.target_placeholder") == "비교할 로그를 선택하세요"
    assert english("search_log_compare.target_placeholder") == "Select a log to compare"
    assert korean("search_log_review.title") == "검색 로그 리뷰"
    assert english("search_log_review.title") == "Search Log Review"
    assert korean("search_feedback.comment_placeholder") == "판정 이유 또는 재검토 메모"
    assert english("search_feedback.comment_placeholder") == "Reason or review note"
    assert korean("search_feedback.comments_title") == "피드백 메모 리뷰"
    assert english("search_feedback.comments_title") == "Feedback Comment Review"


def test_translator_falls_back_to_key_when_missing() -> None:
    translator = get_translator("ko")

    assert translator("missing.translation.key") == "missing.translation.key"


def test_resolve_language_prefers_query_then_cookie_then_default() -> None:
    assert resolve_language(query_language="en", cookie_language="ko") == "en"
    assert resolve_language(query_language=None, cookie_language="en") == "en"
    assert resolve_language(query_language=None, cookie_language=None) == "ko"


def test_normalize_language_rejects_unsupported_language() -> None:
    with pytest.raises(InvalidLanguageError, match="Unsupported language"):
        normalize_language("fr")
