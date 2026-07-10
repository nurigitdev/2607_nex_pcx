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
    assert korean("nav.embedding_routes") == "Provider 라우팅"
    assert english("embedding_routes.title") == "Embedding Provider Routing"
    assert korean("embedding_routes.health_summary") == "Provider Route Health"
    assert korean("embedding_routes.health_loaded") == "Health 확인 완료"
    assert korean("embedding_routes.run_health_check") == "확인"
    assert korean("embedding_routes.health_history") == "Health Snapshot 이력"
    assert english("embedding_routes.status_mismatch") == "Mismatch"
    assert english("embedding_routes.manual_snapshot_saved") == "Snapshot saved"
    assert english("embedding_routes.no_health_snapshots") == "No health snapshots saved."
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
