import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect
from app.core.search_logs import (
    SearchLogInput,
    SearchLogReviewMetadataInput,
    create_search_log,
    update_search_log_review_metadata,
)
from app.main import create_app, search_log_replay_url, search_reproducibility_fingerprint

pytestmark = pytest.mark.integration


def _alice_user_id(database_url: str) -> int:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT user_id FROM app_users WHERE login_id = 'alice.member'")
            return int(cursor.fetchone()["user_id"])


def _delete_search_log(database_url: str, search_log_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM search_logs WHERE search_log_id = %s", (search_log_id,))


def test_search_compare_page_renders_actor_and_profile_options(
    migrated_database_url: str,
) -> None:
    app = create_app(Settings(database_url=migrated_database_url))

    with TestClient(app) as client:
        response = client.get("/search")

    assert response.status_code == 200
    assert "Search Compare" in response.text
    assert "Alice Member / alice.member" in response.text
    assert "kure_v1_1024" in response.text
    assert "bge_m3_1024" in response.text
    assert 'id="search-compare-form"' in response.text
    assert 'id="search-results"' in response.text
    assert 'id="search-runtime-panel"' in response.text
    assert 'id="search-runtime-grid"' in response.text
    assert "검색 Runtime 상태" in response.text
    assert 'id="search-permission-summary"' in response.text
    assert "검색 권한 설명" in response.text
    assert 'id="feedback-summary-grid"' in response.text
    assert "search-feedback-button" in response.text
    assert "search-feedback-comment" in response.text
    assert "판정 이유 또는 재검토 메모" in response.text
    assert "피드백 메모 리뷰" in response.text
    assert "/api/search/feedback/comments" in response.text
    assert "/api/search/feedback" in response.text
    assert "/api/search/feedback/summary" in response.text
    assert "권한 검색 Matrix" in response.text
    assert 'id="permission-matrix-form"' in response.text
    assert "/api/search/permission-matrix" in response.text
    assert "Chunk 정책 비교" in response.text
    assert 'id="chunk-policy-compare-form"' in response.text
    assert "/api/search/compare/chunk-policies" in response.text
    assert "heading_1000_200" in response.text
    assert 'id="search-readiness-panel"' in response.text
    assert "검색 준비 상태" in response.text
    assert "/api/search/compare/readiness" in response.text


def test_search_history_page_renders_filters_and_log_table(
    migrated_database_url: str,
) -> None:
    app = create_app(Settings(database_url=migrated_database_url))

    with TestClient(app) as client:
        response = client.get(
            "/search/logs",
            params={
                "operations_lookback_hours": 168,
                "operations_min_total_elapsed_ms": 1500,
            },
        )

    assert response.status_code == 200
    assert "Search History" in response.text
    assert "Alice Member / alice.member" in response.text
    assert "Search Log Retention" in response.text
    assert 'id="search-retention-form"' in response.text
    assert 'id="search-cleanup-preview"' in response.text
    assert 'id="search-cleanup-run"' in response.text
    assert "/api/search/logs/retention-settings" in response.text
    assert "/api/search/logs/cleanup" in response.text
    assert "검색 Operations Summary" in response.text
    assert "GET /api/search/logs/operations-summary" in response.text
    assert 'name="operations_lookback_hours"' in response.text
    assert 'id="operations_lookback_hours_168"' in response.text
    assert 'value="1500"' in response.text
    assert "7d" in response.text
    assert 'class="search-history-filter"' in response.text
    assert 'class="table table-sm align-middle mb-0 search-log-table"' in response.text


def test_search_history_detail_renders_permission_explainability(
    migrated_database_url: str,
) -> None:
    user_id = _alice_user_id(migrated_database_url)
    search_log = create_search_log(
        migrated_database_url,
        SearchLogInput(
            query_text="History permission explainability",
            normalized_query_text="history permission explainability",
            actor_user_id=user_id,
            requested_search_scope="company",
            effective_search_scope="team",
            permission_filter_metadata={
                "permission_explainability": {
                    "actor_user_id": user_id,
                    "actor_login_id": "alice.member",
                    "actor_display_name": "Alice Member",
                    "role_name": "member",
                    "primary_org_unit_name": "Platform Team",
                    "requested_search_scope": "company",
                    "effective_search_scope": "team",
                    "scope_was_downgraded": True,
                    "ancestor_org_unit_count": 4,
                    "managed_org_unit_count": 0,
                    "includes_company_documents": True,
                    "filter_clause_count": 4,
                    "candidate_document_count": 7,
                    "visible_document_count": 3,
                    "excluded_document_count": 4,
                    "visible_access_scope_counts": {
                        "personal": 1,
                        "team": 1,
                        "org_tree": 0,
                        "company": 1,
                    },
                    "included_access_scopes": ["personal", "team", "company"],
                }
            },
            document_group="history-permission",
            file_type=".md",
            chunk_policy_name="heading_512_64",
            top_k=5,
            profiles=("kure_v1_1024", "bge_m3_1024"),
            query_runtime_metadata={
                "adapter": "query_embedding_bridge",
                "search_mode": "history_reproducibility",
                "query_embedding_bridge": True,
                "profile_status_counts": {"succeeded": 1, "failed": 1},
                "profile_failure_count": 1,
                "profile_query_embeddings": {
                    "kure_v1_1024": {
                        "profile_name": "kure_v1_1024",
                        "dimension": 1024,
                        "provider_type": "remote",
                        "provider_model_id": "nlpai-lab/KURE-v1",
                        "provider_elapsed_ms": 128,
                        "total_elapsed_ms": 142,
                        "runtime_source": "route",
                    }
                },
                "profile_failures": {
                    "bge_m3_1024": {
                        "profile_name": "bge_m3_1024",
                        "status": "failed",
                        "error_code": "query_embedding_failed",
                        "error_message": "Remote provider request failed",
                        "elapsed_ms": 176,
                    }
                },
            },
            total_elapsed_ms=1250,
            created_by_user_id=user_id,
        ),
    )
    duplicate_log = create_search_log(
        migrated_database_url,
        SearchLogInput(
            query_text=search_log.query_text,
            normalized_query_text=search_log.normalized_query_text,
            actor_user_id=search_log.actor_user_id,
            requested_search_scope=search_log.requested_search_scope,
            effective_search_scope=search_log.effective_search_scope,
            permission_filter_metadata=search_log.permission_filter_metadata,
            document_group=search_log.document_group,
            file_type=search_log.file_type,
            chunk_policy_name=search_log.chunk_policy_name,
            top_k=search_log.top_k,
            profiles=search_log.profiles,
            query_runtime_metadata={
                "adapter": "query_embedding_bridge",
                "search_mode": "history_duplicate_fingerprint",
                "profile_status_counts": {"succeeded": 2, "failed": 0},
            },
            total_elapsed_ms=980,
            created_by_user_id=user_id,
        ),
    )
    comparison_log = create_search_log(
        migrated_database_url,
        SearchLogInput(
            query_text="History comparison target",
            normalized_query_text="history comparison target",
            actor_user_id=user_id,
            requested_search_scope="company",
            effective_search_scope="team",
            permission_filter_metadata={
                "permission_explainability": {
                    "actor_user_id": user_id,
                    "actor_login_id": "alice.member",
                    "actor_display_name": "Alice Member",
                    "role_name": "member",
                    "primary_org_unit_name": "Platform Team",
                    "requested_search_scope": "company",
                    "effective_search_scope": "team",
                    "scope_was_downgraded": True,
                    "ancestor_org_unit_count": 4,
                    "managed_org_unit_count": 0,
                    "includes_company_documents": True,
                    "filter_clause_count": 4,
                    "candidate_document_count": 7,
                    "visible_document_count": 3,
                    "excluded_document_count": 4,
                    "visible_access_scope_counts": {
                        "personal": 1,
                        "team": 1,
                        "org_tree": 0,
                        "company": 1,
                    },
                    "included_access_scopes": ["personal", "team", "company"],
                }
            },
            document_group="history-permission",
            file_type=".md",
            chunk_policy_name="heading_512_64",
            top_k=3,
            profiles=("kure_v1_1024",),
            query_runtime_metadata={
                "adapter": "mock",
                "search_mode": "history_compare_target",
                "query_instruction": "none",
            },
            total_elapsed_ms=8,
            created_by_user_id=user_id,
        ),
    )
    update_search_log_review_metadata(
        migrated_database_url,
        SearchLogReviewMetadataInput(
            search_log_id=search_log.search_log_id,
            review_tags=("baseline", "permission-review"),
            review_memo="UI review memo",
            reviewed_by_user_id=user_id,
        ),
    )
    app = create_app(Settings(database_url=migrated_database_url))
    replay_url = search_log_replay_url(search_log)
    fingerprint = search_reproducibility_fingerprint(search_log)
    try:
        with TestClient(app) as client:
            response = client.get(f"/search/logs?search_log_id={search_log.search_log_id}")
            replay_response = client.get(replay_url)
            filtered_response = client.get(f"/search/logs?fingerprint={fingerprint}")
            comparison_response = client.get(
                "/search/logs",
                params={
                    "search_log_id": search_log.search_log_id,
                    "compare_search_log_id": comparison_log.search_log_id,
                },
            )

        assert response.status_code == 200
        assert filtered_response.status_code == 200
        assert comparison_response.status_code == 200
        assert "검색 권한 설명" in response.text
        assert "History permission explainability" in response.text
        assert "Alice Member" in response.text
        assert "회사 전체" in response.text
        assert "팀" in response.text
        assert "3 / 7" in response.text
        assert "4" in response.text
        assert "개인 1" in response.text
        assert "회사 공통 1" in response.text
        assert "검색 재현성 Metadata" in response.text
        assert "Fingerprint" in response.text
        assert fingerprint in response.text
        assert "heading_512_64" in response.text
        assert ".md" in response.text
        assert "cosine" in response.text
        assert "bge_m3_1024" in response.text
        assert "history_reproducibility" in response.text
        assert "검색 이력 Runtime 상태" in response.text
        assert 'id="search-history-runtime-panel"' in response.text
        assert "검색 Runtime Failure Triage" in response.text
        assert "GET /api/search/logs/runtime-failures" in response.text
        assert "표시된 실패 재시도" in response.text
        assert "/api/search/logs/runtime-failures/retry" in response.text
        assert "data-runtime-failure-row" in response.text
        assert "상세 보기" in response.text
        assert "검색 Latency Outlier Triage" in response.text
        assert "GET /api/search/logs/latency-outliers" in response.text
        assert "1250 ms" in response.text
        assert "검색 Operations Summary" in response.text
        assert "GET /api/search/logs/operations-summary" in response.text
        assert "24h / 1000ms" in response.text
        assert 'id="operations_min_total_elapsed_ms"' in response.text
        assert "Runtime Failure" in response.text
        assert "Duplicate 그룹" in response.text
        assert "검색 No Result Triage" in response.text
        assert "GET /api/search/logs/no-results" in response.text
        assert "검색 Duplicate Fingerprint Triage" in response.text
        assert "GET /api/search/logs/duplicate-fingerprints" in response.text
        assert "최신 로그 보기" in response.text
        assert f"/search/logs?search_log_id={duplicate_log.search_log_id}" in response.text
        assert "History permission explainability" in response.text
        assert "nlpai-lab/KURE-v1" in response.text
        assert "Remote provider request failed" in response.text
        assert "query_embedding_failed" in response.text
        assert "성공 1" in response.text
        assert "실패 1" in response.text
        assert "실패 Profile 재시도" in response.text
        assert 'class="btn btn-sm btn-outline-primary search-runtime-retry-button"' in (
            response.text
        )
        assert 'data-profile-name="bge_m3_1024"' in response.text
        assert "/api/search/logs/${searchLogId}/retry-profile" in response.text
        assert "JSON Export" in response.text
        assert "CSV Export" in response.text
        assert "Report Export" in response.text
        assert "검색 로그 리뷰" in response.text
        assert "baseline" in response.text
        assert "permission-review" in response.text
        assert "UI review memo" in response.text
        assert 'id="search_log_review_tags_input"' in response.text
        assert 'value="baseline, permission-review"' in response.text
        assert f'data-search-log-id="{search_log.search_log_id}"' in response.text
        assert f"/api/search/logs/{search_log.search_log_id}/export" in response.text
        assert f"/api/search/logs/{search_log.search_log_id}/export?format=csv" in response.text
        assert f"/api/search/logs/{search_log.search_log_id}/experiment-report" in response.text
        assert "동일 조건으로 재실행" in response.text
        assert f"replay_search_log_id={search_log.search_log_id}" in response.text
        assert "No result rows found" in response.text
        assert f'value="{fingerprint}"' in filtered_response.text
        assert "History permission explainability" in filtered_response.text
        assert "검색 로그 비교" in comparison_response.text
        assert (
            f"/api/search/logs/{search_log.search_log_id}/experiment-report"
            f"?compare_search_log_id={comparison_log.search_log_id}" in comparison_response.text
        )
        assert (
            '<select\n          class="form-select"\n          id="compare_search_log_id"'
            in comparison_response.text
        )
        assert "비교할 로그를 선택하세요" in comparison_response.text
        assert f'value="{comparison_log.search_log_id}"' in comparison_response.text
        assert (
            f"#{comparison_log.search_log_id} · History comparison target"
            in comparison_response.text
        )
        assert f"compare_search_log_id={comparison_log.search_log_id}" in response.text
        assert "Fingerprint 동일" in comparison_response.text
        assert "공통 Chunk" in comparison_response.text
        assert "query_text" in comparison_response.text
        assert "다름" in comparison_response.text
        assert replay_response.status_code == 200
        assert "검색 이력 조건이 적용되었습니다." in replay_response.text
        assert f"#{search_log.search_log_id}" in replay_response.text
        assert 'value="History permission explainability"' in replay_response.text
        assert f'<option value="{user_id}" selected>' in replay_response.text
        assert '<option value="company" selected>' in replay_response.text
        assert 'value="5"' in replay_response.text
        assert 'value="history-permission"' in replay_response.text
        assert '<option value=".md" selected>' in replay_response.text
        assert 'value="heading_512_64"' in replay_response.text
        assert 'value="kure_v1_1024"' in replay_response.text
        assert 'value="bge_m3_1024"' in replay_response.text
    finally:
        _delete_search_log(migrated_database_url, search_log.search_log_id)
        _delete_search_log(migrated_database_url, duplicate_log.search_log_id)
        _delete_search_log(migrated_database_url, comparison_log.search_log_id)
