import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect
from app.core.search_logs import SearchLogInput, create_search_log
from app.main import create_app, search_log_replay_url

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
    assert 'id="search-permission-summary"' in response.text
    assert "검색 권한 설명" in response.text
    assert 'id="feedback-summary-grid"' in response.text
    assert "search-feedback-button" in response.text
    assert "/api/search/feedback" in response.text
    assert "/api/search/feedback/summary" in response.text
    assert "권한 검색 Matrix" in response.text
    assert 'id="permission-matrix-form"' in response.text
    assert "/api/search/permission-matrix" in response.text


def test_search_history_page_renders_filters_and_log_table(
    migrated_database_url: str,
) -> None:
    app = create_app(Settings(database_url=migrated_database_url))

    with TestClient(app) as client:
        response = client.get("/search/logs")

    assert response.status_code == 200
    assert "Search History" in response.text
    assert "Alice Member / alice.member" in response.text
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
                "adapter": "mock",
                "search_mode": "history_reproducibility",
                "query_instruction": "none",
            },
            total_elapsed_ms=12,
            created_by_user_id=user_id,
        ),
    )
    app = create_app(Settings(database_url=migrated_database_url))
    replay_url = search_log_replay_url(search_log)
    try:
        with TestClient(app) as client:
            response = client.get(f"/search/logs?search_log_id={search_log.search_log_id}")
            replay_response = client.get(replay_url)

        assert response.status_code == 200
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
        assert "heading_512_64" in response.text
        assert ".md" in response.text
        assert "cosine" in response.text
        assert "bge_m3_1024" in response.text
        assert "history_reproducibility" in response.text
        assert "query_instruction" in response.text
        assert "JSON Export" in response.text
        assert "CSV Export" in response.text
        assert f"/api/search/logs/{search_log.search_log_id}/export" in response.text
        assert f"/api/search/logs/{search_log.search_log_id}/export?format=csv" in response.text
        assert "동일 조건으로 재실행" in response.text
        assert f"replay_search_log_id={search_log.search_log_id}" in response.text
        assert "No result rows found" in response.text
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
