from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from psycopg.types.json import Json

from app.core.config import Settings
from app.core.database import connect, fetch_one
from app.core.embedding_vectors import (
    EmbeddingVectorInput,
    generate_mock_embedding,
    store_chunk_embedding,
)
from app.core.search_logs import (
    SearchLogInput,
    SearchLogResultInput,
    create_search_log,
    create_search_log_results,
)
from app.main import create_app

pytestmark = pytest.mark.integration


def _seed_ids(database_url: str) -> dict[str, int]:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT login_id, user_id
                FROM app_users
                WHERE login_id IN ('alice.member', 'bob.member')
                """)
            users = {row["login_id"]: int(row["user_id"]) for row in cursor.fetchall()}
            cursor.execute("""
                SELECT org_unit_name, org_unit_id
                FROM org_units
                WHERE org_unit_name IN ('NeX Company', 'Business Team')
                """)
            orgs = {row["org_unit_name"]: int(row["org_unit_id"]) for row in cursor.fetchall()}
    return {**users, **orgs}


def _create_search_compare_chunk(
    database_url: str,
    *,
    title: str,
    owner_user_id: int | None,
    owner_org_unit_id: int,
    access_scope: str,
    chunk_text: str,
    document_group: str,
) -> tuple[int, int]:
    checksum = f"search-compare-{uuid4()}"
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
                    uploaded_by_user_id,
                    document_group
                )
                VALUES (%s, %s, '.md', 1, %s, %s, %s, %s)
                RETURNING file_id
                """,
                (
                    f"{checksum}.md",
                    f"{checksum}.stored.md",
                    checksum,
                    f"/tmp/{checksum}.md",
                    owner_user_id,
                    document_group,
                ),
            )
            file_id = cursor.fetchone()["file_id"]
            cursor.execute(
                """
                INSERT INTO documents (
                    file_id,
                    document_title,
                    document_group,
                    owner_user_id,
                    owner_org_unit_id,
                    access_scope
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING document_id
                """,
                (
                    file_id,
                    title,
                    document_group,
                    owner_user_id,
                    owner_org_unit_id,
                    access_scope,
                ),
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
                    char_count
                )
                VALUES (%s, 0, %s, %s, 'heading_512_64', %s)
                RETURNING chunk_id
                """,
                (document_id, chunk_text, f"chunk-{checksum}", len(chunk_text)),
            )
            chunk_id = cursor.fetchone()["chunk_id"]
    return file_id, chunk_id


def _store_profile_embeddings(database_url: str, chunk_id: int, chunk_text: str) -> None:
    for profile_name in ("kure_v1_1024", "bge_m3_1024"):
        store_chunk_embedding(
            database_url,
            EmbeddingVectorInput(
                chunk_id=chunk_id,
                profile_name=profile_name,
                embedding=generate_mock_embedding(
                    chunk_text,
                    profile_name=profile_name,
                    dimension=1024,
                ),
                elapsed_ms=4,
            ),
        )


def _create_search_result_source_context_fixture(
    database_url: str,
    *,
    owner_user_id: int | None,
    owner_org_unit_id: int,
    document_group: str,
) -> tuple[int, int, int]:
    checksum = f"search-context-{uuid4()}"
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
                    uploaded_by_user_id,
                    document_group
                )
                VALUES (%s, %s, '.md', 3, %s, %s, %s, %s)
                RETURNING file_id
                """,
                (
                    f"{checksum}.md",
                    f"{checksum}.stored.md",
                    checksum,
                    f"/tmp/{checksum}.md",
                    owner_user_id,
                    document_group,
                ),
            )
            file_id = int(cursor.fetchone()["file_id"])
            cursor.execute(
                """
                INSERT INTO documents (
                    file_id,
                    document_title,
                    document_group,
                    owner_user_id,
                    owner_org_unit_id,
                    access_scope
                )
                VALUES (%s, %s, %s, %s, %s, 'company')
                RETURNING document_id
                """,
                (
                    file_id,
                    "source context fixture",
                    document_group,
                    owner_user_id,
                    owner_org_unit_id,
                ),
            )
            document_id = int(cursor.fetchone()["document_id"])
            artifact_text = "\n".join(
                [
                    "# Source Context",
                    "Previous context sentence.",
                    "Current source context anchor sentence.",
                    "Next context sentence.",
                ]
            )
            cursor.execute(
                """
                INSERT INTO extraction_artifacts (
                    file_id,
                    document_id,
                    artifact_type,
                    content_text,
                    content_hash,
                    size_bytes,
                    language,
                    metadata
                )
                VALUES (%s, %s, 'normalized_markdown', %s, %s, %s, 'ko', %s)
                RETURNING artifact_id
                """,
                (
                    file_id,
                    document_id,
                    artifact_text,
                    f"artifact-{checksum}",
                    len(artifact_text.encode("utf-8")),
                    Json({"fixture": "source-context"}),
                ),
            )
            artifact_id = int(cursor.fetchone()["artifact_id"])
            cursor.execute(
                """
                INSERT INTO document_blocks (
                    artifact_id,
                    document_id,
                    block_seq,
                    block_type,
                    content_text,
                    content_markdown,
                    heading_path,
                    source_anchor,
                    char_start,
                    char_end,
                    token_count,
                    metadata
                )
                VALUES (%s, %s, 1, 'paragraph', %s, %s, %s, %s, 31, 70, 6, %s)
                RETURNING block_id
                """,
                (
                    artifact_id,
                    document_id,
                    "Current source context anchor sentence.",
                    "Current source context anchor sentence.",
                    ["Source Context"],
                    Json({"source": "fixture", "start_line": 3, "end_line": 3}),
                    Json({"block_fixture": True}),
                ),
            )
            block_id = int(cursor.fetchone()["block_id"])
            chunk_ids: list[int] = []
            chunk_inputs = [
                ("Previous context sentence.", 0, None, None),
                (
                    "Current source context anchor sentence.",
                    1,
                    artifact_id,
                    block_id,
                ),
                ("Next context sentence.", 2, None, None),
            ]
            for chunk_text, chunk_seq, chunk_artifact_id, chunk_block_id in chunk_inputs:
                cursor.execute(
                    """
                    INSERT INTO chunks (
                        document_id,
                        artifact_id,
                        block_id,
                        chunk_seq,
                        chunk_type,
                        chunk_text,
                        content_markdown,
                        content_hash,
                        chunk_policy_name,
                        heading_path,
                        source_anchor,
                        source_char_start,
                        source_char_end,
                        token_count,
                        char_count,
                        metadata
                    )
                    VALUES (
                        %s, %s, %s, %s, 'text', %s, %s, %s, 'heading_512_64',
                        %s, %s, %s, %s, %s, %s, %s
                    )
                    RETURNING chunk_id
                    """,
                    (
                        document_id,
                        chunk_artifact_id,
                        chunk_block_id,
                        chunk_seq,
                        chunk_text,
                        chunk_text,
                        f"chunk-{checksum}-{chunk_seq}",
                        ["Source Context"],
                        Json({"source": "fixture", "chunk_seq": chunk_seq}),
                        chunk_seq * 30,
                        chunk_seq * 30 + len(chunk_text),
                        5,
                        len(chunk_text),
                        Json({"chunk_fixture": chunk_seq}),
                    ),
                )
                chunk_ids.append(int(cursor.fetchone()["chunk_id"]))
            cursor.execute(
                """
                UPDATE chunks
                SET prev_chunk_id = %s,
                    next_chunk_id = %s
                WHERE chunk_id = %s
                """,
                (None, chunk_ids[1], chunk_ids[0]),
            )
            cursor.execute(
                """
                UPDATE chunks
                SET prev_chunk_id = %s,
                    next_chunk_id = %s
                WHERE chunk_id = %s
                """,
                (chunk_ids[0], chunk_ids[2], chunk_ids[1]),
            )
            cursor.execute(
                """
                UPDATE chunks
                SET prev_chunk_id = %s,
                    next_chunk_id = %s
                WHERE chunk_id = %s
                """,
                (chunk_ids[1], None, chunk_ids[2]),
            )

    search_log = create_search_log(
        database_url,
        SearchLogInput(
            query_text="source context query",
            normalized_query_text="source context query",
            actor_user_id=owner_user_id,
            requested_search_scope="company",
            effective_search_scope="company",
            document_group=document_group,
            top_k=3,
            profiles=("kure_v1_1024",),
            created_by="source-context-test",
        ),
    )
    result = create_search_log_results(
        database_url,
        [
            SearchLogResultInput(
                search_log_id=search_log.search_log_id,
                profile_name="kure_v1_1024",
                rank=1,
                chunk_id=chunk_ids[1],
                distance=0.12,
                score=0.88,
                profile_elapsed_ms=9,
            )
        ],
    )[0]
    return file_id, result.search_log_result_id, chunk_ids[1]


def _cleanup_files(database_url: str, file_ids: list[int]) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            for file_id in file_ids:
                cursor.execute(
                    """
                    DELETE FROM search_logs
                    WHERE search_log_id IN (
                        SELECT DISTINCT sl.search_log_id
                        FROM search_logs sl
                        JOIN search_log_results slr
                          ON slr.search_log_id = sl.search_log_id
                        JOIN chunks c ON c.chunk_id = slr.chunk_id
                        JOIN documents d ON d.document_id = c.document_id
                        WHERE d.file_id = %s
                    )
                    """,
                    (file_id,),
                )
                cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def test_search_log_retention_settings_api_round_trips(
    migrated_database_url: str,
) -> None:
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            update_response = client.put(
                "/api/search/logs/retention-settings",
                json={
                    "enabled": False,
                    "retention_days": 45,
                    "cleanup_batch_size": 250,
                },
            )
            get_response = client.get("/api/search/logs/retention-settings")
            invalid_response = client.put(
                "/api/search/logs/retention-settings",
                json={
                    "enabled": True,
                    "retention_days": 0,
                    "cleanup_batch_size": 250,
                },
            )

        assert update_response.status_code == 200
        assert update_response.json()["settings"] == {
            "enabled": False,
            "retention_days": 45,
            "cleanup_batch_size": 250,
        }
        assert get_response.status_code == 200
        assert get_response.json()["settings"] == update_response.json()["settings"]
        assert invalid_response.status_code == 422
    finally:
        with TestClient(app) as client:
            client.put(
                "/api/search/logs/retention-settings",
                json={
                    "enabled": True,
                    "retention_days": 30,
                    "cleanup_batch_size": 1000,
                },
            )


def _create_cleanup_search_log(database_url: str, query_text: str, days_old: int) -> int:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO search_logs (
                    query_text,
                    normalized_query_text,
                    top_k,
                    profiles,
                    created_by,
                    created_at
                )
                VALUES (
                    %s,
                    %s,
                    5,
                    '["kure_v1_1024"]'::jsonb,
                    'cleanup-test',
                    now() - (%s::int * interval '1 day')
                )
                RETURNING search_log_id
                """,
                (query_text, query_text.casefold(), days_old),
            )
            return int(cursor.fetchone()["search_log_id"])


def _delete_search_logs(database_url: str, search_log_ids: list[int]) -> None:
    if not search_log_ids:
        return
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM search_logs WHERE search_log_id = ANY(%s)", (search_log_ids,)
            )


def test_search_log_cleanup_api_previews_and_deletes_expired_logs(
    migrated_database_url: str,
) -> None:
    app = create_app(Settings(database_url=migrated_database_url))
    created_ids: list[int] = []

    try:
        with TestClient(app) as client:
            client.put(
                "/api/search/logs/retention-settings",
                json={
                    "enabled": True,
                    "retention_days": 3650,
                    "cleanup_batch_size": 1,
                },
            )
            old_log_id = _create_cleanup_search_log(
                migrated_database_url,
                f"cleanup old {uuid4()}",
                4000,
            )
            second_old_log_id = _create_cleanup_search_log(
                migrated_database_url,
                f"cleanup second old {uuid4()}",
                3999,
            )
            recent_log_id = _create_cleanup_search_log(
                migrated_database_url,
                f"cleanup recent {uuid4()}",
                1,
            )
            created_ids.extend([old_log_id, second_old_log_id, recent_log_id])

            preview_response = client.post(
                "/api/search/logs/cleanup",
                json={"dry_run": True},
            )
            cleanup_response = client.post(
                "/api/search/logs/cleanup",
                json={"dry_run": False},
            )

        remaining = fetch_one(
            migrated_database_url,
            """
            SELECT
                count(*) FILTER (
                    WHERE search_log_id IN (%s, %s)
                ) AS old_count,
                count(*) FILTER (
                    WHERE search_log_id = %s
                ) AS recent_count
            FROM search_logs
            """,
            (old_log_id, second_old_log_id, recent_log_id),
        )

        assert preview_response.status_code == 200
        assert preview_response.json()["cleanup"]["dry_run"] is True
        assert preview_response.json()["cleanup"]["expired_count"] >= 2
        assert preview_response.json()["cleanup"]["deleted_count"] == 0
        assert cleanup_response.status_code == 200
        assert cleanup_response.json()["cleanup"]["dry_run"] is False
        assert cleanup_response.json()["cleanup"]["deleted_count"] == 1
        assert remaining["old_count"] == 1
        assert remaining["recent_count"] == 1
    finally:
        _delete_search_logs(migrated_database_url, created_ids)
        with TestClient(app) as client:
            client.put(
                "/api/search/logs/retention-settings",
                json={
                    "enabled": True,
                    "retention_days": 30,
                    "cleanup_batch_size": 1000,
                },
            )


def test_search_result_source_context_api_returns_chunk_trace(
    migrated_database_url: str,
) -> None:
    ids = _seed_ids(migrated_database_url)
    document_group = f"slice-255-{uuid4()}"
    file_id, search_log_result_id, current_chunk_id = (
        _create_search_result_source_context_fixture(
            migrated_database_url,
            owner_user_id=ids["alice.member"],
            owner_org_unit_id=ids["NeX Company"],
            document_group=document_group,
        )
    )
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            response = client.get(
                f"/api/search/results/{search_log_result_id}/source-context"
            )
            missing_response = client.get("/api/search/results/999999999/source-context")
            invalid_response = client.get("/api/search/results/-1/source-context")

        body = response.json()
        with TestClient(app) as client:
            history_page_response = client.get(
                "/search/logs",
                params={"search_log_id": body["search_result"]["search_log_id"]},
            )

        chunks_by_position = {chunk["position"]: chunk for chunk in body["chunks"]}

        assert response.status_code == 200
        assert body["search_result"]["search_log_result_id"] == search_log_result_id
        assert body["search_result"]["chunk_id"] == current_chunk_id
        assert body["document"]["document_title"] == "source context fixture"
        assert body["document"]["document_group"] == document_group
        assert set(chunks_by_position) == {"previous", "current", "next"}
        assert chunks_by_position["current"]["chunk_id"] == current_chunk_id
        assert chunks_by_position["current"]["chunk_preview"] == (
            "Current source context anchor sentence."
        )
        assert chunks_by_position["previous"]["chunk_preview"] == "Previous context sentence."
        assert chunks_by_position["next"]["chunk_preview"] == "Next context sentence."
        assert chunks_by_position["current"]["source_anchor"]["chunk_seq"] == 1
        assert body["source_block"]["block_type"] == "paragraph"
        assert body["source_block"]["content_preview"] == (
            "Current source context anchor sentence."
        )
        assert body["source_block"]["source_anchor"]["start_line"] == 3
        assert body["source_artifact"]["artifact_type"] == "normalized_markdown"
        assert body["source_artifact"]["content_length"] > 0
        assert body["trace_summary"] == {
            "has_previous_chunk": True,
            "has_next_chunk": True,
            "has_source_block": True,
            "has_source_artifact": True,
            "context_chunk_count": 3,
            "current_source_anchor": {"source": "fixture", "chunk_seq": 1},
        }
        assert missing_response.status_code == 404
        assert invalid_response.status_code == 400
        assert history_page_response.status_code == 200
        assert "근거 보기" in history_page_response.text
        assert f'data-search-log-result-id="{search_log_result_id}"' in history_page_response.text
        assert "search-history-source-context-panel" in history_page_response.text
    finally:
        _cleanup_files(migrated_database_url, [file_id])


def test_search_compare_api_returns_permission_filtered_profile_results(
    migrated_database_url: str,
) -> None:
    ids = _seed_ids(migrated_database_url)
    query_text = "Search compare API anchor"
    document_group = f"slice-028-{uuid4()}"
    visible_file_id, visible_chunk_id = _create_search_compare_chunk(
        migrated_database_url,
        title="visible company fixture",
        owner_user_id=None,
        owner_org_unit_id=ids["NeX Company"],
        access_scope="company",
        chunk_text=query_text,
        document_group=document_group,
    )
    hidden_file_id, hidden_chunk_id = _create_search_compare_chunk(
        migrated_database_url,
        title="hidden bob personal fixture",
        owner_user_id=ids["bob.member"],
        owner_org_unit_id=ids["Business Team"],
        access_scope="personal",
        chunk_text=query_text,
        document_group=document_group,
    )
    try:
        _store_profile_embeddings(migrated_database_url, visible_chunk_id, query_text)
        _store_profile_embeddings(migrated_database_url, hidden_chunk_id, query_text)
        app = create_app(Settings(database_url=migrated_database_url))

        with TestClient(app) as client:
            response = client.post(
                "/api/search/compare",
                json={
                    "query_text": query_text,
                    "actor_user_id": ids["alice.member"],
                    "requested_search_scope": "company",
                    "top_k": 5,
                    "profiles": ["kure_v1_1024", "bge_m3_1024"],
                    "chunk_policy_name": "heading_512_64",
                    "document_group": document_group,
                },
            )
            matrix_response = client.post(
                "/api/search/permission-matrix",
                json={
                    "query_text": query_text,
                    "top_k": 5,
                    "profiles": ["kure_v1_1024"],
                    "chunk_policy_name": "heading_512_64",
                    "document_group": document_group,
                    "entries": [
                        {
                            "actor_user_id": ids["alice.member"],
                            "requested_search_scope": "company",
                        },
                        {
                            "actor_user_id": ids["bob.member"],
                            "requested_search_scope": "company",
                        },
                    ],
                },
            )

        body = response.json()
        matrix_body = matrix_response.json()
        profile_results = {
            profile["profile_name"]: profile["results"] for profile in body["profiles"]
        }
        result_chunk_ids = {
            result["chunk_id"] for results in profile_results.values() for result in results
        }
        result_count = fetch_one(
            migrated_database_url,
            """
            SELECT count(*) AS count
            FROM search_log_results
            WHERE search_log_id = %s
            """,
            (body["search_log_id"],),
        )
        first_result_id = profile_results["kure_v1_1024"][0]["search_log_result_id"]

        with TestClient(app) as client:
            feedback_response = client.post(
                "/api/search/feedback",
                json={
                    "search_log_result_id": first_result_id,
                    "relevance_label": "correct",
                    "comment": "Visible fixture matched the query.",
                },
            )
            summary_response = client.get(
                "/api/search/feedback/summary",
                params={"document_group": document_group},
            )
            comments_response = client.get(
                "/api/search/feedback/comments",
                params={"document_group": document_group},
            )
            logs_response = client.get(
                "/api/search/logs",
                params={"document_group": document_group},
            )
            mock_provider_logs_response = client.get(
                "/api/search/logs",
                params={
                    "document_group": document_group,
                    "provider_mode_filter": "mock_fallback_allowed",
                },
            )
            real_provider_logs_response = client.get(
                "/api/search/logs",
                params={
                    "document_group": document_group,
                    "provider_mode_filter": "real_provider_required",
                },
            )
            invalid_provider_logs_response = client.get(
                "/api/search/logs",
                params={
                    "document_group": document_group,
                    "provider_mode_filter": "unsupported",
                },
            )
            filtered_history_page_response = client.get(
                "/search/logs",
                params={
                    "document_group": document_group,
                    "provider_mode_filter": "mock_fallback_allowed",
                },
            )
            search_page_response = client.get(
                "/search",
                params={
                    "query_text": query_text,
                    "document_group": document_group,
                    "chunk_policy_name": "heading_512_64",
                },
            )
            detail_response = client.get(f"/api/search/logs/{body['search_log_id']}")
            export_response = client.get(f"/api/search/logs/{body['search_log_id']}/export")
            export_csv_response = client.get(
                f"/api/search/logs/{body['search_log_id']}/export",
                params={"format": "csv"},
            )
            bad_export_response = client.get(
                f"/api/search/logs/{body['search_log_id']}/export",
                params={"format": "xlsx"},
            )

        feedback_body = feedback_response.json()
        summary_body = summary_response.json()
        comments_body = comments_response.json()
        logs_body = logs_response.json()
        detail_body = detail_response.json()
        export_body = export_response.json()
        feedback_count = fetch_one(
            migrated_database_url,
            """
            SELECT count(*) AS count
            FROM search_result_feedback
            WHERE search_log_result_id = %s
            """,
            (first_result_id,),
        )

        assert response.status_code == 200
        assert matrix_response.status_code == 200
        assert body["requested_search_scope"] == "company"
        assert body["effective_search_scope"] == "company"
        assert body["permission_filter_metadata"]["actor_user_id"] == ids["alice.member"]
        assert body["permission_summary"]["actor_user_id"] == ids["alice.member"]
        assert body["permission_summary"]["actor_login_id"] == "alice.member"
        assert body["permission_summary"]["requested_search_scope"] == "company"
        assert body["permission_summary"]["effective_search_scope"] == "company"
        assert body["permission_summary"]["scope_was_downgraded"] is False
        assert body["permission_summary"]["candidate_document_count"] == 2
        assert body["permission_summary"]["visible_document_count"] == 1
        assert body["permission_summary"]["excluded_document_count"] == 1
        assert body["permission_summary"]["visible_access_scope_counts"]["company"] == 1
        assert "hidden bob personal fixture" not in str(body["permission_summary"])
        assert (
            body["permission_filter_metadata"]["permission_explainability"]
            == body["permission_summary"]
        )
        assert set(profile_results) == {"kure_v1_1024", "bge_m3_1024"}
        assert body["profile_status_counts"] == {"succeeded": 2, "failed": 0}
        assert body["profile_failure_count"] == 0
        assert all(profile["status"] == "succeeded" for profile in body["profiles"])
        assert all(profile["error_code"] is None for profile in body["profiles"])
        assert all(profile["error_message"] is None for profile in body["profiles"])
        assert all(profile["query_runtime_metadata"] for profile in body["profiles"])
        assert result_chunk_ids == {visible_chunk_id}
        assert hidden_chunk_id not in result_chunk_ids
        assert result_count["count"] == 2
        assert all(
            "search_log_result_id" in result
            for results in profile_results.values()
            for result in results
        )
        matrix_by_actor = {entry["actor_user_id"]: entry for entry in matrix_body["entries"]}
        alice_matrix_log_id = matrix_by_actor[ids["alice.member"]]["search_log_id"]
        with TestClient(app) as client:
            compare_response = client.get(
                "/api/search/logs/compare",
                params={
                    "left_search_log_id": body["search_log_id"],
                    "right_search_log_id": alice_matrix_log_id,
                },
            )
            missing_compare_response = client.get(
                "/api/search/logs/compare",
                params={
                    "left_search_log_id": body["search_log_id"],
                    "right_search_log_id": 999999999,
                },
            )
            metadata_response = client.patch(
                f"/api/search/logs/{body['search_log_id']}/metadata",
                json={
                    "review_tags": ["baseline", "permission-review"],
                    "review_memo": "Important search experiment",
                    "reviewed_by_user_id": ids["alice.member"],
                },
            )
            invalid_metadata_response = client.patch(
                f"/api/search/logs/{body['search_log_id']}/metadata",
                json={"review_tags": ["duplicate", "duplicate"]},
            )
            missing_metadata_response = client.patch(
                "/api/search/logs/999999999/metadata",
                json={"review_tags": ["missing"]},
            )
            report_response = client.get(
                f"/api/search/logs/{body['search_log_id']}/experiment-report"
            )
            compare_report_response = client.get(
                f"/api/search/logs/{body['search_log_id']}/experiment-report",
                params={"compare_search_log_id": alice_matrix_log_id},
            )
            missing_compare_report_response = client.get(
                f"/api/search/logs/{body['search_log_id']}/experiment-report",
                params={"compare_search_log_id": 999999999},
            )
        compare_body = compare_response.json()
        compare_fields = {item["field"]: item for item in compare_body["reproducibility"]["fields"]}
        metadata_body = metadata_response.json()
        assert matrix_body["query_text"] == query_text
        assert matrix_body["top_k"] == 5
        assert matrix_by_actor[ids["alice.member"]]["result_count"] == 1
        assert matrix_by_actor[ids["alice.member"]]["unique_chunk_count"] == 1
        assert matrix_by_actor[ids["alice.member"]]["top_result"]["chunk_id"] == visible_chunk_id
        assert matrix_by_actor[ids["bob.member"]]["result_count"] == 2
        assert matrix_by_actor[ids["bob.member"]]["unique_chunk_count"] == 2
        assert matrix_by_actor[ids["bob.member"]]["effective_search_scope"] == "company"
        assert matrix_by_actor[ids["bob.member"]]["permission_filter_metadata"]["login_id"] == (
            "bob.member"
        )
        assert (
            matrix_by_actor[ids["bob.member"]]["permission_summary"]["excluded_document_count"] == 0
        )
        assert (
            matrix_by_actor[ids["bob.member"]]["permission_summary"]["visible_access_scope_counts"][
                "personal"
            ]
            == 1
        )
        assert compare_response.status_code == 200
        assert compare_body["left"]["search_log_id"] == body["search_log_id"]
        assert compare_body["right"]["search_log_id"] == alice_matrix_log_id
        assert compare_body["reproducibility"]["same_fingerprint"] is False
        assert compare_fields["query_text"]["matches"] is True
        assert compare_fields["profiles"]["matches"] is False
        assert compare_body["result_overlap"]["left_result_count"] == 2
        assert compare_body["result_overlap"]["right_result_count"] == 1
        assert compare_body["result_overlap"]["shared_chunk_count"] == 1
        assert compare_body["result_overlap"]["shared_chunk_ids"] == [visible_chunk_id]
        assert missing_compare_response.status_code == 404
        assert metadata_response.status_code == 200
        assert metadata_body["search_log"]["review_tags"] == [
            "baseline",
            "permission-review",
        ]
        assert metadata_body["search_log"]["review_memo"] == "Important search experiment"
        assert metadata_body["search_log"]["reviewed_by_user_id"] == ids["alice.member"]
        assert metadata_body["search_log"]["reviewed_at"]
        assert invalid_metadata_response.status_code == 400
        assert "review_tags must be unique" in invalid_metadata_response.json()["detail"]
        assert missing_metadata_response.status_code == 404
        assert feedback_response.status_code == 201
        assert feedback_body["feedback"]["relevance_label"] == "correct"
        assert feedback_body["feedback"]["comment"] == "Visible fixture matched the query."
        assert feedback_body["feedback"]["created_by"] == "search-compare-ui"
        assert feedback_body["feedback"]["created_by_user_id"] == ids["alice.member"]
        assert feedback_count["count"] == 1
        assert summary_response.status_code == 200
        assert summary_body["feedback_count"] == 1
        assert summary_body["search_log_count"] == 1
        assert summary_body["result_count"] == 1
        profile_summary = {profile["profile_name"]: profile for profile in summary_body["profiles"]}
        assert profile_summary["kure_v1_1024"]["feedback_count"] == 1
        assert profile_summary["kure_v1_1024"]["correct_count"] == 1
        assert profile_summary["kure_v1_1024"]["relevant_count"] == 1
        assert profile_summary["kure_v1_1024"]["correct_rate"] == 1
        assert profile_summary["bge_m3_1024"]["feedback_count"] == 0
        assert comments_response.status_code == 200
        assert len(comments_body["comments"]) == 1
        assert comments_body["comments"][0]["comment"] == "Visible fixture matched the query."
        assert comments_body["comments"][0]["search_log_id"] == body["search_log_id"]
        assert comments_body["comments"][0]["document_title"] == "visible company fixture"
        assert comments_body["comments"][0]["relevance_label"] == "correct"
        assert logs_response.status_code == 200
        assert len(logs_body["logs"]) == 3
        assert mock_provider_logs_response.status_code == 200
        assert real_provider_logs_response.status_code == 200
        assert invalid_provider_logs_response.status_code == 400
        assert len(mock_provider_logs_response.json()["logs"]) == 3
        assert real_provider_logs_response.json()["logs"] == []
        assert "provider_mode_filter" in invalid_provider_logs_response.json()["detail"]
        assert filtered_history_page_response.status_code == 200
        assert "Mock fallback 허용" in filtered_history_page_response.text
        assert str(body["search_log_id"]) in filtered_history_page_response.text
        assert search_page_response.status_code == 200
        assert 'id="chunk_policy_name"' in search_page_response.text
        assert "Chunk 정책" in search_page_response.text
        assert "전체 정책" in search_page_response.text
        assert "heading_512_64" in search_page_response.text
        logs_by_id = {log["search_log_id"]: log for log in logs_body["logs"]}
        matrix_log_ids = {entry["search_log_id"] for entry in matrix_body["entries"]}
        assert {body["search_log_id"], *matrix_log_ids} == set(logs_by_id)
        assert logs_by_id[body["search_log_id"]]["result_count"] == 2
        assert logs_by_id[body["search_log_id"]]["feedback_count"] == 1
        assert (
            logs_by_id[body["search_log_id"]]["permission_summary"]["excluded_document_count"] == 1
        )
        assert logs_by_id[body["search_log_id"]]["reproducibility_summary"]["profiles"] == [
            "kure_v1_1024",
            "bge_m3_1024",
        ]
        fingerprint = logs_by_id[body["search_log_id"]]["reproducibility_summary"]["fingerprint"]
        assert len(fingerprint) == 16
        assert (
            logs_by_id[body["search_log_id"]]["reproducibility_summary"]["fingerprint_algorithm"]
            == "sha256:16"
        )
        with TestClient(app) as client:
            fingerprint_logs_response = client.get(
                "/api/search/logs",
                params={
                    "document_group": document_group,
                    "fingerprint": fingerprint.upper(),
                },
            )
            empty_fingerprint_logs_response = client.get(
                "/api/search/logs",
                params={
                    "document_group": document_group,
                    "fingerprint": "0000000000000000",
                },
            )
        assert fingerprint_logs_response.status_code == 200
        assert [log["search_log_id"] for log in fingerprint_logs_response.json()["logs"]] == [
            body["search_log_id"]
        ]
        assert empty_fingerprint_logs_response.status_code == 200
        assert empty_fingerprint_logs_response.json()["logs"] == []
        assert (
            logs_by_id[body["search_log_id"]]["reproducibility_summary"]["query_runtime_metadata"][
                "search_mode"
            ]
            == "compare_mvp"
        )
        query_runtime_metadata = logs_by_id[body["search_log_id"]][
            "reproducibility_summary"
        ]["query_runtime_metadata"]
        assert query_runtime_metadata["adapter"] == "query_embedding_bridge"
        assert query_runtime_metadata["query_embedding_bridge"] is True
        assert query_runtime_metadata["query_embedding_profile_count"] == 2
        assert query_runtime_metadata["query_embedding_provider_types"] == ["mock"]
        assert query_runtime_metadata["query_embedding_runtime_sources"] == [
            "fallback_runtime_config"
        ]
        assert set(query_runtime_metadata["profile_query_embeddings"]) == {
            "kure_v1_1024",
            "bge_m3_1024",
        }
        assert (
            query_runtime_metadata["profile_query_embeddings"]["kure_v1_1024"]["dimension"]
            == 1024
        )
        assert (
            query_runtime_metadata["profile_query_embeddings"]["kure_v1_1024"][
                "runtime_metadata"
            ]["query_embedding_input_type"]
            == "query"
        )
        assert detail_response.status_code == 200
        assert detail_body["search_log"]["search_log_id"] == body["search_log_id"]
        assert detail_body["search_log"]["actor_login_id"] == "alice.member"
        assert detail_body["search_log"]["chunk_policy_name"] == "heading_512_64"
        assert detail_body["search_log"]["permission_summary"]["visible_document_count"] == 1
        assert detail_body["search_log"]["permission_summary"]["excluded_document_count"] == 1
        assert detail_body["search_log"]["reproducibility_summary"]["top_k"] == 5
        assert detail_body["search_log"]["reproducibility_summary"]["fingerprint"] == fingerprint
        assert detail_body["search_log"]["reproducibility_summary"]["document_group"] == (
            document_group
        )
        assert (
            "adapter"
            in detail_body["search_log"]["reproducibility_summary"]["runtime_metadata_keys"]
        )
        assert (
            detail_body["search_log"]["permission_filter_metadata"]["permission_explainability"][
                "excluded_document_count"
            ]
            == 1
        )
        assert len(detail_body["results"]) == 2
        assert detail_body["results"][0]["document_group"] == document_group
        detail_results = {result["profile_name"]: result for result in detail_body["results"]}
        assert detail_results["kure_v1_1024"]["feedback"][0]["relevance_label"] == "correct"
        assert detail_results["kure_v1_1024"]["feedback"][0]["comment"] == (
            "Visible fixture matched the query."
        )
        assert export_response.status_code == 200
        assert export_response.headers["content-disposition"].endswith(
            f'search-log-{body["search_log_id"]}.json"'
        )
        assert export_body["version"] == 1
        assert export_body["exported_at"]
        assert export_body["search_log"] == detail_body["search_log"]
        assert export_body["results"] == detail_body["results"]
        assert export_csv_response.status_code == 200
        assert export_csv_response.headers["content-type"].startswith("text/csv")
        assert export_csv_response.headers["content-disposition"].endswith(
            f'search-log-{body["search_log_id"]}.csv"'
        )
        assert "search_log_id,query_text,actor_login_id" in export_csv_response.text
        assert str(body["search_log_id"]) in export_csv_response.text
        assert "kure_v1_1024" in export_csv_response.text
        assert "correct" in export_csv_response.text
        assert "Visible fixture matched the query." in str(export_body["results"])
        assert report_response.status_code == 200
        assert report_response.headers["content-type"].startswith("text/markdown")
        assert report_response.headers["content-disposition"].endswith(
            f'search-log-{body["search_log_id"]}-experiment-report.md"'
        )
        assert "# Search Experiment Report" in report_response.text
        assert f"Search Log ID: {body['search_log_id']}" in report_response.text
        assert f"| Query | {query_text} |" in report_response.text
        assert "visible company fixture" in report_response.text
        assert "| Review Tags | baseline, permission-review |" in report_response.text
        assert "| Review Memo | Important search experiment |" in report_response.text
        assert "correct" in report_response.text
        assert compare_report_response.status_code == 200
        assert "## Compare Summary" in compare_report_response.text
        assert f"| Target Search Log ID | #{alice_matrix_log_id} |" in (
            compare_report_response.text
        )
        assert "| Shared Chunks | 1 |" in compare_report_response.text
        assert "## Reproducibility Field Compare" in compare_report_response.text
        assert missing_compare_report_response.status_code == 404
        assert bad_export_response.status_code == 400
        assert bad_export_response.json() == {"detail": "format must be json or csv."}
    finally:
        _cleanup_files(migrated_database_url, [visible_file_id, hidden_file_id])


def test_search_compare_api_returns_profile_failure_for_provider_errors(
    migrated_database_url: str,
) -> None:
    ids = _seed_ids(migrated_database_url)
    app = create_app(
        Settings(
            database_url=migrated_database_url,
            embedding_provider_mode="remote",
            remote_embedding_provider_url="http://127.0.0.1:9",
            remote_embedding_provider_timeout_seconds=0.01,
        )
    )
    search_log_id: int | None = None

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/search/compare",
                json={
                    "query_text": "provider failure fixture",
                    "actor_user_id": ids["alice.member"],
                    "requested_search_scope": "company",
                    "top_k": 5,
                    "profiles": ["kure_v1_1024"],
                    "document_group": f"provider-failure-{uuid4()}",
                },
            )

        assert response.status_code == 200
        body = response.json()
        search_log_id = body["search_log_id"]
        profile = body["profiles"][0]

        assert body["profile_status_counts"] == {"succeeded": 0, "failed": 1}
        assert body["profile_failure_count"] == 1
        assert profile["profile_name"] == "kure_v1_1024"
        assert profile["status"] == "failed"
        assert profile["error_code"] == "query_embedding_failed"
        assert "Remote provider request failed" in profile["error_message"]
        assert profile["results"] == []
        assert profile["query_runtime_metadata"]["error_code"] == "query_embedding_failed"
    finally:
        if search_log_id is not None:
            _delete_search_logs(migrated_database_url, [search_log_id])


def test_search_log_profile_retry_api_replays_source_conditions(
    migrated_database_url: str,
) -> None:
    ids = _seed_ids(migrated_database_url)
    document_group = f"profile-retry-{uuid4()}"
    source_log = create_search_log(
        migrated_database_url,
        SearchLogInput(
            query_text="Retry failed profile fixture",
            normalized_query_text="retry failed profile fixture",
            actor_user_id=ids["alice.member"],
            requested_search_scope="company",
            effective_search_scope="company",
            permission_filter_metadata={"fixture": "profile_retry"},
            document_group=document_group,
            file_type=".md",
            chunk_policy_name="heading_512_64",
            top_k=3,
            profiles=("kure_v1_1024", "bge_m3_1024"),
            query_runtime_metadata={
                "adapter": "query_embedding_bridge",
                "search_mode": "compare_mvp",
                "query_embedding_bridge": True,
                "profile_status_counts": {"succeeded": 1, "failed": 1},
                "profile_failure_count": 1,
                "profile_query_embeddings": {
                    "kure_v1_1024": {
                        "profile_name": "kure_v1_1024",
                        "dimension": 1024,
                        "provider_type": "mock",
                        "runtime_source": "fallback",
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
            total_elapsed_ms=12,
            created_by_user_id=ids["alice.member"],
        ),
    )
    retry_log_id: int | None = None
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/search/logs/{source_log.search_log_id}/retry-profile",
                json={"profile_name": "bge_m3_1024"},
            )
            invalid_profile_response = client.post(
                f"/api/search/logs/{source_log.search_log_id}/retry-profile",
                json={"profile_name": "missing_profile"},
            )
            missing_response = client.post(
                "/api/search/logs/999999999/retry-profile",
                json={"profile_name": "bge_m3_1024"},
            )

        assert response.status_code == 200
        body = response.json()
        retry_log_id = body["search_result"]["search_log_id"]
        retry_profile = body["search_result"]["profiles"][0]

        assert body["source_search_log_id"] == source_log.search_log_id
        assert body["retry_profile_name"] == "bge_m3_1024"
        assert body["retry_search_log_url"] == f"/search/logs?search_log_id={retry_log_id}"
        assert body["search_result"]["query_text"] == source_log.query_text
        assert body["search_result"]["actor_user_id"] == ids["alice.member"]
        assert body["search_result"]["requested_search_scope"] == "company"
        assert body["search_result"]["effective_search_scope"] == "company"
        assert body["search_result"]["top_k"] == 3
        assert body["search_result"]["profile_status_counts"] == {"succeeded": 1, "failed": 0}
        assert retry_profile["profile_name"] == "bge_m3_1024"
        assert retry_profile["status"] == "succeeded"
        assert invalid_profile_response.status_code == 400
        assert invalid_profile_response.json() == {
            "detail": "profile_name is not included in the source search log."
        }
        assert missing_response.status_code == 404
        assert missing_response.json() == {"detail": "Search log not found."}
    finally:
        _delete_search_logs(
            migrated_database_url,
            [
                search_log_id
                for search_log_id in (source_log.search_log_id, retry_log_id)
                if search_log_id is not None
            ],
        )


def test_search_runtime_failures_api_lists_and_filters_profile_failures(
    migrated_database_url: str,
) -> None:
    ids = _seed_ids(migrated_database_url)
    first_log = create_search_log(
        migrated_database_url,
        SearchLogInput(
            query_text="Runtime failure triage first",
            normalized_query_text="runtime failure triage first",
            actor_user_id=ids["alice.member"],
            requested_search_scope="company",
            effective_search_scope="company",
            document_group=f"runtime-failure-triage-{uuid4()}",
            top_k=5,
            profiles=("kure_v1_1024", "bge_m3_1024"),
            query_runtime_metadata={
                "query_embedding_bridge": True,
                "profile_status_counts": {"succeeded": 1, "failed": 1},
                "profile_failures": {
                    "bge_m3_1024": {
                        "profile_name": "bge_m3_1024",
                        "status": "failed",
                        "error_code": "query_embedding_failed",
                        "error_message": "BGE provider timeout",
                        "elapsed_ms": 250,
                    }
                },
            },
            created_by_user_id=ids["alice.member"],
        ),
    )
    second_log = create_search_log(
        migrated_database_url,
        SearchLogInput(
            query_text="Runtime failure triage second",
            normalized_query_text="runtime failure triage second",
            actor_user_id=ids["alice.member"],
            requested_search_scope="team",
            effective_search_scope="team",
            document_group=f"runtime-failure-triage-{uuid4()}",
            top_k=3,
            profiles=("kure_v1_1024",),
            query_runtime_metadata={
                "query_embedding_bridge": True,
                "profile_status_counts": {"succeeded": 0, "failed": 1},
                "profile_failures": {
                    "kure_v1_1024": {
                        "profile_name": "kure_v1_1024",
                        "status": "failed",
                        "error_code": "vector_search_failed",
                        "error_message": "Vector search failed",
                        "elapsed_ms": 88,
                    }
                },
            },
            created_by_user_id=ids["alice.member"],
        ),
    )
    app = create_app(Settings(database_url=migrated_database_url))
    retry_log_ids: list[int] = []

    try:
        with TestClient(app) as client:
            response = client.get("/api/search/logs/runtime-failures", params={"limit": 20})
            filtered_response = client.get(
                "/api/search/logs/runtime-failures",
                params={"profile_name": "bge_m3_1024", "limit": 20},
            )
            invalid_limit_response = client.get(
                "/api/search/logs/runtime-failures",
                params={"limit": 0},
            )
            bulk_retry_response = client.post(
                "/api/search/logs/runtime-failures/retry",
                json={
                    "failures": [
                        {
                            "search_log_id": first_log.search_log_id,
                            "profile_name": "bge_m3_1024",
                        },
                        {
                            "search_log_id": first_log.search_log_id,
                            "profile_name": "missing_profile",
                        },
                        {
                            "search_log_id": 999999999,
                            "profile_name": "bge_m3_1024",
                        },
                    ]
                },
            )
            empty_retry_response = client.post(
                "/api/search/logs/runtime-failures/retry",
                json={"failures": []},
            )
            too_many_retry_response = client.post(
                "/api/search/logs/runtime-failures/retry",
                json={
                    "failures": [
                        {
                            "search_log_id": first_log.search_log_id,
                            "profile_name": "bge_m3_1024",
                        }
                        for _ in range(21)
                    ]
                },
            )

        assert response.status_code == 200
        body = response.json()
        failures_by_log_id = {
            failure["search_log_id"]: failure for failure in body["failures"]
        }
        assert first_log.search_log_id in failures_by_log_id
        assert second_log.search_log_id in failures_by_log_id
        assert failures_by_log_id[first_log.search_log_id]["profile_name"] == "bge_m3_1024"
        assert failures_by_log_id[first_log.search_log_id]["error_code"] == (
            "query_embedding_failed"
        )
        assert failures_by_log_id[first_log.search_log_id]["elapsed_ms"] == 250
        assert failures_by_log_id[first_log.search_log_id]["actor_display_name"] == (
            "Alice Member"
        )
        assert failures_by_log_id[first_log.search_log_id]["search_log_url"] == (
            f"/search/logs?search_log_id={first_log.search_log_id}"
        )
        assert filtered_response.status_code == 200
        filtered_failures = filtered_response.json()["failures"]
        assert any(
            failure["search_log_id"] == first_log.search_log_id
            for failure in filtered_failures
        )
        assert all(failure["profile_name"] == "bge_m3_1024" for failure in filtered_failures)
        assert invalid_limit_response.status_code == 400
        assert "limit" in invalid_limit_response.json()["detail"]
        assert bulk_retry_response.status_code == 200
        bulk_body = bulk_retry_response.json()
        assert bulk_body["requested_count"] == 3
        assert bulk_body["retried_count"] == 1
        assert bulk_body["failed_count"] == 2
        succeeded_retry = next(
            result for result in bulk_body["results"] if result["status"] == "succeeded"
        )
        retry_log_ids.append(succeeded_retry["retry_search_log_id"])
        assert succeeded_retry["source_search_log_id"] == first_log.search_log_id
        assert succeeded_retry["retry_profile_name"] == "bge_m3_1024"
        assert succeeded_retry["retry_search_log_url"] == (
            f"/search/logs?search_log_id={succeeded_retry['retry_search_log_id']}"
        )
        failed_details = [
            result["detail"] for result in bulk_body["results"] if result["status"] == "failed"
        ]
        assert "profile_name is not included in the source search log." in failed_details
        assert "Search log not found." in failed_details
        assert empty_retry_response.status_code == 400
        assert empty_retry_response.json() == {"detail": "failures must not be empty."}
        assert too_many_retry_response.status_code == 400
        assert too_many_retry_response.json() == {
            "detail": "failures must contain at most 20 items."
        }
    finally:
        _delete_search_logs(
            migrated_database_url,
            [first_log.search_log_id, second_log.search_log_id, *retry_log_ids],
        )


def test_search_latency_outliers_api_lists_slow_search_logs(
    migrated_database_url: str,
) -> None:
    ids = _seed_ids(migrated_database_url)
    slow_log = create_search_log(
        migrated_database_url,
        SearchLogInput(
            query_text="Latency outlier slow fixture",
            normalized_query_text="latency outlier slow fixture",
            actor_user_id=ids["alice.member"],
            requested_search_scope="company",
            effective_search_scope="company",
            document_group=f"latency-outlier-{uuid4()}",
            top_k=5,
            profiles=("kure_v1_1024", "bge_m3_1024"),
            query_runtime_metadata={
                "query_embedding_bridge": True,
                "profile_status_counts": {"succeeded": 1, "failed": 1},
            },
            total_elapsed_ms=1750,
            created_by_user_id=ids["alice.member"],
        ),
    )
    fast_log = create_search_log(
        migrated_database_url,
        SearchLogInput(
            query_text="Latency outlier fast fixture",
            normalized_query_text="latency outlier fast fixture",
            actor_user_id=ids["alice.member"],
            requested_search_scope="company",
            effective_search_scope="company",
            document_group=f"latency-outlier-{uuid4()}",
            top_k=5,
            profiles=("kure_v1_1024",),
            query_runtime_metadata={
                "query_embedding_bridge": True,
                "profile_status_counts": {"succeeded": 1, "failed": 0},
            },
            total_elapsed_ms=250,
            created_by_user_id=ids["alice.member"],
        ),
    )
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            response = client.get(
                "/api/search/logs/latency-outliers",
                params={"min_total_elapsed_ms": 1000, "limit": 20},
            )
            low_threshold_response = client.get(
                "/api/search/logs/latency-outliers",
                params={"min_total_elapsed_ms": 0, "limit": 20},
            )
            invalid_threshold_response = client.get(
                "/api/search/logs/latency-outliers",
                params={"min_total_elapsed_ms": -1},
            )

        assert response.status_code == 200
        body = response.json()
        outliers_by_log_id = {
            outlier["search_log_id"]: outlier for outlier in body["outliers"]
        }
        assert body["min_total_elapsed_ms"] == 1000
        assert slow_log.search_log_id in outliers_by_log_id
        assert fast_log.search_log_id not in outliers_by_log_id
        assert outliers_by_log_id[slow_log.search_log_id]["total_elapsed_ms"] == 1750
        assert outliers_by_log_id[slow_log.search_log_id]["profile_count"] == 2
        assert outliers_by_log_id[slow_log.search_log_id]["failed_profile_count"] == 1
        assert outliers_by_log_id[slow_log.search_log_id]["actor_display_name"] == (
            "Alice Member"
        )
        assert low_threshold_response.status_code == 200
        low_threshold_ids = {
            outlier["search_log_id"] for outlier in low_threshold_response.json()["outliers"]
        }
        assert slow_log.search_log_id in low_threshold_ids
        assert fast_log.search_log_id in low_threshold_ids
        assert invalid_threshold_response.status_code == 400
        assert "min_total_elapsed_ms" in invalid_threshold_response.json()["detail"]
    finally:
        _delete_search_logs(
            migrated_database_url,
            [slow_log.search_log_id, fast_log.search_log_id],
        )


def test_search_no_result_logs_api_lists_zero_result_searches(
    migrated_database_url: str,
) -> None:
    ids = _seed_ids(migrated_database_url)
    search_log = create_search_log(
        migrated_database_url,
        SearchLogInput(
            query_text="No result triage fixture",
            normalized_query_text="no result triage fixture",
            actor_user_id=ids["alice.member"],
            requested_search_scope="company",
            effective_search_scope="company",
            document_group=f"no-result-triage-{uuid4()}",
            file_type=".md",
            chunk_policy_name="heading_512_64",
            top_k=5,
            profiles=("kure_v1_1024", "bge_m3_1024"),
            query_runtime_metadata={
                "query_embedding_bridge": True,
                "profile_status_counts": {"succeeded": 2, "failed": 0},
            },
            total_elapsed_ms=640,
            created_by_user_id=ids["alice.member"],
        ),
    )
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            response = client.get("/api/search/logs/no-results", params={"limit": 20})
            invalid_limit_response = client.get(
                "/api/search/logs/no-results",
                params={"limit": 0},
            )

        assert response.status_code == 200
        records_by_log_id = {
            record["search_log_id"]: record for record in response.json()["records"]
        }
        assert search_log.search_log_id in records_by_log_id
        record = records_by_log_id[search_log.search_log_id]
        assert record["query_text"] == "No result triage fixture"
        assert record["profile_count"] == 2
        assert record["failed_profile_count"] == 0
        assert record["total_elapsed_ms"] == 640
        assert len(record["created_at_label"]) == len("2026-07-16 13:02:05")
        assert "T" not in record["created_at_label"]
        assert record["search_log_url"] == f"/search/logs?search_log_id={search_log.search_log_id}"
        assert invalid_limit_response.status_code == 400
        assert "limit" in invalid_limit_response.json()["detail"]
    finally:
        _delete_search_logs(migrated_database_url, [search_log.search_log_id])


def test_search_duplicate_fingerprints_api_groups_repeated_conditions(
    migrated_database_url: str,
) -> None:
    ids = _seed_ids(migrated_database_url)
    document_group = f"duplicate-fingerprint-{uuid4()}"
    first_log = create_search_log(
        migrated_database_url,
        SearchLogInput(
            query_text="Repeated search condition",
            normalized_query_text="repeated search condition",
            actor_user_id=ids["alice.member"],
            requested_search_scope="company",
            effective_search_scope="team",
            permission_filter_metadata={"visible_document_count": 3},
            document_group=document_group,
            file_type=".md",
            chunk_policy_name="heading_512_64",
            top_k=5,
            profiles=("kure_v1_1024", "bge_m3_1024"),
            query_runtime_metadata={
                "profile_status_counts": {"succeeded": 2, "failed": 0},
            },
            total_elapsed_ms=600,
            created_by_user_id=ids["alice.member"],
        ),
    )
    second_log = create_search_log(
        migrated_database_url,
        SearchLogInput(
            query_text="Repeated search condition",
            normalized_query_text="repeated search condition",
            actor_user_id=ids["alice.member"],
            requested_search_scope="company",
            effective_search_scope="team",
            permission_filter_metadata={"visible_document_count": 3},
            document_group=document_group,
            file_type=".md",
            chunk_policy_name="heading_512_64",
            top_k=5,
            profiles=("kure_v1_1024", "bge_m3_1024"),
            query_runtime_metadata={
                "profile_failures": {
                    "bge_m3_1024": {
                        "error_code": "query_embedding_failed",
                        "error_message": "provider failed",
                    }
                },
                "profile_status_counts": {"succeeded": 1, "failed": 1},
            },
            total_elapsed_ms=900,
            created_by_user_id=ids["alice.member"],
        ),
    )
    different_top_k_log = create_search_log(
        migrated_database_url,
        SearchLogInput(
            query_text="Repeated search condition",
            normalized_query_text="repeated search condition",
            actor_user_id=ids["alice.member"],
            requested_search_scope="company",
            effective_search_scope="team",
            permission_filter_metadata={"visible_document_count": 3},
            document_group=document_group,
            file_type=".md",
            chunk_policy_name="heading_512_64",
            top_k=3,
            profiles=("kure_v1_1024", "bge_m3_1024"),
            query_runtime_metadata={
                "profile_status_counts": {"succeeded": 2, "failed": 0},
            },
            total_elapsed_ms=120,
            created_by_user_id=ids["alice.member"],
        ),
    )
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            response = client.get(
                "/api/search/logs/duplicate-fingerprints",
                params={"min_count": 2, "limit": 20},
            )
            min_three_response = client.get(
                "/api/search/logs/duplicate-fingerprints",
                params={"min_count": 3, "limit": 20},
            )
            invalid_min_response = client.get(
                "/api/search/logs/duplicate-fingerprints",
                params={"min_count": 1},
            )

        assert response.status_code == 200
        matching_records = [
            record
            for record in response.json()["records"]
            if record["document_group"] == document_group and record["top_k"] == 5
        ]
        assert len(matching_records) == 1
        record = matching_records[0]
        assert record["duplicate_count"] == 2
        assert record["latest_search_log_id"] == second_log.search_log_id
        assert record["first_search_log_id"] == first_log.search_log_id
        assert record["profile_count"] == 2
        assert record["zero_result_count"] == 2
        assert record["runtime_failure_count"] == 1
        assert record["average_total_elapsed_ms"] == 750.0
        assert len(record["condition_fingerprint"]) == 32
        assert "T" not in record["latest_created_at_label"]
        assert record["latest_search_log_url"] == (
            f"/search/logs?search_log_id={second_log.search_log_id}"
        )
        assert response.json()["min_count"] == 2
        assert all(
            record["document_group"] != document_group
            for record in min_three_response.json()["records"]
        )
        assert invalid_min_response.status_code == 400
        assert "min_count" in invalid_min_response.json()["detail"]
    finally:
        _delete_search_logs(
            migrated_database_url,
            [
                first_log.search_log_id,
                second_log.search_log_id,
                different_top_k_log.search_log_id,
            ],
        )


def test_search_operations_summary_api_returns_recent_signal_deltas(
    migrated_database_url: str,
) -> None:
    ids = _seed_ids(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))
    document_group = f"operations-summary-{uuid4()}"
    file_id, chunk_id = _create_search_compare_chunk(
        migrated_database_url,
        title="Operations summary result fixture",
        owner_user_id=ids["alice.member"],
        owner_org_unit_id=ids["Business Team"],
        access_scope="team",
        chunk_text="operations summary chunk",
        document_group=document_group,
    )

    with TestClient(app) as client:
        baseline_response = client.get("/api/search/logs/operations-summary")
    assert baseline_response.status_code == 200
    baseline = baseline_response.json()["operations_summary"]

    result_log = create_search_log(
        migrated_database_url,
        SearchLogInput(
            query_text="Operations summary repeated condition",
            normalized_query_text="operations summary repeated condition",
            actor_user_id=ids["alice.member"],
            requested_search_scope="company",
            effective_search_scope="team",
            permission_filter_metadata={"visible_document_count": 1},
            document_group=document_group,
            file_type=".md",
            chunk_policy_name="heading_512_64",
            top_k=5,
            profiles=("kure_v1_1024",),
            query_runtime_metadata={
                "allow_mock_fallback": True,
                "real_provider_required": False,
                "profile_status_counts": {"succeeded": 1, "failed": 0},
            },
            total_elapsed_ms=1200,
            created_by_user_id=ids["alice.member"],
        ),
    )
    create_search_log_results(
        migrated_database_url,
        [
            SearchLogResultInput(
                search_log_id=result_log.search_log_id,
                profile_name="kure_v1_1024",
                rank=1,
                chunk_id=chunk_id,
                score=0.94,
                profile_elapsed_ms=33,
            )
        ],
    )
    no_result_log = create_search_log(
        migrated_database_url,
        SearchLogInput(
            query_text="Operations summary repeated condition",
            normalized_query_text="operations summary repeated condition",
            actor_user_id=ids["alice.member"],
            requested_search_scope="company",
            effective_search_scope="team",
            permission_filter_metadata={"visible_document_count": 1},
            document_group=document_group,
            file_type=".md",
            chunk_policy_name="heading_512_64",
            top_k=5,
            profiles=("kure_v1_1024",),
            query_runtime_metadata={
                "allow_mock_fallback": False,
                "real_provider_required": True,
                "profile_failures": {
                    "kure_v1_1024": {
                        "error_code": "query_embedding_failed",
                        "error_message": "provider failed",
                    }
                },
                "profile_status_counts": {"succeeded": 0, "failed": 1},
            },
            total_elapsed_ms=1500,
            created_by_user_id=ids["alice.member"],
        ),
    )

    try:
        with TestClient(app) as client:
            response = client.get(
                "/api/search/logs/operations-summary",
                params={"lookback_hours": 24, "min_total_elapsed_ms": 1000},
            )
            invalid_lookback_response = client.get(
                "/api/search/logs/operations-summary",
                params={"lookback_hours": 0},
            )
            invalid_threshold_response = client.get(
                "/api/search/logs/operations-summary",
                params={"min_total_elapsed_ms": -1},
            )
            detail_page_response = client.get(
                "/search/logs",
                params={"search_log_id": no_result_log.search_log_id},
            )

        assert response.status_code == 200
        summary = response.json()["operations_summary"]
        assert summary["lookback_hours"] == 24
        assert summary["min_total_elapsed_ms"] == 1000
        assert summary["search_count"] == baseline["search_count"] + 2
        assert summary["result_row_count"] == baseline["result_row_count"] + 1
        assert summary["no_result_count"] == baseline["no_result_count"] + 1
        assert summary["runtime_failure_count"] == baseline["runtime_failure_count"] + 1
        assert summary["latency_outlier_count"] == baseline["latency_outlier_count"] + 2
        assert summary["real_provider_required_count"] == (
            baseline["real_provider_required_count"] + 1
        )
        assert summary["mock_fallback_allowed_count"] == (
            baseline["mock_fallback_allowed_count"] + 1
        )
        assert summary["duplicate_fingerprint_count"] == (
            baseline["duplicate_fingerprint_count"] + 1
        )
        assert summary["max_duplicate_count"] >= 2
        assert summary["average_total_elapsed_ms"] is not None
        assert "T" not in summary["latest_search_at_label"]
        assert invalid_lookback_response.status_code == 400
        assert "lookback_hours" in invalid_lookback_response.json()["detail"]
        assert invalid_threshold_response.status_code == 400
        assert "min_total_elapsed_ms" in invalid_threshold_response.json()["detail"]
        assert detail_page_response.status_code == 200
        assert "실제 provider 필수" in detail_page_response.text
        assert "mock fallback 차단" in detail_page_response.text
    finally:
        _delete_search_logs(migrated_database_url, [no_result_log.search_log_id])
        _cleanup_files(migrated_database_url, [file_id])


def test_search_compare_api_handles_invalid_scope(
    migrated_database_url: str,
) -> None:
    ids = _seed_ids(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))

    with TestClient(app) as client:
        response = client.post(
            "/api/search/compare",
            json={
                "query_text": "hello",
                "actor_user_id": ids["alice.member"],
                "requested_search_scope": "all",
            },
        )

    assert response.status_code == 400
    assert "requested_search_scope" in response.json()["detail"]

    with TestClient(app) as client:
        matrix_response = client.post(
            "/api/search/permission-matrix",
            json={
                "query_text": "hello",
                "top_k": 5,
                "entries": [
                    {
                        "actor_user_id": ids["alice.member"],
                        "requested_search_scope": "company",
                    },
                    {
                        "actor_user_id": ids["alice.member"],
                        "requested_search_scope": "company",
                    },
                ],
            },
        )

    assert matrix_response.status_code == 400
    assert "entries must be unique" in matrix_response.json()["detail"]


def test_search_feedback_api_returns_not_found_for_missing_result(
    migrated_database_url: str,
) -> None:
    app = create_app(Settings(database_url=migrated_database_url))

    with TestClient(app) as client:
        response = client.post(
            "/api/search/feedback",
            json={
                "search_log_result_id": 999999999,
                "relevance_label": "correct",
            },
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "Search result not found."}


def test_search_log_detail_api_returns_not_found_for_missing_log(
    migrated_database_url: str,
) -> None:
    app = create_app(Settings(database_url=migrated_database_url))

    with TestClient(app) as client:
        response = client.get("/api/search/logs/999999999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Search log not found."}
