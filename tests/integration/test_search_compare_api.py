from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect, fetch_one
from app.core.embedding_vectors import (
    EmbeddingVectorInput,
    generate_mock_embedding,
    store_chunk_embedding,
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
                    "document_group": document_group,
                },
            )
            matrix_response = client.post(
                "/api/search/permission-matrix",
                json={
                    "query_text": query_text,
                    "top_k": 5,
                    "profiles": ["kure_v1_1024"],
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
        assert detail_response.status_code == 200
        assert detail_body["search_log"]["search_log_id"] == body["search_log_id"]
        assert detail_body["search_log"]["actor_login_id"] == "alice.member"
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
