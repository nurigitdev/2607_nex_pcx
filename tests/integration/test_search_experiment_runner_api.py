from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect
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
                WHERE login_id = 'pcx.admin'
                """)
            users = {row["login_id"]: int(row["user_id"]) for row in cursor.fetchall()}
    return users


def _create_api_runner_document(
    database_url: str,
    *,
    document_group: str,
) -> tuple[int, str]:
    checksum = f"search-experiment-api-{uuid4()}"
    chunk_text = "search experiment runner API inverter guide"
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
                    parse_status
                )
                VALUES (%s, %s, '.md', 1, %s, %s, %s, 'succeeded')
                RETURNING file_id
                """,
                (
                    f"{checksum}.md",
                    f"{checksum}.stored.md",
                    checksum,
                    f"/tmp/{checksum}.md",
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
                    access_scope
                )
                VALUES (%s, 'Search Experiment API Fixture', %s, 'company')
                RETURNING document_id
                """,
                (file_id, document_group),
            )
            document_id = int(cursor.fetchone()["document_id"])
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
            chunk_id = int(cursor.fetchone()["chunk_id"])

    for profile_name in ("bge_m3_1024", "kure_v1_1024"):
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
                elapsed_ms=3,
            ),
        )
    return file_id, chunk_text


def _cleanup_api_runner_fixture(
    database_url: str,
    *,
    file_id: int | None,
    experiment_run_id: int | None,
    search_log_id: int | None,
) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            if experiment_run_id is not None:
                cursor.execute(
                    "DELETE FROM search_experiment_runs WHERE experiment_run_id = %s",
                    (experiment_run_id,),
                )
            if search_log_id is not None:
                cursor.execute(
                    "DELETE FROM search_logs WHERE search_log_id = %s",
                    (search_log_id,),
                )
            if file_id is not None:
                cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def test_search_experiment_run_api_executes_experiment(
    migrated_database_url: str,
) -> None:
    ids = _seed_ids(migrated_database_url)
    document_group = f"slice-168-{uuid4()}"
    file_id, chunk_text = _create_api_runner_document(
        migrated_database_url,
        document_group=document_group,
    )
    body: dict[str, object] | None = None
    try:
        app = create_app(Settings(database_url=migrated_database_url))
        with TestClient(app) as client:
            response = client.post(
                "/api/search/experiments/run",
                json={
                    "run_name": f"Slice 168 API {uuid4()}",
                    "query_text": chunk_text,
                    "actor_user_id": ids["pcx.admin"],
                    "requested_search_scope": "company",
                    "profiles": ["bge_m3_1024", "kure_v1_1024"],
                    "strategy_name": "vector_cosine_threshold",
                    "top_k": 1,
                    "score_threshold": 0.0,
                    "chunk_policy_name": "heading_512_64",
                    "document_group": document_group,
                    "file_type": ".md",
                    "runtime_metadata": {"slice": 168},
                },
            )
            body = response.json()
            experiment_run_id = body.get("experiment_run", {}).get("experiment_run_id")
            list_response = client.get(
                "/api/search/experiments",
                params={"status": "succeeded", "limit": 10},
            )
            detail_response = client.get(f"/api/search/experiments/{experiment_run_id}")
            missing_detail_response = client.get("/api/search/experiments/999999999")

        assert response.status_code == 200
        experiment_run = body["experiment_run"]
        assert experiment_run["status"] == "succeeded"
        assert experiment_run["result_count"] == 2
        assert experiment_run["strategy_name"] == "vector_cosine_threshold"
        assert experiment_run["runtime_metadata"]["slice"] == 168
        assert body["strategy"]["score_threshold"] == 0.0
        assert body["search_result"]["search_log_id"] > 0
        assert [item["retained_result_count"] for item in body["profile_summaries"]] == [1, 1]
        list_body = list_response.json()
        detail_body = detail_response.json()
        assert list_response.status_code == 200
        assert detail_response.status_code == 200
        assert missing_detail_response.status_code == 404
        assert any(
            item["experiment_run_id"] == experiment_run["experiment_run_id"]
            for item in list_body["experiments"]
        )
        assert detail_body["experiment_run"]["experiment_run_id"] == experiment_run[
            "experiment_run_id"
        ]
        assert [profile["result_count"] for profile in detail_body["profiles"]] == [1, 1]
    finally:
        experiment_run_id = None
        search_log_id = None
        if body is not None and "experiment_run" in body:
            experiment_run_id = int(body["experiment_run"]["experiment_run_id"])
            search_log_id = int(body["search_result"]["search_log_id"])
        _cleanup_api_runner_fixture(
            migrated_database_url,
            file_id=file_id,
            experiment_run_id=experiment_run_id,
            search_log_id=search_log_id,
        )


def test_search_experiment_run_api_rejects_invalid_strategy(
    migrated_database_url: str,
) -> None:
    ids = _seed_ids(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))

    with TestClient(app) as client:
        response = client.post(
            "/api/search/experiments/run",
            json={
                "run_name": "invalid strategy",
                "query_text": "hello",
                "actor_user_id": ids["pcx.admin"],
                "strategy_name": "unknown_strategy",
            },
        )

    assert response.status_code == 400
    assert "Unsupported search strategy" in response.json()["detail"]
