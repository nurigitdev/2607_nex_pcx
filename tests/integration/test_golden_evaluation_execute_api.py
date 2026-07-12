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
from app.core.golden_questions import (
    GoldenQuestionExpectedTargetInput,
    GoldenQuestionInput,
    GoldenQuestionSetInput,
    create_expected_target,
    create_golden_question,
    create_golden_question_set,
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
    checksum = f"golden-execute-{uuid4()}"
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


def _create_execute_fixture(database_url: str) -> dict[str, object]:
    ids = _seed_ids(database_url)
    query_text = f"Golden execute API anchor {uuid4()}"
    document_group = f"slice-044-{uuid4()}"
    file_id, chunk_id = _create_search_compare_chunk(
        database_url,
        title="golden execute visible fixture",
        owner_user_id=None,
        owner_org_unit_id=ids["NeX Company"],
        access_scope="company",
        chunk_text=query_text,
        document_group=document_group,
    )
    _store_profile_embeddings(database_url, chunk_id, query_text)
    question_set = create_golden_question_set(
        database_url,
        GoldenQuestionSetInput(
            set_name=f"slice-044-{uuid4()}",
            description="Slice 044 execute fixture",
            created_by_user_id=ids["alice.member"],
        ),
    )
    question = create_golden_question(
        database_url,
        GoldenQuestionInput(
            question_set_id=question_set.question_set_id,
            question_text=query_text,
            question_type="single_fact",
            actor_user_id=ids["alice.member"],
            requested_search_scope="company",
            document_group=document_group,
            file_type=".md",
            chunk_policy_name="heading_512_64",
            top_k=3,
            created_by_user_id=ids["alice.member"],
        ),
    )
    create_expected_target(
        database_url,
        GoldenQuestionExpectedTargetInput(
            question_id=question.question_id,
            chunk_id=chunk_id,
            expectation_type="visible",
            relevance_grade=3,
        ),
    )
    return {
        "file_id": file_id,
        "chunk_id": chunk_id,
        "question_set_id": question_set.question_set_id,
        "question_id": question.question_id,
        "query_text": query_text,
        "alice_user_id": ids["alice.member"],
    }


def _cleanup_question_set(database_url: str, question_set_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM golden_question_sets WHERE question_set_id = %s",
                (question_set_id,),
            )


def _cleanup_search_experiment_batch(database_url: str, body: dict[str, object]) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            for item in body.get("questions", []):
                experiment = item["experiment"]
                cursor.execute(
                    "DELETE FROM search_experiment_runs WHERE experiment_run_id = %s",
                    (experiment["experiment_run"]["experiment_run_id"],),
                )
                cursor.execute(
                    "DELETE FROM search_logs WHERE search_log_id = %s",
                    (experiment["search_result"]["search_log_id"],),
                )


def test_golden_evaluation_execute_api_runs_search_and_persists_results(
    migrated_database_url: str,
) -> None:
    fixture = _create_execute_fixture(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/evaluations/runs/execute",
                json={
                    "question_set_id": fixture["question_set_id"],
                    "profile_name": "kure_v1_1024",
                    "run_name": f"slice-044-execute-{uuid4()}",
                    "chunk_policy_name": "heading_512_64",
                    "top_k": 3,
                    "runtime_metadata": {"slice": "044"},
                },
            )

        body = response.json()["execution"]
        run = body["run"]
        result = body["results"][0]
        search_log_mapping = body["search_log_ids_by_question"]

        with TestClient(app) as client:
            detail_response = client.get(f"/api/evaluations/runs/{run['evaluation_run_id']}")
            permission_audit_response = client.get(
                f"/api/evaluations/runs/{run['evaluation_run_id']}/permission-audit"
            )

        assert response.status_code == 201
        assert run["status"] == "succeeded"
        assert run["question_set_id"] == fixture["question_set_id"]
        assert run["profile_name"] == "kure_v1_1024"
        assert run["question_count"] == 1
        assert run["mean_recall_at_k"] == pytest.approx(1)
        assert run["mean_reciprocal_rank"] == pytest.approx(1)
        assert run["mean_ndcg"] == pytest.approx(1)
        assert run["runtime_metadata"]["slice"] == "044"
        assert run["runtime_metadata"]["executor"] == "api"
        assert result["question_id"] == fixture["question_id"]
        assert result["search_log_id"] == search_log_mapping[str(fixture["question_id"])]
        assert result["matched_chunk_ids"] == [fixture["chunk_id"]]
        assert result["recall_at_k"] == pytest.approx(1)
        assert body["summary"]["question_count"] == 1
        assert detail_response.status_code == 200
        assert detail_response.json()["run"]["evaluation_run_id"] == run["evaluation_run_id"]
        assert detail_response.json()["results"][0]["matched_chunk_ids"] == [fixture["chunk_id"]]
        assert permission_audit_response.status_code == 200
        audit = permission_audit_response.json()["audit"][0]
        assert audit["question_id"] == fixture["question_id"]
        assert audit["actor_login_id"] == "alice.member"
        assert audit["requested_search_scope"] == "company"
        assert audit["effective_search_scope"] == "company"
        assert audit["permission_filter_metadata"]["actor_user_id"] == fixture["alice_user_id"]
        assert audit["search_log_id"] == search_log_mapping[str(fixture["question_id"])]
        assert audit["hidden_violation_count"] == 0
        assert audit["permission_status"] == "clean"
    finally:
        _cleanup_question_set(migrated_database_url, fixture["question_set_id"])
        _cleanup_files(migrated_database_url, [fixture["file_id"]])


def test_golden_search_experiment_batch_api_runs_question_set(
    migrated_database_url: str,
) -> None:
    fixture = _create_execute_fixture(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))
    batch: dict[str, object] | None = None
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/search/experiments/golden-question-set/run",
                json={
                    "question_set_id": fixture["question_set_id"],
                    "run_name_prefix": f"slice-170-batch-{uuid4()}",
                    "profiles": ["kure_v1_1024", "bge_m3_1024"],
                    "strategy_name": "vector_cosine_threshold",
                    "top_k": 2,
                    "score_threshold": 0.0,
                    "chunk_policy_name": "heading_512_64",
                    "runtime_metadata": {"slice": "170"},
                },
            )
            batch = response.json()["batch"]
            experiment_run_id = batch["questions"][0]["experiment"]["experiment_run"][
                "experiment_run_id"
            ]
            detail_response = client.get(f"/api/search/experiments/{experiment_run_id}")

        question_report = batch["questions"][0]
        experiment = question_report["experiment"]
        experiment_run = experiment["experiment_run"]
        runtime_metadata = experiment_run["runtime_metadata"]

        assert response.status_code == 201
        assert batch["question_count"] == 1
        assert batch["question_set"]["question_set_id"] == fixture["question_set_id"]
        assert str(fixture["question_id"]) in batch["experiment_run_ids_by_question"]
        assert question_report["question"]["question_id"] == fixture["question_id"]
        assert experiment_run["status"] == "succeeded"
        assert experiment_run["profile_names"] == ["kure_v1_1024", "bge_m3_1024"]
        assert experiment_run["result_count"] == 2
        assert runtime_metadata["question_set_id"] == fixture["question_set_id"]
        assert runtime_metadata["question_id"] == fixture["question_id"]
        assert runtime_metadata["golden_question_batch"] is True
        assert detail_response.status_code == 200
        assert detail_response.json()["experiment_run"]["experiment_run_id"] == experiment_run_id
    finally:
        if batch is not None:
            _cleanup_search_experiment_batch(migrated_database_url, batch)
        _cleanup_question_set(migrated_database_url, fixture["question_set_id"])
        _cleanup_files(migrated_database_url, [fixture["file_id"]])


def test_golden_evaluation_execute_api_returns_not_found_for_missing_set(
    migrated_database_url: str,
) -> None:
    app = create_app(Settings(database_url=migrated_database_url))

    with TestClient(app) as client:
        response = client.post(
            "/api/evaluations/runs/execute",
            json={"question_set_id": 999999999, "profile_name": "kure_v1_1024"},
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "Golden question set not found."}


def test_golden_search_experiment_batch_api_returns_not_found_for_missing_set(
    migrated_database_url: str,
) -> None:
    app = create_app(Settings(database_url=migrated_database_url))

    with TestClient(app) as client:
        response = client.post(
            "/api/search/experiments/golden-question-set/run",
            json={"question_set_id": 999999999, "profiles": ["kure_v1_1024"]},
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "Golden question set not found."}


def test_golden_evaluation_execute_api_rejects_empty_question_set(
    migrated_database_url: str,
) -> None:
    ids = _seed_ids(migrated_database_url)
    question_set = create_golden_question_set(
        migrated_database_url,
        GoldenQuestionSetInput(
            set_name=f"slice-044-empty-{uuid4()}",
            created_by_user_id=ids["alice.member"],
        ),
    )
    app = create_app(Settings(database_url=migrated_database_url))
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/evaluations/runs/execute",
                json={
                    "question_set_id": question_set.question_set_id,
                    "profile_name": "kure_v1_1024",
                },
            )

        assert response.status_code == 400
        assert response.json() == {"detail": "question_set has no golden questions"}
    finally:
        _cleanup_question_set(migrated_database_url, question_set.question_set_id)
