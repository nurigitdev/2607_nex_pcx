from uuid import uuid4

import pytest

from app.core.database import connect
from app.core.embedding_vectors import (
    EmbeddingVectorInput,
    generate_mock_embedding,
    store_chunk_embedding,
)
from app.core.search_experiment_runner import (
    SearchExperimentExecutionInput,
    execute_search_experiment,
)
from app.core.search_experiments import get_search_experiment_run_detail

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


def _create_runner_document(database_url: str, *, document_group: str) -> tuple[int, int, str]:
    checksum = f"search-experiment-runner-{uuid4()}"
    chunk_text = "search experiment runner inverter guide"
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
                VALUES (%s, 'Search Experiment Runner Fixture', %s, 'company')
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
    return file_id, chunk_id, chunk_text


def _cleanup_runner_fixture(
    database_url: str,
    *,
    file_id: int,
    experiment_run_id: int,
    search_log_id: int,
) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM search_experiment_runs WHERE experiment_run_id = %s",
                (experiment_run_id,),
            )
            cursor.execute("DELETE FROM search_logs WHERE search_log_id = %s", (search_log_id,))
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def test_search_experiment_runner_executes_vector_strategy(
    migrated_database_url: str,
) -> None:
    ids = _seed_ids(migrated_database_url)
    document_group = f"slice-167-{uuid4()}"
    file_id, _chunk_id, chunk_text = _create_runner_document(
        migrated_database_url,
        document_group=document_group,
    )
    report = None
    try:
        report = execute_search_experiment(
            migrated_database_url,
            SearchExperimentExecutionInput(
                run_name=f"Slice 167 Runner {uuid4()}",
                query_text=chunk_text,
                actor_user_id=ids["pcx.admin"],
                requested_search_scope="company",
                profiles=("bge_m3_1024", "kure_v1_1024"),
                strategy_name="vector_cosine_threshold",
                top_k=1,
                score_threshold=0.0,
                chunk_policy_name="heading_512_64",
                document_group=document_group,
                file_type=".md",
                runtime_metadata={"slice": 167},
                created_by="integration-test",
            ),
        )
        detail = get_search_experiment_run_detail(
            migrated_database_url,
            report.run.experiment_run_id,
        )

        assert report.run.status == "succeeded"
        assert report.run.total_profile_count == 2
        assert report.run.completed_profile_count == 2
        assert report.run.result_count == 2
        assert report.search_result.search_log_id > 0
        assert report.strategy_selection.strategy.strategy_name == "vector_cosine_threshold"
        assert [summary.retained_result_count for summary in report.profile_summaries] == [1, 1]
        assert detail is not None
        assert [profile.status for profile in detail.profiles] == ["succeeded", "succeeded"]
        assert [profile.result_count for profile in detail.profiles] == [1, 1]
        assert all(
            profile.search_log_id == report.search_result.search_log_id
            for profile in detail.profiles
        )
    finally:
        if report is not None:
            _cleanup_runner_fixture(
                migrated_database_url,
                file_id=file_id,
                experiment_run_id=report.run.experiment_run_id,
                search_log_id=report.search_result.search_log_id,
            )
        else:
            with connect(migrated_database_url) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
