from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.chunks import DEFAULT_CHUNK_POLICY_NAME, ChunkInput, create_chunk
from app.core.config import Settings
from app.core.database import connect
from app.core.embedding_coverage import (
    get_embedding_coverage_matrix,
    get_multi_policy_ingestion_coverage_detail,
    get_multi_policy_ingestion_coverage_matrix,
)
from app.core.embedding_jobs import (
    EmbeddingJobInput,
    create_embedding_job,
    mark_embedding_job_failed,
    mark_embedding_job_succeeded,
)
from app.core.embedding_vectors import (
    EmbeddingVectorInput,
    get_embedding_vector_table,
    store_chunk_embedding,
)
from app.core.file_metadata import (
    FileMetadataInput,
    create_file_metadata,
    mark_file_parse_succeeded,
)
from app.main import create_app

pytestmark = pytest.mark.integration


COMPLETE_PROFILE = "bge_m3_1024"
FAILED_PROFILE = "kure_v1_1024"
MISSING_PROFILE = "qwen3_4b_1000"
SECOND_MISSING_PROFILE = "qwen3_4b_2560"
LARGE_POLICY = "heading_1000_200"
LONG_POLICY = "heading_1500_200"


def _embedding_values(profile_name: str) -> tuple[float, ...]:
    table = get_embedding_vector_table(profile_name)
    return tuple(0.001 for _ in range(table.dimension))


def _create_coverage_fixture(database_url: str) -> dict[str, object]:
    suffix = str(uuid4())
    checksum = f"slice-164-{suffix}"
    group = f"slice-164-{suffix}"
    created = create_file_metadata(
        database_url,
        FileMetadataInput(
            original_file_name=f"coverage-{suffix}.md",
            stored_file_name=f"coverage-{suffix}.stored.md",
            file_size_bytes=128,
            sha256_checksum=checksum,
            storage_path=f"/tmp/nex_pcx/coverage-{suffix}.stored.md",
            mime_type="text/markdown",
            document_group=group,
            security_level="internal",
            uploaded_by="integration-test",
            document_title=f"Coverage Document {suffix}",
        ),
    )
    document_id = created.file.document_id
    assert document_id is not None

    chunks = [
        create_chunk(
            database_url,
            ChunkInput(
                document_id=document_id,
                chunk_seq=0,
                chunk_text="Coverage first chunk",
                token_count=3,
            ),
        ),
        create_chunk(
            database_url,
            ChunkInput(
                document_id=document_id,
                chunk_seq=1,
                chunk_text="Coverage second chunk",
                token_count=4,
            ),
        ),
    ]
    mark_file_parse_succeeded(
        database_url,
        created.file.file_id,
        parser_name="markdown",
        parser_version="slice-164",
        extracted_text_size=42,
    )

    for chunk in chunks:
        job = create_embedding_job(
            database_url,
            EmbeddingJobInput(chunk_id=chunk.chunk_id, profile_name=COMPLETE_PROFILE),
        ).job
        store_chunk_embedding(
            database_url,
            EmbeddingVectorInput(
                chunk_id=chunk.chunk_id,
                profile_name=COMPLETE_PROFILE,
                embedding=_embedding_values(COMPLETE_PROFILE),
                elapsed_ms=12,
            ),
        )
        mark_embedding_job_succeeded(database_url, job.job_id)

    failed_job = create_embedding_job(
        database_url,
        EmbeddingJobInput(chunk_id=chunks[0].chunk_id, profile_name=FAILED_PROFILE),
    ).job
    mark_embedding_job_failed(
        database_url,
        failed_job.job_id,
        error_code="TEST_PROVIDER_ERROR",
        error_message="coverage fixture failure",
    )

    return {
        "checksum": checksum,
        "document_group": group,
        "document_id": document_id,
    }


def _cleanup_coverage_fixture(database_url: str, fixture: dict[str, object]) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM files WHERE sha256_checksum = %s",
                (fixture["checksum"],),
            )


def _create_multi_policy_coverage_fixture(database_url: str) -> dict[str, object]:
    suffix = str(uuid4())
    checksum = f"slice-260-{suffix}"
    group = f"slice-260-{suffix}"
    created = create_file_metadata(
        database_url,
        FileMetadataInput(
            original_file_name=f"multi-policy-coverage-{suffix}.md",
            stored_file_name=f"multi-policy-coverage-{suffix}.stored.md",
            file_size_bytes=256,
            sha256_checksum=checksum,
            storage_path=f"/tmp/nex_pcx/multi-policy-coverage-{suffix}.stored.md",
            mime_type="text/markdown",
            document_group=group,
            security_level="internal",
            uploaded_by="integration-test",
            document_title=f"Multi Policy Coverage Document {suffix}",
        ),
    )
    document_id = created.file.document_id
    assert document_id is not None

    default_chunks = [
        create_chunk(
            database_url,
            ChunkInput(
                document_id=document_id,
                chunk_seq=0,
                chunk_text="Multi policy default first chunk",
                chunk_policy_name=DEFAULT_CHUNK_POLICY_NAME,
                token_count=5,
            ),
        ),
        create_chunk(
            database_url,
            ChunkInput(
                document_id=document_id,
                chunk_seq=1,
                chunk_text="Multi policy default second chunk",
                chunk_policy_name=DEFAULT_CHUNK_POLICY_NAME,
                token_count=5,
            ),
        ),
    ]
    large_chunk = create_chunk(
        database_url,
        ChunkInput(
            document_id=document_id,
            chunk_seq=0,
            chunk_text="Multi policy large chunk",
            chunk_policy_name=LARGE_POLICY,
            token_count=4,
        ),
    )
    mark_file_parse_succeeded(
        database_url,
        created.file.file_id,
        parser_name="markdown",
        parser_version="slice-260",
        extracted_text_size=84,
    )

    for chunk in default_chunks:
        job = create_embedding_job(
            database_url,
            EmbeddingJobInput(chunk_id=chunk.chunk_id, profile_name=COMPLETE_PROFILE),
        ).job
        store_chunk_embedding(
            database_url,
            EmbeddingVectorInput(
                chunk_id=chunk.chunk_id,
                profile_name=COMPLETE_PROFILE,
                embedding=_embedding_values(COMPLETE_PROFILE),
                elapsed_ms=10,
            ),
        )
        mark_embedding_job_succeeded(database_url, job.job_id)

    failed_job = create_embedding_job(
        database_url,
        EmbeddingJobInput(chunk_id=large_chunk.chunk_id, profile_name=FAILED_PROFILE),
    ).job
    mark_embedding_job_failed(
        database_url,
        failed_job.job_id,
        error_code="TEST_POLICY_PROVIDER_ERROR",
        error_message="multi-policy coverage fixture failure",
    )

    return {
        "checksum": checksum,
        "document_group": group,
        "document_id": document_id,
    }


def test_embedding_coverage_matrix_repository_api_and_page(
    migrated_database_url: str,
) -> None:
    fixture = _create_coverage_fixture(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))
    try:
        matrix = get_embedding_coverage_matrix(
            migrated_database_url,
            document_group=str(fixture["document_group"]),
        )

        with TestClient(app) as client:
            api_response = client.get(
                "/api/admin/embedding-coverage",
                params={"document_group": fixture["document_group"]},
            )
            filtered_api_response = client.get(
                "/api/admin/embedding-coverage",
                params={
                    "document_group": fixture["document_group"],
                    "profile_name": COMPLETE_PROFILE,
                },
            )
            bad_response = client.get("/api/admin/embedding-coverage", params={"limit": 0})
            page_response = client.get(
                "/admin/embedding-coverage",
                params={"document_group": fixture["document_group"]},
            )
            english_page_response = client.get(
                "/admin/embedding-coverage",
                params={"document_group": fixture["document_group"], "lang": "en"},
            )

        document = matrix.documents[0]
        cells = {cell.profile_name: cell for cell in document.profiles}
        payload = api_response.json()
        payload_document = payload["documents"][0]
        payload_cells = {cell["profile_name"]: cell for cell in payload_document["profiles"]}

        assert matrix.summary.document_count == 1
        assert matrix.summary.profile_count >= 2
        assert matrix.summary.total_chunk_count == 2
        assert cells[COMPLETE_PROFILE].status == "complete"
        assert cells[COMPLETE_PROFILE].embedded_chunk_count == 2
        assert cells[COMPLETE_PROFILE].coverage_percent == 100
        assert cells[FAILED_PROFILE].status == "failed"
        assert cells[FAILED_PROFILE].failed_count == 1
        assert api_response.status_code == 200
        assert payload["summary"]["document_count"] == 1
        assert payload_cells[COMPLETE_PROFILE]["status"] == "complete"
        assert payload_cells[COMPLETE_PROFILE]["coverage_label"] == "100.00%"
        assert payload_cells[FAILED_PROFILE]["status"] == "failed"
        assert filtered_api_response.status_code == 200
        assert filtered_api_response.json()["summary"]["profile_count"] == 1
        assert bad_response.status_code == 400
        assert page_response.status_code == 200
        assert "data-embedding-coverage-page" in page_response.text
        assert "임베딩 Coverage Matrix" in page_response.text
        assert COMPLETE_PROFILE in page_response.text
        assert "2 / 2" in page_response.text
        assert english_page_response.status_code == 200
        assert "Embedding Coverage Matrix" in english_page_response.text
    finally:
        _cleanup_coverage_fixture(migrated_database_url, fixture)


def test_multi_policy_ingestion_coverage_matrix_repository_api_and_page(
    migrated_database_url: str,
) -> None:
    fixture = _create_multi_policy_coverage_fixture(migrated_database_url)
    app = create_app(Settings(database_url=migrated_database_url))
    try:
        matrix = get_multi_policy_ingestion_coverage_matrix(
            migrated_database_url,
            document_group=str(fixture["document_group"]),
        )

        with TestClient(app) as client:
            api_response = client.get(
                "/api/admin/multi-policy-ingestion-coverage",
                params={"document_group": fixture["document_group"]},
            )
            policy_filtered_api_response = client.get(
                "/api/admin/multi-policy-ingestion-coverage",
                params={
                    "document_group": fixture["document_group"],
                    "chunk_policy_name": DEFAULT_CHUNK_POLICY_NAME,
                    "profile_name": COMPLETE_PROFILE,
                },
            )
            bad_response = client.get(
                "/api/admin/multi-policy-ingestion-coverage",
                params={"chunk_policy_name": " "},
            )
            page_response = client.get(
                "/admin/multi-policy-ingestion-coverage",
                params={"document_group": fixture["document_group"]},
            )
            english_page_response = client.get(
                "/admin/multi-policy-ingestion-coverage",
                params={"document_group": fixture["document_group"], "lang": "en"},
            )
            detail_api_response = client.get(
                "/api/admin/multi-policy-ingestion-coverage/detail",
                params={
                    "document_id": fixture["document_id"],
                    "chunk_policy_name": DEFAULT_CHUNK_POLICY_NAME,
                    "profile_name": COMPLETE_PROFILE,
                },
            )
            missing_detail_api_response = client.get(
                "/api/admin/multi-policy-ingestion-coverage/detail",
                params={
                    "document_id": fixture["document_id"],
                    "chunk_policy_name": DEFAULT_CHUNK_POLICY_NAME,
                    "profile_name": MISSING_PROFILE,
                },
            )
            reconcile_response = client.post(
                "/api/admin/multi-policy-ingestion-coverage/reconcile-missing-jobs",
                json={
                    "document_id": fixture["document_id"],
                    "chunk_policy_name": DEFAULT_CHUNK_POLICY_NAME,
                    "profile_name": MISSING_PROFILE,
                },
            )
            reconcile_repeat_response = client.post(
                "/api/admin/multi-policy-ingestion-coverage/reconcile-missing-jobs",
                json={
                    "document_id": fixture["document_id"],
                    "chunk_policy_name": DEFAULT_CHUNK_POLICY_NAME,
                    "profile_name": MISSING_PROFILE,
                },
            )
            reconcile_bad_response = client.post(
                "/api/admin/multi-policy-ingestion-coverage/reconcile-missing-jobs",
                json={
                    "document_id": fixture["document_id"],
                    "chunk_policy_name": DEFAULT_CHUNK_POLICY_NAME,
                    "profile_name": "missing_profile",
                },
            )
            reconcile_page_response = client.post(
                "/admin/multi-policy-ingestion-coverage/reconcile-missing-jobs",
                data={
                    "document_id": fixture["document_id"],
                    "detail_chunk_policy_name": DEFAULT_CHUNK_POLICY_NAME,
                    "detail_profile_name": SECOND_MISSING_PROFILE,
                    "document_group": fixture["document_group"],
                    "limit": "100",
                    "lang": "ko",
                },
            )
            missing_detail_response = client.get(
                "/api/admin/multi-policy-ingestion-coverage/detail",
                params={
                    "document_id": 999999999,
                    "chunk_policy_name": DEFAULT_CHUNK_POLICY_NAME,
                    "profile_name": COMPLETE_PROFILE,
                },
            )
            bad_detail_response = client.get(
                "/api/admin/multi-policy-ingestion-coverage/detail",
                params={
                    "document_id": 0,
                    "chunk_policy_name": DEFAULT_CHUNK_POLICY_NAME,
                    "profile_name": COMPLETE_PROFILE,
                },
            )
            detail_page_response = client.get(
                "/admin/multi-policy-ingestion-coverage",
                params={
                    "document_group": fixture["document_group"],
                    "lang": "ko",
                    "detail_document_id": fixture["document_id"],
                    "detail_chunk_policy_name": DEFAULT_CHUNK_POLICY_NAME,
                    "detail_profile_name": COMPLETE_PROFILE,
                },
            )

        rows_by_policy = {row.chunk_policy_name: row for row in matrix.rows}
        default_detail = get_multi_policy_ingestion_coverage_detail(
            migrated_database_url,
            document_id=int(fixture["document_id"]),
            chunk_policy_name=DEFAULT_CHUNK_POLICY_NAME,
            profile_name=COMPLETE_PROFILE,
        )
        failed_detail = get_multi_policy_ingestion_coverage_detail(
            migrated_database_url,
            document_id=int(fixture["document_id"]),
            chunk_policy_name=LARGE_POLICY,
            profile_name=FAILED_PROFILE,
        )
        not_chunked_detail = get_multi_policy_ingestion_coverage_detail(
            migrated_database_url,
            document_id=int(fixture["document_id"]),
            chunk_policy_name=LONG_POLICY,
            profile_name=COMPLETE_PROFILE,
        )
        reconciled_detail = get_multi_policy_ingestion_coverage_detail(
            migrated_database_url,
            document_id=int(fixture["document_id"]),
            chunk_policy_name=DEFAULT_CHUNK_POLICY_NAME,
            profile_name=MISSING_PROFILE,
        )
        default_cells = {
            cell.profile_name: cell for cell in rows_by_policy[DEFAULT_CHUNK_POLICY_NAME].profiles
        }
        large_cells = {cell.profile_name: cell for cell in rows_by_policy[LARGE_POLICY].profiles}
        long_cells = {cell.profile_name: cell for cell in rows_by_policy[LONG_POLICY].profiles}
        payload = api_response.json()
        payload_rows = {row["chunk_policy_name"]: row for row in payload["rows"]}
        default_payload_cells = {
            cell["profile_name"]: cell
            for cell in payload_rows[DEFAULT_CHUNK_POLICY_NAME]["profiles"]
        }
        detail_payload = detail_api_response.json()
        missing_detail_payload = missing_detail_api_response.json()
        reconcile_payload = reconcile_response.json()
        reconcile_repeat_payload = reconcile_repeat_response.json()

        assert matrix.summary.document_count == 1
        assert matrix.summary.policy_count >= 3
        assert matrix.summary.document_policy_count >= 3
        assert rows_by_policy[DEFAULT_CHUNK_POLICY_NAME].chunk_count == 2
        assert rows_by_policy[LARGE_POLICY].chunk_count == 1
        assert rows_by_policy[LONG_POLICY].chunk_count == 0
        assert default_cells[COMPLETE_PROFILE].status == "complete"
        assert default_cells[COMPLETE_PROFILE].embedded_chunk_count == 2
        assert large_cells[FAILED_PROFILE].status == "failed"
        assert large_cells[FAILED_PROFILE].failed_count == 1
        assert long_cells[COMPLETE_PROFILE].status == "not_chunked"
        assert default_detail is not None
        assert default_detail.profile.status == "complete"
        assert len(default_detail.chunks) == 2
        assert all(chunk.vector_present for chunk in default_detail.chunks)
        assert failed_detail is not None
        assert failed_detail.profile.status == "failed"
        assert failed_detail.chunks[0].error_code == "TEST_POLICY_PROVIDER_ERROR"
        assert not_chunked_detail is not None
        assert not_chunked_detail.profile.status == "not_chunked"
        assert not_chunked_detail.chunks == ()
        assert reconciled_detail is not None
        assert reconciled_detail.profile.status == "pending"
        assert reconciled_detail.profile.job_count == 2
        assert api_response.status_code == 200
        assert payload["summary"]["document_count"] == 1
        assert payload["summary"]["policy_count"] >= 3
        assert default_payload_cells[COMPLETE_PROFILE]["coverage_label"] == "100.00%"
        assert detail_api_response.status_code == 200
        assert detail_payload["profile"]["status"] == "complete"
        assert detail_payload["chunks"][0]["vector_present"] is True
        assert detail_payload["chunks"][0]["job_status"] == "succeeded"
        assert missing_detail_api_response.status_code == 200
        assert missing_detail_payload["profile"]["status"] == "missing"
        assert missing_detail_payload["chunks"][0]["job_id"] is None
        assert reconcile_response.status_code == 200
        assert reconcile_payload["missing_job_count"] == 2
        assert reconcile_payload["created_job_count"] == 2
        assert reconcile_payload["created_jobs"][0]["status"] == "pending"
        assert reconcile_repeat_response.status_code == 200
        assert reconcile_repeat_payload["missing_job_count"] == 0
        assert reconcile_repeat_payload["created_job_count"] == 0
        assert reconcile_bad_response.status_code == 400
        assert reconcile_page_response.status_code == 200
        assert "누락 Job 생성 완료" in reconcile_page_response.text
        assert missing_detail_response.status_code == 404
        assert bad_detail_response.status_code == 400
        assert policy_filtered_api_response.status_code == 200
        assert policy_filtered_api_response.json()["summary"]["policy_count"] == 1
        assert policy_filtered_api_response.json()["summary"]["profile_count"] == 1
        assert bad_response.status_code == 400
        assert page_response.status_code == 200
        assert "data-multi-policy-ingestion-coverage-page" in page_response.text
        assert "다중 Chunk 정책 Coverage" in page_response.text
        assert DEFAULT_CHUNK_POLICY_NAME in page_response.text
        assert "2 / 2" in page_response.text
        assert detail_page_response.status_code == 200
        assert "data-multi-policy-coverage-detail" in detail_page_response.text
        assert "Coverage 상세" in detail_page_response.text
        assert "Multi policy default first chunk" in detail_page_response.text
        assert english_page_response.status_code == 200
        assert "Document x Policy x Profile Matrix" in english_page_response.text
    finally:
        _cleanup_coverage_fixture(migrated_database_url, fixture)
