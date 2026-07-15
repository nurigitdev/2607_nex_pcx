from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.chunks import ChunkInput, create_chunk
from app.core.config import Settings
from app.core.database import connect
from app.core.extraction_runtime import ExtractionRuntimeRequest
from app.core.local_extraction import (
    LOCAL_MARKDOWN_PROFILE_NAME,
    persist_extraction_runtime_result,
    run_local_extraction,
)
from app.main import create_app

pytestmark = pytest.mark.integration


def _create_document(database_url: str, source_path: Path) -> tuple[int, int]:
    checksum = f"artifact-api-{uuid4()}"
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO files (
                    original_file_name,
                    stored_file_name,
                    file_ext,
                    mime_type,
                    file_size_bytes,
                    sha256_checksum,
                    storage_path,
                    detected_file_type
                )
                VALUES (%s, %s, '.md', 'text/markdown', %s, %s, %s, 'md')
                RETURNING file_id
                """,
                (
                    f"{checksum}.md",
                    f"{checksum}.stored.md",
                    source_path.stat().st_size,
                    checksum,
                    str(source_path),
                ),
            )
            file_id = cursor.fetchone()["file_id"]
            cursor.execute(
                """
                INSERT INTO documents (file_id, document_title)
                VALUES (%s, %s)
                RETURNING document_id
                """,
                (file_id, f"Artifact API Fixture {checksum}"),
            )
            document_id = cursor.fetchone()["document_id"]
    return file_id, document_id


def _cleanup_file(database_url: str, file_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def test_document_ingestion_artifact_api_and_page(
    migrated_database_url: str,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "slice-220.md"
    source_path.write_text("# Artifact API\n\nInspect this block.", encoding="utf-8")
    file_id, document_id = _create_document(migrated_database_url, source_path)
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        request = ExtractionRuntimeRequest(
            file_id=file_id,
            document_id=document_id,
            storage_path=str(source_path),
            extraction_profile_name=LOCAL_MARKDOWN_PROFILE_NAME,
            mime_type="text/markdown",
            detected_file_type="md",
        )
        runtime_result = run_local_extraction(request)
        persisted = persist_extraction_runtime_result(
            migrated_database_url,
            request,
            runtime_result,
        )
        artifact_id = persisted.artifacts[0].artifact_id
        first_block = persisted.blocks[0]
        second_block = persisted.blocks[1]
        create_chunk(
            migrated_database_url,
            ChunkInput(
                document_id=document_id,
                artifact_id=artifact_id,
                block_id=first_block.block_id,
                chunk_seq=0,
                chunk_text=first_block.content_markdown or first_block.content_text or "heading",
                content_markdown=first_block.content_markdown,
                heading_path=first_block.heading_path,
                source_anchor=first_block.source_anchor,
                token_count=first_block.token_count,
            ),
        )
        create_chunk(
            migrated_database_url,
            ChunkInput(
                document_id=document_id,
                artifact_id=artifact_id,
                block_id=second_block.block_id,
                chunk_seq=1,
                chunk_text=(
                    second_block.content_markdown
                    or second_block.content_text
                    or "paragraph"
                ),
                content_markdown=second_block.content_markdown,
                heading_path=second_block.heading_path,
                source_anchor=second_block.source_anchor,
                token_count=second_block.token_count,
            ),
        )

        with TestClient(app) as client:
            api_response = client.get(f"/api/documents/{document_id}/ingestion-artifacts")
            selected_response = client.get(
                f"/api/documents/{document_id}/ingestion-artifacts",
                params={"artifact_id": artifact_id},
            )
            missing_artifact_response = client.get(
                f"/api/documents/{document_id}/ingestion-artifacts",
                params={"artifact_id": 999_999_999},
            )
            preview_response = client.get(f"/api/documents/{document_id}/extraction-preview")
            selected_preview_response = client.get(
                f"/api/documents/{document_id}/extraction-preview",
                params={"artifact_id": artifact_id},
            )
            missing_preview_artifact_response = client.get(
                f"/api/documents/{document_id}/extraction-preview",
                params={"artifact_id": 999_999_999},
            )
            chunk_source_trace_response = client.get(
                f"/api/documents/{document_id}/chunk-source-trace",
                params={"artifact_id": artifact_id},
            )
            missing_chunk_source_trace_response = client.get(
                f"/api/documents/{document_id}/chunk-source-trace",
                params={"artifact_id": 999_999_999},
            )
            invalid_chunk_source_trace_response = client.get(
                f"/api/documents/{document_id}/chunk-source-trace",
                params={"artifact_id": artifact_id, "chunk_policy_name": " "},
            )
            quality_response = client.get(
                f"/api/documents/{document_id}/extraction-quality",
                params={"artifact_id": artifact_id},
            )
            missing_quality_artifact_response = client.get(
                f"/api/documents/{document_id}/extraction-quality",
                params={"artifact_id": 999_999_999},
            )
            initial_snapshots_response = client.get(
                f"/api/documents/{document_id}/extraction-quality-snapshots",
                params={"artifact_id": artifact_id},
            )
            initial_snapshot_summary_response = client.get(
                f"/api/documents/{document_id}/extraction-quality-summary",
                params={"artifact_id": artifact_id},
            )
            snapshot_create_response = client.post(
                f"/api/documents/{document_id}/extraction-quality-snapshots",
                json={"artifact_id": artifact_id, "created_by": "api-test"},
            )
            snapshots_response = client.get(
                f"/api/documents/{document_id}/extraction-quality-snapshots",
                params={"artifact_id": artifact_id, "limit": 5},
            )
            snapshot_summary_response = client.get(
                f"/api/documents/{document_id}/extraction-quality-summary",
                params={"artifact_id": artifact_id},
            )
            invalid_snapshot_limit_response = client.get(
                f"/api/documents/{document_id}/extraction-quality-snapshots",
                params={"limit": 0},
            )
            missing_snapshot_artifact_response = client.post(
                f"/api/documents/{document_id}/extraction-quality-snapshots",
                json={"artifact_id": 999_999_999},
            )
            markdown_export_response = client.get(
                f"/api/documents/{document_id}/extraction-export",
                params={"artifact_id": artifact_id, "format": "markdown"},
            )
            blocks_export_response = client.get(
                f"/api/documents/{document_id}/extraction-export",
                params={"artifact_id": artifact_id, "format": "blocks_json"},
            )
            metadata_export_response = client.get(
                f"/api/documents/{document_id}/extraction-export",
                params={"artifact_id": artifact_id, "format": "metadata_json"},
            )
            quality_export_response = client.get(
                f"/api/documents/{document_id}/extraction-export",
                params={"artifact_id": artifact_id, "format": "quality_json"},
            )
            bundle_export_response = client.get(
                f"/api/documents/{document_id}/extraction-export",
                params={"artifact_id": artifact_id, "format": "bundle_json"},
            )
            invalid_export_response = client.get(
                f"/api/documents/{document_id}/extraction-export",
                params={"artifact_id": artifact_id, "format": "csv"},
            )
            missing_export_artifact_response = client.get(
                f"/api/documents/{document_id}/extraction-export",
                params={"artifact_id": 999_999_999, "format": "markdown"},
            )
            page_response = client.get(
                f"/documents/{document_id}/artifacts",
                params={"lang": "ko", "artifact_id": artifact_id},
            )
            missing_artifact_page_response = client.get(
                f"/documents/{document_id}/artifacts",
                params={"lang": "ko", "artifact_id": 999_999_999},
            )

        payload = api_response.json()
        selected_payload = selected_response.json()
        preview_payload = preview_response.json()
        selected_preview_payload = selected_preview_response.json()
        chunk_source_trace_payload = chunk_source_trace_response.json()
        quality_payload = quality_response.json()
        initial_snapshots_payload = initial_snapshots_response.json()
        initial_snapshot_summary_payload = initial_snapshot_summary_response.json()
        snapshot_create_payload = snapshot_create_response.json()
        snapshots_payload = snapshots_response.json()
        snapshot_summary_payload = snapshot_summary_response.json()
        blocks_export_payload = blocks_export_response.json()
        metadata_export_payload = metadata_export_response.json()
        quality_export_payload = quality_export_response.json()
        bundle_export_payload = bundle_export_response.json()

        assert api_response.status_code == 200
        assert payload["document"]["document_id"] == document_id
        assert payload["selected_artifact_id"] == artifact_id
        assert payload["extraction_runs"][0]["status"] == "succeeded"
        assert payload["artifacts"][0]["artifact_id"] == artifact_id
        assert payload["artifacts"][0]["content_preview"].startswith("# Artifact API")
        assert [block["block_type"] for block in payload["blocks"]] == [
            "heading",
            "paragraph",
        ]
        assert selected_payload["selected_artifact_id"] == artifact_id
        assert missing_artifact_response.status_code == 404
        assert preview_response.status_code == 200
        assert preview_payload["selected_artifact_id"] == artifact_id
        assert preview_payload["selected_artifact"]["artifact_id"] == artifact_id
        assert preview_payload["selected_artifact"]["content_text"].startswith("# Artifact API")
        assert preview_payload["selected_artifact"]["content_lines"] == 3
        assert preview_payload["block_summary"] == {
            "block_count": 2,
            "block_type_counts": {"heading": 1, "paragraph": 1},
            "source_anchor_count": 2,
            "page_count": 0,
            "slide_count": 0,
            "sheet_count": 0,
            "sheet_names": [],
        }
        assert selected_preview_payload["selected_artifact"]["artifact_id"] == artifact_id
        assert missing_preview_artifact_response.status_code == 404
        assert chunk_source_trace_response.status_code == 200
        assert chunk_source_trace_payload["trace"]["selected_artifact_id"] == artifact_id
        assert chunk_source_trace_payload["trace"]["summary"]["block_count"] == 2
        assert chunk_source_trace_payload["trace"]["summary"]["chunk_count"] >= 1
        assert chunk_source_trace_payload["trace"]["summary"]["traced_block_count"] >= 1
        assert chunk_source_trace_payload["trace"]["block_traces"][0]["block"][
            "block_type"
        ] == "heading"
        assert missing_chunk_source_trace_response.status_code == 404
        assert invalid_chunk_source_trace_response.status_code == 400
        assert quality_response.status_code == 200
        assert quality_payload["selected_artifact_id"] == artifact_id
        assert quality_payload["quality_check"]["status"] == "warning"
        assert quality_payload["quality_check"]["warning_count"] == 1
        assert quality_payload["quality_check"]["issues"][0]["code"] == "short_content_text"
        assert quality_payload["quality_check"]["source_anchor_coverage_percent"] == 100.0
        assert missing_quality_artifact_response.status_code == 404
        assert initial_snapshots_response.status_code == 200
        assert initial_snapshots_payload["snapshots"] == []
        assert initial_snapshot_summary_response.status_code == 200
        assert initial_snapshot_summary_payload["summary"]["snapshot_count"] == 0
        assert initial_snapshot_summary_payload["summary"]["latest_snapshot"] is None
        assert snapshot_create_response.status_code == 201
        assert snapshot_create_payload["snapshot"]["artifact_id"] == artifact_id
        assert snapshot_create_payload["snapshot"]["status"] == "warning"
        assert snapshot_create_payload["snapshot"]["created_by"] == "api-test"
        assert snapshot_create_payload["snapshot"]["quality_payload"]["issue_count"] == 1
        assert snapshot_create_payload["snapshot"]["source_anchor_coverage_label"] == "100.00%"
        assert snapshots_response.status_code == 200
        assert snapshots_payload["snapshot_count"] == 1
        created_snapshot_id = snapshot_create_payload["snapshot"]["snapshot_id"]
        assert snapshots_payload["snapshots"][0]["snapshot_id"] == created_snapshot_id
        assert snapshot_summary_response.status_code == 200
        assert snapshot_summary_payload["summary"]["snapshot_count"] == 1
        assert snapshot_summary_payload["summary"]["warning_count"] == 1
        assert snapshot_summary_payload["summary"]["latest_snapshot"]["snapshot_id"] == (
            created_snapshot_id
        )
        assert invalid_snapshot_limit_response.status_code == 422
        assert missing_snapshot_artifact_response.status_code == 404
        assert markdown_export_response.status_code == 200
        assert markdown_export_response.headers["content-type"].startswith("text/markdown")
        assert "document-" in markdown_export_response.headers["content-disposition"]
        assert markdown_export_response.headers["content-disposition"].endswith(
            "-markdown.md\""
        )
        assert markdown_export_response.text.startswith("# Artifact API")
        assert blocks_export_response.status_code == 200
        assert blocks_export_response.headers["content-disposition"].endswith(
            "-blocks_json.json\""
        )
        assert blocks_export_payload["block_summary"]["block_count"] == 2
        assert [block["block_type"] for block in blocks_export_payload["blocks"]] == [
            "heading",
            "paragraph",
        ]
        assert metadata_export_response.status_code == 200
        assert metadata_export_payload["metadata"]["parser_name"]
        assert metadata_export_payload["metadata"]["block_count"] == 2
        assert quality_export_response.status_code == 200
        assert quality_export_payload["quality_check"]["status"] == "warning"
        assert bundle_export_response.status_code == 200
        assert bundle_export_payload["selected_artifact"]["content_text"].startswith(
            "# Artifact API"
        )
        assert bundle_export_payload["extraction_runs"][0]["status"] == "succeeded"
        assert invalid_export_response.status_code == 400
        assert missing_export_artifact_response.status_code == 404
        assert page_response.status_code == 200
        assert "Artifact API" in page_response.text
        assert "Blocks" in page_response.text
        assert "Artifact 내보내기" in page_response.text
        assert "Bundle JSON" in page_response.text
        assert "정규화 텍스트 Preview" in page_response.text
        assert "Block 요약" in page_response.text
        assert "품질 점검" in page_response.text
        assert "품질 Snapshot 요약" in page_response.text
        assert "최신 Snapshot" in page_response.text
        assert "Chunk Source Trace" in page_response.text
        assert "Trace Block" in page_response.text
        assert "short_content_text" in page_response.text
        assert "100.00%" in page_response.text
        assert "Source Anchor" in page_response.text
        assert "heading 1" in page_response.text
        assert "paragraph 1" in page_response.text
        assert missing_artifact_page_response.status_code == 200
        assert "Extraction artifact not found for document" in missing_artifact_page_response.text
        assert "선택된 artifact가 없습니다." in missing_artifact_page_response.text
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_document_extraction_preview_api_handles_document_without_artifacts(
    migrated_database_url: str,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "slice-228-empty.md"
    source_path.write_text("Waiting for extraction.", encoding="utf-8")
    file_id, document_id = _create_document(migrated_database_url, source_path)
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            response = client.get(f"/api/documents/{document_id}/extraction-preview")
            chunk_source_trace_response = client.get(
                f"/api/documents/{document_id}/chunk-source-trace"
            )
            quality_response = client.get(f"/api/documents/{document_id}/extraction-quality")
            snapshots_response = client.get(
                f"/api/documents/{document_id}/extraction-quality-snapshots"
            )
            snapshot_summary_response = client.get(
                f"/api/documents/{document_id}/extraction-quality-summary"
            )
            snapshot_create_response = client.post(
                f"/api/documents/{document_id}/extraction-quality-snapshots",
                json={"created_by": "api-test"},
            )
            export_response = client.get(f"/api/documents/{document_id}/extraction-export")
            missing_document_response = client.get(
                f"/api/documents/{document_id + 999_999_999}/extraction-preview"
            )
            missing_quality_document_response = client.get(
                f"/api/documents/{document_id + 999_999_999}/extraction-quality"
            )
            missing_export_document_response = client.get(
                f"/api/documents/{document_id + 999_999_999}/extraction-export"
            )
            page_response = client.get(
                f"/documents/{document_id}/artifacts",
                params={"lang": "ko"},
            )

        payload = response.json()
        quality_payload = quality_response.json()

        assert response.status_code == 200
        assert payload["document"]["document_id"] == document_id
        assert payload["extraction_runs"] == []
        assert payload["artifacts"] == []
        assert payload["selected_artifact_id"] is None
        assert payload["selected_artifact"] is None
        assert payload["blocks"] == []
        assert payload["block_summary"] == {
            "block_count": 0,
            "block_type_counts": {},
            "source_anchor_count": 0,
            "page_count": 0,
            "slide_count": 0,
            "sheet_count": 0,
            "sheet_names": [],
        }
        assert chunk_source_trace_response.status_code == 200
        assert chunk_source_trace_response.json()["trace"]["summary"]["chunk_count"] == 0
        assert quality_response.status_code == 200
        assert quality_payload["quality_check"]["status"] == "not_available"
        assert quality_payload["quality_check"]["issues"][0]["code"] == "no_artifact_selected"
        assert snapshots_response.status_code == 200
        assert snapshots_response.json()["snapshots"] == []
        assert snapshot_summary_response.status_code == 200
        assert snapshot_summary_response.json()["summary"]["snapshot_count"] == 0
        assert snapshot_create_response.status_code == 404
        assert export_response.status_code == 404
        assert missing_document_response.status_code == 404
        assert missing_quality_document_response.status_code == 404
        assert missing_export_document_response.status_code == 404
        assert page_response.status_code == 200
        assert "선택된 artifact가 없습니다." in page_response.text
        assert "집계할 block 유형이 없습니다." in page_response.text
        assert "확인 불가" in page_response.text
        assert "저장된 품질 snapshot이 없습니다." in page_response.text
        assert "표시할 chunk source trace가 없습니다." in page_response.text
    finally:
        _cleanup_file(migrated_database_url, file_id)
