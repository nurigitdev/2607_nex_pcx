from uuid import uuid4

import pytest

from app.core.database import connect, fetch_one
from app.core.ingestion_artifacts import (
    DocumentBlockInput,
    ExtractionArtifactInput,
    ExtractionProfileInput,
    ExtractionQualitySnapshotInput,
    ExtractionRunInput,
    ImageArtifactInput,
    InvalidIngestionArtifactError,
    TableArtifactInput,
    create_document_block,
    create_extraction_artifact,
    create_extraction_quality_snapshot,
    create_extraction_run,
    create_image_artifact,
    create_table_artifact,
    get_extraction_quality_snapshot_summary,
    get_extraction_artifact,
    get_extraction_profile,
    get_extraction_run,
    list_document_blocks,
    list_document_extraction_artifacts,
    list_document_extraction_runs,
    list_extraction_quality_snapshots,
    list_extraction_profiles,
    upsert_extraction_profile,
)

pytestmark = pytest.mark.integration


def _create_document(database_url: str) -> tuple[int, int]:
    checksum = f"ingestion-repo-{uuid4()}"
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
                    storage_path
                )
                VALUES (%s, %s, '.md', 'text/markdown', 100, %s, %s)
                RETURNING file_id
                """,
                (
                    f"{checksum}.md",
                    f"{checksum}.stored.md",
                    checksum,
                    f"/tmp/{checksum}.md",
                ),
            )
            file_id = cursor.fetchone()["file_id"]
            cursor.execute(
                """
                INSERT INTO documents (file_id, document_title)
                VALUES (%s, %s)
                RETURNING document_id
                """,
                (file_id, f"Ingestion repository fixture {checksum}"),
            )
            document_id = cursor.fetchone()["document_id"]
    return file_id, document_id


def _cleanup_file(database_url: str, file_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def test_extraction_profile_upsert_get_and_list(migrated_database_url: str) -> None:
    profile_name = f"local_markdown_{uuid4().hex}"

    created = upsert_extraction_profile(
        migrated_database_url,
        ExtractionProfileInput(
            extraction_profile_name=profile_name,
            extractor_name="markdown",
            extractor_version="0.1.0",
            supported_file_types=("md",),
            default_options={"normalize_headings": True},
        ),
    )
    updated = upsert_extraction_profile(
        migrated_database_url,
        ExtractionProfileInput(
            extraction_profile_name=profile_name,
            extractor_name="markdown",
            extractor_version="0.2.0",
            supported_file_types=("md", "txt"),
            provider_mode="remote",
            default_options={"normalize_headings": False},
            is_active=False,
        ),
    )
    stored = get_extraction_profile(migrated_database_url, profile_name)
    active_profiles = list_extraction_profiles(migrated_database_url, active_only=True)

    assert created.extraction_profile_name == profile_name
    assert updated.extractor_version == "0.2.0"
    assert updated.provider_mode == "remote"
    assert updated.supported_file_types == ("md", "txt")
    assert updated.default_options == {"normalize_headings": False}
    assert stored == updated
    assert profile_name not in {profile.extraction_profile_name for profile in active_profiles}


def test_ingestion_artifact_repository_round_trip(migrated_database_url: str) -> None:
    file_id, document_id = _create_document(migrated_database_url)
    profile_name = f"local_markdown_{uuid4().hex}"
    try:
        upsert_extraction_profile(
            migrated_database_url,
            ExtractionProfileInput(
                extraction_profile_name=profile_name,
                extractor_name="markdown",
                extractor_version="0.1.0",
                supported_file_types=("md",),
            ),
        )
        run = create_extraction_run(
            migrated_database_url,
            ExtractionRunInput(
                file_id=file_id,
                document_id=document_id,
                extraction_profile_name=profile_name,
                status="succeeded",
                extractor_name="markdown",
                extractor_version="0.1.0",
                elapsed_ms=15,
                warning_count=1,
                runtime_metadata={"provider_mode": "local"},
            ),
        )
        artifact = create_extraction_artifact(
            migrated_database_url,
            ExtractionArtifactInput(
                extraction_run_id=run.extraction_run_id,
                file_id=file_id,
                document_id=document_id,
                artifact_type="normalized_markdown",
                content_text="# Title\n\n| A | B |",
                content_hash="artifact-round-trip",
                size_bytes=20,
                language="ko",
                metadata={"normalizer": "markdown"},
            ),
        )
        heading = create_document_block(
            migrated_database_url,
            DocumentBlockInput(
                artifact_id=artifact.artifact_id,
                document_id=document_id,
                block_seq=0,
                block_type="heading",
                content_text="Title",
                content_markdown="# Title",
                heading_path=("Title",),
                source_anchor={"line": 1},
                char_start=0,
                char_end=7,
                token_count=1,
            ),
        )
        table_block = create_document_block(
            migrated_database_url,
            DocumentBlockInput(
                artifact_id=artifact.artifact_id,
                document_id=document_id,
                parent_block_id=heading.block_id,
                block_seq=1,
                block_type="table",
                content_text="A B",
                content_markdown="| A | B |",
                heading_path=("Title",),
                source_anchor={"line": 3, "table_index": 0},
                token_count=2,
            ),
        )
        table = create_table_artifact(
            migrated_database_url,
            TableArtifactInput(
                block_id=table_block.block_id,
                content_markdown="| A | B |",
                content_json={"rows": [["A", "B"]]},
                row_count=1,
                column_count=2,
                source_anchor={"table_index": 0},
            ),
        )
        image_block = create_document_block(
            migrated_database_url,
            DocumentBlockInput(
                artifact_id=artifact.artifact_id,
                document_id=document_id,
                block_seq=2,
                block_type="image",
                content_text="figure caption",
                source_anchor={"image_index": 0},
            ),
        )
        image = create_image_artifact(
            migrated_database_url,
            ImageArtifactInput(
                block_id=image_block.block_id,
                storage_path="/tmp/figure.png",
                mime_type="image/png",
                width_px=320,
                height_px=200,
                caption_text="figure caption",
                source_anchor={"image_index": 0},
            ),
        )

        stored_run = get_extraction_run(migrated_database_url, run.extraction_run_id)
        stored_artifact = get_extraction_artifact(migrated_database_url, artifact.artifact_id)
        document_runs = list_document_extraction_runs(migrated_database_url, document_id)
        document_artifacts = list_document_extraction_artifacts(
            migrated_database_url,
            document_id,
            artifact_type="normalized_markdown",
        )
        blocks = list_document_blocks(
            migrated_database_url,
            document_id,
            artifact_id=artifact.artifact_id,
        )
        snapshot = create_extraction_quality_snapshot(
            migrated_database_url,
            ExtractionQualitySnapshotInput(
                document_id=document_id,
                file_id=file_id,
                artifact_id=artifact.artifact_id,
                extraction_run_id=run.extraction_run_id,
                artifact_type=artifact.artifact_type,
                extraction_profile_name=profile_name,
                extractor_name="markdown",
                extractor_version="0.1.0",
                status="warning",
                content_length=20,
                content_lines=3,
                block_count=3,
                source_anchor_count=3,
                source_anchor_coverage_percent=100.0,
                issue_count=1,
                warning_count=1,
                failed_count=0,
                block_summary={"block_count": 3},
                quality_payload={"status": "warning", "issues": [{"code": "fixture"}]},
                created_by="repository-test",
            ),
        )
        snapshots = list_extraction_quality_snapshots(
            migrated_database_url,
            document_id,
            artifact_id=artifact.artifact_id,
        )
        summary = get_extraction_quality_snapshot_summary(
            migrated_database_url,
            document_id,
            artifact_id=artifact.artifact_id,
        )

        assert stored_run == run
        assert run.status == "succeeded"
        assert run.finished_at is not None
        assert run.runtime_metadata == {"provider_mode": "local"}
        assert stored_artifact == artifact
        assert artifact.metadata == {"normalizer": "markdown"}
        assert document_runs[0].extraction_run_id == run.extraction_run_id
        assert [item.artifact_id for item in document_artifacts] == [artifact.artifact_id]
        assert [block.block_type for block in blocks] == ["heading", "table", "image"]
        assert blocks[1].parent_block_id == heading.block_id
        assert blocks[1].source_anchor == {"line": 3, "table_index": 0}
        assert table.content_json == {"rows": [["A", "B"]]}
        assert image.storage_path == "/tmp/figure.png"
        assert image.caption_text == "figure caption"
        assert snapshots[0] == snapshot
        assert snapshot.status == "warning"
        assert snapshot.source_anchor_coverage_percent == 100.0
        assert snapshot.quality_payload["issues"] == [{"code": "fixture"}]
        assert summary.snapshot_count == 1
        assert summary.warning_count == 1
        assert summary.latest_snapshot == snapshot
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_repository_returns_none_or_empty_lists_for_missing_records(
    migrated_database_url: str,
) -> None:
    assert get_extraction_profile(migrated_database_url, f"missing-{uuid4()}") is None
    assert get_extraction_run(migrated_database_url, 999_999_999) is None
    assert get_extraction_artifact(migrated_database_url, 999_999_999) is None
    assert list_document_extraction_runs(migrated_database_url, 999_999_999) == []
    assert list_document_extraction_artifacts(migrated_database_url, 999_999_999) == []
    assert list_document_blocks(migrated_database_url, 999_999_999) == []
    assert list_extraction_quality_snapshots(migrated_database_url, 999_999_999) == []
    summary = get_extraction_quality_snapshot_summary(migrated_database_url, 999_999_999)
    assert summary.snapshot_count == 0
    assert summary.latest_snapshot is None


def test_extraction_quality_snapshot_validates_inputs(
    migrated_database_url: str,
) -> None:
    with pytest.raises(InvalidIngestionArtifactError, match="limit"):
        list_extraction_quality_snapshots(migrated_database_url, 1, limit=0)
    with pytest.raises(InvalidIngestionArtifactError, match="artifact_id"):
        list_extraction_quality_snapshots(migrated_database_url, 1, artifact_id=0)
    with pytest.raises(InvalidIngestionArtifactError, match="artifact_id"):
        get_extraction_quality_snapshot_summary(migrated_database_url, 1, artifact_id=0)
    with pytest.raises(
        InvalidIngestionArtifactError,
        match="Unsupported extraction quality status",
    ):
        create_extraction_quality_snapshot(
            migrated_database_url,
            ExtractionQualitySnapshotInput(
                document_id=1,
                file_id=1,
                artifact_id=1,
                artifact_type="normalized_markdown",
                status="not_available",
                block_count=0,
                source_anchor_count=0,
                issue_count=0,
                warning_count=0,
                failed_count=0,
                block_summary={},
                quality_payload={},
            ),
        )


def test_create_extraction_run_running_sets_started_at(
    migrated_database_url: str,
) -> None:
    file_id, _document_id = _create_document(migrated_database_url)
    try:
        run = create_extraction_run(
            migrated_database_url,
            ExtractionRunInput(
                file_id=file_id,
                status="running",
                provider_mode="local",
            ),
        )

        assert run.status == "running"
        assert run.started_at is not None
        assert run.finished_at is None
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_repository_persists_artifact_storage_path_without_text(
    migrated_database_url: str,
) -> None:
    file_id, document_id = _create_document(migrated_database_url)
    try:
        artifact = create_extraction_artifact(
            migrated_database_url,
            ExtractionArtifactInput(
                file_id=file_id,
                document_id=document_id,
                artifact_type="warning_report",
                storage_path="/tmp/warnings.json",
                content_hash="warning-hash",
                metadata={"warning_count": 1},
            ),
        )
        row = fetch_one(
            migrated_database_url,
            """
            SELECT storage_path, content_text, metadata
            FROM extraction_artifacts
            WHERE artifact_id = %s
            """,
            (artifact.artifact_id,),
        )

        assert row == {
            "storage_path": "/tmp/warnings.json",
            "content_text": None,
            "metadata": {"warning_count": 1},
        }
    finally:
        _cleanup_file(migrated_database_url, file_id)
