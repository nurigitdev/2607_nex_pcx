from datetime import UTC, datetime
from decimal import Decimal

from app.core.document_inventory import DocumentInventoryItem
from app.core.chunks import ChunkRecord
from app.core.ingestion_artifacts import (
    DocumentBlockRecord,
    ExtractionArtifactRecord,
    ExtractionQualitySnapshotRecord,
    ExtractionQualitySnapshotSummary,
    ExtractionRunRecord,
)
from app.main import (
    _extraction_artifact_export_filename,
    _chunks_for_source_trace,
    _percent_label,
    _percent_value,
    chunk_source_trace_preview_payload,
    document_block_summary_payload,
    extraction_artifact_export_payload,
    extraction_artifact_preview_payload,
    extraction_quality_check_payload,
    extraction_quality_snapshot_input_from_context,
    extraction_quality_snapshot_payload,
    extraction_quality_snapshot_summary_payload,
)

NOW = datetime(2026, 7, 15, tzinfo=UTC)


def make_extraction_artifact(**overrides) -> ExtractionArtifactRecord:
    values = {
        "artifact_id": 1,
        "extraction_run_id": 2,
        "file_id": 3,
        "document_id": 4,
        "artifact_type": "normalized_markdown",
        "content_text": (
            "# Quality Fixture\n\n"
            "This extracted artifact has enough normalized text for the baseline "
            "quality review threshold and lineage checks."
        ),
        "storage_path": None,
        "content_hash": "hash",
        "size_bytes": None,
        "language": "ko",
        "metadata": {},
        "created_at": NOW,
    }
    values.update(overrides)
    return ExtractionArtifactRecord(**values)


def make_document_block(**overrides) -> DocumentBlockRecord:
    values = {
        "block_id": 1,
        "artifact_id": 1,
        "document_id": 4,
        "parent_block_id": None,
        "block_seq": 0,
        "block_type": "heading",
        "content_text": "Quality Fixture",
        "content_markdown": "# Quality Fixture",
        "heading_path": ("Quality Fixture",),
        "source_anchor": {"block_seq": 0},
        "page_no": None,
        "slide_no": None,
        "sheet_name": None,
        "cell_range": None,
        "char_start": 0,
        "char_end": 17,
        "token_count": 3,
        "metadata": {},
        "created_at": NOW,
    }
    values.update(overrides)
    return DocumentBlockRecord(**values)


def make_chunk(**overrides) -> ChunkRecord:
    values = {
        "chunk_id": 1,
        "document_id": 4,
        "chunk_seq": 0,
        "chunk_text": "Quality chunk",
        "content_hash": "chunk-hash",
        "chunk_policy_name": "heading_512_64",
        "artifact_id": 1,
        "block_id": 1,
        "chunk_type": "text",
        "content_markdown": "Quality chunk",
        "parser_name": "markdown",
        "parser_version": "0.1.0",
        "heading_path": ("Quality Fixture",),
        "source_anchor": {"block_seq": 0},
        "page_no": None,
        "slide_no": None,
        "sheet_name": None,
        "cell_range": None,
        "source_char_start": 0,
        "source_char_end": 13,
        "token_count": 2,
        "char_count": 13,
        "prev_chunk_id": None,
        "next_chunk_id": None,
        "metadata": {},
    }
    values.update(overrides)
    return ChunkRecord(**values)


def make_extraction_run(**overrides) -> ExtractionRunRecord:
    values = {
        "extraction_run_id": 2,
        "file_id": 3,
        "document_id": 4,
        "extraction_profile_name": "local_markdown_default",
        "status": "succeeded",
        "provider_mode": "local",
        "extractor_name": "markdown",
        "extractor_version": "0.1.0",
        "started_at": NOW,
        "finished_at": NOW,
        "elapsed_ms": 10,
        "warning_count": 0,
        "error_count": 0,
        "error_code": None,
        "error_message": None,
        "runtime_metadata": {},
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(overrides)
    return ExtractionRunRecord(**values)


def make_extraction_quality_snapshot(**overrides) -> ExtractionQualitySnapshotRecord:
    values = {
        "snapshot_id": 1,
        "document_id": 4,
        "file_id": 3,
        "artifact_id": 1,
        "extraction_run_id": 2,
        "artifact_type": "normalized_markdown",
        "extraction_profile_name": "local_markdown_default",
        "extractor_name": "markdown",
        "extractor_version": "0.1.0",
        "status": "passed",
        "content_length": 100,
        "content_lines": 3,
        "block_count": 2,
        "source_anchor_count": 2,
        "source_anchor_coverage_percent": 100.0,
        "issue_count": 0,
        "warning_count": 0,
        "failed_count": 0,
        "block_summary": {"block_count": 2},
        "quality_payload": {"status": "passed", "issues": []},
        "created_by": "unit-test",
        "created_by_user_id": None,
        "created_at": NOW,
    }
    values.update(overrides)
    return ExtractionQualitySnapshotRecord(**values)


def make_document_inventory_item(**overrides) -> DocumentInventoryItem:
    values = {
        "document_id": 4,
        "file_id": 3,
        "document_title": "Quality Fixture",
        "original_file_name": "quality.md",
        "file_ext": ".md",
        "mime_type": "text/markdown",
        "file_size_bytes": 120,
        "document_group": "general",
        "security_level": "internal",
        "document_status": "active",
        "parse_status": "succeeded",
        "owner_user_id": None,
        "owner_login_id": None,
        "owner_display_name": None,
        "owner_org_unit_id": None,
        "owner_org_unit_name": None,
        "access_scope": "personal",
        "uploaded_by": "tester",
        "uploaded_by_user_id": None,
        "uploaded_by_login_id": None,
        "uploaded_by_display_name": None,
        "chunk_count": 0,
        "total_token_count": None,
        "total_char_count": 0,
        "latest_pipeline_job_id": None,
        "latest_pipeline_status": None,
        "latest_pipeline_stage": None,
        "latest_pipeline_progress_percent": None,
        "uploaded_at": NOW,
        "updated_at": NOW,
    }
    values.update(overrides)
    return DocumentInventoryItem(**values)


def test_percent_value_formats_decimal_and_numeric_values_to_two_places() -> None:
    assert _percent_value(Decimal("0E-20")) == "0.00"
    assert _percent_value(Decimal("33.33333333333333333333")) == "33.33"
    assert _percent_value(100) == "100.00"


def test_percent_label_appends_unit_and_handles_missing_values() -> None:
    assert _percent_label(Decimal("50")) == "50.00%"
    assert _percent_label(None) == "-"


def test_extraction_artifact_preview_payload_handles_missing_content_text() -> None:
    artifact = make_extraction_artifact(content_text=None)

    payload = extraction_artifact_preview_payload(artifact)

    assert payload["content_text"] is None
    assert payload["content_lines"] is None
    assert payload["content_preview"] is None
    assert payload["content_length"] is None


def test_extraction_artifact_export_filename_uses_expected_extension() -> None:
    assert (
        _extraction_artifact_export_filename(
            document_id=10,
            artifact_id=20,
            export_format="markdown",
        )
        == "document-10-artifact-20-markdown.md"
    )
    assert (
        _extraction_artifact_export_filename(
            document_id=10,
            artifact_id=20,
            export_format="bundle_json",
        )
        == "document-10-artifact-20-bundle_json.json"
    )


def test_extraction_artifact_export_payload_formats_supported_json_exports() -> None:
    document = make_document_inventory_item()
    artifact = make_extraction_artifact(metadata={"parser_name": "markdown"})
    blocks = [make_document_block()]
    extraction_runs = [make_extraction_run()]

    blocks_payload = extraction_artifact_export_payload(
        document=document,
        artifact=artifact,
        blocks=blocks,
        extraction_runs=extraction_runs,
        export_format="blocks_json",
    )
    metadata_payload = extraction_artifact_export_payload(
        document=document,
        artifact=artifact,
        blocks=blocks,
        extraction_runs=extraction_runs,
        export_format="metadata_json",
    )
    quality_payload = extraction_artifact_export_payload(
        document=document,
        artifact=artifact,
        blocks=blocks,
        extraction_runs=extraction_runs,
        export_format="quality_json",
    )
    bundle_payload = extraction_artifact_export_payload(
        document=document,
        artifact=artifact,
        blocks=blocks,
        extraction_runs=extraction_runs,
        export_format="bundle_json",
    )

    assert blocks_payload["block_summary"]["block_count"] == 1
    assert blocks_payload["blocks"][0]["block_id"] == 1
    assert metadata_payload["metadata"] == {"parser_name": "markdown"}
    assert quality_payload["quality_check"]["status"] == "passed"
    assert bundle_payload["selected_artifact"]["content_text"].startswith(
        "# Quality Fixture"
    )
    assert bundle_payload["extraction_runs"][0]["status"] == "succeeded"


def test_extraction_artifact_export_payload_rejects_unsupported_format() -> None:
    try:
        extraction_artifact_export_payload(
            document=make_document_inventory_item(),
            artifact=make_extraction_artifact(),
            blocks=[],
            extraction_runs=[],
            export_format="csv",
        )
    except ValueError as exc:
        assert "Unsupported extraction artifact export format" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_chunk_source_trace_preview_payload_groups_chunks_by_block() -> None:
    artifact = make_extraction_artifact()
    block = make_document_block()
    linked_chunk = make_chunk()
    unlinked_chunk = make_chunk(
        chunk_id=2,
        chunk_seq=1,
        block_id=None,
        chunk_text="Artifact level chunk",
        content_markdown="Artifact level chunk",
    )

    payload = chunk_source_trace_preview_payload(
        artifact,
        [block],
        [linked_chunk, unlinked_chunk],
    )

    assert payload["summary"] == {
        "block_count": 1,
        "chunk_count": 2,
        "traced_block_count": 1,
        "unlinked_chunk_count": 1,
        "chunk_policy_names": ["heading_512_64"],
    }
    assert payload["block_traces"][0]["chunk_count"] == 1
    assert payload["block_traces"][0]["chunks"][0]["chunk_id"] == linked_chunk.chunk_id
    assert payload["unlinked_chunks"][0]["chunk_id"] == unlinked_chunk.chunk_id


def test_chunk_source_trace_preview_payload_handles_no_artifact() -> None:
    payload = chunk_source_trace_preview_payload(None, [], [])

    assert payload["selected_artifact_id"] is None
    assert payload["selected_artifact"] is None
    assert payload["summary"]["chunk_count"] == 0


def test_chunks_for_source_trace_filters_by_artifact_or_block() -> None:
    chunks = [
        make_chunk(chunk_id=1, artifact_id=10, block_id=None),
        make_chunk(chunk_id=2, artifact_id=None, block_id=20),
        make_chunk(chunk_id=3, artifact_id=99, block_id=99),
    ]

    filtered = _chunks_for_source_trace(chunks, artifact_id=10, block_ids={20})

    assert [chunk.chunk_id for chunk in filtered] == [1, 2]


def test_extraction_quality_snapshot_payload_formats_datetime_and_percent() -> None:
    payload = extraction_quality_snapshot_payload(make_extraction_quality_snapshot())

    assert payload["snapshot_id"] == 1
    assert payload["source_anchor_coverage_label"] == "100.00%"
    assert payload["created_at"] == "2026-07-15T00:00:00+00:00"
    assert payload["quality_payload"] == {"status": "passed", "issues": []}


def test_extraction_quality_snapshot_summary_payload_handles_latest_snapshot() -> None:
    snapshot = make_extraction_quality_snapshot()
    payload = extraction_quality_snapshot_summary_payload(
        ExtractionQualitySnapshotSummary(
            document_id=4,
            artifact_id=1,
            snapshot_count=1,
            passed_count=1,
            warning_count=0,
            failed_count=0,
            latest_snapshot=snapshot,
        )
    )

    assert payload["snapshot_count"] == 1
    assert payload["latest_snapshot"]["snapshot_id"] == snapshot.snapshot_id


def test_extraction_quality_snapshot_summary_payload_handles_empty_summary() -> None:
    payload = extraction_quality_snapshot_summary_payload(
        ExtractionQualitySnapshotSummary(
            document_id=4,
            artifact_id=None,
            snapshot_count=0,
            passed_count=0,
            warning_count=0,
            failed_count=0,
            latest_snapshot=None,
        )
    )

    assert payload["snapshot_count"] == 0
    assert payload["latest_snapshot"] is None


def test_extraction_quality_snapshot_input_from_context_uses_selected_run_metadata() -> None:
    document = make_document_inventory_item()
    artifact = make_extraction_artifact()
    blocks = [
        make_document_block(),
        make_document_block(
            block_id=2,
            block_seq=1,
            block_type="paragraph",
            content_text="Enough paragraph text.",
            content_markdown="Enough paragraph text.",
            heading_path=("Quality Fixture",),
            source_anchor={"block_seq": 1},
        ),
    ]

    snapshot_input = extraction_quality_snapshot_input_from_context(
        document=document,
        artifact=artifact,
        blocks=blocks,
        extraction_runs=[make_extraction_run()],
        created_by="unit-test",
        created_by_user_id=None,
    )

    assert snapshot_input.document_id == document.document_id
    assert snapshot_input.extraction_profile_name == "local_markdown_default"
    assert snapshot_input.status == "passed"
    assert snapshot_input.block_summary["block_count"] == 2
    assert snapshot_input.quality_payload["issue_count"] == 0


def test_extraction_quality_snapshot_input_from_context_allows_missing_run() -> None:
    snapshot_input = extraction_quality_snapshot_input_from_context(
        document=make_document_inventory_item(),
        artifact=make_extraction_artifact(extraction_run_id=999),
        blocks=[],
        extraction_runs=[],
        created_by=None,
        created_by_user_id=None,
    )

    assert snapshot_input.extraction_profile_name is None
    assert snapshot_input.status == "failed"
    assert snapshot_input.failed_count == 1


def test_document_block_summary_payload_counts_source_coordinates() -> None:
    blocks = [
        make_document_block(
            block_id=1,
            artifact_id=10,
            document_id=20,
            block_seq=1,
            block_type="page",
            content_text="Page text",
            content_markdown="Page text",
            heading_path=(),
            source_anchor={"page": 1},
            page_no=1,
            slide_no=None,
            sheet_name=None,
            cell_range=None,
            char_start=0,
            char_end=9,
            token_count=2,
        ),
        make_document_block(
            block_id=2,
            artifact_id=10,
            document_id=20,
            block_seq=2,
            block_type="sheet",
            content_text="Sheet text",
            content_markdown="Sheet text",
            heading_path=(),
            source_anchor={},
            page_no=None,
            slide_no=3,
            sheet_name="Sheet1",
            cell_range="A1:B2",
            char_start=10,
            char_end=20,
            token_count=2,
        ),
    ]

    assert document_block_summary_payload(blocks) == {
        "block_count": 2,
        "block_type_counts": {"page": 1, "sheet": 1},
        "source_anchor_count": 1,
        "page_count": 1,
        "slide_count": 1,
        "sheet_count": 1,
        "sheet_names": ["Sheet1"],
    }


def test_extraction_quality_check_payload_passes_well_formed_artifact() -> None:
    payload = extraction_quality_check_payload(
        make_extraction_artifact(),
        [
            make_document_block(),
            make_document_block(
                block_id=2,
                block_seq=1,
                block_type="paragraph",
                content_text="Enough paragraph text.",
                content_markdown="Enough paragraph text.",
                heading_path=("Quality Fixture",),
                source_anchor={"block_seq": 1},
            ),
        ],
        [make_extraction_run()],
    )

    assert payload["status"] == "passed"
    assert payload["issue_count"] == 0
    assert payload["source_anchor_coverage_percent"] == 100.0


def test_extraction_quality_check_payload_handles_no_selected_artifact() -> None:
    payload = extraction_quality_check_payload(None, [], [])

    assert payload["status"] == "not_available"
    assert payload["issues"] == [
        {
            "code": "no_artifact_selected",
            "severity": "info",
            "message": "No extraction artifact is selected.",
            "metric": None,
        }
    ]
    assert payload["warning_count"] == 0
    assert payload["failed_count"] == 0


def test_extraction_quality_check_payload_fails_blank_artifact_without_blocks() -> None:
    payload = extraction_quality_check_payload(
        make_extraction_artifact(content_text="  "),
        [],
        [make_extraction_run(status="failed", error_count=1)],
    )

    assert payload["status"] == "failed"
    assert payload["failed_count"] == 3
    assert {issue["code"] for issue in payload["issues"]} == {
        "missing_content_text",
        "missing_blocks",
        "extraction_run_errors",
    }


def test_extraction_quality_check_payload_warns_for_partial_lineage() -> None:
    payload = extraction_quality_check_payload(
        make_extraction_artifact(),
        [
            make_document_block(
                block_type="paragraph",
                source_anchor={"block_seq": 0},
            ),
            make_document_block(
                block_id=2,
                block_seq=1,
                block_type="paragraph",
                source_anchor={},
            ),
        ],
        [make_extraction_run(warning_count=1)],
    )

    assert payload["status"] == "warning"
    assert payload["warning_count"] == 3
    assert {issue["code"] for issue in payload["issues"]} == {
        "low_source_anchor_coverage",
        "missing_heading_blocks",
        "extraction_run_warnings",
    }
    assert payload["source_anchor_coverage_percent"] == 50.0


def test_extraction_quality_check_payload_warns_when_all_source_anchors_are_missing() -> None:
    payload = extraction_quality_check_payload(
        make_extraction_artifact(),
        [make_document_block(source_anchor={})],
        [make_extraction_run()],
    )

    assert payload["status"] == "warning"
    assert payload["warning_count"] == 1
    assert payload["issues"][0]["code"] == "missing_source_anchors"
