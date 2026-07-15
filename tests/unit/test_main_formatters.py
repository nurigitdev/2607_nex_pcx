from datetime import UTC, datetime
from decimal import Decimal

from app.core.ingestion_artifacts import (
    DocumentBlockRecord,
    ExtractionArtifactRecord,
    ExtractionRunRecord,
)
from app.main import (
    _percent_label,
    _percent_value,
    document_block_summary_payload,
    extraction_artifact_preview_payload,
    extraction_quality_check_payload,
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
