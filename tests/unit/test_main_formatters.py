from datetime import UTC, datetime
from decimal import Decimal

from app.core.ingestion_artifacts import DocumentBlockRecord, ExtractionArtifactRecord
from app.main import (
    _percent_label,
    _percent_value,
    document_block_summary_payload,
    extraction_artifact_preview_payload,
)


def test_percent_value_formats_decimal_and_numeric_values_to_two_places() -> None:
    assert _percent_value(Decimal("0E-20")) == "0.00"
    assert _percent_value(Decimal("33.33333333333333333333")) == "33.33"
    assert _percent_value(100) == "100.00"


def test_percent_label_appends_unit_and_handles_missing_values() -> None:
    assert _percent_label(Decimal("50")) == "50.00%"
    assert _percent_label(None) == "-"


def test_extraction_artifact_preview_payload_handles_missing_content_text() -> None:
    artifact = ExtractionArtifactRecord(
        artifact_id=1,
        extraction_run_id=2,
        file_id=3,
        document_id=4,
        artifact_type="normalized_markdown",
        content_text=None,
        storage_path=None,
        content_hash=None,
        size_bytes=None,
        language=None,
        metadata={},
        created_at=datetime(2026, 7, 15, tzinfo=UTC),
    )

    payload = extraction_artifact_preview_payload(artifact)

    assert payload["content_text"] is None
    assert payload["content_lines"] is None
    assert payload["content_preview"] is None
    assert payload["content_length"] is None


def test_document_block_summary_payload_counts_source_coordinates() -> None:
    blocks = [
        DocumentBlockRecord(
            block_id=1,
            artifact_id=10,
            document_id=20,
            parent_block_id=None,
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
            metadata={},
            created_at=datetime(2026, 7, 15, tzinfo=UTC),
        ),
        DocumentBlockRecord(
            block_id=2,
            artifact_id=10,
            document_id=20,
            parent_block_id=None,
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
            metadata={},
            created_at=datetime(2026, 7, 15, tzinfo=UTC),
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
