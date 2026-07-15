import pytest

from app.core.ingestion_artifacts import (
    DocumentBlockInput,
    ExtractionArtifactInput,
    ExtractionProfileInput,
    ExtractionRunInput,
    ImageArtifactInput,
    InvalidIngestionArtifactError,
    TableArtifactInput,
    validate_document_block_input,
    validate_extraction_artifact_input,
    validate_extraction_profile_input,
    validate_extraction_run_input,
    validate_image_artifact_input,
    validate_table_artifact_input,
)


def make_profile(**overrides) -> ExtractionProfileInput:
    values = {
        "extraction_profile_name": "local_markdown",
        "extractor_name": "markdown",
        "extractor_version": "0.1.0",
        "supported_file_types": ("md",),
    }
    values.update(overrides)
    return ExtractionProfileInput(**values)


def make_run(**overrides) -> ExtractionRunInput:
    values = {"file_id": 1, "document_id": 2}
    values.update(overrides)
    return ExtractionRunInput(**values)


def make_artifact(**overrides) -> ExtractionArtifactInput:
    values = {
        "file_id": 1,
        "document_id": 2,
        "artifact_type": "normalized_markdown",
        "content_text": "# Title",
    }
    values.update(overrides)
    return ExtractionArtifactInput(**values)


def make_block(**overrides) -> DocumentBlockInput:
    values = {
        "artifact_id": 1,
        "document_id": 2,
        "block_seq": 0,
        "block_type": "paragraph",
    }
    values.update(overrides)
    return DocumentBlockInput(**values)


def test_validate_extraction_profile_accepts_valid_input() -> None:
    validate_extraction_profile_input(make_profile())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"extraction_profile_name": " "}, "extraction_profile_name"),
        ({"extractor_name": " "}, "extractor_name"),
        ({"extractor_version": " "}, "extractor_version"),
        ({"provider_mode": "sidecar"}, "provider_mode"),
        ({"supported_file_types": ()}, "supported_file_types"),
        ({"supported_file_types": (" ",)}, "supported_file_types"),
    ],
)
def test_validate_extraction_profile_rejects_invalid_values(overrides, message: str) -> None:
    with pytest.raises(InvalidIngestionArtifactError, match=message):
        validate_extraction_profile_input(make_profile(**overrides))


def test_validate_extraction_run_accepts_valid_input() -> None:
    validate_extraction_run_input(
        make_run(
            status="failed",
            provider_mode="remote",
            extraction_profile_name="remote_pdf",
            extractor_name="pdf",
            extractor_version="1.2.3",
            elapsed_ms=100,
            warning_count=1,
            error_count=1,
            error_code="parse_failed",
            error_message="failed",
        )
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"file_id": 0}, "file_id"),
        ({"document_id": -1}, "document_id"),
        ({"extraction_profile_name": " "}, "extraction_profile_name"),
        ({"extractor_name": " "}, "extractor_name"),
        ({"extractor_version": " "}, "extractor_version"),
        ({"status": "done"}, "status"),
        ({"provider_mode": "worker"}, "provider_mode"),
        ({"elapsed_ms": -1}, "elapsed_ms"),
        ({"warning_count": -1}, "warning_count"),
        ({"error_count": -1}, "error_count"),
        ({"error_code": " "}, "error_code"),
        ({"error_message": " "}, "error_message"),
    ],
)
def test_validate_extraction_run_rejects_invalid_values(overrides, message: str) -> None:
    with pytest.raises(InvalidIngestionArtifactError, match=message):
        validate_extraction_run_input(make_run(**overrides))


def test_validate_extraction_artifact_accepts_text_or_storage_path() -> None:
    validate_extraction_artifact_input(make_artifact(content_text=None, storage_path="/tmp/a.md"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"file_id": 0}, "file_id"),
        ({"extraction_run_id": -1}, "extraction_run_id"),
        ({"document_id": -1}, "document_id"),
        ({"artifact_type": "markdown"}, "artifact_type"),
        ({"content_text": None, "storage_path": None}, "content_text or storage_path"),
        ({"content_text": None, "storage_path": " "}, "storage_path"),
        ({"content_hash": " "}, "content_hash"),
        ({"size_bytes": -1}, "size_bytes"),
    ],
)
def test_validate_extraction_artifact_rejects_invalid_values(overrides, message: str) -> None:
    with pytest.raises(InvalidIngestionArtifactError, match=message):
        validate_extraction_artifact_input(make_artifact(**overrides))


def test_validate_document_block_accepts_source_anchor_metadata() -> None:
    validate_document_block_input(
        make_block(
            heading_path=("Title",),
            source_anchor={"page_no": 1},
            page_no=1,
            char_start=0,
            char_end=12,
            token_count=3,
        )
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"artifact_id": 0}, "artifact_id"),
        ({"document_id": 0}, "document_id"),
        ({"parent_block_id": 0}, "parent_block_id"),
        ({"block_seq": -1}, "block_seq"),
        ({"block_type": "unknown"}, "block_type"),
        ({"page_no": 0}, "page_no"),
        ({"slide_no": -1}, "slide_no"),
        ({"token_count": -1}, "token_count"),
        ({"char_start": -1}, "char_start"),
        ({"char_start": 9, "char_end": 3}, "char_end"),
    ],
)
def test_validate_document_block_rejects_invalid_values(overrides, message: str) -> None:
    with pytest.raises(InvalidIngestionArtifactError, match=message):
        validate_document_block_input(make_block(**overrides))


def test_validate_table_artifact_accepts_any_content_representation() -> None:
    validate_table_artifact_input(TableArtifactInput(block_id=1, content_json={"rows": []}))


@pytest.mark.parametrize(
    ("table_input", "message"),
    [
        (TableArtifactInput(block_id=0, content_markdown="| A |"), "block_id"),
        (TableArtifactInput(block_id=1), "content_markdown"),
        (TableArtifactInput(block_id=1, storage_path=" "), "storage_path"),
        (TableArtifactInput(block_id=1, content_markdown="x", row_count=-1), "row_count"),
        (
            TableArtifactInput(block_id=1, content_markdown="x", column_count=-1),
            "column_count",
        ),
    ],
)
def test_validate_table_artifact_rejects_invalid_values(
    table_input: TableArtifactInput,
    message: str,
) -> None:
    with pytest.raises(InvalidIngestionArtifactError, match=message):
        validate_table_artifact_input(table_input)


def test_validate_image_artifact_accepts_valid_input() -> None:
    validate_image_artifact_input(
        ImageArtifactInput(block_id=1, storage_path="/tmp/image.png", width_px=10, height_px=20)
    )


@pytest.mark.parametrize(
    ("image_input", "message"),
    [
        (ImageArtifactInput(block_id=0, storage_path="/tmp/image.png"), "block_id"),
        (ImageArtifactInput(block_id=1, storage_path=" "), "storage_path"),
        (ImageArtifactInput(block_id=1, storage_path="/tmp/image.png", width_px=0), "width_px"),
        (
            ImageArtifactInput(block_id=1, storage_path="/tmp/image.png", height_px=-1),
            "height_px",
        ),
    ],
)
def test_validate_image_artifact_rejects_invalid_values(
    image_input: ImageArtifactInput,
    message: str,
) -> None:
    with pytest.raises(InvalidIngestionArtifactError, match=message):
        validate_image_artifact_input(image_input)
