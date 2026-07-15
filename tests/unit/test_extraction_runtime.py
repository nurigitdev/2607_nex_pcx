import pytest

from app.core.extraction_runtime import (
    ExtractionRuntimeArtifact,
    ExtractionRuntimeBlock,
    ExtractionRuntimeRequest,
    ExtractionRuntimeResult,
    validate_runtime_artifact,
    validate_runtime_block,
    validate_runtime_request,
    validate_runtime_result,
)
from app.core.ingestion_artifacts import InvalidIngestionArtifactError


def test_validate_runtime_request_accepts_profile_and_file_metadata() -> None:
    validate_runtime_request(
        ExtractionRuntimeRequest(
            file_id=1,
            document_id=2,
            storage_path="/tmp/source.md",
            extraction_profile_name="local_markdown_default",
            mime_type="text/markdown",
            detected_file_type="md",
            trace_id="trace-1",
            options={"normalize_line_endings": True},
        )
    )


@pytest.mark.parametrize(
    ("runtime_request", "message"),
    [
        (
            ExtractionRuntimeRequest(
                file_id=0,
                storage_path="/tmp/source.md",
                extraction_profile_name="local_markdown_default",
            ),
            "file_id",
        ),
        (
            ExtractionRuntimeRequest(
                file_id=1,
                document_id=-1,
                storage_path="/tmp/source.md",
                extraction_profile_name="local_markdown_default",
            ),
            "document_id",
        ),
        (
            ExtractionRuntimeRequest(
                file_id=1,
                storage_path=" ",
                extraction_profile_name="local_markdown_default",
            ),
            "storage_path",
        ),
        (
            ExtractionRuntimeRequest(
                file_id=1,
                storage_path="/tmp/source.md",
                extraction_profile_name=" ",
            ),
            "extraction_profile_name",
        ),
        (
            ExtractionRuntimeRequest(
                file_id=1,
                storage_path="/tmp/source.md",
                extraction_profile_name="local_markdown_default",
                detected_file_type=" ",
            ),
            "detected_file_type",
        ),
        (
            ExtractionRuntimeRequest(
                file_id=1,
                storage_path="/tmp/source.md",
                extraction_profile_name="local_markdown_default",
                mime_type=" ",
            ),
            "mime_type",
        ),
        (
            ExtractionRuntimeRequest(
                file_id=1,
                storage_path="/tmp/source.md",
                extraction_profile_name="local_markdown_default",
                trace_id=" ",
            ),
            "trace_id",
        ),
    ],
)
def test_validate_runtime_request_rejects_invalid_values(
    runtime_request: ExtractionRuntimeRequest,
    message: str,
) -> None:
    with pytest.raises(InvalidIngestionArtifactError, match=message):
        validate_runtime_request(runtime_request)


def test_validate_runtime_artifact_accepts_text_or_storage_path() -> None:
    validate_runtime_artifact(
        ExtractionRuntimeArtifact(
            artifact_type="normalized_markdown",
            content_text="# Title",
            content_hash="abc",
            size_bytes=7,
            language="ko",
        )
    )
    validate_runtime_artifact(
        ExtractionRuntimeArtifact(
            artifact_type="warning_report",
            storage_path="/tmp/warnings.json",
        )
    )


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        (ExtractionRuntimeArtifact(artifact_type="markdown", content_text="x"), "artifact_type"),
        (ExtractionRuntimeArtifact(artifact_type="plain_text"), "content_text or storage_path"),
        (
            ExtractionRuntimeArtifact(artifact_type="plain_text", storage_path=" "),
            "storage_path",
        ),
        (
            ExtractionRuntimeArtifact(
                artifact_type="plain_text",
                content_text="x",
                content_hash=" ",
            ),
            "content_hash",
        ),
        (
            ExtractionRuntimeArtifact(
                artifact_type="plain_text",
                content_text="x",
                size_bytes=-1,
            ),
            "size_bytes",
        ),
    ],
)
def test_validate_runtime_artifact_rejects_invalid_values(
    artifact: ExtractionRuntimeArtifact,
    message: str,
) -> None:
    with pytest.raises(InvalidIngestionArtifactError, match=message):
        validate_runtime_artifact(artifact)


def test_validate_runtime_block_accepts_source_anchor() -> None:
    validate_runtime_block(
        ExtractionRuntimeBlock(
            block_seq=0,
            block_type="paragraph",
            content_text="hello",
            heading_path=("Title",),
            source_anchor={"line": 1},
            char_start=0,
            char_end=5,
            token_count=1,
        )
    )


@pytest.mark.parametrize(
    ("block", "message"),
    [
        (ExtractionRuntimeBlock(block_seq=-1, block_type="paragraph"), "block_seq"),
        (
            ExtractionRuntimeBlock(block_seq=0, parent_block_seq=-1, block_type="paragraph"),
            "parent_block_seq",
        ),
        (
            ExtractionRuntimeBlock(block_seq=0, parent_block_seq=0, block_type="paragraph"),
            "parent_block_seq",
        ),
        (ExtractionRuntimeBlock(block_seq=0, block_type="unknown"), "block_type"),
        (ExtractionRuntimeBlock(block_seq=0, block_type="page", page_no=0), "page_no"),
        (ExtractionRuntimeBlock(block_seq=0, block_type="slide", slide_no=-1), "slide_no"),
        (ExtractionRuntimeBlock(block_seq=0, block_type="paragraph", token_count=-1), "token"),
        (
            ExtractionRuntimeBlock(
                block_seq=0,
                block_type="paragraph",
                char_start=5,
                char_end=3,
            ),
            "char_end",
        ),
    ],
)
def test_validate_runtime_block_rejects_invalid_values(
    block: ExtractionRuntimeBlock,
    message: str,
) -> None:
    with pytest.raises(InvalidIngestionArtifactError, match=message):
        validate_runtime_block(block)


def test_validate_runtime_result_accepts_success_and_failure_contracts() -> None:
    validate_runtime_result(
        ExtractionRuntimeResult(
            status="succeeded",
            artifacts=(
                ExtractionRuntimeArtifact(
                    artifact_type="normalized_markdown",
                    content_text="# Title",
                ),
            ),
            blocks=(ExtractionRuntimeBlock(block_seq=0, block_type="heading"),),
            warnings=("table normalized",),
            elapsed_ms=10,
            runtime_metadata={"extractor": "local_markdown"},
        )
    )
    validate_runtime_result(
        ExtractionRuntimeResult(
            status="failed",
            errors=("parse failed",),
            elapsed_ms=1,
        )
    )


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (ExtractionRuntimeResult(status="running"), "terminal"),
        (ExtractionRuntimeResult(status="unknown"), "status"),
        (ExtractionRuntimeResult(status="succeeded"), "requires artifacts"),
        (ExtractionRuntimeResult(status="failed"), "requires errors"),
        (
            ExtractionRuntimeResult(
                status="skipped",
                elapsed_ms=-1,
            ),
            "elapsed_ms",
        ),
    ],
)
def test_validate_runtime_result_rejects_invalid_values(
    result: ExtractionRuntimeResult,
    message: str,
) -> None:
    with pytest.raises(InvalidIngestionArtifactError, match=message):
        validate_runtime_result(result)
