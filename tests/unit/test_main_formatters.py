from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.core.chunks import ChunkRecord
from app.core.config import Settings
from app.core.document_inventory import DocumentInventoryItem
from app.core.file_metadata import FileMetadataRecord
from app.core.ingestion_artifacts import (
    DocumentBlockRecord,
    ExtractionArtifactRecord,
    ExtractionQualitySnapshotRecord,
    ExtractionQualitySnapshotSummary,
    ExtractionRunRecord,
    InvalidIngestionArtifactError,
)
from app.main import (
    CHAT_SESSION_DEFAULT_GENERATION_TEMPLATE_KEY,
    ExtractionRerunRequest,
    _chunks_for_source_trace,
    _extraction_artifact_export_filename,
    _percent_label,
    _percent_value,
    _resolve_extraction_rerun_profile_name,
    build_extraction_rerun_request,
    chat_generation_template_default_options,
    chat_session_default_generation_template_key,
    chunk_source_trace_preview_payload,
    document_artifacts_redirect_url,
    document_block_summary_payload,
    extraction_artifact_export_payload,
    extraction_artifact_preview_payload,
    extraction_quality_check_payload,
    extraction_quality_snapshot_input_from_context,
    extraction_quality_snapshot_payload,
    extraction_quality_snapshot_summary_payload,
    extraction_rerun_feedback_payload,
    search_log_bm25_tokenizer_name,
    search_log_reranked_vector_profile_name,
    search_reranker_runtime_control_payload,
    select_default_chat_generation_template_key,
)

NOW = datetime(2026, 7, 15, tzinfo=UTC)


def _chat_template(template_key: str, *, is_default: bool = False) -> SimpleNamespace:
    return SimpleNamespace(template_key=template_key, is_default=is_default)


def test_chat_session_default_generation_template_key_reads_metadata() -> None:
    session = SimpleNamespace(metadata={CHAT_SESSION_DEFAULT_GENERATION_TEMPLATE_KEY: " report "})

    assert chat_session_default_generation_template_key(session) == "report"
    assert (
        chat_session_default_generation_template_key(
            SimpleNamespace(metadata={CHAT_SESSION_DEFAULT_GENERATION_TEMPLATE_KEY: " "})
        )
        is None
    )
    assert (
        chat_session_default_generation_template_key(
            SimpleNamespace(metadata={CHAT_SESSION_DEFAULT_GENERATION_TEMPLATE_KEY: 42})
        )
        is None
    )


def test_chat_generation_template_default_options_filters_supported_templates() -> None:
    templates = (
        _chat_template("report"),
        _chat_template("grounded_answer"),
        _chat_template("proposal"),
    )

    options = chat_generation_template_default_options(templates)

    assert [template.template_key for template in options] == ["report", "proposal"]


def test_select_default_chat_generation_template_key_prefers_default_then_report() -> None:
    assert (
        select_default_chat_generation_template_key(
            (
                _chat_template("proposal"),
                _chat_template("summary", is_default=True),
                _chat_template("report"),
            )
        )
        == "summary"
    )
    assert (
        select_default_chat_generation_template_key(
            (_chat_template("proposal"), _chat_template("report"))
        )
        == "report"
    )
    assert select_default_chat_generation_template_key((_chat_template("proposal"),)) == "proposal"
    assert select_default_chat_generation_template_key(()) == ""


def test_search_log_bm25_tokenizer_name_reads_direct_and_profile_metadata() -> None:
    assert (
        search_log_bm25_tokenizer_name(
            {
                "profile_keyword_searches": {
                    "bm25_keyword": {"tokenizer_name": "unicode_word_ko_2_3gram_v1"}
                }
            }
        )
        == "unicode_word_ko_2_3gram_v1"
    )
    assert (
        search_log_bm25_tokenizer_name(
            {
                "bm25_tokenizer_name": "unicode_word_v1",
                "profile_keyword_searches": {
                    "bm25_keyword": {"tokenizer_name": "unicode_word_ko_2_3gram_v1"}
                },
            }
        )
        == "unicode_word_v1"
    )
    assert (
        search_log_bm25_tokenizer_name(
            {
                "bm25_tokenizer_name": "unknown",
                "profile_keyword_searches": {"bm25_keyword": {"tokenizer_name": "unicode_word_v1"}},
            }
        )
        == "unicode_word_v1"
    )
    assert search_log_bm25_tokenizer_name({"profile_keyword_searches": {}}) is None


def test_search_log_reranked_vector_profile_name_reads_profile_metadata() -> None:
    assert (
        search_log_reranked_vector_profile_name(
            {
                "profile_reranked_searches": {
                    "reranked_vector_cosine": {"reranked_vector_profile_name": "qwen3_4b_2560"}
                }
            }
        )
        == "qwen3_4b_2560"
    )
    assert (
        search_log_reranked_vector_profile_name(
            {
                "profile_reranked_searches": {
                    "reranked_vector_cosine": {"source_vector_profile_name": "kure_v1_1024"}
                }
            }
        )
        == "kure_v1_1024"
    )
    assert search_log_reranked_vector_profile_name({}) is None
    assert (
        search_log_reranked_vector_profile_name(
            {"profile_reranked_searches": {"reranked_vector_cosine": {}}}
        )
        is None
    )


def test_search_reranker_runtime_control_payload_reports_mock_defaults() -> None:
    payload = search_reranker_runtime_control_payload(Settings())

    assert payload["status"] == "configured"
    assert payload["mode"] == "mock"
    assert payload["remote_base_url"] == ""
    assert payload["timeout_seconds"] == 60.0
    assert payload["reranker_profile_name"] == "qwen3_reranker_4b"
    assert payload["reranker_model_id"] == "Qwen/Qwen3-Reranker-4B"


def test_search_reranker_runtime_control_payload_normalizes_remote_config() -> None:
    payload = search_reranker_runtime_control_payload(
        Settings(
            reranker_provider_mode=" REMOTE ",
            remote_reranker_provider_url=" http://reranker.local:9104/ ",
            remote_reranker_provider_timeout_seconds=90.0,
        )
    )

    assert payload["status"] == "configured"
    assert payload["mode"] == "remote"
    assert payload["remote_base_url"] == "http://reranker.local:9104"
    assert payload["timeout_seconds"] == 90.0
    assert payload["validation_error"] == ""


@pytest.mark.parametrize(
    ("settings", "expected_error", "expected_timeout"),
    [
        (
            Settings(reranker_provider_mode="remote"),
            "remote_reranker_provider_url is required",
            60.0,
        ),
        (
            Settings(
                reranker_provider_mode="remote",
                remote_reranker_provider_url="http://reranker.local:9104",
                remote_reranker_provider_timeout_seconds=0,
            ),
            "remote_reranker_provider_timeout_seconds must be greater than 0",
            0.0,
        ),
    ],
)
def test_search_reranker_runtime_control_payload_reports_invalid_config(
    settings: Settings,
    expected_error: str,
    expected_timeout: float,
) -> None:
    payload = search_reranker_runtime_control_payload(settings)

    assert payload["status"] == "invalid"
    assert payload["mode"] == "remote"
    assert expected_error in str(payload["validation_error"])
    assert payload["timeout_seconds"] == expected_timeout


def test_search_reranker_runtime_control_payload_preserves_unparseable_timeout() -> None:
    class BrokenSettings:
        reranker_provider_mode = "remote"
        remote_reranker_provider_url = " http://reranker.local:9104/ "
        remote_reranker_provider_timeout_seconds = "slow"

    payload = search_reranker_runtime_control_payload(BrokenSettings())

    assert payload["status"] == "invalid"
    assert payload["remote_base_url"] == "http://reranker.local:9104"
    assert payload["timeout_seconds"] is None
    assert "could not convert string to float" in str(payload["validation_error"])


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


def make_file_metadata_record(**overrides) -> FileMetadataRecord:
    values = {
        "file_id": 3,
        "document_id": 4,
        "original_file_name": "quality.md",
        "stored_file_name": "quality-stored.md",
        "file_ext": ".md",
        "mime_type": "text/markdown",
        "file_size_bytes": 120,
        "sha256_checksum": "checksum",
        "storage_path": "/tmp/quality.md",
        "document_group": "general",
        "security_level": "internal",
        "parse_status": "succeeded",
        "uploaded_by_user_id": None,
        "owner_user_id": None,
        "owner_org_unit_id": None,
        "access_scope": "personal",
    }
    values.update(overrides)
    return FileMetadataRecord(**values)


def test_build_extraction_rerun_request_defaults_profile_and_metadata() -> None:
    request = build_extraction_rerun_request(
        document=make_document_inventory_item(),
        file_record=make_file_metadata_record(),
        payload=ExtractionRerunRequest(
            requested_by="unit-test",
            options={"reason": "quality-review"},
        ),
    )

    assert request.file_id == 3
    assert request.document_id == 4
    assert request.storage_path == "/tmp/quality.md"
    assert request.extraction_profile_name == "local_markdown_default"
    assert request.detected_file_type == "md"
    assert request.options["reason"] == "quality-review"
    assert request.options["rerun_request"] == {
        "source": "extraction_rerun_api",
        "requested_by": "unit-test",
        "document_id": 4,
        "file_id": 3,
    }
    assert request.trace_id is not None
    assert request.trace_id.startswith("extraction-rerun-4-")


def test_resolve_extraction_rerun_profile_name_uses_explicit_profile() -> None:
    profile_name = _resolve_extraction_rerun_profile_name(
        make_file_metadata_record(file_ext=".pdf"),
        " local_pdf_text_default ",
    )

    assert profile_name == "local_pdf_text_default"


def test_resolve_extraction_rerun_profile_name_rejects_invalid_values() -> None:
    with pytest.raises(InvalidIngestionArtifactError, match="must not be blank"):
        _resolve_extraction_rerun_profile_name(make_file_metadata_record(), " ")

    with pytest.raises(InvalidIngestionArtifactError, match="No local extraction profile"):
        _resolve_extraction_rerun_profile_name(
            make_file_metadata_record(file_ext=".zip"),
            None,
        )


def test_extraction_rerun_feedback_payload_formats_success_and_error() -> None:
    success_payload = extraction_rerun_feedback_payload(
        run_id=12,
        status_value="succeeded",
        artifact_count=1,
        block_count=3,
    )
    failed_payload = extraction_rerun_feedback_payload(
        error_message="Source file missing",
    )

    assert success_payload == {
        "ok": True,
        "status": "succeeded",
        "run_id": 12,
        "artifact_count": 1,
        "block_count": 3,
        "error_message": None,
    }
    assert failed_payload == {
        "ok": False,
        "status": "failed",
        "run_id": None,
        "artifact_count": 0,
        "block_count": 0,
        "error_message": "Source file missing",
    }
    assert extraction_rerun_feedback_payload(status_value="succeeded") is None


def test_document_artifacts_redirect_url_omits_empty_values() -> None:
    url = document_artifacts_redirect_url(
        4,
        {
            "artifact_id": 10,
            "rerun_status": "succeeded",
            "rerun_error": "",
            "empty": None,
        },
    )

    assert url == "/documents/4/artifacts?artifact_id=10&rerun_status=succeeded"


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
    assert bundle_payload["selected_artifact"]["content_text"].startswith("# Quality Fixture")
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
