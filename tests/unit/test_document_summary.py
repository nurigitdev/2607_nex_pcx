from types import SimpleNamespace

import pytest

from app.core.document_summary import (
    DEFAULT_DOCUMENT_SUMMARY_TEMPLATE_KEY,
    DocumentSummaryHistoryFilter,
    DocumentSummaryInput,
    InvalidDocumentSummaryError,
    run_document_summary_generation,
    validate_document_summary_history_filter,
)
from app.core.generation_runs import (
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
)


def _document() -> SimpleNamespace:
    return SimpleNamespace(
        document_id=5,
        file_id=9,
        document_title="요약 대상 문서",
        original_file_name="summary-target.md",
        document_group="unit-summary",
        file_ext=".md",
        access_scope="company",
        owner_user_id=3,
        owner_org_unit_id=4,
    )


def test_run_document_summary_generation_orchestrates_mock_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    document = _document()
    package = SimpleNamespace(
        search_log=SimpleNamespace(search_log=SimpleNamespace(search_log_id=44))
    )
    report = object()

    monkeypatch.setattr(
        "app.core.document_summary.get_document_inventory_item",
        lambda database_url, document_id: document,
    )
    monkeypatch.setattr(
        "app.core.document_summary._fetch_document_summary_chunk_ids",
        lambda *_args, **_kwargs: (101, 102),
    )

    def fake_create_search_log(database_url, *, document, summary_input, query_text, chunk_ids):
        calls["search_log"] = {
            "database_url": database_url,
            "document": document,
            "summary_input": summary_input,
            "query_text": query_text,
            "chunk_ids": chunk_ids,
        }
        return 44

    def fake_build_package(database_url, context_input):
        calls["context_input"] = context_input
        return package

    def fake_execute_mock(database_url, retrieval_package, **kwargs):
        calls["mock"] = {
            "database_url": database_url,
            "retrieval_package": retrieval_package,
            "kwargs": kwargs,
        }
        return report

    monkeypatch.setattr(
        "app.core.document_summary._create_document_summary_search_log",
        fake_create_search_log,
    )
    monkeypatch.setattr(
        "app.core.document_summary.build_retrieval_context_package",
        fake_build_package,
    )
    monkeypatch.setattr(
        "app.core.document_summary.execute_mock_generation_run",
        fake_execute_mock,
    )

    result = run_document_summary_generation(
        "postgresql://example",
        DocumentSummaryInput(
            document_id=5,
            actor_user_id=7,
            summary_instruction=" 핵심 리스크 중심 ",
            provider_mode=GENERATION_PROVIDER_MODE_MOCK,
            generation_template_key=" summary ",
            max_chunks=2,
            max_context_chars=2000,
            include_neighbors=True,
            chunk_policy_name=" heading_512_64 ",
        ),
    )

    search_log = calls["search_log"]
    context_input = calls["context_input"]
    assert result.document is document
    assert result.source_chunk_ids == (101, 102)
    assert result.retrieval_package is package
    assert result.generation_report is report
    assert search_log["query_text"] == "요약 대상 문서 요약 요청: 핵심 리스크 중심"
    assert search_log["summary_input"].summary_instruction == "핵심 리스크 중심"
    assert search_log["summary_input"].generation_template_key == "summary"
    assert search_log["summary_input"].chunk_policy_name == "heading_512_64"
    assert context_input.search_log_id == 44
    assert context_input.max_context_chars == 2000
    assert context_input.include_neighbors is True
    assert context_input.max_items == 2
    assert calls["mock"] == {
        "database_url": "postgresql://example",
        "retrieval_package": package,
        "kwargs": {
            "generation_template_key": "summary",
            "created_by": "api_document_summary",
            "created_by_user_id": 7,
        },
    }


def test_run_document_summary_generation_orchestrates_remote_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    document = _document()
    package = SimpleNamespace(
        search_log=SimpleNamespace(search_log=SimpleNamespace(search_log_id=55))
    )
    provider_client = object()

    monkeypatch.setattr(
        "app.core.document_summary.get_document_inventory_item",
        lambda *_args, **_kwargs: document,
    )
    monkeypatch.setattr(
        "app.core.document_summary._fetch_document_summary_chunk_ids",
        lambda *_args, **_kwargs: (201,),
    )
    monkeypatch.setattr(
        "app.core.document_summary._create_document_summary_search_log",
        lambda *_args, **_kwargs: 55,
    )
    monkeypatch.setattr(
        "app.core.document_summary.build_retrieval_context_package",
        lambda *_args, **_kwargs: package,
    )

    def fake_execute_remote(database_url, retrieval_package, **kwargs):
        calls["remote"] = {
            "database_url": database_url,
            "retrieval_package": retrieval_package,
            "kwargs": kwargs,
        }
        return "remote-report"

    monkeypatch.setattr(
        "app.core.document_summary.execute_remote_generation_run",
        fake_execute_remote,
    )

    result = run_document_summary_generation(
        "postgresql://example",
        DocumentSummaryInput(
            document_id=5,
            actor_user_id=7,
            provider_mode=GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
            generation_template_key=None,
        ),
        api_key="secret",
        generation_provider_client=provider_client,  # type: ignore[arg-type]
    )

    assert result.generation_report == "remote-report"
    assert calls["remote"] == {
        "database_url": "postgresql://example",
        "retrieval_package": package,
        "kwargs": {
            "generation_template_key": DEFAULT_DOCUMENT_SUMMARY_TEMPLATE_KEY,
            "provider_client": provider_client,
            "api_key": "secret",
            "created_by": "api_document_summary",
            "created_by_user_id": 7,
        },
    }


@pytest.mark.parametrize(
    ("summary_input", "message"),
    (
        (DocumentSummaryInput(document_id=0, actor_user_id=1), "document_id"),
        (DocumentSummaryInput(document_id=1, actor_user_id=0), "actor_user_id"),
        (DocumentSummaryInput(document_id=1, actor_user_id=1, max_chunks=0), "max_chunks"),
        (DocumentSummaryInput(document_id=1, actor_user_id=1, max_chunks=101), "max_chunks"),
        (
            DocumentSummaryInput(document_id=1, actor_user_id=1, provider_mode="unsupported"),
            "provider_mode",
        ),
    ),
)
def test_run_document_summary_generation_rejects_invalid_input(
    summary_input: DocumentSummaryInput,
    message: str,
) -> None:
    with pytest.raises(InvalidDocumentSummaryError, match=message):
        run_document_summary_generation("postgresql://example", summary_input)


def test_run_document_summary_generation_reports_missing_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.core.document_summary.get_document_inventory_item",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(InvalidDocumentSummaryError, match="document was not found"):
        run_document_summary_generation(
            "postgresql://example",
            DocumentSummaryInput(document_id=1, actor_user_id=1),
        )


def test_run_document_summary_generation_reports_missing_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.core.document_summary.get_document_inventory_item",
        lambda *_args, **_kwargs: _document(),
    )
    monkeypatch.setattr(
        "app.core.document_summary._fetch_document_summary_chunk_ids",
        lambda *_args, **_kwargs: (),
    )

    with pytest.raises(InvalidDocumentSummaryError, match="no summary-ready chunks"):
        run_document_summary_generation(
            "postgresql://example",
            DocumentSummaryInput(document_id=1, actor_user_id=1),
        )


def test_run_document_summary_generation_reports_missing_retrieval_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.core.document_summary.get_document_inventory_item",
        lambda *_args, **_kwargs: _document(),
    )
    monkeypatch.setattr(
        "app.core.document_summary._fetch_document_summary_chunk_ids",
        lambda *_args, **_kwargs: (301,),
    )
    monkeypatch.setattr(
        "app.core.document_summary._create_document_summary_search_log",
        lambda *_args, **_kwargs: 99,
    )
    monkeypatch.setattr(
        "app.core.document_summary.build_retrieval_context_package",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(InvalidDocumentSummaryError, match="retrieval context package"):
        run_document_summary_generation(
            "postgresql://example",
            DocumentSummaryInput(document_id=1, actor_user_id=1),
        )


def test_validate_document_summary_history_filter_normalizes_values() -> None:
    history_filter = validate_document_summary_history_filter(
        DocumentSummaryHistoryFilter(
            limit=10,
            run_status=" SUCCEEDED ",
            generation_template_key=" SUMMARY_RISK_ACTION ",
        )
    )
    blank_filter = validate_document_summary_history_filter(
        DocumentSummaryHistoryFilter(
            limit=1,
            run_status="",
            generation_template_key=" ",
        )
    )

    assert history_filter.limit == 10
    assert history_filter.run_status == "succeeded"
    assert history_filter.generation_template_key == "summary_risk_action"
    assert blank_filter.run_status == "all"
    assert blank_filter.generation_template_key == "all"


@pytest.mark.parametrize(
    ("history_filter", "message"),
    (
        (DocumentSummaryHistoryFilter(limit=0), "limit"),
        (DocumentSummaryHistoryFilter(limit=501), "limit"),
        (DocumentSummaryHistoryFilter(run_status="bad_status"), "run_status"),
    ),
)
def test_validate_document_summary_history_filter_rejects_invalid_values(
    history_filter: DocumentSummaryHistoryFilter,
    message: str,
) -> None:
    with pytest.raises(InvalidDocumentSummaryError, match=message):
        validate_document_summary_history_filter(history_filter)
