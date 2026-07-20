from decimal import Decimal

import pytest

from app.core import bm25_index_refresh
from app.core.bm25_index_refresh import (
    BM25_REFRESH_STATUS_FAILED,
    BM25_REFRESH_STATUS_SUCCEEDED,
    BM25IndexRefreshOptions,
    bm25_index_refresh_report_payload,
    refresh_bm25_keyword_indexes,
)
from app.core.bm25_keyword_index import (
    DEFAULT_BM25_TOKENIZER_NAME,
    BM25IndexRefreshResult,
    InvalidBM25KeywordIndexError,
)


def make_refresh_result(
    policy_name: str,
    *,
    chunk_count: int = 2,
) -> BM25IndexRefreshResult:
    return BM25IndexRefreshResult(
        chunk_policy_name=policy_name,
        tokenizer_name=DEFAULT_BM25_TOKENIZER_NAME,
        chunk_count=chunk_count,
        term_row_count=4,
        statistics_row_count=3,
        average_document_length=Decimal("2.5000"),
    )


def test_refresh_bm25_keyword_indexes_uses_all_policies_when_omitted(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        bm25_index_refresh,
        "list_bm25_refresh_chunk_policy_names",
        lambda database_url: ("heading_512_64", "heading_1000_200"),
    )

    def fake_refresh(database_url: str, **kwargs):
        calls.append((database_url, kwargs))
        return make_refresh_result(kwargs["chunk_policy_name"])

    monkeypatch.setattr(bm25_index_refresh, "refresh_chunk_policy_keyword_index", fake_refresh)

    report = refresh_bm25_keyword_indexes("postgresql://example/db")
    payload = bm25_index_refresh_report_payload(report)

    assert report.status == BM25_REFRESH_STATUS_SUCCEEDED
    assert report.policy_count == 2
    assert report.succeeded_count == 2
    assert report.failed_count == 0
    assert calls == [
        (
            "postgresql://example/db",
            {
                "chunk_policy_name": "heading_512_64",
                "tokenizer_name": DEFAULT_BM25_TOKENIZER_NAME,
            },
        ),
        (
            "postgresql://example/db",
            {
                "chunk_policy_name": "heading_1000_200",
                "tokenizer_name": DEFAULT_BM25_TOKENIZER_NAME,
            },
        ),
    ]
    assert payload["results"][0]["average_document_length"] == "2.5000"


def test_refresh_bm25_keyword_indexes_deduplicates_policy_names_and_records_failures(
    monkeypatch,
) -> None:
    calls = []

    def fake_refresh(database_url: str, **kwargs):
        calls.append(kwargs["chunk_policy_name"])
        if kwargs["chunk_policy_name"] == "bad_policy":
            raise RuntimeError("policy refresh failed")
        return make_refresh_result(kwargs["chunk_policy_name"], chunk_count=0)

    monkeypatch.setattr(bm25_index_refresh, "refresh_chunk_policy_keyword_index", fake_refresh)

    report = refresh_bm25_keyword_indexes(
        "postgresql://example/db",
        options=BM25IndexRefreshOptions(
            chunk_policy_names=(" heading_512_64 ", "bad_policy", "heading_512_64"),
        ),
    )
    payload = bm25_index_refresh_report_payload(report)

    assert calls == ["heading_512_64", "bad_policy"]
    assert report.status == BM25_REFRESH_STATUS_FAILED
    assert report.policy_count == 2
    assert report.succeeded_count == 1
    assert report.failed_count == 1
    assert report.empty_policy_count == 1
    assert payload["results"][1]["status"] == BM25_REFRESH_STATUS_FAILED
    assert payload["results"][1]["error_message"] == "policy refresh failed"


def test_refresh_bm25_keyword_indexes_can_stop_on_first_error(monkeypatch) -> None:
    def fake_refresh(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(bm25_index_refresh, "refresh_chunk_policy_keyword_index", fake_refresh)

    with pytest.raises(RuntimeError, match="boom"):
        refresh_bm25_keyword_indexes(
            "postgresql://example/db",
            options=BM25IndexRefreshOptions(
                chunk_policy_names=("heading_512_64",),
                continue_on_error=False,
            ),
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            BM25IndexRefreshOptions(tokenizer_name=" "),
            "tokenizer_name",
        ),
        (
            BM25IndexRefreshOptions(tokenizer_name="custom"),
            "Unsupported tokenizer_name",
        ),
        (
            BM25IndexRefreshOptions(chunk_policy_names=(" ",)),
            "chunk_policy_name",
        ),
    ],
)
def test_refresh_bm25_keyword_indexes_rejects_invalid_options(
    options: BM25IndexRefreshOptions,
    message: str,
) -> None:
    with pytest.raises(InvalidBM25KeywordIndexError, match=message):
        refresh_bm25_keyword_indexes("postgresql://example/db", options=options)
