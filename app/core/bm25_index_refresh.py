"""BM25 keyword index refresh runner."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from app.core.bm25_keyword_index import (
    DEFAULT_BM25_TOKENIZER_NAME,
    BM25IndexRefreshResult,
    InvalidBM25KeywordIndexError,
    refresh_chunk_policy_keyword_index,
    validate_bm25_tokenizer_name,
)
from app.core.database import connect

BM25_REFRESH_STATUS_SUCCEEDED = "succeeded"
BM25_REFRESH_STATUS_FAILED = "failed"


@dataclass(frozen=True)
class BM25IndexRefreshOptions:
    chunk_policy_names: tuple[str, ...] = ()
    tokenizer_name: str = DEFAULT_BM25_TOKENIZER_NAME
    continue_on_error: bool = True


@dataclass(frozen=True)
class BM25IndexRefreshPolicyResult:
    chunk_policy_name: str
    tokenizer_name: str
    status: str
    chunk_count: int
    term_row_count: int
    statistics_row_count: int
    average_document_length: Decimal
    error_message: str | None = None


@dataclass(frozen=True)
class BM25IndexRefreshReport:
    status: str
    tokenizer_name: str
    policy_count: int
    succeeded_count: int
    failed_count: int
    empty_policy_count: int
    results: tuple[BM25IndexRefreshPolicyResult, ...]


def _validate_policy_name(chunk_policy_name: str) -> str:
    normalized = chunk_policy_name.strip()
    if not normalized:
        raise InvalidBM25KeywordIndexError("chunk_policy_name must not be blank")
    return normalized


def _normalize_policy_names(chunk_policy_names: tuple[str, ...]) -> tuple[str, ...]:
    normalized_names: list[str] = []
    seen_names: set[str] = set()
    for chunk_policy_name in chunk_policy_names:
        normalized_name = _validate_policy_name(chunk_policy_name)
        if normalized_name in seen_names:
            continue
        normalized_names.append(normalized_name)
        seen_names.add(normalized_name)
    return tuple(normalized_names)


def list_bm25_refresh_chunk_policy_names(database_url: str) -> tuple[str, ...]:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT chunk_policy_name
                FROM chunk_policies
                ORDER BY chunk_policy_name ASC
                """)
            rows = cursor.fetchall()
    return tuple(str(row["chunk_policy_name"]) for row in rows)


def refresh_bm25_keyword_indexes(
    database_url: str,
    *,
    options: BM25IndexRefreshOptions | None = None,
) -> BM25IndexRefreshReport:
    refresh_options = options or BM25IndexRefreshOptions()
    tokenizer_name = validate_bm25_tokenizer_name(refresh_options.tokenizer_name)

    policy_names = _normalize_policy_names(refresh_options.chunk_policy_names)
    if not policy_names:
        policy_names = list_bm25_refresh_chunk_policy_names(database_url)

    results: list[BM25IndexRefreshPolicyResult] = []
    for policy_name in policy_names:
        try:
            refresh_result = refresh_chunk_policy_keyword_index(
                database_url,
                chunk_policy_name=policy_name,
                tokenizer_name=tokenizer_name,
            )
        except Exception as exc:
            if not refresh_options.continue_on_error:
                raise
            results.append(_failed_policy_result(policy_name, tokenizer_name, exc))
            continue
        results.append(_succeeded_policy_result(refresh_result))

    failed_count = sum(1 for result in results if result.status == BM25_REFRESH_STATUS_FAILED)
    succeeded_count = len(results) - failed_count
    return BM25IndexRefreshReport(
        status=BM25_REFRESH_STATUS_FAILED if failed_count else BM25_REFRESH_STATUS_SUCCEEDED,
        tokenizer_name=tokenizer_name,
        policy_count=len(results),
        succeeded_count=succeeded_count,
        failed_count=failed_count,
        empty_policy_count=sum(
            1
            for result in results
            if result.status == BM25_REFRESH_STATUS_SUCCEEDED and result.chunk_count == 0
        ),
        results=tuple(results),
    )


def _succeeded_policy_result(
    refresh_result: BM25IndexRefreshResult,
) -> BM25IndexRefreshPolicyResult:
    return BM25IndexRefreshPolicyResult(
        chunk_policy_name=refresh_result.chunk_policy_name,
        tokenizer_name=refresh_result.tokenizer_name,
        status=BM25_REFRESH_STATUS_SUCCEEDED,
        chunk_count=refresh_result.chunk_count,
        term_row_count=refresh_result.term_row_count,
        statistics_row_count=refresh_result.statistics_row_count,
        average_document_length=refresh_result.average_document_length,
    )


def _failed_policy_result(
    chunk_policy_name: str,
    tokenizer_name: str,
    exc: Exception,
) -> BM25IndexRefreshPolicyResult:
    return BM25IndexRefreshPolicyResult(
        chunk_policy_name=chunk_policy_name,
        tokenizer_name=tokenizer_name,
        status=BM25_REFRESH_STATUS_FAILED,
        chunk_count=0,
        term_row_count=0,
        statistics_row_count=0,
        average_document_length=Decimal("0.0000"),
        error_message=str(exc),
    )


def bm25_index_refresh_report_payload(
    report: BM25IndexRefreshReport,
) -> dict[str, Any]:
    return {
        "status": report.status,
        "tokenizer_name": report.tokenizer_name,
        "policy_count": report.policy_count,
        "succeeded_count": report.succeeded_count,
        "failed_count": report.failed_count,
        "empty_policy_count": report.empty_policy_count,
        "results": [
            {
                "chunk_policy_name": result.chunk_policy_name,
                "tokenizer_name": result.tokenizer_name,
                "status": result.status,
                "chunk_count": result.chunk_count,
                "term_row_count": result.term_row_count,
                "statistics_row_count": result.statistics_row_count,
                "average_document_length": f"{result.average_document_length:.4f}",
                "error_message": result.error_message,
            }
            for result in report.results
        ],
    }
