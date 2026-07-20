"""BM25 keyword index persistence helpers."""

import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from psycopg import Connection

from app.core.database import connect

DEFAULT_BM25_TOKENIZER_NAME = "unicode_word_v1"
SUPPORTED_BM25_TOKENIZERS = {DEFAULT_BM25_TOKENIZER_NAME}
AVERAGE_DOCUMENT_LENGTH_QUANT = Decimal("0.0001")
_WORD_PATTERN = re.compile(r"(?u)\b\w+\b")


@dataclass(frozen=True)
class BM25ChunkTermRecord:
    chunk_id: int
    chunk_policy_name: str
    tokenizer_name: str
    term: str
    term_frequency: int


@dataclass(frozen=True)
class BM25KeywordStatisticsRecord:
    chunk_policy_name: str
    tokenizer_name: str
    term: str
    document_frequency: int
    corpus_chunk_count: int
    average_document_length: Decimal


@dataclass(frozen=True)
class BM25IndexRefreshResult:
    chunk_policy_name: str
    tokenizer_name: str
    chunk_count: int
    term_row_count: int
    statistics_row_count: int
    average_document_length: Decimal


class InvalidBM25KeywordIndexError(ValueError):
    """Raised when BM25 keyword index input is invalid."""


def _require_positive_id(value: int, field_name: str) -> None:
    if value <= 0:
        raise InvalidBM25KeywordIndexError(f"{field_name} must be greater than 0")


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidBM25KeywordIndexError(f"{field_name} must not be blank")
    return normalized


def _validate_tokenizer_name(tokenizer_name: str) -> str:
    normalized = _validate_nonblank(tokenizer_name, "tokenizer_name")
    if normalized not in SUPPORTED_BM25_TOKENIZERS:
        raise InvalidBM25KeywordIndexError(f"Unsupported tokenizer_name: {tokenizer_name}")
    return normalized


def validate_bm25_tokenizer_name(tokenizer_name: str) -> str:
    return _validate_tokenizer_name(tokenizer_name)


def tokenize_bm25_text(
    text: str,
    *,
    tokenizer_name: str = DEFAULT_BM25_TOKENIZER_NAME,
) -> tuple[str, ...]:
    _validate_tokenizer_name(tokenizer_name)
    return tuple(match.group(0).casefold() for match in _WORD_PATTERN.finditer(text))


def build_bm25_term_frequencies(
    text: str,
    *,
    tokenizer_name: str = DEFAULT_BM25_TOKENIZER_NAME,
) -> dict[str, int]:
    terms = tokenize_bm25_text(text, tokenizer_name=tokenizer_name)
    return dict(Counter(terms))


def _row_to_chunk_term_record(row: dict[str, Any]) -> BM25ChunkTermRecord:
    return BM25ChunkTermRecord(
        chunk_id=int(row["chunk_id"]),
        chunk_policy_name=str(row["chunk_policy_name"]),
        tokenizer_name=str(row["tokenizer_name"]),
        term=str(row["term"]),
        term_frequency=int(row["term_frequency"]),
    )


def _row_to_keyword_statistics_record(row: dict[str, Any]) -> BM25KeywordStatisticsRecord:
    return BM25KeywordStatisticsRecord(
        chunk_policy_name=str(row["chunk_policy_name"]),
        tokenizer_name=str(row["tokenizer_name"]),
        term=str(row["term"]),
        document_frequency=int(row["document_frequency"]),
        corpus_chunk_count=int(row["corpus_chunk_count"]),
        average_document_length=row["average_document_length"],
    )


def replace_chunk_keyword_terms_in_connection(
    connection: Connection,
    *,
    chunk_id: int,
    chunk_policy_name: str,
    chunk_text: str,
    tokenizer_name: str = DEFAULT_BM25_TOKENIZER_NAME,
) -> list[BM25ChunkTermRecord]:
    _require_positive_id(chunk_id, "chunk_id")
    validated_policy_name = _validate_nonblank(chunk_policy_name, "chunk_policy_name")
    validated_tokenizer_name = _validate_tokenizer_name(tokenizer_name)
    term_frequencies = build_bm25_term_frequencies(
        chunk_text,
        tokenizer_name=validated_tokenizer_name,
    )

    with connection.cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM chunk_keyword_terms
            WHERE chunk_id = %s
              AND tokenizer_name = %s
            """,
            (chunk_id, validated_tokenizer_name),
        )
        if not term_frequencies:
            return []
        cursor.executemany(
            """
            INSERT INTO chunk_keyword_terms (
                chunk_id,
                chunk_policy_name,
                tokenizer_name,
                term,
                term_frequency
            )
            VALUES (%s, %s, %s, %s, %s)
            """,
            [
                (
                    chunk_id,
                    validated_policy_name,
                    validated_tokenizer_name,
                    term,
                    frequency,
                )
                for term, frequency in sorted(term_frequencies.items())
            ],
        )
        cursor.execute(
            """
            SELECT
                chunk_id,
                chunk_policy_name,
                tokenizer_name,
                term,
                term_frequency
            FROM chunk_keyword_terms
            WHERE chunk_id = %s
              AND tokenizer_name = %s
            ORDER BY term ASC
            """,
            (chunk_id, validated_tokenizer_name),
        )
        rows = cursor.fetchall()
    return [_row_to_chunk_term_record(dict(row)) for row in rows]


def replace_chunk_keyword_terms(
    database_url: str,
    *,
    chunk_id: int,
    chunk_policy_name: str,
    chunk_text: str,
    tokenizer_name: str = DEFAULT_BM25_TOKENIZER_NAME,
) -> list[BM25ChunkTermRecord]:
    with connect(database_url) as connection:
        return replace_chunk_keyword_terms_in_connection(
            connection,
            chunk_id=chunk_id,
            chunk_policy_name=chunk_policy_name,
            chunk_text=chunk_text,
            tokenizer_name=tokenizer_name,
        )


def refresh_chunk_policy_keyword_index_in_connection(
    connection: Connection,
    *,
    chunk_policy_name: str,
    tokenizer_name: str = DEFAULT_BM25_TOKENIZER_NAME,
) -> BM25IndexRefreshResult:
    validated_policy_name = _validate_nonblank(chunk_policy_name, "chunk_policy_name")
    validated_tokenizer_name = _validate_tokenizer_name(tokenizer_name)

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT chunk_id, chunk_text
            FROM chunks
            WHERE chunk_policy_name = %s
            ORDER BY chunk_id ASC
            """,
            (validated_policy_name,),
        )
        chunk_rows = cursor.fetchall()
        cursor.execute(
            """
            DELETE FROM chunk_keyword_terms
            WHERE chunk_policy_name = %s
              AND tokenizer_name = %s
            """,
            (validated_policy_name, validated_tokenizer_name),
        )
        cursor.execute(
            """
            DELETE FROM chunk_keyword_statistics
            WHERE chunk_policy_name = %s
              AND tokenizer_name = %s
            """,
            (validated_policy_name, validated_tokenizer_name),
        )

    chunk_lengths: list[int] = []
    term_document_counts: Counter[str] = Counter()
    term_row_count = 0
    for row in chunk_rows:
        term_frequencies = build_bm25_term_frequencies(
            str(row["chunk_text"]),
            tokenizer_name=validated_tokenizer_name,
        )
        chunk_lengths.append(sum(term_frequencies.values()))
        term_document_counts.update(term_frequencies.keys())
        if term_frequencies:
            replace_chunk_keyword_terms_in_connection(
                connection,
                chunk_id=int(row["chunk_id"]),
                chunk_policy_name=validated_policy_name,
                chunk_text=str(row["chunk_text"]),
                tokenizer_name=validated_tokenizer_name,
            )
            term_row_count += len(term_frequencies)

    chunk_count = len(chunk_rows)
    average_document_length = (
        Decimal(sum(chunk_lengths)) / Decimal(chunk_count) if chunk_count else Decimal("0")
    ).quantize(AVERAGE_DOCUMENT_LENGTH_QUANT)
    with connection.cursor() as cursor:
        if term_document_counts:
            cursor.executemany(
                """
                INSERT INTO chunk_keyword_statistics (
                    chunk_policy_name,
                    tokenizer_name,
                    term,
                    document_frequency,
                    corpus_chunk_count,
                    average_document_length
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        validated_policy_name,
                        validated_tokenizer_name,
                        term,
                        document_frequency,
                        chunk_count,
                        average_document_length,
                    )
                    for term, document_frequency in sorted(term_document_counts.items())
                ],
            )
    return BM25IndexRefreshResult(
        chunk_policy_name=validated_policy_name,
        tokenizer_name=validated_tokenizer_name,
        chunk_count=chunk_count,
        term_row_count=term_row_count,
        statistics_row_count=len(term_document_counts),
        average_document_length=average_document_length,
    )


def refresh_chunk_policy_keyword_index(
    database_url: str,
    *,
    chunk_policy_name: str,
    tokenizer_name: str = DEFAULT_BM25_TOKENIZER_NAME,
) -> BM25IndexRefreshResult:
    with connect(database_url) as connection:
        return refresh_chunk_policy_keyword_index_in_connection(
            connection,
            chunk_policy_name=chunk_policy_name,
            tokenizer_name=tokenizer_name,
        )


def list_chunk_keyword_terms(
    database_url: str,
    *,
    chunk_id: int,
    tokenizer_name: str = DEFAULT_BM25_TOKENIZER_NAME,
) -> list[BM25ChunkTermRecord]:
    _require_positive_id(chunk_id, "chunk_id")
    validated_tokenizer_name = _validate_tokenizer_name(tokenizer_name)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    chunk_id,
                    chunk_policy_name,
                    tokenizer_name,
                    term,
                    term_frequency
                FROM chunk_keyword_terms
                WHERE chunk_id = %s
                  AND tokenizer_name = %s
                ORDER BY term ASC
                """,
                (chunk_id, validated_tokenizer_name),
            )
            rows = cursor.fetchall()
    return [_row_to_chunk_term_record(dict(row)) for row in rows]


def list_chunk_keyword_statistics(
    database_url: str,
    *,
    chunk_policy_name: str,
    tokenizer_name: str = DEFAULT_BM25_TOKENIZER_NAME,
) -> list[BM25KeywordStatisticsRecord]:
    validated_policy_name = _validate_nonblank(chunk_policy_name, "chunk_policy_name")
    validated_tokenizer_name = _validate_tokenizer_name(tokenizer_name)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    chunk_policy_name,
                    tokenizer_name,
                    term,
                    document_frequency,
                    corpus_chunk_count,
                    average_document_length
                FROM chunk_keyword_statistics
                WHERE chunk_policy_name = %s
                  AND tokenizer_name = %s
                ORDER BY term ASC
                """,
                (validated_policy_name, validated_tokenizer_name),
            )
            rows = cursor.fetchall()
    return [_row_to_keyword_statistics_record(dict(row)) for row in rows]
