"""BM25 keyword index persistence helpers."""

import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from typing import Any

from psycopg import Connection

from app.core.database import connect

DEFAULT_BM25_TOKENIZER_NAME = "unicode_word_v1"
KOREAN_NGRAM_BM25_TOKENIZER_NAME = "unicode_word_ko_2_3gram_v1"
MECAB_KO_BM25_TOKENIZER_NAME = "mecab_ko_morph_v1"
AVERAGE_DOCUMENT_LENGTH_QUANT = Decimal("0.0001")
_WORD_PATTERN = re.compile(r"(?u)\b\w+\b")
_HANGUL_SYLLABLE_PATTERN = re.compile(r"[가-힣]+")
_KOREAN_NGRAM_SIZES = (2, 3)


@dataclass(frozen=True)
class BM25TokenizerDefinition:
    tokenizer_name: str
    display_name: str
    description: str
    dependency_mode: str
    status: str
    available: bool = True
    unavailable_reason: str | None = None
    install_hint: str | None = None


BM25_TOKENIZER_DEFINITIONS = {
    DEFAULT_BM25_TOKENIZER_NAME: BM25TokenizerDefinition(
        tokenizer_name=DEFAULT_BM25_TOKENIZER_NAME,
        display_name="Unicode Word",
        description="Case-folded Unicode word tokens for the reproducible BM25 baseline.",
        dependency_mode="builtin",
        status="default",
    ),
    KOREAN_NGRAM_BM25_TOKENIZER_NAME: BM25TokenizerDefinition(
        tokenizer_name=KOREAN_NGRAM_BM25_TOKENIZER_NAME,
        display_name="Unicode Word + Korean 2/3-gram",
        description=(
            "Unicode word tokens plus Hangul syllable 2-gram and 3-gram terms for "
            "Korean spacing and compound-noun recall experiments."
        ),
        dependency_mode="builtin",
        status="experimental",
    ),
    MECAB_KO_BM25_TOKENIZER_NAME: BM25TokenizerDefinition(
        tokenizer_name=MECAB_KO_BM25_TOKENIZER_NAME,
        display_name="Mecab-ko Morph",
        description=(
            "Optional Mecab-ko morphological tokens for Korean BM25 experiments. "
            "Requires mecab-ko-python and mecab-ko-dic."
        ),
        dependency_mode="optional",
        status="experimental",
        install_hint="pip install 'nex-pcx[korean-tokenizers]'",
    ),
}
SUPPORTED_BM25_TOKENIZERS = frozenset(BM25_TOKENIZER_DEFINITIONS)


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


def list_bm25_tokenizers() -> tuple[BM25TokenizerDefinition, ...]:
    definitions: list[BM25TokenizerDefinition] = []
    for definition in BM25_TOKENIZER_DEFINITIONS.values():
        if definition.tokenizer_name != MECAB_KO_BM25_TOKENIZER_NAME:
            definitions.append(definition)
            continue
        available, unavailable_reason = get_mecab_ko_tokenizer_availability()
        definitions.append(
            BM25TokenizerDefinition(
                tokenizer_name=definition.tokenizer_name,
                display_name=definition.display_name,
                description=definition.description,
                dependency_mode=definition.dependency_mode,
                status=definition.status,
                available=available,
                unavailable_reason=unavailable_reason,
                install_hint=definition.install_hint,
            )
        )
    return tuple(definitions)


def _tokenize_unicode_words(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _WORD_PATTERN.finditer(text))


def _iter_korean_ngrams(token: str) -> tuple[str, ...]:
    ngrams: list[str] = []
    normalized_token = token.casefold()
    for match in _HANGUL_SYLLABLE_PATTERN.finditer(normalized_token):
        sequence = match.group(0)
        for size in _KOREAN_NGRAM_SIZES:
            if len(sequence) < size:
                continue
            for start in range(0, len(sequence) - size + 1):
                term = sequence[start : start + size]
                if term == normalized_token:
                    continue
                ngrams.append(term)
    return tuple(ngrams)


def _tokenize_unicode_words_with_korean_ngrams(text: str) -> tuple[str, ...]:
    terms: list[str] = []
    for token in _tokenize_unicode_words(text):
        terms.append(token)
        terms.extend(_iter_korean_ngrams(token))
    return tuple(terms)


@lru_cache(maxsize=1)
def _load_mecab_ko_tokenizer() -> Any:
    try:
        from mecab_ko import Mecab
    except ModuleNotFoundError as exc:
        raise InvalidBM25KeywordIndexError(
            "mecab_ko module is not installed. Install nex-pcx[korean-tokenizers]."
        ) from exc

    try:
        import mecab_ko_dic
    except ModuleNotFoundError:
        dicpath = None
    else:
        dicpath = mecab_ko_dic.DICDIR

    try:
        return Mecab(dicpath=dicpath) if dicpath is not None else Mecab()
    except RuntimeError as exc:
        raise InvalidBM25KeywordIndexError(f"mecab_ko tokenizer is unavailable: {exc}") from exc


def get_mecab_ko_tokenizer_availability() -> tuple[bool, str | None]:
    try:
        _load_mecab_ko_tokenizer()
    except InvalidBM25KeywordIndexError as exc:
        return False, str(exc)
    return True, None


def _tokenize_mecab_ko_morphs(text: str) -> tuple[str, ...]:
    tokenizer = _load_mecab_ko_tokenizer()
    terms: list[str] = []
    for token in tokenizer.morphs(text):
        normalized = token.casefold()
        if _WORD_PATTERN.fullmatch(normalized) is None:
            continue
        terms.append(normalized)
    return tuple(terms)


_BM25_TOKENIZER_FUNCTIONS = {
    DEFAULT_BM25_TOKENIZER_NAME: _tokenize_unicode_words,
    KOREAN_NGRAM_BM25_TOKENIZER_NAME: _tokenize_unicode_words_with_korean_ngrams,
    MECAB_KO_BM25_TOKENIZER_NAME: _tokenize_mecab_ko_morphs,
}


def tokenize_bm25_text(
    text: str,
    *,
    tokenizer_name: str = DEFAULT_BM25_TOKENIZER_NAME,
) -> tuple[str, ...]:
    validated_tokenizer_name = _validate_tokenizer_name(tokenizer_name)
    return _BM25_TOKENIZER_FUNCTIONS[validated_tokenizer_name](text)


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
