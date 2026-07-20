import pytest

from app.core.bm25_keyword_index import (
    DEFAULT_BM25_TOKENIZER_NAME,
    InvalidBM25KeywordIndexError,
    build_bm25_term_frequencies,
    replace_chunk_keyword_terms_in_connection,
    tokenize_bm25_text,
)


def test_tokenize_bm25_text_normalizes_unicode_word_tokens() -> None:
    tokens = tokenize_bm25_text("NeX-PCX 문서 검색 2.0, BM25 bm25!")

    assert tokens == ("nex", "pcx", "문서", "검색", "2", "0", "bm25", "bm25")


def test_build_bm25_term_frequencies_counts_casefolded_tokens() -> None:
    frequencies = build_bm25_term_frequencies("Alpha alpha BETA 한국어 한국어.")

    assert frequencies == {"alpha": 2, "beta": 1, "한국어": 2}


def test_build_bm25_term_frequencies_returns_empty_for_punctuation_only() -> None:
    assert build_bm25_term_frequencies("... --- !!!") == {}


@pytest.mark.parametrize(
    ("tokenizer_name", "message"),
    [
        (" ", "tokenizer_name"),
        ("unknown_tokenizer", "Unsupported tokenizer_name"),
    ],
)
def test_tokenize_bm25_text_rejects_invalid_tokenizer(
    tokenizer_name: str,
    message: str,
) -> None:
    with pytest.raises(InvalidBM25KeywordIndexError, match=message):
        tokenize_bm25_text("content", tokenizer_name=tokenizer_name)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "chunk_id": 0,
                "chunk_policy_name": "heading_512_64",
                "chunk_text": "content",
            },
            "chunk_id",
        ),
        (
            {
                "chunk_id": 1,
                "chunk_policy_name": " ",
                "chunk_text": "content",
            },
            "chunk_policy_name",
        ),
        (
            {
                "chunk_id": 1,
                "chunk_policy_name": "heading_512_64",
                "chunk_text": "content",
                "tokenizer_name": "custom",
            },
            "Unsupported tokenizer_name",
        ),
    ],
)
def test_replace_chunk_keyword_terms_validates_before_db(
    kwargs: dict[str, object],
    message: str,
) -> None:
    call_kwargs = {"tokenizer_name": DEFAULT_BM25_TOKENIZER_NAME, **kwargs}
    with pytest.raises(InvalidBM25KeywordIndexError, match=message):
        replace_chunk_keyword_terms_in_connection(
            None,  # type: ignore[arg-type]
            **call_kwargs,
        )
