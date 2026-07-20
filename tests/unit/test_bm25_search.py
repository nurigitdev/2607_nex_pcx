import pytest

from app.core.bm25_search import (
    BM25SearchInput,
    InvalidBM25SearchError,
    validate_bm25_search_input,
)


def test_validate_bm25_search_input_returns_query_term_frequencies() -> None:
    terms = validate_bm25_search_input(
        BM25SearchInput(
            query_text="Alpha alpha 한국어",
            top_k=3,
            chunk_policy_name="heading_512_64",
            document_group="default",
            file_type=".md",
        )
    )

    assert terms == {"alpha": 2, "한국어": 1}


def test_validate_bm25_search_input_allows_punctuation_query_for_empty_result() -> None:
    assert validate_bm25_search_input(BM25SearchInput(query_text="... !!!")) == {}


@pytest.mark.parametrize(
    ("query_input", "message"),
    [
        (
            BM25SearchInput(query_text=" "),
            "query_text",
        ),
        (
            BM25SearchInput(query_text="hello", top_k=0),
            "top_k",
        ),
        (
            BM25SearchInput(query_text="hello", top_k=101),
            "top_k",
        ),
        (
            BM25SearchInput(query_text="hello", chunk_policy_name=" "),
            "chunk_policy_name",
        ),
        (
            BM25SearchInput(query_text="hello", tokenizer_name="custom"),
            "Unsupported tokenizer_name",
        ),
        (
            BM25SearchInput(query_text="hello", k1=0),
            "k1",
        ),
        (
            BM25SearchInput(query_text="hello", b=-0.1),
            "b",
        ),
        (
            BM25SearchInput(query_text="hello", b=1.1),
            "b",
        ),
        (
            BM25SearchInput(query_text="hello", document_group=" "),
            "document_group",
        ),
        (
            BM25SearchInput(query_text="hello", file_type=" "),
            "file_type",
        ),
    ],
)
def test_validate_bm25_search_input_rejects_invalid_values(
    query_input: BM25SearchInput,
    message: str,
) -> None:
    with pytest.raises(InvalidBM25SearchError, match=message):
        validate_bm25_search_input(query_input)
