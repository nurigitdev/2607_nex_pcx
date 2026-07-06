import pytest

from app.core.chunks import (
    ChunkInput,
    InvalidChunkError,
    calculate_chunk_content_hash,
    validate_chunk_input,
)


def make_chunk(**overrides) -> ChunkInput:
    values = {
        "document_id": 1,
        "chunk_seq": 0,
        "chunk_text": "A chunk of text.",
    }
    values.update(overrides)
    return ChunkInput(**values)


def test_calculate_chunk_content_hash_uses_utf8_sha256() -> None:
    assert (
        calculate_chunk_content_hash("NeX PCX")
        == "9c407bc4f7c6832609a443475808506cbe058c49b152e059a47d7eb5603e84ab"
    )


def test_validate_chunk_input_computes_hash_when_missing() -> None:
    assert validate_chunk_input(make_chunk()) == calculate_chunk_content_hash("A chunk of text.")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"document_id": 0}, "document_id"),
        ({"chunk_seq": -1}, "chunk_seq"),
        ({"chunk_text": " "}, "chunk_text"),
        ({"chunk_policy_name": " "}, "chunk_policy_name"),
        ({"page_no": 0}, "page_no"),
        ({"slide_no": -1}, "slide_no"),
        ({"token_count": -1}, "token_count"),
        ({"content_hash": " "}, "content_hash"),
    ],
)
def test_validate_chunk_input_rejects_invalid_values(overrides, message: str) -> None:
    with pytest.raises(InvalidChunkError, match=message):
        validate_chunk_input(make_chunk(**overrides))
