import pytest

from app.core.chunking import (
    ChunkPolicy,
    InvalidChunkPolicyError,
    chunk_parsed_document,
    count_chunk_tokens,
    get_chunk_policy,
    validate_chunk_policy,
)
from app.core.document_parsers import MarkdownParser


def test_get_chunk_policy_returns_seeded_policy() -> None:
    policy = get_chunk_policy("heading_1000_200")

    assert policy.target_token_size == 1000
    assert policy.overlap_token_size == 200


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (ChunkPolicy("", 512, 64), "chunk_policy_name"),
        (ChunkPolicy("bad", 0, 0), "target_token_size"),
        (ChunkPolicy("bad", 512, -1), "overlap_token_size"),
        (ChunkPolicy("bad", 512, 512), "overlap_token_size"),
        (ChunkPolicy("bad", 512, 64, split_strategy="plain"), "Unsupported split strategy"),
    ],
)
def test_validate_chunk_policy_rejects_invalid_values(
    policy: ChunkPolicy,
    message: str,
) -> None:
    with pytest.raises(InvalidChunkPolicyError, match=message):
        validate_chunk_policy(policy)


def test_get_chunk_policy_rejects_unknown_policy() -> None:
    with pytest.raises(InvalidChunkPolicyError, match="Unsupported chunk policy"):
        get_chunk_policy("missing")


def test_chunk_parsed_document_splits_on_heading_boundaries() -> None:
    document = MarkdownParser().parse_text("# Root\n\nIntro\n\n## Child\n\nBody")

    chunks = chunk_parsed_document(document, document_id=1)

    assert [chunk.heading_path for chunk in chunks] == [
        ("Root",),
        ("Root", "Child"),
    ]
    assert chunks[0].chunk_text == "# Root\n\nIntro"
    assert chunks[1].chunk_text == "## Child\n\nBody"


def test_chunk_parsed_document_preserves_code_fence_and_table() -> None:
    document = MarkdownParser().parse_text(
        "# Root\n\n```python\nprint('hello')\n```\n\n| A | B |\n| --- | --- |\n| 1 | 2 |",
    )

    chunks = chunk_parsed_document(document, document_id=1)

    assert len(chunks) == 1
    assert "```python\nprint('hello')\n```" in chunks[0].chunk_text
    assert "| A | B |" in chunks[0].chunk_text
    assert chunks[0].metadata["block_types"] == ["heading", "code_block", "table"]


def test_chunk_parsed_document_applies_block_overlap_when_target_is_exceeded() -> None:
    document = MarkdownParser().parse_text(
        "# Root\n\none two three\n\nfour five six\n\nseven eight nine",
    )
    policy = ChunkPolicy(
        chunk_policy_name="test_7_3",
        target_token_size=7,
        overlap_token_size=3,
    )

    chunks = chunk_parsed_document(document, document_id=1, policy=policy)

    assert [chunk.chunk_text for chunk in chunks] == [
        "# Root\n\none two three",
        "one two three\n\nfour five six",
        "four five six\n\nseven eight nine",
    ]
    assert [chunk.token_count for chunk in chunks] == [
        count_chunk_tokens(chunk.chunk_text) for chunk in chunks
    ]


def test_chunk_parsed_document_rejects_invalid_document_id() -> None:
    document = MarkdownParser().parse_text("# Root")

    with pytest.raises(ValueError, match="document_id"):
        chunk_parsed_document(document, document_id=0)
