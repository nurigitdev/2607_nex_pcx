from datetime import UTC, datetime

import pytest

from app.core.chunking import (
    ChunkPolicy,
    InvalidChunkPolicyError,
    chunk_document_blocks,
    chunk_parsed_document,
    count_chunk_tokens,
    get_chunk_policy,
    validate_chunk_policy,
)
from app.core.document_parsers import MarkdownParser
from app.core.ingestion_artifacts import DocumentBlockRecord


def make_document_block(**overrides) -> DocumentBlockRecord:
    values = {
        "block_id": 10,
        "artifact_id": 20,
        "document_id": 1,
        "parent_block_id": None,
        "block_seq": 0,
        "block_type": "paragraph",
        "content_text": "Example block",
        "content_markdown": "Example block",
        "heading_path": ("Root",),
        "source_anchor": {"start_line": 1, "end_line": 1},
        "page_no": None,
        "slide_no": None,
        "sheet_name": None,
        "cell_range": None,
        "char_start": 0,
        "char_end": 13,
        "token_count": 2,
        "metadata": {},
        "created_at": datetime(2026, 7, 15, tzinfo=UTC),
    }
    values.update(overrides)
    return DocumentBlockRecord(**values)


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


def test_chunk_document_blocks_preserves_source_lineage() -> None:
    blocks = [
        make_document_block(
            block_id=101,
            block_seq=0,
            block_type="heading",
            content_text="Root",
            content_markdown="# Root",
            source_anchor={"start_line": 1, "end_line": 1},
            char_start=0,
            char_end=6,
            token_count=2,
        ),
        make_document_block(
            block_id=102,
            block_seq=1,
            block_type="paragraph",
            content_text="Body text",
            content_markdown="Body text",
            source_anchor={"start_line": 3, "end_line": 3},
            char_start=8,
            char_end=17,
            token_count=2,
        ),
    ]

    chunks = chunk_document_blocks(
        blocks,
        document_id=1,
        parser_name="markdown",
        parser_version="0.1.0",
    )

    assert len(chunks) == 1
    assert chunks[0].artifact_id == 20
    assert chunks[0].block_id == 101
    assert chunks[0].chunk_type == "text"
    assert chunks[0].content_markdown == "# Root\n\nBody text"
    assert chunks[0].parser_name == "markdown"
    assert chunks[0].source_anchor["start_line"] == 1
    assert chunks[0].source_anchor["end_line"] == 3
    assert chunks[0].source_char_start == 0
    assert chunks[0].source_char_end == 17
    assert chunks[0].metadata["block_ids"] == [101, 102]
    assert chunks[0].metadata["block_types"] == ["heading", "paragraph"]


def test_chunk_document_blocks_uses_single_block_semantic_chunk_type() -> None:
    chunks = chunk_document_blocks(
        [
            make_document_block(
                block_id=103,
                block_seq=0,
                block_type="table",
                content_text="A B",
                content_markdown="| A | B |",
                source_anchor={"start_line": 4, "end_line": 6, "table_index": 0},
                char_start=20,
                char_end=50,
            )
        ],
        document_id=1,
    )

    assert chunks[0].chunk_type == "table"
    assert chunks[0].block_id == 103
    assert chunks[0].source_anchor["start_line"] == 4
    assert chunks[0].source_anchor["end_line"] == 6


def test_chunk_document_blocks_splits_on_heading_boundaries() -> None:
    blocks = [
        make_document_block(
            block_id=101,
            block_seq=0,
            block_type="heading",
            content_text="Root",
            content_markdown="# Root",
            heading_path=("Root",),
        ),
        make_document_block(
            block_id=102,
            block_seq=1,
            block_type="paragraph",
            content_text="Intro",
            content_markdown="Intro",
            heading_path=("Root",),
        ),
        make_document_block(
            block_id=103,
            block_seq=2,
            block_type="heading",
            content_text="Child",
            content_markdown="## Child",
            heading_path=("Root", "Child"),
            source_anchor={},
            char_start=None,
            char_end=None,
        ),
        make_document_block(
            block_id=104,
            block_seq=3,
            block_type="paragraph",
            content_text="Body",
            content_markdown=None,
            heading_path=("Root", "Child"),
            source_anchor={},
            char_start=None,
            char_end=None,
        ),
    ]

    chunks = chunk_document_blocks(blocks, document_id=1)

    assert [chunk.chunk_text for chunk in chunks] == [
        "# Root\n\nIntro",
        "## Child\n\nBody",
    ]
    assert chunks[1].heading_path == ("Root", "Child")
    assert "start_line" not in chunks[1].source_anchor
    assert chunks[1].source_char_start is None
    assert chunks[1].source_char_end is None


def test_chunk_document_blocks_applies_block_overlap() -> None:
    blocks = [
        make_document_block(block_id=101, block_seq=0, content_markdown="one two three"),
        make_document_block(block_id=102, block_seq=1, content_markdown="four five six"),
        make_document_block(block_id=103, block_seq=2, content_markdown="seven eight nine"),
    ]
    policy = ChunkPolicy(
        chunk_policy_name="document_7_4",
        target_token_size=7,
        overlap_token_size=4,
    )

    chunks = chunk_document_blocks(blocks, document_id=1, policy=policy)

    assert [chunk.chunk_text for chunk in chunks] == [
        "one two three\n\nfour five six",
        "four five six\n\nseven eight nine",
    ]
    assert chunks[1].metadata["block_ids"] == [102, 103]


def test_chunk_document_blocks_allows_zero_overlap() -> None:
    blocks = [
        make_document_block(block_id=101, block_seq=0, content_markdown="one two three"),
        make_document_block(block_id=102, block_seq=1, content_markdown="four five six"),
    ]
    policy = ChunkPolicy(
        chunk_policy_name="document_5_0",
        target_token_size=5,
        overlap_token_size=0,
    )

    chunks = chunk_document_blocks(blocks, document_id=1, policy=policy)

    assert [chunk.chunk_text for chunk in chunks] == [
        "one two three",
        "four five six",
    ]


def test_chunk_document_blocks_rejects_wrong_document_id() -> None:
    with pytest.raises(ValueError, match="document_id"):
        chunk_document_blocks([], document_id=0)

    with pytest.raises(ValueError, match="target document_id"):
        chunk_document_blocks(
            [make_document_block(document_id=2)],
            document_id=1,
        )


def test_chunk_parsed_document_rejects_invalid_document_id() -> None:
    document = MarkdownParser().parse_text("# Root")

    with pytest.raises(ValueError, match="document_id"):
        chunk_parsed_document(document, document_id=0)
