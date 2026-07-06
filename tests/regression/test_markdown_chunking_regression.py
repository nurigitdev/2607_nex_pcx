import json
from pathlib import Path

from app.core.chunking import chunk_parsed_document
from app.core.document_parsers import MarkdownParser

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def _chunk_to_dict(chunk) -> dict:
    return {
        "chunk_seq": chunk.chunk_seq,
        "chunk_text": chunk.chunk_text,
        "chunk_policy_name": chunk.chunk_policy_name,
        "parser_name": chunk.parser_name,
        "parser_version": chunk.parser_version,
        "heading_path": list(chunk.heading_path),
        "token_count": chunk.token_count,
        "content_hash": chunk.content_hash,
        "metadata": chunk.metadata,
    }


def test_markdown_chunking_fixture_matches_expected_chunks() -> None:
    source_path = FIXTURES_DIR / "files" / "markdown_parser_sample.md"
    expected_path = FIXTURES_DIR / "expected_chunks" / "markdown_chunk_sample.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))

    document = MarkdownParser().parse_text(source_path.read_text(encoding="utf-8"))
    chunks = chunk_parsed_document(document, document_id=1)

    assert [_chunk_to_dict(chunk) for chunk in chunks] == expected
