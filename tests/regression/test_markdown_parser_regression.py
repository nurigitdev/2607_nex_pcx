import json
from pathlib import Path

from app.core.document_parsers import MarkdownParser

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def test_markdown_parser_fixture_matches_expected_structure() -> None:
    source_path = FIXTURES_DIR / "files" / "markdown_parser_sample.md"
    expected_path = FIXTURES_DIR / "expected_parses" / "markdown_parser_sample.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))

    document = MarkdownParser().parse_text(source_path.read_text(encoding="utf-8"))

    assert document.to_dict(include_source_path=False) == expected
    assert document.extracted_text_size == len(document.extracted_text)
