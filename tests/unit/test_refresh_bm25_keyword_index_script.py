import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.bm25_index_refresh import (
    BM25_REFRESH_STATUS_FAILED,
    BM25_REFRESH_STATUS_SUCCEEDED,
    BM25IndexRefreshPolicyResult,
    BM25IndexRefreshReport,
)
from app.core.bm25_keyword_index import DEFAULT_BM25_TOKENIZER_NAME


def _load_refresh_bm25_keyword_index_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "refresh_bm25_keyword_index.py"
    spec = importlib.util.spec_from_file_location("refresh_bm25_keyword_index_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


refresh_bm25_keyword_index = _load_refresh_bm25_keyword_index_module()


def make_report(status: str = BM25_REFRESH_STATUS_SUCCEEDED) -> BM25IndexRefreshReport:
    failed = status == BM25_REFRESH_STATUS_FAILED
    return BM25IndexRefreshReport(
        status=status,
        tokenizer_name=DEFAULT_BM25_TOKENIZER_NAME,
        policy_count=1,
        succeeded_count=0 if failed else 1,
        failed_count=1 if failed else 0,
        empty_policy_count=0,
        results=(
            BM25IndexRefreshPolicyResult(
                chunk_policy_name="heading_512_64",
                tokenizer_name=DEFAULT_BM25_TOKENIZER_NAME,
                status=status,
                chunk_count=3,
                term_row_count=8,
                statistics_row_count=6,
                average_document_length=Decimal("2.6667"),
                error_message="failed" if failed else None,
            ),
        ),
    )


def test_main_passes_cli_options_and_writes_json(monkeypatch, tmp_path) -> None:
    captured = {}
    json_output = tmp_path / "bm25" / "refresh.json"

    def fake_refresh(database_url: str, *, options):
        captured["database_url"] = database_url
        captured["options"] = options
        return make_report()

    monkeypatch.setattr(
        refresh_bm25_keyword_index,
        "get_settings",
        lambda: SimpleNamespace(database_url="postgresql://original/db"),
    )
    monkeypatch.setattr(refresh_bm25_keyword_index, "refresh_bm25_keyword_indexes", fake_refresh)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refresh_bm25_keyword_index.py",
            "--database-url",
            "postgresql://override/db",
            "--chunk-policy",
            "heading_512_64",
            "--chunk-policy",
            "heading_1000_200",
            "--json-output",
            str(json_output),
            "--pretty",
        ],
    )

    exit_code = refresh_bm25_keyword_index.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert captured["database_url"] == "postgresql://override/db"
    assert captured["options"].chunk_policy_names == ("heading_512_64", "heading_1000_200")
    assert captured["options"].tokenizer_name == DEFAULT_BM25_TOKENIZER_NAME
    assert captured["options"].continue_on_error is True
    assert payload["status"] == BM25_REFRESH_STATUS_SUCCEEDED
    assert payload["results"][0]["average_document_length"] == "2.6667"


def test_main_prints_json_and_returns_nonzero_when_refresh_fails(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        refresh_bm25_keyword_index,
        "get_settings",
        lambda: SimpleNamespace(database_url="postgresql://example/db"),
    )
    monkeypatch.setattr(
        refresh_bm25_keyword_index,
        "refresh_bm25_keyword_indexes",
        lambda *args, **kwargs: make_report(BM25_REFRESH_STATUS_FAILED),
    )
    monkeypatch.setattr(sys, "argv", ["refresh_bm25_keyword_index.py"])

    exit_code = refresh_bm25_keyword_index.main()
    output = capsys.readouterr().out

    assert exit_code == 1
    assert '"failed_count": 1' in output


def test_main_allows_failed_refresh_when_requested(monkeypatch) -> None:
    monkeypatch.setattr(
        refresh_bm25_keyword_index,
        "get_settings",
        lambda: SimpleNamespace(database_url="postgresql://example/db"),
    )
    monkeypatch.setattr(
        refresh_bm25_keyword_index,
        "refresh_bm25_keyword_indexes",
        lambda *args, **kwargs: make_report(BM25_REFRESH_STATUS_FAILED),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["refresh_bm25_keyword_index.py", "--allow-failures"],
    )

    assert refresh_bm25_keyword_index.main() == 0


def test_main_requires_database_url(monkeypatch) -> None:
    monkeypatch.setattr(
        refresh_bm25_keyword_index,
        "get_settings",
        lambda: SimpleNamespace(database_url=None),
    )
    monkeypatch.setattr(sys, "argv", ["refresh_bm25_keyword_index.py"])

    with pytest.raises(SystemExit) as exc_info:
        refresh_bm25_keyword_index.main()

    assert exc_info.value.code == 2
