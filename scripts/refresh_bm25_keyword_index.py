"""Refresh BM25 keyword indexes from the command line."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.bm25_index_refresh import (  # noqa: E402
    BM25IndexRefreshOptions,
    bm25_index_refresh_report_payload,
    refresh_bm25_keyword_indexes,
)
from app.core.bm25_keyword_index import (  # noqa: E402
    DEFAULT_BM25_TOKENIZER_NAME,
    InvalidBM25KeywordIndexError,
)
from app.core.config import get_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh materialized BM25 keyword terms and corpus statistics.",
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument(
        "--chunk-policy",
        action="append",
        dest="chunk_policy_names",
        default=None,
        help="Chunk policy to refresh. Repeat to refresh multiple policies. "
        "Omit to refresh every policy.",
    )
    parser.add_argument(
        "--tokenizer-name",
        default=DEFAULT_BM25_TOKENIZER_NAME,
        help=f"Tokenizer name. Default: {DEFAULT_BM25_TOKENIZER_NAME}.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional JSON output path. Prints JSON to stdout when omitted.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return exit code 0 even when one or more policies fail.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    args = parser.parse_args()

    settings = get_settings()
    database_url = args.database_url or settings.database_url
    if not database_url:
        parser.error("--database-url or NEX_PCX_DATABASE_URL is required")

    try:
        report = refresh_bm25_keyword_indexes(
            database_url,
            options=BM25IndexRefreshOptions(
                chunk_policy_names=tuple(args.chunk_policy_names or ()),
                tokenizer_name=args.tokenizer_name,
                continue_on_error=True,
            ),
        )
    except InvalidBM25KeywordIndexError as exc:
        parser.error(str(exc))

    json_text = json.dumps(
        bm25_index_refresh_report_payload(report),
        ensure_ascii=False,
        indent=2 if args.pretty else None,
    )
    if args.json_output:
        _write_text(Path(args.json_output), json_text + "\n")
    else:
        print(json_text)

    return 0 if args.allow_failures or report.failed_count == 0 else 1


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
