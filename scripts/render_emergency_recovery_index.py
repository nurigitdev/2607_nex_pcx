"""Render the NeX_PCX emergency recovery command index."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.emergency_recovery_index import (  # noqa: E402
    build_emergency_recovery_index,
    emergency_recovery_index_payload,
    payload_to_json,
    render_emergency_recovery_index_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render NeX_PCX emergency recovery commands and checklists.",
    )
    parser.add_argument("--workdir", default=str(PROJECT_ROOT))
    parser.add_argument("--app-url", default="http://127.0.0.1:8000")
    parser.add_argument("--provider-host", default="192.168.20.243")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument(
        "--print-markdown",
        action="store_true",
        help="Print Markdown instead of JSON when no output path is supplied.",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    index = build_emergency_recovery_index(
        workdir=Path(args.workdir),
        app_url=args.app_url,
        provider_host=args.provider_host,
        artifacts_dir=Path(args.artifacts_dir),
    )
    payload = emergency_recovery_index_payload(index)
    json_text = payload_to_json(payload, pretty=args.pretty)
    markdown_text = render_emergency_recovery_index_markdown(payload)

    if args.json_output:
        _write_text(Path(args.json_output), json_text + "\n")
    if args.markdown_output:
        _write_text(Path(args.markdown_output), markdown_text)
    if not args.json_output and not args.markdown_output:
        print(markdown_text if args.print_markdown else json_text)

    return 0


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
