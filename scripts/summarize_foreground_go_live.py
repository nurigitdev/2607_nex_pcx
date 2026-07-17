"""Summarize foreground go-live evidence files."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.foreground_go_live_summary import (  # noqa: E402
    ForegroundGoLiveSummaryOptions,
    build_foreground_go_live_summary,
    foreground_go_live_summary_payload,
    payload_to_json,
    render_foreground_go_live_summary_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize NeX-PCX foreground go-live evidence.",
    )
    parser.add_argument("--workdir", default=str(PROJECT_ROOT))
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    summary = build_foreground_go_live_summary(ForegroundGoLiveSummaryOptions(workdir=args.workdir))
    payload = foreground_go_live_summary_payload(summary)
    json_text = payload_to_json(payload, pretty=args.pretty)
    markdown_text = render_foreground_go_live_summary_markdown(payload)
    if args.json_output:
        _write_text(Path(args.json_output), json_text + "\n")
    else:
        print(json_text)
    if args.markdown_output:
        _write_text(Path(args.markdown_output), markdown_text)
    return 0 if summary.status in {"ready", "warning"} else 1


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
