"""Export a NeX_PCX release version snapshot and tag command guide."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.release_version_snapshot import (  # noqa: E402
    RELEASE_STATUS_WARNING,
    build_release_version_snapshot,
    payload_to_json,
    release_version_snapshot_payload,
    render_release_version_snapshot_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export NeX_PCX release version metadata and tag commands.",
    )
    parser.add_argument("--workdir", default=str(PROJECT_ROOT))
    parser.add_argument("--release-version", default=None)
    parser.add_argument("--tag-prefix", default="nex-pcx-v")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument(
        "--allow-warning",
        action="store_true",
        help="Return zero even when the snapshot status is warning.",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    snapshot = build_release_version_snapshot(
        workdir=Path(args.workdir),
        release_version=args.release_version,
        tag_prefix=args.tag_prefix,
    )
    payload = release_version_snapshot_payload(snapshot)
    json_text = payload_to_json(payload, pretty=args.pretty)
    markdown_text = render_release_version_snapshot_markdown(payload)

    if args.json_output:
        _write_text(Path(args.json_output), json_text + "\n")
    if args.markdown_output:
        _write_text(Path(args.markdown_output), markdown_text)
    if not args.json_output and not args.markdown_output:
        print(json_text)

    if snapshot.status == RELEASE_STATUS_WARNING and not args.allow_warning:
        return 1
    return 0


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
