"""Export a NeX_PCX operator handoff bundle."""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.operator_handoff_bundle import (  # noqa: E402
    export_operator_handoff_bundle,
    operator_handoff_bundle_payload,
    payload_to_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export NeX_PCX go-live evidence for operator handoff.",
    )
    parser.add_argument("--workdir", default=str(PROJECT_ROOT))
    parser.add_argument("--output-dir", default="artifacts/operator_handoff/latest")
    parser.add_argument("--app-url", default="http://127.0.0.1:8000")
    parser.add_argument("--provider-host", default="192.168.20.243")
    parser.add_argument("--git-commit", default=None)
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Write manifest and Markdown only without copying evidence files.",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    git_commit = args.git_commit or _git_commit(workdir)
    bundle = export_operator_handoff_bundle(
        workdir=workdir,
        bundle_dir=Path(args.output_dir),
        git_commit=git_commit,
        app_url=args.app_url,
        provider_host=args.provider_host,
        copy_files=not args.no_copy,
    )
    payload = operator_handoff_bundle_payload(bundle)
    print(payload_to_json(payload, pretty=args.pretty))
    return 1 if bundle.missing_required_count else 0


def _git_commit(workdir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


if __name__ == "__main__":
    raise SystemExit(main())
