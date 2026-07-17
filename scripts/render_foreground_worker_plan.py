"""Render foreground worker command plan."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.foreground_worker_plan import (  # noqa: E402
    DEFAULT_EMBEDDING_LIMIT,
    DEFAULT_EMBEDDING_WORKER_NAME,
    DEFAULT_LEASE_SECONDS,
    DEFAULT_PIPELINE_WORKER_NAME,
    build_foreground_worker_plan,
    foreground_worker_plan_payload,
    payload_to_json,
    render_foreground_worker_plan_markdown,
)
from app.core.service_startup_templates import DEFAULT_CHUNK_POLICY_NAMES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render NeX-PCX foreground worker command plan.",
    )
    parser.add_argument("--workdir", default=str(PROJECT_ROOT))
    parser.add_argument("--python-bin", default="./.venv/bin/python")
    parser.add_argument("--database-url-source", default="${NEX_PCX_DATABASE_URL}")
    parser.add_argument("--pipeline-worker-name", default=DEFAULT_PIPELINE_WORKER_NAME)
    parser.add_argument("--embedding-worker-name", default=DEFAULT_EMBEDDING_WORKER_NAME)
    parser.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    parser.add_argument("--embedding-limit", type=int, default=DEFAULT_EMBEDDING_LIMIT)
    parser.add_argument(
        "--chunk-policy-names",
        nargs="+",
        default=list(DEFAULT_CHUNK_POLICY_NAMES),
    )
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    plan = build_foreground_worker_plan(
        workdir=args.workdir,
        python_bin=args.python_bin,
        database_url_source=args.database_url_source,
        pipeline_worker_name=args.pipeline_worker_name,
        embedding_worker_name=args.embedding_worker_name,
        lease_seconds=args.lease_seconds,
        embedding_limit=args.embedding_limit,
        chunk_policy_names=tuple(args.chunk_policy_names),
    )
    payload = foreground_worker_plan_payload(plan)
    json_text = payload_to_json(payload, pretty=args.pretty)
    markdown_text = render_foreground_worker_plan_markdown(payload)
    if args.json_output:
        _write_text(Path(args.json_output), json_text + "\n")
    else:
        print(json_text)
    if args.markdown_output:
        _write_text(Path(args.markdown_output), markdown_text)
    return 0


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
