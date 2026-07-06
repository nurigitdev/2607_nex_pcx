"""Process one queued Markdown pipeline job."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.chunks import DEFAULT_CHUNK_POLICY_NAME  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.pipeline_jobs import DEFAULT_LEASE_SECONDS  # noqa: E402
from app.core.pipeline_worker import process_next_markdown_pipeline_job  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Process one queued Markdown pipeline job.")
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--worker-name", default="markdown-worker")
    parser.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    parser.add_argument("--chunk-policy-name", default=DEFAULT_CHUNK_POLICY_NAME)
    args = parser.parse_args()

    database_url = args.database_url or get_settings().database_url
    if not database_url:
        parser.error("--database-url or NEX_PCX_DATABASE_URL is required")

    result = process_next_markdown_pipeline_job(
        database_url,
        worker_name=args.worker_name,
        lease_seconds=args.lease_seconds,
        chunk_policy_name=args.chunk_policy_name,
    )
    payload = {
        "processed": result.processed,
        "job_id": result.job.job_id if result.job else None,
        "status": result.job.status if result.job else None,
        "stage": result.job.stage if result.job else None,
        "chunk_count": result.chunk_count,
        "message": result.message,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
