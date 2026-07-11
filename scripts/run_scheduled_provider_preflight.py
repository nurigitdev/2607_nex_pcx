"""Run due provider route preflight schedules from the command line."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.embedding_provider_preflight_schedules import (  # noqa: E402
    InvalidEmbeddingProviderPreflightScheduleError,
    run_due_embedding_provider_preflight_schedules,
)


def run_scheduled_preflight(
    database_url: str,
    *,
    limit: int = 20,
    schedule_name: str | None = None,
) -> dict[str, object]:
    runs = run_due_embedding_provider_preflight_schedules(
        database_url,
        limit=limit,
        schedule_name=schedule_name,
    )
    failed_count = sum(1 for run in runs if run.status != "succeeded")
    return {
        "run_count": len(runs),
        "failed_count": failed_count,
        "results": [
            {
                "schedule_name": run.schedule.schedule_name,
                "run_id": run.run_record.run_id,
                "status": run.status,
                "profile_name": run.schedule.profile_name,
                "active_only": run.schedule.active_only,
                "next_run_at": (
                    run.updated_schedule.next_run_at.isoformat()
                    if run.updated_schedule.next_run_at is not None
                    else None
                ),
                "result": run.result,
            }
            for run in runs
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run due scheduled provider route preflight checks.",
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--schedule-name", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return exit code 0 even when one or more scheduled preflight runs fail.",
    )
    args = parser.parse_args()

    settings = get_settings()
    database_url = args.database_url or settings.database_url
    if not database_url:
        parser.error("--database-url or NEX_PCX_DATABASE_URL is required")

    try:
        payload = run_scheduled_preflight(
            database_url,
            limit=args.limit,
            schedule_name=args.schedule_name,
        )
    except InvalidEmbeddingProviderPreflightScheduleError as exc:
        parser.error(str(exc))

    print(json.dumps(payload, ensure_ascii=False))
    return 0 if args.allow_failures or payload["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
