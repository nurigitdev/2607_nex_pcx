"""Smoke test upload queue auto-processing through foreground worker runtime."""

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_APP_URL = "http://127.0.0.1:8000"
DEFAULT_FILENAME = "slice300-auto-processing.md"
DEFAULT_CONTENT = (
    "# Slice 300 Auto Processing Smoke\n\n"
    "This small markdown document verifies that uploads enqueue pipeline work "
    "and that foreground worker runtime visibility is reachable.\n"
)
DEFAULT_JSON_OUTPUT = "artifacts/upload_queue_auto_processing_smoke.json"
DEFAULT_MARKDOWN_OUTPUT = "artifacts/upload_queue_auto_processing_smoke.md"

SMOKE_STATUS_PLANNED = "planned"
SMOKE_STATUS_READY = "ready"
SMOKE_STATUS_WARNING = "warning"
SMOKE_STATUS_BLOCKED = "blocked"
SMOKE_STATUS_FAILED = "failed"

TERMINAL_PIPELINE_STATUSES = {"succeeded", "failed", "cancelled"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify upload queue auto-processing visibility for foreground workers.",
    )
    parser.add_argument("--app-url", default=DEFAULT_APP_URL)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--content", default=DEFAULT_CONTENT)
    parser.add_argument("--poll-attempts", type=int, default=12)
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-output", default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", default=DEFAULT_MARKDOWN_OUTPUT)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    try:
        plan = _smoke_plan(
            app_url=args.app_url,
            filename=args.filename,
            poll_attempts=args.poll_attempts,
            poll_interval_seconds=args.poll_interval_seconds,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run:
        payload = _smoke_payload(
            status=SMOKE_STATUS_PLANNED,
            plan=plan,
            message="Dry-run smoke plan generated; no upload was submitted.",
        )
        _write_outputs(
            payload,
            json_output=args.json_output,
            markdown_output=args.markdown_output,
            pretty=args.pretty,
        )
        return 0

    started_at = datetime.now(UTC)
    try:
        runtime_before = _get_json(plan["runtime_url"])
        upload_payload = _post_upload(
            upload_url=plan["upload_url"],
            filename=args.filename,
            content=args.content.encode("utf-8"),
            content_type="text/markdown",
        )
        pipeline_job_id = _pipeline_job_id(upload_payload)
        poll_results = _poll_pipeline_job(
            job_url=f"{plan['pipeline_job_url_base']}/{pipeline_job_id}",
            attempts=args.poll_attempts,
            interval_seconds=args.poll_interval_seconds,
        )
        runtime_after = _get_json(plan["runtime_url"])
        status = _smoke_status(
            runtime_before=runtime_before,
            runtime_after=runtime_after,
            poll_results=poll_results,
        )
        payload = _smoke_payload(
            status=status,
            plan=plan,
            message=_smoke_message(status),
            started_at=started_at,
            completed_at=datetime.now(UTC),
            runtime_before=runtime_before,
            upload_response=upload_payload,
            pipeline_job_id=pipeline_job_id,
            poll_results=poll_results,
            runtime_after=runtime_after,
        )
    except (OSError, urllib.error.URLError, ValueError) as exc:
        payload = _smoke_payload(
            status=SMOKE_STATUS_FAILED,
            plan=plan,
            message=f"Upload queue auto-processing smoke failed: {exc}",
            started_at=started_at,
            completed_at=datetime.now(UTC),
            error=str(exc),
        )

    _write_outputs(
        payload,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
        pretty=args.pretty,
    )
    return 0 if payload["status"] in {SMOKE_STATUS_READY, SMOKE_STATUS_WARNING} else 1


def _smoke_plan(
    *,
    app_url: str,
    filename: str,
    poll_attempts: int,
    poll_interval_seconds: float,
) -> dict[str, object]:
    selected_app_url = app_url.strip().rstrip("/")
    if not selected_app_url:
        raise ValueError("app-url is required")
    selected_filename = filename.strip()
    if not selected_filename:
        raise ValueError("filename is required")
    if poll_attempts <= 0:
        raise ValueError("poll-attempts must be greater than zero")
    if poll_interval_seconds < 0:
        raise ValueError("poll-interval-seconds must be greater than or equal to zero")
    return {
        "app_url": selected_app_url,
        "upload_url": f"{selected_app_url}/api/files",
        "runtime_url": f"{selected_app_url}/api/admin/foreground-worker-runtime",
        "pipeline_job_url_base": f"{selected_app_url}/api/pipeline/jobs",
        "filename": selected_filename,
        "poll_attempts": poll_attempts,
        "poll_interval_seconds": poll_interval_seconds,
    }


def _post_upload(
    *,
    upload_url: str,
    filename: str,
    content: bytes,
    content_type: str,
) -> dict[str, object]:
    boundary = f"----nex-pcx-smoke-{uuid.uuid4().hex}"
    body = _multipart_body(
        boundary=boundary,
        filename=filename,
        content=content,
        content_type=content_type,
    )
    request = urllib.request.Request(
        upload_url,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )
    return _read_json_response(request)


def _multipart_body(
    *,
    boundary: str,
    filename: str,
    content: bytes,
    content_type: str,
) -> bytes:
    fields = [
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode(),
        content,
        f"\r\n--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="document_group"\r\n\r\ndefault\r\n',
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="security_level"\r\n\r\ninternal\r\n',
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="access_scope"\r\n\r\ncommon\r\n',
        f"--{boundary}--\r\n".encode(),
    ]
    return b"".join(fields)


def _get_json(url: str) -> dict[str, object]:
    return _read_json_response(urllib.request.Request(url, method="GET"))


def _read_json_response(request: urllib.request.Request) -> dict[str, object]:
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("HTTP response was not a JSON object")
    return payload


def _pipeline_job_id(upload_payload: dict[str, object]) -> int:
    value = upload_payload.get("pipeline_job_id")
    if value is None:
        raise ValueError("upload response did not include pipeline_job_id")
    return int(value)


def _poll_pipeline_job(
    *,
    job_url: str,
    attempts: int,
    interval_seconds: float,
) -> list[dict[str, object]]:
    results = []
    for attempt in range(1, attempts + 1):
        payload = _get_json(job_url)
        job = payload.get("job")
        if not isinstance(job, dict):
            raise ValueError("pipeline job response did not include job")
        results.append({"attempt": attempt, "job": job})
        if str(job.get("status")) in TERMINAL_PIPELINE_STATUSES:
            break
        if attempt < attempts and interval_seconds > 0:
            time.sleep(interval_seconds)
    return results


def _smoke_status(
    *,
    runtime_before: dict[str, object],
    runtime_after: dict[str, object],
    poll_results: list[dict[str, object]],
) -> str:
    after_status = _runtime_status(runtime_after)
    if after_status == "blocked":
        return SMOKE_STATUS_BLOCKED
    if not poll_results:
        return SMOKE_STATUS_FAILED
    latest_job = poll_results[-1]["job"]
    if not isinstance(latest_job, dict):
        return SMOKE_STATUS_FAILED
    latest_status = str(latest_job.get("status"))
    if latest_status == "succeeded":
        return SMOKE_STATUS_READY
    if latest_status in {"pending", "running", "failed", "cancelled"}:
        return SMOKE_STATUS_WARNING
    if _runtime_status(runtime_before) == "blocked":
        return SMOKE_STATUS_BLOCKED
    return SMOKE_STATUS_WARNING


def _runtime_status(payload: dict[str, object]) -> str:
    runtime = payload.get("foreground_worker_runtime")
    if isinstance(runtime, dict):
        return str(runtime.get("status") or "")
    return str(payload.get("status") or "")


def _smoke_message(status: str) -> str:
    if status == SMOKE_STATUS_READY:
        return "Upload was queued and pipeline processing reached succeeded."
    if status == SMOKE_STATUS_WARNING:
        return "Upload was queued, but processing did not reach succeeded before timeout."
    if status == SMOKE_STATUS_BLOCKED:
        return "Foreground worker runtime is blocked; inspect runtime visibility."
    return "Upload queue auto-processing smoke finished."


def _smoke_payload(
    *,
    status: str,
    plan: dict[str, object],
    message: str,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    runtime_before: dict[str, object] | None = None,
    upload_response: dict[str, object] | None = None,
    pipeline_job_id: int | None = None,
    poll_results: list[dict[str, object]] | None = None,
    runtime_after: dict[str, object] | None = None,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(),
        "message": message,
        "started_at": started_at.isoformat() if started_at else None,
        "completed_at": completed_at.isoformat() if completed_at else None,
        "plan": dict(plan),
        "runtime_before": runtime_before,
        "upload_response": upload_response,
        "pipeline_job_id": pipeline_job_id,
        "poll_results": list(poll_results or []),
        "runtime_after": runtime_after,
        "error": error,
    }


def _write_outputs(
    payload: dict[str, object],
    *,
    json_output: str | None,
    markdown_output: str | None,
    pretty: bool,
) -> None:
    if json_output:
        _write_text(
            Path(json_output),
            json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None) + "\n",
        )
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None))
    if markdown_output:
        _write_text(Path(markdown_output), _render_markdown(payload))


def _render_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Upload Queue Auto-Processing Smoke",
        "",
        f"- Status: `{payload.get('status')}`",
        f"- Message: {payload.get('message')}",
        f"- Pipeline Job ID: {payload.get('pipeline_job_id')}",
        "",
        "## Poll Results",
        "",
        "| Attempt | Job Status | Stage | Progress |",
        "| ---: | --- | --- | ---: |",
    ]
    for result in payload.get("poll_results") or []:
        job = result.get("job") if isinstance(result, dict) else {}
        job_payload = job if isinstance(job, dict) else {}
        lines.append(
            "| "
            f"{result.get('attempt') if isinstance(result, dict) else ''} | "
            f"{job_payload.get('status')} | "
            f"{job_payload.get('stage')} | "
            f"{job_payload.get('progress_percent')} |"
        )
    lines.extend(
        [
            "",
            "## Operator Notes",
            "",
            "- `ready` means the uploaded job reached pipeline `succeeded`.",
            "- `warning` means the upload queued but processing did not finish in time.",
            "- `blocked` means foreground worker runtime visibility reported a blocked state.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
