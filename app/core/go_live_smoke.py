"""HTTP go-live smoke checks for NeX_PCX."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

GO_LIVE_SMOKE_VERSION = 1

SMOKE_CHECK_PASSED = "passed"
SMOKE_CHECK_WARNING = "warning"
SMOKE_CHECK_FAILED = "failed"

SMOKE_STATUS_READY = "ready"
SMOKE_STATUS_WARNING = "warning"
SMOKE_STATUS_BLOCKED = "blocked"


@dataclass(frozen=True)
class SmokeEndpoint:
    code: str
    path: str
    description: str


@dataclass(frozen=True)
class HttpJsonResult:
    status_code: int | None
    payload: dict[str, object] | None = None
    error: str | None = None


@dataclass(frozen=True)
class GoLiveSmokeCheck:
    code: str
    status: str
    detail: str
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class GoLiveSmokeReport:
    status: str
    checked_at: datetime
    app_base_url: str
    checks: tuple[GoLiveSmokeCheck, ...]

    @property
    def check_count(self) -> int:
        return len(self.checks)

    @property
    def passed_count(self) -> int:
        return self._count_status(SMOKE_CHECK_PASSED)

    @property
    def warning_count(self) -> int:
        return self._count_status(SMOKE_CHECK_WARNING)

    @property
    def failed_count(self) -> int:
        return self._count_status(SMOKE_CHECK_FAILED)

    def _count_status(self, status: str) -> int:
        return sum(1 for check in self.checks if check.status == status)


class FetchJson(Protocol):
    def __call__(self, url: str, timeout_seconds: float) -> HttpJsonResult: ...


SMOKE_ENDPOINTS = (
    SmokeEndpoint(
        code="healthz",
        path="/healthz",
        description="Application process health.",
    ),
    SmokeEndpoint(
        code="go_live_readiness",
        path="/api/admin/go-live-readiness",
        description="Go-live readiness API.",
    ),
    SmokeEndpoint(
        code="pipeline_queue",
        path="/api/dashboard/pipeline-queue",
        description="Pipeline queue dashboard API.",
    ),
    SmokeEndpoint(
        code="embedding_backlog",
        path="/api/dashboard/embedding-backlog",
        description="Embedding backlog dashboard API.",
    ),
    SmokeEndpoint(
        code="provider_readiness",
        path="/api/admin/embedding-provider-routes/readiness",
        description="Provider route readiness API.",
    ),
    SmokeEndpoint(
        code="search_operations",
        path="/api/search/logs/operations-summary",
        description="Search operations summary API.",
    ),
)


def run_go_live_smoke(
    app_base_url: str,
    *,
    timeout_seconds: float = 5.0,
    checked_at: datetime | None = None,
    fetch_json: FetchJson | None = None,
) -> GoLiveSmokeReport:
    normalized_base_url = _normalize_base_url(app_base_url)
    selected_fetch_json = fetch_json or fetch_json_with_httpx
    checks = tuple(
        _endpoint_check(
            endpoint,
            base_url=normalized_base_url,
            timeout_seconds=timeout_seconds,
            fetch_json=selected_fetch_json,
        )
        for endpoint in SMOKE_ENDPOINTS
    )
    return GoLiveSmokeReport(
        status=_overall_status(checks),
        checked_at=checked_at or datetime.now(UTC),
        app_base_url=normalized_base_url,
        checks=checks,
    )


def go_live_smoke_report_payload(report: GoLiveSmokeReport) -> dict[str, object]:
    return {
        "version": GO_LIVE_SMOKE_VERSION,
        "status": report.status,
        "checked_at": report.checked_at.isoformat(),
        "checked_at_label": report.checked_at.strftime("%Y-%m-%d %H:%M:%S"),
        "app_base_url": report.app_base_url,
        "check_count": report.check_count,
        "passed_count": report.passed_count,
        "warning_count": report.warning_count,
        "failed_count": report.failed_count,
        "checks": [
            {
                "code": check.code,
                "status": check.status,
                "detail": check.detail,
                "metadata": dict(check.metadata or {}),
            }
            for check in report.checks
        ],
    }


def render_go_live_smoke_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# NeX_PCX End-to-End Go-Live Smoke Report",
        "",
        f"- Checked At: {_text(payload.get('checked_at_label'))}",
        f"- Overall Status: {_text(payload.get('status'))}",
        f"- App URL: {_text(payload.get('app_base_url'))}",
        f"- Passed: {_text(payload.get('passed_count'))}",
        f"- Warning: {_text(payload.get('warning_count'))}",
        f"- Failed: {_text(payload.get('failed_count'))}",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for check in payload.get("checks", []):
        check_payload = _dict(check)
        lines.append(
            "| "
            f"{_md_cell(check_payload.get('code'))} | "
            f"{_md_cell(check_payload.get('status'))} | "
            f"{_md_cell(check_payload.get('detail'))} |"
        )

    lines.extend(["", "## Metadata", ""])
    for check in payload.get("checks", []):
        check_payload = _dict(check)
        metadata = _dict(check_payload.get("metadata"))
        if not metadata:
            continue
        lines.extend(
            [
                f"### {_text(check_payload.get('code'))}",
                "",
                "```json",
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def fetch_json_with_httpx(url: str, timeout_seconds: float) -> HttpJsonResult:
    try:
        import httpx

        response = httpx.get(url, timeout=timeout_seconds)
        payload = response.json() if response.content else {}
        return HttpJsonResult(status_code=response.status_code, payload=payload)
    except Exception as exc:
        return HttpJsonResult(status_code=None, error=str(exc))


def _endpoint_check(
    endpoint: SmokeEndpoint,
    *,
    base_url: str,
    timeout_seconds: float,
    fetch_json: FetchJson,
) -> GoLiveSmokeCheck:
    url = f"{base_url}{endpoint.path}"
    result = fetch_json(url, timeout_seconds)
    metadata = {
        "url": url,
        "description": endpoint.description,
        "status_code": result.status_code,
        "payload_preview": _payload_preview(result.payload),
    }
    if result.error:
        return GoLiveSmokeCheck(
            code=endpoint.code,
            status=SMOKE_CHECK_FAILED,
            detail=f"{endpoint.path} request failed: {result.error}",
            metadata={**metadata, "error": result.error},
        )
    if result.status_code != 200:
        return GoLiveSmokeCheck(
            code=endpoint.code,
            status=SMOKE_CHECK_FAILED,
            detail=f"{endpoint.path} returned HTTP {result.status_code}.",
            metadata=metadata,
        )
    if endpoint.code == "healthz":
        return _healthz_check(endpoint, metadata, result.payload)
    if endpoint.code == "go_live_readiness":
        return _go_live_readiness_check(endpoint, metadata, result.payload)
    return GoLiveSmokeCheck(
        code=endpoint.code,
        status=SMOKE_CHECK_PASSED,
        detail=f"{endpoint.path} returned HTTP 200.",
        metadata=metadata,
    )


def _healthz_check(
    endpoint: SmokeEndpoint,
    metadata: dict[str, object],
    payload: dict[str, object] | None,
) -> GoLiveSmokeCheck:
    if _dict(payload).get("status") == "ok":
        return GoLiveSmokeCheck(
            code=endpoint.code,
            status=SMOKE_CHECK_PASSED,
            detail="/healthz returned status ok.",
            metadata=metadata,
        )
    return GoLiveSmokeCheck(
        code=endpoint.code,
        status=SMOKE_CHECK_FAILED,
        detail="/healthz did not return status ok.",
        metadata=metadata,
    )


def _go_live_readiness_check(
    endpoint: SmokeEndpoint,
    metadata: dict[str, object],
    payload: dict[str, object] | None,
) -> GoLiveSmokeCheck:
    readiness = _dict(_dict(payload).get("go_live_readiness"))
    readiness_status = readiness.get("status")
    metadata = {**metadata, "readiness_status": readiness_status}
    if readiness_status == "ready":
        return GoLiveSmokeCheck(
            code=endpoint.code,
            status=SMOKE_CHECK_PASSED,
            detail="Go-live readiness API reports ready.",
            metadata=metadata,
        )
    if readiness_status == "warning":
        return GoLiveSmokeCheck(
            code=endpoint.code,
            status=SMOKE_CHECK_WARNING,
            detail="Go-live readiness API reports warning.",
            metadata=metadata,
        )
    return GoLiveSmokeCheck(
        code=endpoint.code,
        status=SMOKE_CHECK_FAILED,
        detail=f"Go-live readiness API reports {readiness_status or 'unknown'}.",
        metadata=metadata,
    )


def _overall_status(checks: tuple[GoLiveSmokeCheck, ...]) -> str:
    if any(check.status == SMOKE_CHECK_FAILED for check in checks):
        return SMOKE_STATUS_BLOCKED
    if any(check.status == SMOKE_CHECK_WARNING for check in checks):
        return SMOKE_STATUS_WARNING
    return SMOKE_STATUS_READY


def _normalize_base_url(app_base_url: str) -> str:
    normalized = app_base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("app_base_url is required")
    if not normalized.startswith(("http://", "https://")):
        raise ValueError("app_base_url must be an absolute http(s) URL")
    return normalized


def _payload_preview(payload: dict[str, object] | None) -> dict[str, object]:
    if not payload:
        return {}
    return {key: payload[key] for key in list(payload)[:8]}


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str:
    if value is None:
        return "-"
    return str(value)


def _md_cell(value: object) -> str:
    return _text(value).replace("\n", " ").replace("|", "\\|")
