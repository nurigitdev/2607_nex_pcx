from datetime import UTC, datetime

import pytest

from app.core.go_live_smoke import (
    SMOKE_STATUS_BLOCKED,
    SMOKE_STATUS_READY,
    SMOKE_STATUS_WARNING,
    HttpJsonResult,
    go_live_smoke_report_payload,
    render_go_live_smoke_markdown,
    run_go_live_smoke,
)


def _ready_fetch(url: str, timeout_seconds: float) -> HttpJsonResult:
    if url.endswith("/healthz"):
        return HttpJsonResult(status_code=200, payload={"status": "ok"})
    if url.endswith("/api/admin/go-live-readiness"):
        return HttpJsonResult(
            status_code=200,
            payload={"go_live_readiness": {"status": "ready"}},
        )
    return HttpJsonResult(status_code=200, payload={"ok": True})


def test_go_live_smoke_ready_when_all_endpoints_pass() -> None:
    report = run_go_live_smoke(
        "http://127.0.0.1:8000/",
        checked_at=datetime(2026, 7, 17, 9, 10, 11, tzinfo=UTC),
        fetch_json=_ready_fetch,
    )
    payload = go_live_smoke_report_payload(report)
    markdown = render_go_live_smoke_markdown(payload)

    assert report.status == SMOKE_STATUS_READY
    assert report.passed_count == 6
    assert payload["checked_at_label"] == "2026-07-17 09:10:11"
    assert payload["app_base_url"] == "http://127.0.0.1:8000"
    assert "End-to-End Go-Live Smoke" in markdown


def test_go_live_smoke_warns_when_readiness_reports_warning() -> None:
    def fetch(url: str, timeout_seconds: float) -> HttpJsonResult:
        if url.endswith("/api/admin/go-live-readiness"):
            return HttpJsonResult(
                status_code=200,
                payload={"go_live_readiness": {"status": "warning"}},
            )
        return _ready_fetch(url, timeout_seconds)

    report = run_go_live_smoke("http://app", fetch_json=fetch)

    assert report.status == SMOKE_STATUS_WARNING
    assert report.warning_count == 1


def test_go_live_smoke_blocks_failed_health_and_readiness() -> None:
    def fetch(url: str, timeout_seconds: float) -> HttpJsonResult:
        if url.endswith("/healthz"):
            return HttpJsonResult(status_code=200, payload={"status": "warming"})
        if url.endswith("/api/admin/go-live-readiness"):
            return HttpJsonResult(
                status_code=200,
                payload={"go_live_readiness": {"status": "blocked"}},
            )
        return _ready_fetch(url, timeout_seconds)

    report = run_go_live_smoke("http://app", fetch_json=fetch)
    details = " ".join(check.detail for check in report.checks)

    assert report.status == SMOKE_STATUS_BLOCKED
    assert report.failed_count == 2
    assert "did not return status ok" in details
    assert "reports blocked" in details


def test_go_live_smoke_blocks_http_errors_and_non_200() -> None:
    def fetch(url: str, timeout_seconds: float) -> HttpJsonResult:
        if url.endswith("/healthz"):
            return HttpJsonResult(status_code=None, error="connection refused")
        return HttpJsonResult(status_code=503, payload={"detail": "database missing"})

    report = run_go_live_smoke("http://app", fetch_json=fetch)

    assert report.status == SMOKE_STATUS_BLOCKED
    assert report.failed_count == 6
    assert "connection refused" in report.checks[0].detail


@pytest.mark.parametrize("url", ["", "127.0.0.1:8000"])
def test_go_live_smoke_requires_absolute_app_url(url: str) -> None:
    with pytest.raises(ValueError):
        run_go_live_smoke(url, fetch_json=_ready_fetch)
