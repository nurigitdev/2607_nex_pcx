"""Foreground go-live evidence summary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SUMMARY_CHECK_PASSED = "passed"
SUMMARY_CHECK_WARNING = "warning"
SUMMARY_CHECK_FAILED = "failed"

SUMMARY_STATUS_READY = "ready"
SUMMARY_STATUS_WARNING = "warning"
SUMMARY_STATUS_BLOCKED = "blocked"

DEFAULT_REQUIRED_EVIDENCE = (
    (
        "foreground_operations",
        "artifacts/foreground_operations_validation.json",
        ("ready", "warning"),
    ),
    ("production_environment", "artifacts/production_environment_validation.json", ("ready",)),
    ("go_live_evidence", "artifacts/go_live_evidence.json", ("ready", "warning")),
    ("go_live_smoke", "artifacts/go_live_smoke.json", ("ready",)),
)
DEFAULT_OPTIONAL_HARDENING_EVIDENCE = (
    ("app_host_service_restart", "artifacts/app_host_service_restart_validation.json", ("ready",)),
)


@dataclass(frozen=True)
class ForegroundGoLiveEvidenceSpec:
    code: str
    path: str
    accepted_statuses: tuple[str, ...]
    required: bool = True


@dataclass(frozen=True)
class ForegroundGoLiveSummaryOptions:
    workdir: str | Path = "."
    required_evidence: tuple[ForegroundGoLiveEvidenceSpec, ...] = tuple(
        ForegroundGoLiveEvidenceSpec(code=code, path=path, accepted_statuses=statuses)
        for code, path, statuses in DEFAULT_REQUIRED_EVIDENCE
    )
    optional_hardening_evidence: tuple[ForegroundGoLiveEvidenceSpec, ...] = tuple(
        ForegroundGoLiveEvidenceSpec(
            code=code,
            path=path,
            accepted_statuses=statuses,
            required=False,
        )
        for code, path, statuses in DEFAULT_OPTIONAL_HARDENING_EVIDENCE
    )


@dataclass(frozen=True)
class ForegroundGoLiveSummaryCheck:
    code: str
    status: str
    detail: str
    path: str
    required: bool
    evidence_status: str | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class ForegroundGoLiveSummary:
    status: str
    checked_at: datetime
    workdir: str
    checks: tuple[ForegroundGoLiveSummaryCheck, ...]

    @property
    def check_count(self) -> int:
        return len(self.checks)

    @property
    def passed_count(self) -> int:
        return self._count_status(SUMMARY_CHECK_PASSED)

    @property
    def warning_count(self) -> int:
        return self._count_status(SUMMARY_CHECK_WARNING)

    @property
    def failed_count(self) -> int:
        return self._count_status(SUMMARY_CHECK_FAILED)

    def _count_status(self, status: str) -> int:
        return sum(1 for check in self.checks if check.status == status)


def build_foreground_go_live_summary(
    options: ForegroundGoLiveSummaryOptions | None = None,
    *,
    checked_at: datetime | None = None,
) -> ForegroundGoLiveSummary:
    selected_options = _validate_options(options or ForegroundGoLiveSummaryOptions())
    root = Path(selected_options.workdir)
    checks = tuple(
        _evaluate_evidence(root=root, spec=spec)
        for spec in (
            *selected_options.required_evidence,
            *selected_options.optional_hardening_evidence,
        )
    )
    return ForegroundGoLiveSummary(
        status=_overall_status(checks),
        checked_at=checked_at or datetime.now(UTC),
        workdir=str(root),
        checks=checks,
    )


def foreground_go_live_summary_payload(
    summary: ForegroundGoLiveSummary,
) -> dict[str, object]:
    return {
        "status": summary.status,
        "checked_at": summary.checked_at.isoformat(),
        "checked_at_label": summary.checked_at.strftime("%Y-%m-%d %H:%M:%S"),
        "workdir": summary.workdir,
        "check_count": summary.check_count,
        "passed_count": summary.passed_count,
        "warning_count": summary.warning_count,
        "failed_count": summary.failed_count,
        "checks": [
            {
                "code": check.code,
                "status": check.status,
                "detail": check.detail,
                "path": check.path,
                "required": check.required,
                "evidence_status": check.evidence_status,
                "metadata": dict(check.metadata or {}),
            }
            for check in summary.checks
        ],
    }


def render_foreground_go_live_summary_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Foreground Go-Live Evidence Summary",
        "",
        f"- Checked At: {_text(payload.get('checked_at_label'))}",
        f"- Status: `{_text(payload.get('status'))}`",
        f"- Workdir: `{_text(payload.get('workdir'))}`",
        f"- Checks: {_text(payload.get('check_count'))}",
        f"- Passed: {_text(payload.get('passed_count'))}",
        f"- Warning: {_text(payload.get('warning_count'))}",
        f"- Failed: {_text(payload.get('failed_count'))}",
        "",
        "## Evidence Checks",
        "",
        "| Code | Required | Evidence Status | Result | Detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for check in payload.get("checks", []):
        check_payload = _dict(check)
        lines.append(
            "| "
            f"{_md_cell(check_payload.get('code'))} | "
            f"{_md_cell(check_payload.get('required'))} | "
            f"{_md_cell(check_payload.get('evidence_status'))} | "
            f"{_md_cell(check_payload.get('status'))} | "
            f"{_md_cell(check_payload.get('detail'))} |"
        )
    lines.extend(
        [
            "",
            "## Operator Notes",
            "",
            "- `warning` is acceptable for foreground mode when all required evidence passes.",
            "- Optional hardening evidence can remain warning/blocked while service "
            "registration is deferred.",
            "- Treat `blocked` as a stop signal before supervised go-live use.",
        ]
    )
    return "\n".join(lines) + "\n"


def payload_to_json(payload: dict[str, object], *, pretty: bool = False) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)


def _validate_options(options: ForegroundGoLiveSummaryOptions) -> ForegroundGoLiveSummaryOptions:
    required = tuple(options.required_evidence)
    optional = tuple(options.optional_hardening_evidence)
    if not required:
        raise ValueError("at least one required evidence spec is required")
    for spec in (*required, *optional):
        _validate_spec(spec)
    return ForegroundGoLiveSummaryOptions(
        workdir=Path(options.workdir),
        required_evidence=required,
        optional_hardening_evidence=optional,
    )


def _validate_spec(spec: ForegroundGoLiveEvidenceSpec) -> None:
    if not spec.code.strip():
        raise ValueError("evidence code is required")
    if not spec.path.strip():
        raise ValueError("evidence path is required")
    if not spec.accepted_statuses:
        raise ValueError("at least one accepted evidence status is required")


def _evaluate_evidence(
    *,
    root: Path,
    spec: ForegroundGoLiveEvidenceSpec,
) -> ForegroundGoLiveSummaryCheck:
    evidence_path = root / spec.path
    if not evidence_path.exists():
        status = SUMMARY_CHECK_FAILED if spec.required else SUMMARY_CHECK_WARNING
        return ForegroundGoLiveSummaryCheck(
            code=spec.code,
            status=status,
            detail=f"Evidence file is missing: {spec.path}",
            path=spec.path,
            required=spec.required,
        )
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except Exception as exc:
        status = SUMMARY_CHECK_FAILED if spec.required else SUMMARY_CHECK_WARNING
        return ForegroundGoLiveSummaryCheck(
            code=spec.code,
            status=status,
            detail=f"Evidence file could not be parsed: {exc}",
            path=spec.path,
            required=spec.required,
            metadata={"error": str(exc)},
        )
    evidence_status = _text(_dict(payload).get("status"))
    metadata = _evidence_metadata(payload)
    if evidence_status not in spec.accepted_statuses:
        status = SUMMARY_CHECK_FAILED if spec.required else SUMMARY_CHECK_WARNING
        return ForegroundGoLiveSummaryCheck(
            code=spec.code,
            status=status,
            detail=(
                f"Evidence status is {evidence_status!r}; expected one of "
                f"{', '.join(spec.accepted_statuses)}."
            ),
            path=spec.path,
            required=spec.required,
            evidence_status=evidence_status,
            metadata=metadata,
        )
    return ForegroundGoLiveSummaryCheck(
        code=spec.code,
        status=SUMMARY_CHECK_PASSED,
        detail=f"Evidence status {evidence_status!r} is accepted.",
        path=spec.path,
        required=spec.required,
        evidence_status=evidence_status,
        metadata=metadata,
    )


def _evidence_metadata(payload: object) -> dict[str, object]:
    payload_dict = _dict(payload)
    metadata: dict[str, object] = {}
    for key in (
        "checked_at_label",
        "generated_at_label",
        "passed_count",
        "warning_count",
        "failed_count",
        "missing_required_count",
    ):
        if key in payload_dict:
            metadata[key] = payload_dict[key]
    return metadata


def _overall_status(checks: tuple[ForegroundGoLiveSummaryCheck, ...]) -> str:
    if any(check.status == SUMMARY_CHECK_FAILED for check in checks):
        return SUMMARY_STATUS_BLOCKED
    if any(check.status == SUMMARY_CHECK_WARNING for check in checks):
        return SUMMARY_STATUS_WARNING
    return SUMMARY_STATUS_READY


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _md_cell(value: object) -> str:
    return _text(value).replace("\n", " ").replace("|", "\\|")
