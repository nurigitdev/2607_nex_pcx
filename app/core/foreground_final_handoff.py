"""Foreground final handoff checklist for supervised go-live."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

FINAL_HANDOFF_CHECK_PASSED = "passed"
FINAL_HANDOFF_CHECK_WARNING = "warning"
FINAL_HANDOFF_CHECK_FAILED = "failed"

FINAL_HANDOFF_STATUS_READY = "ready"
FINAL_HANDOFF_STATUS_WARNING = "warning"
FINAL_HANDOFF_STATUS_BLOCKED = "blocked"

DEFAULT_FINAL_HANDOFF_EVIDENCE = (
    (
        "foreground_production_launch",
        "artifacts/foreground_production_launch.json",
        ("running",),
        ("planned", "exited"),
    ),
    (
        "foreground_production_shutdown",
        "artifacts/foreground_production_shutdown.json",
        ("stopped",),
        ("planned", "no_process"),
    ),
    (
        "foreground_operations_validation",
        "artifacts/foreground_operations_validation.json",
        ("ready",),
        ("warning",),
    ),
    (
        "foreground_go_live_summary",
        "artifacts/foreground_go_live_summary.json",
        ("ready",),
        ("warning",),
    ),
)

DEFAULT_FOREGROUND_WORKER_PLAN_PATH = "artifacts/foreground_worker_plan.json"
DEFAULT_OPERATOR_HANDOFF_MANIFEST_PATH = "artifacts/operator_handoff/latest/manifest.json"
REQUIRED_WORKER_COMMAND_CODES = (
    "pipeline_worker_help",
    "embedding_worker_help",
    "pipeline_worker_once",
    "embedding_worker_batch",
)


@dataclass(frozen=True)
class ForegroundFinalEvidenceSpec:
    code: str
    path: str
    ready_statuses: tuple[str, ...] = ("ready",)
    warning_statuses: tuple[str, ...] = ()


@dataclass(frozen=True)
class ForegroundFinalHandoffOptions:
    workdir: str | Path = "."
    evidence_specs: tuple[ForegroundFinalEvidenceSpec, ...] = tuple(
        ForegroundFinalEvidenceSpec(
            code=code,
            path=path,
            ready_statuses=ready_statuses,
            warning_statuses=warning_statuses,
        )
        for code, path, ready_statuses, warning_statuses in DEFAULT_FINAL_HANDOFF_EVIDENCE
    )
    worker_plan_path: str = DEFAULT_FOREGROUND_WORKER_PLAN_PATH
    handoff_manifest_path: str = DEFAULT_OPERATOR_HANDOFF_MANIFEST_PATH


@dataclass(frozen=True)
class ForegroundFinalHandoffCheck:
    code: str
    status: str
    detail: str
    path: str
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class ForegroundFinalHandoffReport:
    status: str
    generated_at: datetime
    workdir: str
    checks: tuple[ForegroundFinalHandoffCheck, ...]

    @property
    def check_count(self) -> int:
        return len(self.checks)

    @property
    def passed_count(self) -> int:
        return self._count_status(FINAL_HANDOFF_CHECK_PASSED)

    @property
    def warning_count(self) -> int:
        return self._count_status(FINAL_HANDOFF_CHECK_WARNING)

    @property
    def failed_count(self) -> int:
        return self._count_status(FINAL_HANDOFF_CHECK_FAILED)

    def _count_status(self, status: str) -> int:
        return sum(1 for check in self.checks if check.status == status)


def build_foreground_final_handoff_report(
    options: ForegroundFinalHandoffOptions | None = None,
    *,
    generated_at: datetime | None = None,
) -> ForegroundFinalHandoffReport:
    selected_options = _validate_options(options or ForegroundFinalHandoffOptions())
    root = Path(selected_options.workdir)
    checks = (
        *(
            _evaluate_status_evidence(root=root, spec=spec)
            for spec in selected_options.evidence_specs
        ),
        _evaluate_worker_plan(root=root, path=selected_options.worker_plan_path),
        _evaluate_handoff_manifest(root=root, path=selected_options.handoff_manifest_path),
    )
    return ForegroundFinalHandoffReport(
        status=_overall_status(checks),
        generated_at=generated_at or datetime.now(UTC),
        workdir=str(root),
        checks=checks,
    )


def foreground_final_handoff_payload(
    report: ForegroundFinalHandoffReport,
) -> dict[str, object]:
    return {
        "status": report.status,
        "generated_at": report.generated_at.isoformat(),
        "generated_at_label": report.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        "workdir": report.workdir,
        "check_count": report.check_count,
        "passed_count": report.passed_count,
        "warning_count": report.warning_count,
        "failed_count": report.failed_count,
        "checks": [
            {
                "code": check.code,
                "status": check.status,
                "detail": check.detail,
                "path": check.path,
                "metadata": dict(check.metadata or {}),
            }
            for check in report.checks
        ],
    }


def render_foreground_final_handoff_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Foreground Final Handoff Checklist",
        "",
        f"- Generated At: {_text(payload.get('generated_at_label'))}",
        f"- Status: `{_text(payload.get('status'))}`",
        f"- Workdir: `{_text(payload.get('workdir'))}`",
        f"- Checks: {_text(payload.get('check_count'))}",
        f"- Passed: {_text(payload.get('passed_count'))}",
        f"- Warning: {_text(payload.get('warning_count'))}",
        f"- Failed: {_text(payload.get('failed_count'))}",
        "",
        "## Checks",
        "",
        "| Code | Status | Detail | Path |",
        "| --- | --- | --- | --- |",
    ]
    for check in payload.get("checks", []):
        check_payload = _dict(check)
        lines.append(
            "| "
            f"{_md_cell(check_payload.get('code'))} | "
            f"{_md_cell(check_payload.get('status'))} | "
            f"{_md_cell(check_payload.get('detail'))} | "
            f"`{_md_cell(check_payload.get('path'))}` |"
        )
    lines.extend(
        [
            "",
            "## Operator Notes",
            "",
            "- `warning` is acceptable for supervised foreground operation when no "
            "required check fails.",
            "- Keep the web process and worker terminals supervised during operation.",
            "- Treat `blocked` as a stop signal before operator handoff.",
        ]
    )
    return "\n".join(lines) + "\n"


def payload_to_json(payload: dict[str, object], *, pretty: bool = False) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)


def _validate_options(
    options: ForegroundFinalHandoffOptions,
) -> ForegroundFinalHandoffOptions:
    specs = tuple(options.evidence_specs)
    if not specs:
        raise ValueError("at least one evidence spec is required")
    for spec in specs:
        _validate_spec(spec)
    worker_plan_path = options.worker_plan_path.strip()
    if not worker_plan_path:
        raise ValueError("worker_plan_path is required")
    handoff_manifest_path = options.handoff_manifest_path.strip()
    if not handoff_manifest_path:
        raise ValueError("handoff_manifest_path is required")
    return ForegroundFinalHandoffOptions(
        workdir=Path(options.workdir),
        evidence_specs=specs,
        worker_plan_path=worker_plan_path,
        handoff_manifest_path=handoff_manifest_path,
    )


def _validate_spec(spec: ForegroundFinalEvidenceSpec) -> None:
    if not spec.code.strip():
        raise ValueError("evidence code is required")
    if not spec.path.strip():
        raise ValueError("evidence path is required")
    if not spec.ready_statuses and not spec.warning_statuses:
        raise ValueError("at least one accepted status is required")


def _evaluate_status_evidence(
    *,
    root: Path,
    spec: ForegroundFinalEvidenceSpec,
) -> ForegroundFinalHandoffCheck:
    payload_result = _read_json(root / spec.path)
    if payload_result.error:
        return ForegroundFinalHandoffCheck(
            code=spec.code,
            status=FINAL_HANDOFF_CHECK_FAILED,
            detail=payload_result.error,
            path=spec.path,
        )
    payload = _dict(payload_result.payload)
    evidence_status = _text(payload.get("status"))
    metadata = _metadata_from_payload(payload)
    if evidence_status in spec.ready_statuses:
        return ForegroundFinalHandoffCheck(
            code=spec.code,
            status=FINAL_HANDOFF_CHECK_PASSED,
            detail=f"Evidence status {evidence_status!r} is ready.",
            path=spec.path,
            metadata=metadata,
        )
    if evidence_status in spec.warning_statuses:
        return ForegroundFinalHandoffCheck(
            code=spec.code,
            status=FINAL_HANDOFF_CHECK_WARNING,
            detail=f"Evidence status {evidence_status!r} is accepted with operator awareness.",
            path=spec.path,
            metadata=metadata,
        )
    return ForegroundFinalHandoffCheck(
        code=spec.code,
        status=FINAL_HANDOFF_CHECK_FAILED,
        detail=(
            f"Evidence status is {evidence_status!r}; expected ready statuses "
            f"{', '.join(spec.ready_statuses)} or warning statuses "
            f"{', '.join(spec.warning_statuses)}."
        ),
        path=spec.path,
        metadata=metadata,
    )


def _evaluate_worker_plan(*, root: Path, path: str) -> ForegroundFinalHandoffCheck:
    payload_result = _read_json(root / path)
    if payload_result.error:
        return ForegroundFinalHandoffCheck(
            code="foreground_worker_plan",
            status=FINAL_HANDOFF_CHECK_FAILED,
            detail=payload_result.error,
            path=path,
        )
    payload = _dict(payload_result.payload)
    commands = tuple(_dict(command) for command in _list(payload.get("commands")))
    command_codes = {str(command.get("code")) for command in commands}
    missing_codes = tuple(
        code for code in REQUIRED_WORKER_COMMAND_CODES if code not in command_codes
    )
    unbounded_codes = tuple(
        str(command.get("code")) for command in commands if command.get("bounded") is not True
    )
    metadata = {
        "command_count": len(commands),
        "lease_seconds": payload.get("lease_seconds"),
        "embedding_limit": payload.get("embedding_limit"),
    }
    if missing_codes:
        return ForegroundFinalHandoffCheck(
            code="foreground_worker_plan",
            status=FINAL_HANDOFF_CHECK_FAILED,
            detail=f"Worker plan is missing required command codes: {', '.join(missing_codes)}.",
            path=path,
            metadata=metadata,
        )
    if unbounded_codes:
        return ForegroundFinalHandoffCheck(
            code="foreground_worker_plan",
            status=FINAL_HANDOFF_CHECK_FAILED,
            detail=f"Worker plan contains unbounded commands: {', '.join(unbounded_codes)}.",
            path=path,
            metadata=metadata,
        )
    return ForegroundFinalHandoffCheck(
        code="foreground_worker_plan",
        status=FINAL_HANDOFF_CHECK_PASSED,
        detail="Worker plan includes bounded pipeline and embedding commands.",
        path=path,
        metadata=metadata,
    )


def _evaluate_handoff_manifest(*, root: Path, path: str) -> ForegroundFinalHandoffCheck:
    payload_result = _read_json(root / path)
    if payload_result.error:
        return ForegroundFinalHandoffCheck(
            code="operator_handoff_bundle",
            status=FINAL_HANDOFF_CHECK_FAILED,
            detail=payload_result.error,
            path=path,
        )
    payload = _dict(payload_result.payload)
    file_count = _int(payload.get("file_count"))
    included_count = _int(payload.get("included_count"))
    missing_required_count = _int(payload.get("missing_required_count"))
    metadata = {
        "file_count": file_count,
        "included_count": included_count,
        "missing_required_count": missing_required_count,
        "git_commit": payload.get("git_commit"),
    }
    if missing_required_count > 0:
        return ForegroundFinalHandoffCheck(
            code="operator_handoff_bundle",
            status=FINAL_HANDOFF_CHECK_FAILED,
            detail=f"Handoff bundle has {missing_required_count} missing required files.",
            path=path,
            metadata=metadata,
        )
    if included_count < file_count:
        return ForegroundFinalHandoffCheck(
            code="operator_handoff_bundle",
            status=FINAL_HANDOFF_CHECK_WARNING,
            detail="Handoff manifest has no missing files, but not every file was copied.",
            path=path,
            metadata=metadata,
        )
    return ForegroundFinalHandoffCheck(
        code="operator_handoff_bundle",
        status=FINAL_HANDOFF_CHECK_PASSED,
        detail="Handoff bundle manifest has no missing required files.",
        path=path,
        metadata=metadata,
    )


@dataclass(frozen=True)
class _JsonReadResult:
    payload: object | None = None
    error: str | None = None


def _read_json(path: Path) -> _JsonReadResult:
    if not path.exists():
        return _JsonReadResult(error=f"Evidence file is missing: {path}")
    try:
        return _JsonReadResult(payload=json.loads(path.read_text(encoding="utf-8")))
    except Exception as exc:
        return _JsonReadResult(error=f"Evidence file could not be parsed: {exc}")


def _metadata_from_payload(payload: dict[str, object]) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for key in (
        "checked_at_label",
        "generated_at_label",
        "passed_count",
        "warning_count",
        "failed_count",
    ):
        if key in payload:
            metadata[key] = payload[key]
    return metadata


def _overall_status(checks: tuple[ForegroundFinalHandoffCheck, ...]) -> str:
    if any(check.status == FINAL_HANDOFF_CHECK_FAILED for check in checks):
        return FINAL_HANDOFF_STATUS_BLOCKED
    if any(check.status == FINAL_HANDOFF_CHECK_WARNING for check in checks):
        return FINAL_HANDOFF_STATUS_WARNING
    return FINAL_HANDOFF_STATUS_READY


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _int(value: object) -> int:
    return value if isinstance(value, int) else 0


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _md_cell(value: object) -> str:
    return _text(value).replace("\n", " ").replace("|", "\\|")
