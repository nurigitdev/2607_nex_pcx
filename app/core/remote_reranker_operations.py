"""Read-only operations status for the remote reranker provider."""

import math
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.core.config import Settings
from app.core.rerankers import (
    DEFAULT_RERANKER_MODEL_ID,
    DEFAULT_RERANKER_PROFILE_NAME,
    REMOTE_RERANKER_PROVIDER_MODE,
    RERANK_RETRIEVAL_STRATEGY,
    InvalidRerankerError,
    RemoteRerankerProviderClient,
    RerankCandidate,
    RerankRequest,
    RerankResult,
    reranker_runtime_config_from_settings,
)

DEFAULT_REMOTE_RERANKER_HOST = "192.168.20.243"
DEFAULT_REMOTE_RERANKER_SSH_USER = "nexpcx"
DEFAULT_REMOTE_RERANKER_WORKDIR = "/home/nexpcx/2607_nex_pcx"
DEFAULT_REMOTE_RERANKER_PORT = 9104
DEFAULT_REMOTE_RERANKER_PROVIDER_HOST = "0.0.0.0"
DEFAULT_REMOTE_RERANKER_PROVIDER_NAME = "qwen-reranker-primary"
DEFAULT_REMOTE_RERANKER_SYSTEMD_UNIT_NAME = "nex-pcx-reranker-provider.service"
DEFAULT_REMOTE_RERANKER_BACKEND = "qwen_reranker"
DEFAULT_REMOTE_RERANKER_DEVICE = "cuda:0"
DEFAULT_REMOTE_RERANKER_HEALTH_TIMEOUT_SECONDS = 5.0
DEFAULT_REMOTE_RERANKER_REQUEST_TIMEOUT_SECONDS = 300.0
DEFAULT_REMOTE_RERANKER_SSH_TIMEOUT_SECONDS = 8
DEFAULT_REQUEST_TOP_K = 2
DEFAULT_REQUEST_SOURCE_PROFILE_NAME = "qwen3_4b_2560"
DEFAULT_REQUEST_SOURCE_RETRIEVAL_STRATEGY = "vector_cosine"
DEFAULT_REQUEST_QUERY_TEXT = "사내 문서 검색 권한과 업무 규칙"
DEFAULT_REQUEST_CANDIDATE_TEXTS = (
    "사내 공통 업무 규칙 문서는 모든 직원에게 공개되며 검색 범위에 포함된다.",
    "개인 업로드 문서는 작성자와 권한을 가진 상위 조직 사용자에게만 검색된다.",
    "reranker provider 상태 점검은 운영자가 관리자 화면에서 확인할 수 있다.",
)


@dataclass(frozen=True)
class RemoteRerankerOperationsPlan:
    provider_name: str
    backend: str
    reranker_profile_name: str
    provider_model_id: str
    device: str
    ssh_target: str
    workdir: str
    base_url: str
    health_url: str
    route_host: str
    host: str
    port: int
    pid_file: str
    log_file: str
    process_pattern: str
    systemd_unit_name: str
    remote_status_command: str


@dataclass(frozen=True)
class RemoteCommandObservation:
    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    values: dict[str, str]


@dataclass(frozen=True)
class RemoteRerankerHealthObservation:
    ok: bool
    status_code: int | None
    payload: dict[str, Any] | None
    error: str | None


@dataclass(frozen=True)
class RemoteRerankerOperationsReport:
    plan: RemoteRerankerOperationsPlan
    status: str
    pid: str | None
    command_observation: RemoteCommandObservation
    health_checked: bool
    health_ok: bool
    health_status_code: int | None
    health_payload: dict[str, Any] | None
    health_error: str | None
    health_mismatches: tuple[str, ...]
    request_smoke_checked: bool
    request_smoke_passed: bool | None
    request_smoke_summary: dict[str, Any] | None
    elapsed_ms: int
    error: str | None = None

    @property
    def passed(self) -> bool:
        if self.error is not None or not self.command_observation.ok:
            return False
        request_ok = not self.request_smoke_checked or self.request_smoke_passed is True
        return (
            self.status == "running"
            and self.health_ok
            and not self.health_mismatches
            and request_ok
        )


@dataclass(frozen=True)
class RemoteRerankerOperationsStatus:
    """HTTP-friendly remote reranker operations status."""

    status_code: int
    payload: dict[str, Any]


def get_remote_reranker_operations_status(
    settings: Settings,
    *,
    request_smoke: bool = False,
    host: str = DEFAULT_REMOTE_RERANKER_HOST,
    ssh_user: str = DEFAULT_REMOTE_RERANKER_SSH_USER,
    workdir: str = DEFAULT_REMOTE_RERANKER_WORKDIR,
    health_timeout_seconds: float = DEFAULT_REMOTE_RERANKER_HEALTH_TIMEOUT_SECONDS,
    request_timeout_seconds: float = DEFAULT_REMOTE_RERANKER_REQUEST_TIMEOUT_SECONDS,
) -> RemoteRerankerOperationsStatus:
    """Check the DGX reranker process, health endpoint, and optional request smoke."""

    checked_at = datetime.now(UTC).isoformat()
    runtime_config = _runtime_config_payload(settings)
    try:
        if runtime_config["status"] == "misconfigured":
            raise InvalidRerankerError(str(runtime_config["error"]))
        route_host, port = _operations_route_from_runtime_config(
            runtime_config,
            default_host=host,
            default_port=DEFAULT_REMOTE_RERANKER_PORT,
        )
        plan = build_remote_reranker_operations_plan(
            host=host,
            ssh_user=ssh_user,
            workdir=workdir,
            route_host=route_host,
            port=port,
            provider_model_id=str(runtime_config["reranker_model_id"]),
            reranker_profile_name=str(runtime_config["reranker_profile_name"]),
        )
        report = run_remote_reranker_operations_status(
            plan,
            request_smoke=request_smoke,
            health_timeout_seconds=health_timeout_seconds,
            request_timeout_seconds=request_timeout_seconds,
        )
    except (InvalidRerankerError, ValueError, OSError) as exc:
        return RemoteRerankerOperationsStatus(
            status_code=503,
            payload=_misconfigured_payload(
                checked_at=checked_at,
                runtime_config=runtime_config,
                request_smoke=request_smoke,
                error=str(exc),
            ),
        )

    operations_status = _classify_operations_status(report)
    status_code = 200 if report.passed else 503
    return RemoteRerankerOperationsStatus(
        status_code=status_code,
        payload=_status_payload(
            checked_at=checked_at,
            report=report,
            runtime_config=runtime_config,
            request_smoke=request_smoke,
            operations_status=operations_status,
        ),
    )


def build_remote_reranker_operations_plan(
    *,
    host: str = DEFAULT_REMOTE_RERANKER_HOST,
    ssh_user: str = DEFAULT_REMOTE_RERANKER_SSH_USER,
    workdir: str = DEFAULT_REMOTE_RERANKER_WORKDIR,
    route_host: str | None = None,
    port: int = DEFAULT_REMOTE_RERANKER_PORT,
    provider_host: str = DEFAULT_REMOTE_RERANKER_PROVIDER_HOST,
    provider_model_id: str = DEFAULT_RERANKER_MODEL_ID,
    reranker_profile_name: str = DEFAULT_RERANKER_PROFILE_NAME,
) -> RemoteRerankerOperationsPlan:
    selected_host = _validate_nonblank(host, "host")
    selected_ssh_user = _validate_nonblank(ssh_user, "ssh_user")
    selected_workdir = _validate_nonblank(workdir, "workdir")
    selected_route_host = _validate_nonblank(route_host or selected_host, "route_host")
    selected_provider_host = _validate_nonblank(provider_host, "provider_host")
    selected_provider_model_id = _validate_nonblank(provider_model_id, "provider_model_id")
    selected_reranker_profile_name = _validate_nonblank(
        reranker_profile_name,
        "reranker_profile_name",
    )
    if port <= 0:
        raise ValueError("port must be greater than 0")

    pid_file = f"run/remote_reranker_provider_{port}.pid"
    log_file = f"logs/remote_reranker_provider_{port}.log"
    process_pattern = (
        f"[u]vicorn app.reranker_provider_service:app --host "
        f"{selected_provider_host} --port {port}"
    )
    plan = RemoteRerankerOperationsPlan(
        provider_name=DEFAULT_REMOTE_RERANKER_PROVIDER_NAME,
        backend=DEFAULT_REMOTE_RERANKER_BACKEND,
        reranker_profile_name=selected_reranker_profile_name,
        provider_model_id=selected_provider_model_id,
        device=DEFAULT_REMOTE_RERANKER_DEVICE,
        ssh_target=f"{selected_ssh_user}@{selected_host}",
        workdir=selected_workdir,
        base_url=f"http://{selected_route_host}:{port}",
        health_url=f"http://{selected_route_host}:{port}/healthz",
        route_host=selected_route_host,
        host=selected_provider_host,
        port=port,
        pid_file=pid_file,
        log_file=log_file,
        process_pattern=process_pattern,
        systemd_unit_name=DEFAULT_REMOTE_RERANKER_SYSTEMD_UNIT_NAME,
        remote_status_command="",
    )
    return RemoteRerankerOperationsPlan(
        **{
            **asdict(plan),
            "remote_status_command": _remote_status_command(plan),
        }
    )


def run_remote_reranker_operations_status(
    plan: RemoteRerankerOperationsPlan,
    *,
    request_smoke: bool = False,
    health_timeout_seconds: float = DEFAULT_REMOTE_RERANKER_HEALTH_TIMEOUT_SECONDS,
    request_timeout_seconds: float = DEFAULT_REMOTE_RERANKER_REQUEST_TIMEOUT_SECONDS,
) -> RemoteRerankerOperationsReport:
    started_at = time.perf_counter()
    try:
        command_observation = _run_remote_status_command(plan)
    except (OSError, subprocess.TimeoutExpired) as exc:
        command_observation = RemoteCommandObservation(
            ok=False,
            exit_code=1,
            stdout="",
            stderr=str(exc),
            values={},
        )
        health_observation = RemoteRerankerHealthObservation(
            ok=False,
            status_code=None,
            payload=None,
            error=None,
        )
        return _report(
            plan,
            command_observation=command_observation,
            health_observation=health_observation,
            health_mismatches=(),
            request_smoke_checked=request_smoke,
            request_smoke_summary=None,
            elapsed_ms=_elapsed_ms(started_at),
            error=str(exc),
        )

    health_observation = _probe_health_once(
        plan,
        timeout_seconds=health_timeout_seconds,
    )
    health_mismatches = (
        _health_mismatches(health_observation.payload, plan=plan) if health_observation.ok else ()
    )
    request_smoke_summary = None
    if request_smoke and health_observation.ok and not health_mismatches:
        request_smoke_summary = _run_request_smoke(
            plan,
            timeout_seconds=request_timeout_seconds,
        )

    return _report(
        plan,
        command_observation=command_observation,
        health_observation=health_observation,
        health_mismatches=health_mismatches,
        request_smoke_checked=request_smoke,
        request_smoke_summary=request_smoke_summary,
        elapsed_ms=_elapsed_ms(started_at),
        error=None,
    )


def _runtime_config_payload(settings: Settings) -> dict[str, Any]:
    raw_reranker_profile_name = str(
        getattr(settings, "reranker_profile_name", DEFAULT_RERANKER_PROFILE_NAME)
    ).strip()
    raw_reranker_model_id = str(
        getattr(settings, "reranker_model_id", DEFAULT_RERANKER_MODEL_ID)
    ).strip()
    try:
        config = reranker_runtime_config_from_settings(settings)
    except InvalidRerankerError as exc:
        return {
            "status": "misconfigured",
            "mode": "invalid",
            "remote_base_url": None,
            "timeout_seconds": None,
            "reranker_profile_name": raw_reranker_profile_name,
            "reranker_model_id": raw_reranker_model_id,
            "configured_for_remote": False,
            "error": str(exc),
        }
    return {
        "status": (
            "remote_selected" if config.mode == REMOTE_RERANKER_PROVIDER_MODE else "mock_selected"
        ),
        "mode": config.mode,
        "remote_base_url": config.remote_base_url,
        "timeout_seconds": config.remote_timeout_seconds,
        "reranker_profile_name": config.reranker_profile_name,
        "reranker_model_id": config.reranker_model_id,
        "configured_for_remote": config.mode == REMOTE_RERANKER_PROVIDER_MODE,
        "error": None,
    }


def _operations_route_from_runtime_config(
    runtime_config: dict[str, Any],
    *,
    default_host: str,
    default_port: int,
) -> tuple[str, int]:
    remote_base_url = runtime_config.get("remote_base_url")
    if not remote_base_url:
        return default_host, default_port
    parsed = urlparse(str(remote_base_url))
    if not parsed.hostname:
        raise ValueError("remote_reranker_provider_url must include a hostname")
    return parsed.hostname, parsed.port or _default_port_for_scheme(parsed.scheme)


def _default_port_for_scheme(scheme: str) -> int:
    if scheme == "https":
        return 443
    return 80


def _run_remote_status_command(
    plan: RemoteRerankerOperationsPlan,
) -> RemoteCommandObservation:
    completed = subprocess.run(
        (
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            plan.ssh_target,
            plan.remote_status_command,
        ),
        capture_output=True,
        check=False,
        text=True,
        timeout=DEFAULT_REMOTE_RERANKER_SSH_TIMEOUT_SECONDS,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    return RemoteCommandObservation(
        ok=completed.returncode == 0,
        exit_code=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        values=_parse_key_values(stdout),
    )


def _remote_status_command(plan: RemoteRerankerOperationsPlan) -> str:
    return _remote_prelude(plan) + (
        "unit_active='unavailable'; unit_enabled='unavailable'; unit_main_pid=''; "
        "if command -v systemctl >/dev/null 2>&1; then "
        f"unit_active=$(systemctl --user is-active {shlex.quote(plan.systemd_unit_name)} "
        "2>/dev/null || true); "
        f"unit_enabled=$(systemctl --user is-enabled {shlex.quote(plan.systemd_unit_name)} "
        "2>/dev/null || true); "
        f"unit_main_pid=$(systemctl --user show {shlex.quote(plan.systemd_unit_name)} "
        "-p MainPID --value 2>/dev/null || true); "
        "fi; "
        f"pid=$(pgrep -f {shlex.quote(plan.process_pattern)} | head -n 1 || true); "
        f"file_pid=''; if [ -f {shlex.quote(plan.pid_file)} ]; then "
        f"file_pid=$(cat {shlex.quote(plan.pid_file)} 2>/dev/null || true); fi; "
        'if [ -n "$pid" ]; then '
        f"printf '%s\\n' \"$pid\" > {shlex.quote(plan.pid_file)}; "
        "printf 'status=running\\npid=%s\\n' \"$pid\"; "
        "else "
        "printf 'status=stopped\\npid=\\n'; "
        "fi; "
        "printf 'file_pid=%s\\n' \"$file_pid\"; "
        f"printf 'pid_file=%s\\n' {shlex.quote(plan.pid_file)}; "
        f"printf 'log_file=%s\\n' {shlex.quote(plan.log_file)}; "
        f"printf '\\nsystemd_unit=%s\\n' {shlex.quote(plan.systemd_unit_name)}; "
        "printf 'systemd_active=%s\\n' \"$unit_active\"; "
        "printf 'systemd_enabled=%s\\n' \"$unit_enabled\"; "
        "printf 'systemd_main_pid=%s\\n' \"$unit_main_pid\""
    )


def _remote_prelude(plan: RemoteRerankerOperationsPlan) -> str:
    return (
        f"cd {shlex.quote(plan.workdir)} && "
        f"mkdir -p {shlex.quote(str(Path(plan.pid_file).parent))} "
        f"{shlex.quote(str(Path(plan.log_file).parent))} && "
    )


def _probe_health_once(
    plan: RemoteRerankerOperationsPlan,
    *,
    timeout_seconds: float,
) -> RemoteRerankerHealthObservation:
    client = RemoteRerankerProviderClient(plan.base_url, timeout_seconds=timeout_seconds)
    try:
        health = client.health()
    except InvalidRerankerError as exc:
        return RemoteRerankerHealthObservation(
            ok=False,
            status_code=None,
            payload=None,
            error=str(exc),
        )
    finally:
        client.close()

    payload = {
        "ready": health.ready,
        "provider_type": health.provider_type,
        "provider_model_id": health.provider_model_id,
        "reranker_profile_name": health.reranker_profile_name,
        "device": health.device,
        "runtime_metadata": dict(health.runtime_metadata),
    }
    return RemoteRerankerHealthObservation(
        ok=health.ready,
        status_code=200 if health.ready else 503,
        payload=payload,
        error=None if health.ready else "Remote reranker reported ready=false",
    )


def _run_request_smoke(
    plan: RemoteRerankerOperationsPlan,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    request = _build_request_smoke_request(plan)
    client = RemoteRerankerProviderClient(plan.base_url, timeout_seconds=timeout_seconds)
    started_at = time.perf_counter()
    try:
        result = client.rerank(request)
    except InvalidRerankerError as exc:
        return {
            "passed": False,
            "request_elapsed_ms": _elapsed_ms(started_at),
            "provider_elapsed_ms": None,
            "candidate_count": len(request.candidates),
            "returned_count": None,
            "result_previews": [],
            "runtime_metadata": {},
            "mismatches": [],
            "error": str(exc),
        }
    finally:
        client.close()

    mismatches = _request_smoke_mismatches(result, plan=plan, request=request)
    runtime_metadata = dict(result.runtime_metadata)
    return {
        "passed": not mismatches,
        "request_elapsed_ms": _elapsed_ms(started_at),
        "provider_elapsed_ms": _metadata_int(runtime_metadata, "elapsed_ms"),
        "candidate_count": result.candidate_count,
        "returned_count": result.returned_count,
        "result_previews": [
            {
                "candidate_key": item.candidate.candidate_key,
                "rank": item.rank,
                "score": round(float(item.score), 6),
                "source_rank": _metadata_int(item.score_components, "source_rank"),
                "score_components": dict(item.score_components),
            }
            for item in result.results
        ],
        "runtime_metadata": runtime_metadata,
        "mismatches": list(mismatches),
        "error": None,
    }


def _build_request_smoke_request(plan: RemoteRerankerOperationsPlan) -> RerankRequest:
    return RerankRequest(
        query_text=DEFAULT_REQUEST_QUERY_TEXT,
        candidates=tuple(
            RerankCandidate(
                candidate_key=f"candidate-{index}",
                rank=index,
                text=text,
                source_profile_name=DEFAULT_REQUEST_SOURCE_PROFILE_NAME,
                source_retrieval_strategy=DEFAULT_REQUEST_SOURCE_RETRIEVAL_STRATEGY,
                source_score=max(0.0, 1.0 - (index * 0.01)),
                chunk_id=1000 + index,
                metadata={"smoke": "remote_reranker_operations"},
            )
            for index, text in enumerate(DEFAULT_REQUEST_CANDIDATE_TEXTS, start=1)
        ),
        top_k=DEFAULT_REQUEST_TOP_K,
        reranker_profile_name=plan.reranker_profile_name,
        reranker_model_id=plan.provider_model_id,
    )


def _health_mismatches(
    payload: dict[str, Any] | None,
    *,
    plan: RemoteRerankerOperationsPlan,
) -> tuple[str, ...]:
    if payload is None:
        return ()
    expected_values = {
        "provider_type": REMOTE_RERANKER_PROVIDER_MODE,
        "provider_model_id": plan.provider_model_id,
        "reranker_profile_name": plan.reranker_profile_name,
        "device": plan.device,
    }
    mismatches: list[str] = []
    for key, expected in expected_values.items():
        actual = payload.get(key)
        if actual != expected:
            mismatches.append(f"{key}: expected {expected!r}, got {actual!r}")

    runtime_metadata = dict(payload.get("runtime_metadata") or {})
    for key, expected in {
        "service": "nex_pcx_reranker_provider_service",
        "backend": plan.backend,
    }.items():
        actual = runtime_metadata.get(key)
        if actual != expected:
            mismatches.append(f"runtime_metadata.{key}: expected {expected!r}, got {actual!r}")
    return tuple(mismatches)


def _request_smoke_mismatches(
    result: RerankResult,
    *,
    plan: RemoteRerankerOperationsPlan,
    request: RerankRequest,
) -> tuple[str, ...]:
    expected_values = {
        "provider_type": REMOTE_RERANKER_PROVIDER_MODE,
        "reranker_model_id": plan.provider_model_id,
        "reranker_profile_name": plan.reranker_profile_name,
        "retrieval_strategy": RERANK_RETRIEVAL_STRATEGY,
        "candidate_count": len(request.candidates),
        "returned_count": min(request.top_k, len(request.candidates)),
        "top_k": min(request.top_k, len(request.candidates)),
    }
    actual_values = {
        "provider_type": result.provider_type,
        "reranker_model_id": result.reranker_model_id,
        "reranker_profile_name": result.reranker_profile_name,
        "retrieval_strategy": result.retrieval_strategy,
        "candidate_count": result.candidate_count,
        "returned_count": result.returned_count,
        "top_k": result.top_k,
    }
    mismatches = [
        f"{key}: expected {expected!r}, got {actual_values[key]!r}"
        for key, expected in expected_values.items()
        if actual_values[key] != expected
    ]
    if len(result.results) != result.returned_count:
        mismatches.append(
            f"result_count: expected {result.returned_count!r}, got {len(result.results)!r}"
        )
    expected_ranks = tuple(range(1, len(result.results) + 1))
    actual_ranks = tuple(item.rank for item in result.results)
    if actual_ranks != expected_ranks:
        mismatches.append(f"result ranks: expected {expected_ranks!r}, got {actual_ranks!r}")
    for item in result.results:
        if not math.isfinite(item.score):
            mismatches.append(f"score must be finite for {item.candidate.candidate_key!r}")
    mismatches.extend(_runtime_metadata_mismatches(result.runtime_metadata, plan=plan))
    return tuple(mismatches)


def _runtime_metadata_mismatches(
    runtime_metadata: dict[str, Any],
    *,
    plan: RemoteRerankerOperationsPlan,
) -> tuple[str, ...]:
    mismatches: list[str] = []
    expected_values = {
        "service": "nex_pcx_reranker_provider_service",
        "backend": plan.backend,
        "device": plan.device,
    }
    for key, expected in expected_values.items():
        actual = runtime_metadata.get(key)
        if actual != expected:
            mismatches.append(f"runtime_metadata.{key}: expected {expected!r}, got {actual!r}")
    return tuple(mismatches)


def _report(
    plan: RemoteRerankerOperationsPlan,
    *,
    command_observation: RemoteCommandObservation,
    health_observation: RemoteRerankerHealthObservation,
    health_mismatches: tuple[str, ...],
    request_smoke_checked: bool,
    request_smoke_summary: dict[str, Any] | None,
    elapsed_ms: int,
    error: str | None,
) -> RemoteRerankerOperationsReport:
    values = command_observation.values
    status = values.get("status") or ("failed" if not command_observation.ok else "unknown")
    pid = values.get("pid") or None
    request_smoke_passed = (
        bool(request_smoke_summary["passed"]) if request_smoke_summary is not None else None
    )
    return RemoteRerankerOperationsReport(
        plan=plan,
        status=status,
        pid=pid,
        command_observation=command_observation,
        health_checked=True,
        health_ok=health_observation.ok,
        health_status_code=health_observation.status_code,
        health_payload=health_observation.payload,
        health_error=health_observation.error,
        health_mismatches=health_mismatches,
        request_smoke_checked=request_smoke_checked,
        request_smoke_passed=request_smoke_passed,
        request_smoke_summary=request_smoke_summary,
        elapsed_ms=elapsed_ms,
        error=error,
    )


def _status_payload(
    *,
    checked_at: str,
    report: RemoteRerankerOperationsReport,
    runtime_config: dict[str, Any],
    request_smoke: bool,
    operations_status: str,
) -> dict[str, Any]:
    return {
        "checked_at": checked_at,
        "passed": report.passed,
        "operations_status": operations_status,
        "app_runtime": runtime_config,
        "request_smoke_requested": request_smoke,
        "status": report.status,
        "pid": report.pid,
        "provider": _provider_payload(report.plan),
        "plan": asdict(report.plan),
        "command_observation": _command_observation_payload(report),
        "health": {
            "checked": report.health_checked,
            "ok": report.health_ok,
            "status_code": report.health_status_code,
            "payload": report.health_payload,
            "error": report.health_error,
            "mismatches": list(report.health_mismatches),
        },
        "request_smoke": {
            "checked": report.request_smoke_checked,
            "passed": report.request_smoke_passed,
            "summary": report.request_smoke_summary,
        },
        "elapsed_ms": report.elapsed_ms,
        "error": report.error,
    }


def _provider_payload(plan: RemoteRerankerOperationsPlan) -> dict[str, Any]:
    return {
        "provider_name": plan.provider_name,
        "provider_model_id": plan.provider_model_id,
        "reranker_profile_name": plan.reranker_profile_name,
        "backend": plan.backend,
        "device": plan.device,
        "base_url": plan.base_url,
        "health_url": plan.health_url,
        "ssh_target": plan.ssh_target,
        "workdir": plan.workdir,
        "pid_file": f"{plan.workdir}/{plan.pid_file}",
        "log_file": f"{plan.workdir}/{plan.log_file}",
        "systemd_unit_name": plan.systemd_unit_name,
    }


def _misconfigured_payload(
    *,
    checked_at: str,
    runtime_config: dict[str, Any],
    request_smoke: bool,
    error: str,
) -> dict[str, Any]:
    return {
        "checked_at": checked_at,
        "passed": False,
        "operations_status": "misconfigured",
        "app_runtime": runtime_config,
        "error": error,
        "request_smoke_requested": request_smoke,
        "plan": None,
        "command_observation": None,
        "health": {
            "checked": False,
            "ok": False,
            "status_code": None,
            "payload": None,
            "error": error,
            "mismatches": [],
        },
        "request_smoke": {
            "checked": request_smoke,
            "passed": None,
            "summary": None,
        },
    }


def _classify_operations_status(report: RemoteRerankerOperationsReport) -> str:
    if report.passed:
        return "ready"
    if not report.command_observation.ok:
        return "command_failed"
    if report.error:
        return "failed"
    if report.status in {"stopped", "not_running"}:
        return "stopped"
    if report.health_mismatches:
        return "contract_mismatch"
    if not report.health_ok:
        return "health_unreachable"
    if report.request_smoke_checked and report.request_smoke_passed is not True:
        return "request_smoke_failed"
    return "warning"


def _command_observation_payload(report: RemoteRerankerOperationsReport) -> dict[str, Any]:
    command_observation = report.command_observation
    return {
        **asdict(command_observation),
        "stdout_line_count": len(command_observation.stdout.splitlines()),
        "stderr_line_count": len(command_observation.stderr.splitlines()),
    }


def _parse_key_values(stdout: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            values[key.strip()] = value.strip()
    return values


def _elapsed_ms(started_at: float) -> int:
    return max(0, int((time.perf_counter() - started_at) * 1000))


def _metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized
