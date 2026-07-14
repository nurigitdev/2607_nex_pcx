"""Run a DGX-backed route-aware embedding worker E2E smoke."""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from psycopg.types.json import Json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.database import connect  # noqa: E402
from app.core.embedding_jobs import (  # noqa: E402
    EmbeddingJobInput,
    EmbeddingJobRecord,
    create_embedding_job,
)
from app.core.embedding_provider_presets import (  # noqa: E402
    InvalidEmbeddingProviderPresetError,
    get_embedding_provider_preset,
    list_embedding_provider_presets,
)
from app.core.embedding_providers import EmbeddingProviderRuntimeConfig  # noqa: E402
from app.core.embedding_vectors import (  # noqa: E402
    EmbeddingVectorRecord,
    get_chunk_embedding,
)
from app.core.embedding_worker import (  # noqa: E402
    EmbeddingWorkerResult,
    process_next_embedding_job_with_provider_routes,
)
from app.core.pipeline_jobs import DEFAULT_LEASE_SECONDS  # noqa: E402
from scripts.plan_remote_provider_foreground_smoke import (  # noqa: E402
    DEFAULT_DEVICE,
    DEFAULT_GPU_HOST,
    DEFAULT_GPU_USER,
    DEFAULT_GPU_WORKDIR,
    RemoteProviderForegroundSmokePlan,
    build_foreground_smoke_plan,
)
from scripts.run_dgx_provider_route_preflight_verification import (  # noqa: E402
    DgxProviderRoutePreflightProfileResult,
    _redact_database_url,
    _run_profile_preflight,
)
from scripts.run_remote_provider_embedding_smoke_suite import (  # noqa: E402
    _confirm_remote_stop,
)
from scripts.run_remote_provider_foreground_smoke import (  # noqa: E402
    DEFAULT_HEALTH_TIMEOUT_SECONDS,
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
    HealthObservation,
    _automated_ssh_launch_command,
    _health_mismatches,
    _probe_health_once,
    _read_tail,
    _stop_process,
    _validate_positive,
)

DEFAULT_PROVIDER_ORDER = ("kure", "bge", "qwen")
DEFAULT_STARTUP_TIMEOUT_SECONDS = 300.0
DEFAULT_REMOTE_TIMEOUT_SECONDS = 300.0
DEFAULT_FIXTURE_TEXT = (
    "NeX-PCX DGX remote embedding worker E2E smoke fixture. "
    "This short chunk validates route-aware worker vector persistence."
)
SMOKE_JOB_PRIORITY_TIMESTAMP = "1970-01-01T00:00:00+00:00"


@dataclass(frozen=True)
class DgxRemoteEmbeddingWorkerProviderPlan:
    provider: str
    foreground_plan: RemoteProviderForegroundSmokePlan
    profile_names: tuple[str, ...]
    startup_timeout_seconds: float


@dataclass(frozen=True)
class DgxRemoteEmbeddingWorkerSmokePlan:
    database_url: str
    providers: tuple[DgxRemoteEmbeddingWorkerProviderPlan, ...]
    fixture_text: str
    preflight_before_worker: bool
    active_only_preflight: bool
    cleanup_fixture: bool
    worker_name_prefix: str
    lease_seconds: int
    remote_timeout_seconds: float
    readiness_gate_defer_seconds: int
    health_timeout_seconds: float
    poll_interval_seconds: float
    shutdown_timeout_seconds: float
    fail_fast: bool


@dataclass(frozen=True)
class DgxEmbeddingWorkerSmokeFixture:
    smoke_run_key: str
    file_id: int
    document_id: int
    chunk_id: int
    job_ids_by_profile: dict[str, int]


@dataclass(frozen=True)
class DgxEmbeddingWorkerProfileSmokeResult:
    profile_name: str
    job_id: int | None
    chunk_id: int | None
    processed: bool
    job_status: str | None
    vector_table_name: str | None
    vector_dimension: int | None
    vector_storage_type: str | None
    provider_route_id: int | None
    provider_route_name: str | None
    provider_runtime_base_url: str | None
    provider_model_id: str | None
    provider_type: str | None
    provider_elapsed_ms: int | None
    elapsed_ms: int | None
    readiness_status: str | None
    readiness_health_snapshot_id: int | None
    readiness_contract_snapshot_id: int | None
    message: str | None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return (
            self.processed
            and self.job_status == "succeeded"
            and self.vector_table_name is not None
            and self.vector_dimension is not None
            and self.provider_runtime_base_url is not None
            and self.provider_type == "remote"
            and self.error is None
        )


@dataclass(frozen=True)
class DgxRemoteEmbeddingWorkerProviderResult:
    provider: str
    provider_name: str
    base_url: str
    health_url: str
    launch_command: tuple[str, ...]
    profile_names: tuple[str, ...]
    startup_timeout_seconds: float
    health_timeout_seconds: float
    poll_interval_seconds: float
    shutdown_timeout_seconds: float
    pre_launch_health_reachable: bool
    launched: bool
    health_checked: bool
    health_ok: bool
    health_attempts: int
    health_status_code: int | None
    health_payload: dict[str, Any] | None
    health_error: str | None
    health_mismatches: tuple[str, ...]
    preflight_results: tuple[DgxProviderRoutePreflightProfileResult, ...]
    profile_results: tuple[DgxEmbeddingWorkerProfileSmokeResult, ...]
    process_exit_code_before_stop: int | None
    process_exit_code_after_stop: int | None
    stopped: bool
    stop_confirmed: bool
    remote_stop_attempted: bool
    remote_stop_exit_code: int | None
    remote_stop_stdout: str
    remote_stop_stderr: str
    post_stop_health_reachable: bool
    elapsed_seconds: float
    stdout_tail: str
    stderr_tail: str
    error: str | None = None

    @property
    def passed(self) -> bool:
        preflight_ok = all(result.passed for result in self.preflight_results)
        return (
            self.launched
            and self.health_ok
            and preflight_ok
            and bool(self.profile_results)
            and all(result.passed for result in self.profile_results)
            and self.stopped
            and self.stop_confirmed
            and (not self.remote_stop_attempted or self.remote_stop_exit_code == 0)
            and not self.post_stop_health_reachable
            and self.error is None
        )


@dataclass(frozen=True)
class DgxRemoteEmbeddingWorkerSmokeReport:
    plan: DgxRemoteEmbeddingWorkerSmokePlan
    fixture: DgxEmbeddingWorkerSmokeFixture | None
    results: tuple[DgxRemoteEmbeddingWorkerProviderResult, ...]
    cleanup_attempted: bool
    cleanup_confirmed: bool
    total_elapsed_seconds: float

    @property
    def passed(self) -> bool:
        return (
            self.fixture is not None
            and bool(self.results)
            and all(result.passed for result in self.results)
            and (not self.cleanup_attempted or self.cleanup_confirmed)
        )


def build_dgx_remote_embedding_worker_smoke_plan(
    provider_names: tuple[str, ...] | None = None,
    *,
    database_url: str,
    host: str = DEFAULT_GPU_HOST,
    ssh_user: str = DEFAULT_GPU_USER,
    workdir: str = DEFAULT_GPU_WORKDIR,
    models_dir: str | None = None,
    python_bin: str | None = None,
    provider_host: str = "0.0.0.0",
    route_host: str | None = None,
    device: str = DEFAULT_DEVICE,
    startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
    health_timeout_seconds: float = DEFAULT_HEALTH_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    shutdown_timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
    fixture_text: str = DEFAULT_FIXTURE_TEXT,
    preflight_before_worker: bool = True,
    active_only_preflight: bool = True,
    cleanup_fixture: bool = True,
    worker_name_prefix: str = "dgx-worker-e2e",
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    remote_timeout_seconds: float = DEFAULT_REMOTE_TIMEOUT_SECONDS,
    readiness_gate_defer_seconds: int = 300,
    fail_fast: bool = False,
) -> DgxRemoteEmbeddingWorkerSmokePlan:
    if not database_url.strip():
        raise ValueError("database_url is required")
    if not fixture_text.strip():
        raise ValueError("fixture_text is required")
    if not worker_name_prefix.strip():
        raise ValueError("worker_name_prefix is required")
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be greater than 0")
    if readiness_gate_defer_seconds <= 0:
        raise ValueError("readiness_gate_defer_seconds must be greater than 0")
    startup_timeout_seconds = _validate_positive(
        startup_timeout_seconds,
        name="startup_timeout_seconds",
    )
    health_timeout_seconds = _validate_positive(
        health_timeout_seconds,
        name="health_timeout_seconds",
    )
    poll_interval_seconds = _validate_positive(
        poll_interval_seconds,
        name="poll_interval_seconds",
    )
    shutdown_timeout_seconds = _validate_positive(
        shutdown_timeout_seconds,
        name="shutdown_timeout_seconds",
    )
    remote_timeout_seconds = _validate_positive(
        remote_timeout_seconds,
        name="remote_timeout_seconds",
    )

    provider_plans: list[DgxRemoteEmbeddingWorkerProviderPlan] = []
    for provider_name in _normalize_provider_names(provider_names):
        preset = get_embedding_provider_preset(provider_name)
        foreground_plan = build_foreground_smoke_plan(
            preset,
            host=host,
            ssh_user=ssh_user,
            workdir=workdir,
            models_dir=models_dir,
            python_bin=python_bin,
            provider_host=provider_host,
            route_host=route_host,
            device=device,
        )
        provider_plans.append(
            DgxRemoteEmbeddingWorkerProviderPlan(
                provider=provider_name,
                foreground_plan=foreground_plan,
                profile_names=tuple(foreground_plan.profile_names),
                startup_timeout_seconds=startup_timeout_seconds,
            )
        )

    return DgxRemoteEmbeddingWorkerSmokePlan(
        database_url=database_url,
        providers=tuple(provider_plans),
        fixture_text=fixture_text.strip(),
        preflight_before_worker=preflight_before_worker,
        active_only_preflight=active_only_preflight,
        cleanup_fixture=cleanup_fixture,
        worker_name_prefix=worker_name_prefix.strip(),
        lease_seconds=lease_seconds,
        remote_timeout_seconds=remote_timeout_seconds,
        readiness_gate_defer_seconds=readiness_gate_defer_seconds,
        health_timeout_seconds=health_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        shutdown_timeout_seconds=shutdown_timeout_seconds,
        fail_fast=fail_fast,
    )


def run_dgx_remote_embedding_worker_smoke(
    plan: DgxRemoteEmbeddingWorkerSmokePlan,
) -> DgxRemoteEmbeddingWorkerSmokeReport:
    started_at = time.monotonic()
    fixture: DgxEmbeddingWorkerSmokeFixture | None = None
    cleanup_attempted = False
    cleanup_confirmed = False
    results: list[DgxRemoteEmbeddingWorkerProviderResult] = []
    try:
        fixture = create_smoke_fixture(
            plan.database_url,
            fixture_text=plan.fixture_text,
            profile_names=_plan_profile_names(plan),
        )
        for provider_plan in plan.providers:
            result = run_provider_worker_e2e_session(
                provider_plan,
                database_url=plan.database_url,
                fixture=fixture,
                preflight_before_worker=plan.preflight_before_worker,
                active_only_preflight=plan.active_only_preflight,
                worker_name_prefix=plan.worker_name_prefix,
                lease_seconds=plan.lease_seconds,
                remote_timeout_seconds=plan.remote_timeout_seconds,
                readiness_gate_defer_seconds=plan.readiness_gate_defer_seconds,
                health_timeout_seconds=plan.health_timeout_seconds,
                poll_interval_seconds=plan.poll_interval_seconds,
                shutdown_timeout_seconds=plan.shutdown_timeout_seconds,
                fail_fast=plan.fail_fast,
            )
            results.append(result)
            if plan.fail_fast and not result.passed:
                break
    finally:
        if fixture is not None and plan.cleanup_fixture:
            cleanup_attempted = True
            cleanup_confirmed = cleanup_smoke_fixture(plan.database_url, fixture.file_id)

    return DgxRemoteEmbeddingWorkerSmokeReport(
        plan=plan,
        fixture=fixture,
        results=tuple(results),
        cleanup_attempted=cleanup_attempted,
        cleanup_confirmed=cleanup_confirmed,
        total_elapsed_seconds=max(0.0, time.monotonic() - started_at),
    )


def run_provider_worker_e2e_session(
    provider_plan: DgxRemoteEmbeddingWorkerProviderPlan,
    *,
    database_url: str,
    fixture: DgxEmbeddingWorkerSmokeFixture,
    preflight_before_worker: bool,
    active_only_preflight: bool,
    worker_name_prefix: str,
    lease_seconds: int,
    remote_timeout_seconds: float,
    readiness_gate_defer_seconds: int,
    health_timeout_seconds: float,
    poll_interval_seconds: float,
    shutdown_timeout_seconds: float,
    fail_fast: bool,
) -> DgxRemoteEmbeddingWorkerProviderResult:
    foreground_plan = provider_plan.foreground_plan
    launch_command = _automated_ssh_launch_command(foreground_plan)
    started_at = time.monotonic()
    health_checked = False
    health_attempts = 0
    health_observation = HealthObservation(
        ok=False,
        status_code=None,
        payload=None,
        error="Health check was not attempted.",
    )
    health_mismatches: tuple[str, ...] = ()
    preflight_results: list[DgxProviderRoutePreflightProfileResult] = []
    profile_results: list[DgxEmbeddingWorkerProfileSmokeResult] = []
    exit_code_before_stop: int | None = None
    exit_code_after_stop: int | None = None
    stopped = False
    stop_confirmed = False
    remote_stop_attempted = False
    remote_stop_exit_code: int | None = None
    remote_stop_stdout = ""
    remote_stop_stderr = ""
    post_stop_health_reachable = False
    launched = False
    error: str | None = None

    pre_launch_observation = _probe_health_once(
        foreground_plan.health_url,
        timeout_seconds=min(health_timeout_seconds, 1.0),
    )
    pre_launch_health_reachable = pre_launch_observation.ok
    if pre_launch_health_reachable:
        return _provider_result(
            provider_plan,
            launch_command=launch_command,
            health_timeout_seconds=health_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            shutdown_timeout_seconds=shutdown_timeout_seconds,
            pre_launch_health_reachable=True,
            launched=False,
            health_checked=False,
            health_ok=False,
            health_attempts=0,
            health_status_code=pre_launch_observation.status_code,
            health_payload=pre_launch_observation.payload,
            health_error=pre_launch_observation.error,
            health_mismatches=(),
            preflight_results=(),
            profile_results=(),
            process_exit_code_before_stop=None,
            process_exit_code_after_stop=None,
            stopped=False,
            stop_confirmed=False,
            remote_stop_attempted=False,
            remote_stop_exit_code=None,
            remote_stop_stdout="",
            remote_stop_stderr="",
            post_stop_health_reachable=True,
            elapsed_seconds=max(0.0, time.monotonic() - started_at),
            stdout_tail="",
            stderr_tail="",
            error=(
                "Health URL is already reachable before launch; "
                "stop the existing provider or use another port."
            ),
        )

    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8", errors="replace") as stdout:
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8", errors="replace") as stderr:
            try:
                process = subprocess.Popen(
                    launch_command,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                )
                launched = True
            except OSError as exc:
                return _provider_result(
                    provider_plan,
                    launch_command=launch_command,
                    health_timeout_seconds=health_timeout_seconds,
                    poll_interval_seconds=poll_interval_seconds,
                    shutdown_timeout_seconds=shutdown_timeout_seconds,
                    pre_launch_health_reachable=pre_launch_health_reachable,
                    launched=False,
                    health_checked=False,
                    health_ok=False,
                    health_attempts=0,
                    health_status_code=None,
                    health_payload=None,
                    health_error=None,
                    health_mismatches=(),
                    preflight_results=(),
                    profile_results=(),
                    process_exit_code_before_stop=None,
                    process_exit_code_after_stop=None,
                    stopped=False,
                    stop_confirmed=False,
                    remote_stop_attempted=False,
                    remote_stop_exit_code=None,
                    remote_stop_stdout="",
                    remote_stop_stderr="",
                    post_stop_health_reachable=False,
                    elapsed_seconds=max(0.0, time.monotonic() - started_at),
                    stdout_tail=_read_tail(stdout),
                    stderr_tail=_read_tail(stderr),
                    error=str(exc),
                )

            deadline = time.monotonic() + provider_plan.startup_timeout_seconds
            while time.monotonic() < deadline:
                exit_code_before_stop = process.poll()
                if exit_code_before_stop is not None:
                    error = (
                        f"Remote provider process exited early with code {exit_code_before_stop}."
                    )
                    break

                health_checked = True
                health_attempts += 1
                health_observation = _probe_health_once(
                    foreground_plan.health_url,
                    timeout_seconds=health_timeout_seconds,
                )
                if health_observation.ok:
                    health_mismatches = _health_mismatches(
                        health_observation.payload,
                        plan=foreground_plan,
                    )
                    break
                time.sleep(min(poll_interval_seconds, max(0.0, deadline - time.monotonic())))
            else:
                error = (
                    "Health check did not pass within "
                    f"{provider_plan.startup_timeout_seconds:.1f} seconds."
                )

            if health_observation.ok and health_mismatches:
                error = "Health response did not match the expected provider plan."

            if health_observation.ok and not health_mismatches:
                if preflight_before_worker:
                    for profile_name in provider_plan.profile_names:
                        preflight_result = _run_profile_preflight(
                            database_url,
                            profile_name=profile_name,
                            active_only=active_only_preflight,
                        )
                        preflight_results.append(preflight_result)
                        if fail_fast and not preflight_result.passed:
                            break
                if any(not result.passed for result in preflight_results):
                    error = "One or more route preflight checks failed before worker smoke."

            if health_observation.ok and not health_mismatches and error is None:
                for profile_name in provider_plan.profile_names:
                    profile_result = _run_worker_profile_smoke(
                        database_url,
                        fixture=fixture,
                        profile_name=profile_name,
                        worker_name=f"{worker_name_prefix}-{profile_name}",
                        lease_seconds=lease_seconds,
                        remote_timeout_seconds=remote_timeout_seconds,
                        readiness_gate_defer_seconds=readiness_gate_defer_seconds,
                    )
                    profile_results.append(profile_result)
                    if fail_fast and not profile_result.passed:
                        break
                if any(not result.passed for result in profile_results) and error is None:
                    error = "One or more embedding worker profile smokes failed."

            exit_code_before_stop = process.poll()
            stopped, exit_code_after_stop = _stop_process(
                process,
                shutdown_timeout_seconds=shutdown_timeout_seconds,
            )
            stop_confirmed = exit_code_after_stop is not None
            if health_observation.ok:
                (
                    post_stop_health_reachable,
                    remote_stop_attempted,
                    remote_stop_exit_code,
                    remote_stop_stdout,
                    remote_stop_stderr,
                    stop_error,
                ) = _confirm_remote_stop(
                    foreground_plan,
                    health_timeout_seconds=health_timeout_seconds,
                    poll_interval_seconds=poll_interval_seconds,
                    shutdown_timeout_seconds=shutdown_timeout_seconds,
                )
                if stop_error and error is None:
                    error = stop_error

            return _provider_result(
                provider_plan,
                launch_command=launch_command,
                health_timeout_seconds=health_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
                shutdown_timeout_seconds=shutdown_timeout_seconds,
                pre_launch_health_reachable=pre_launch_health_reachable,
                launched=launched,
                health_checked=health_checked,
                health_ok=health_observation.ok and not health_mismatches,
                health_attempts=health_attempts,
                health_status_code=health_observation.status_code,
                health_payload=health_observation.payload,
                health_error=health_observation.error,
                health_mismatches=health_mismatches,
                preflight_results=tuple(preflight_results),
                profile_results=tuple(profile_results),
                process_exit_code_before_stop=exit_code_before_stop,
                process_exit_code_after_stop=exit_code_after_stop,
                stopped=stopped,
                stop_confirmed=stop_confirmed,
                remote_stop_attempted=remote_stop_attempted,
                remote_stop_exit_code=remote_stop_exit_code,
                remote_stop_stdout=remote_stop_stdout,
                remote_stop_stderr=remote_stop_stderr,
                post_stop_health_reachable=post_stop_health_reachable,
                elapsed_seconds=max(0.0, time.monotonic() - started_at),
                stdout_tail=_read_tail(stdout),
                stderr_tail=_read_tail(stderr),
                error=error,
            )


def create_smoke_fixture(
    database_url: str,
    *,
    fixture_text: str,
    profile_names: tuple[str, ...],
) -> DgxEmbeddingWorkerSmokeFixture:
    smoke_run_key = f"dgx-worker-e2e-{int(time.time() * 1000)}"
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO files (
                    original_file_name,
                    stored_file_name,
                    file_ext,
                    file_size_bytes,
                    sha256_checksum,
                    storage_path
                )
                VALUES (%s, %s, '.md', %s, %s, %s)
                RETURNING file_id
                """,
                (
                    f"{smoke_run_key}.md",
                    f"{smoke_run_key}.stored.md",
                    len(fixture_text.encode("utf-8")),
                    smoke_run_key,
                    f"/tmp/nex_pcx/{smoke_run_key}.md",
                ),
            )
            file_id = int(cursor.fetchone()["file_id"])
            cursor.execute(
                """
                INSERT INTO documents (file_id, document_title)
                VALUES (%s, %s)
                RETURNING document_id
                """,
                (file_id, f"DGX worker E2E smoke {smoke_run_key}"),
            )
            document_id = int(cursor.fetchone()["document_id"])
            cursor.execute(
                """
                INSERT INTO chunks (
                    document_id,
                    chunk_seq,
                    chunk_text,
                    content_hash,
                    chunk_policy_name,
                    char_count,
                    token_count,
                    metadata
                )
                VALUES (%s, 0, %s, %s, 'heading_512_64', %s, %s, %s)
                RETURNING chunk_id
                """,
                (
                    document_id,
                    fixture_text,
                    f"chunk-{smoke_run_key}",
                    len(fixture_text),
                    max(1, len(fixture_text.split())),
                    Json({"script": "run_dgx_remote_embedding_worker_e2e_smoke.py"}),
                ),
            )
            chunk_id = int(cursor.fetchone()["chunk_id"])

    job_ids_by_profile: dict[str, int] = {}
    for profile_name in profile_names:
        result = create_embedding_job(
            database_url,
            EmbeddingJobInput(
                chunk_id=chunk_id,
                profile_name=profile_name,
                runtime_metadata={
                    "script": "run_dgx_remote_embedding_worker_e2e_smoke.py",
                    "smoke_run_key": smoke_run_key,
                },
            ),
        )
        job_ids_by_profile[profile_name] = result.job.job_id

    _prioritize_smoke_jobs(database_url, tuple(job_ids_by_profile.values()))
    return DgxEmbeddingWorkerSmokeFixture(
        smoke_run_key=smoke_run_key,
        file_id=file_id,
        document_id=document_id,
        chunk_id=chunk_id,
        job_ids_by_profile=job_ids_by_profile,
    )


def cleanup_smoke_fixture(database_url: str, file_id: int) -> bool:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
            cursor.execute("SELECT 1 FROM files WHERE file_id = %s", (file_id,))
            return cursor.fetchone() is None


def _prioritize_smoke_jobs(database_url: str, job_ids: tuple[int, ...]) -> None:
    if not job_ids:
        return
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE embedding_jobs
                SET created_at = %s::timestamptz,
                    updated_at = now()
                WHERE job_id = ANY(%s)
                """,
                (SMOKE_JOB_PRIORITY_TIMESTAMP, list(job_ids)),
            )


def _run_worker_profile_smoke(
    database_url: str,
    *,
    fixture: DgxEmbeddingWorkerSmokeFixture,
    profile_name: str,
    worker_name: str,
    lease_seconds: int,
    remote_timeout_seconds: float,
    readiness_gate_defer_seconds: int,
) -> DgxEmbeddingWorkerProfileSmokeResult:
    expected_job_id = fixture.job_ids_by_profile.get(profile_name)
    fallback_runtime_config = EmbeddingProviderRuntimeConfig(
        mode="remote",
        remote_base_url="http://127.0.0.1:1",
        remote_timeout_seconds=1.0,
    )
    try:
        result = process_next_embedding_job_with_provider_routes(
            database_url,
            worker_name=worker_name,
            profile_name=profile_name,
            lease_seconds=lease_seconds,
            fallback_runtime_config=fallback_runtime_config,
            require_route_readiness=True,
            readiness_gate_failure_mode="fail",
            readiness_gate_defer_seconds=readiness_gate_defer_seconds,
        )
    except Exception as exc:
        return DgxEmbeddingWorkerProfileSmokeResult(
            profile_name=profile_name,
            job_id=expected_job_id,
            chunk_id=fixture.chunk_id,
            processed=False,
            job_status=None,
            vector_table_name=None,
            vector_dimension=None,
            vector_storage_type=None,
            provider_route_id=None,
            provider_route_name=None,
            provider_runtime_base_url=None,
            provider_model_id=None,
            provider_type=None,
            provider_elapsed_ms=None,
            elapsed_ms=None,
            readiness_status=None,
            readiness_health_snapshot_id=None,
            readiness_contract_snapshot_id=None,
            message=None,
            error=str(exc),
        )

    job = result.job
    vector = (
        get_chunk_embedding(
            database_url,
            profile_name=profile_name,
            chunk_id=fixture.chunk_id,
        )
        if job is not None
        else None
    )
    error = _worker_profile_error(
        result,
        expected_job_id=expected_job_id,
        expected_chunk_id=fixture.chunk_id,
        vector=vector,
    )
    return _profile_result_from_worker_result(
        profile_name,
        result,
        vector=vector,
        error=error,
    )


def _worker_profile_error(
    result: EmbeddingWorkerResult,
    *,
    expected_job_id: int | None,
    expected_chunk_id: int,
    vector: EmbeddingVectorRecord | None,
) -> str | None:
    if not result.processed:
        return "Worker did not process a job."
    if result.job is None:
        return "Worker result did not include a job."
    if expected_job_id is not None and result.job.job_id != expected_job_id:
        return f"Worker claimed unexpected job_id={result.job.job_id}; expected {expected_job_id}."
    if result.job.chunk_id != expected_chunk_id:
        return (
            f"Worker claimed unexpected chunk_id={result.job.chunk_id}; "
            f"expected {expected_chunk_id}."
        )
    if result.job.status != "succeeded":
        return result.job.error_message or f"Worker finished with status {result.job.status}."
    if vector is None:
        return "Succeeded job did not persist a vector."
    return None


def _profile_result_from_worker_result(
    profile_name: str,
    result: EmbeddingWorkerResult,
    *,
    vector: EmbeddingVectorRecord | None,
    error: str | None,
) -> DgxEmbeddingWorkerProfileSmokeResult:
    job: EmbeddingJobRecord | None = result.job
    metadata = job.runtime_metadata if job is not None else {}
    return DgxEmbeddingWorkerProfileSmokeResult(
        profile_name=profile_name,
        job_id=job.job_id if job is not None else None,
        chunk_id=job.chunk_id if job is not None else None,
        processed=result.processed,
        job_status=job.status if job is not None else None,
        vector_table_name=vector.table_name if vector is not None else None,
        vector_dimension=vector.dimension if vector is not None else None,
        vector_storage_type=vector.storage_type if vector is not None else None,
        provider_route_id=_int_or_none(metadata.get("provider_route_id")),
        provider_route_name=_str_or_none(metadata.get("provider_route_name")),
        provider_runtime_base_url=_str_or_none(metadata.get("provider_runtime_base_url")),
        provider_model_id=_str_or_none(metadata.get("provider_model_id")),
        provider_type=_str_or_none(metadata.get("provider_type")),
        provider_elapsed_ms=_int_or_none(metadata.get("provider_elapsed_ms")),
        elapsed_ms=result.elapsed_ms,
        readiness_status=_str_or_none(metadata.get("provider_route_readiness_status")),
        readiness_health_snapshot_id=_int_or_none(
            metadata.get("provider_route_health_snapshot_id")
        ),
        readiness_contract_snapshot_id=_int_or_none(
            metadata.get("provider_route_contract_snapshot_id")
        ),
        message=result.message,
        error=error,
    )


def _provider_result(
    provider_plan: DgxRemoteEmbeddingWorkerProviderPlan,
    *,
    launch_command: tuple[str, ...],
    health_timeout_seconds: float,
    poll_interval_seconds: float,
    shutdown_timeout_seconds: float,
    pre_launch_health_reachable: bool,
    launched: bool,
    health_checked: bool,
    health_ok: bool,
    health_attempts: int,
    health_status_code: int | None,
    health_payload: dict[str, Any] | None,
    health_error: str | None,
    health_mismatches: tuple[str, ...],
    preflight_results: tuple[DgxProviderRoutePreflightProfileResult, ...],
    profile_results: tuple[DgxEmbeddingWorkerProfileSmokeResult, ...],
    process_exit_code_before_stop: int | None,
    process_exit_code_after_stop: int | None,
    stopped: bool,
    stop_confirmed: bool,
    remote_stop_attempted: bool,
    remote_stop_exit_code: int | None,
    remote_stop_stdout: str,
    remote_stop_stderr: str,
    post_stop_health_reachable: bool,
    elapsed_seconds: float,
    stdout_tail: str,
    stderr_tail: str,
    error: str | None,
) -> DgxRemoteEmbeddingWorkerProviderResult:
    foreground_plan = provider_plan.foreground_plan
    return DgxRemoteEmbeddingWorkerProviderResult(
        provider=provider_plan.provider,
        provider_name=foreground_plan.provider_name,
        base_url=foreground_plan.base_url,
        health_url=foreground_plan.health_url,
        launch_command=launch_command,
        profile_names=provider_plan.profile_names,
        startup_timeout_seconds=provider_plan.startup_timeout_seconds,
        health_timeout_seconds=health_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        shutdown_timeout_seconds=shutdown_timeout_seconds,
        pre_launch_health_reachable=pre_launch_health_reachable,
        launched=launched,
        health_checked=health_checked,
        health_ok=health_ok,
        health_attempts=health_attempts,
        health_status_code=health_status_code,
        health_payload=health_payload,
        health_error=health_error,
        health_mismatches=health_mismatches,
        preflight_results=preflight_results,
        profile_results=profile_results,
        process_exit_code_before_stop=process_exit_code_before_stop,
        process_exit_code_after_stop=process_exit_code_after_stop,
        stopped=stopped,
        stop_confirmed=stop_confirmed,
        remote_stop_attempted=remote_stop_attempted,
        remote_stop_exit_code=remote_stop_exit_code,
        remote_stop_stdout=remote_stop_stdout,
        remote_stop_stderr=remote_stop_stderr,
        post_stop_health_reachable=post_stop_health_reachable,
        elapsed_seconds=elapsed_seconds,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        error=error,
    )


def _plan_profile_names(plan: DgxRemoteEmbeddingWorkerSmokePlan) -> tuple[str, ...]:
    profile_names: list[str] = []
    for provider in plan.providers:
        profile_names.extend(provider.profile_names)
    return tuple(profile_names)


def _normalize_provider_names(provider_names: tuple[str, ...] | None) -> tuple[str, ...]:
    selected_names = provider_names or DEFAULT_PROVIDER_ORDER
    normalized_names = tuple(name.strip().lower() for name in selected_names if name.strip())
    if not normalized_names:
        raise InvalidEmbeddingProviderPresetError("At least one provider is required")
    seen: set[str] = set()
    unique_names: list[str] = []
    for name in normalized_names:
        get_embedding_provider_preset(name)
        if name not in seen:
            unique_names.append(name)
            seen.add(name)
    return tuple(unique_names)


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: object) -> str | None:
    return str(value) if value is not None else None


def _plan_payload(plan: DgxRemoteEmbeddingWorkerSmokePlan) -> dict[str, Any]:
    return {
        "database_url": _redact_database_url(plan.database_url),
        "providers": [_provider_plan_payload(provider) for provider in plan.providers],
        "fixture_text": "<fixture_text>",
        "preflight_before_worker": plan.preflight_before_worker,
        "active_only_preflight": plan.active_only_preflight,
        "cleanup_fixture": plan.cleanup_fixture,
        "worker_name_prefix": plan.worker_name_prefix,
        "lease_seconds": plan.lease_seconds,
        "remote_timeout_seconds": plan.remote_timeout_seconds,
        "readiness_gate_defer_seconds": plan.readiness_gate_defer_seconds,
        "health_timeout_seconds": plan.health_timeout_seconds,
        "poll_interval_seconds": plan.poll_interval_seconds,
        "shutdown_timeout_seconds": plan.shutdown_timeout_seconds,
        "fail_fast": plan.fail_fast,
    }


def _provider_plan_payload(
    provider_plan: DgxRemoteEmbeddingWorkerProviderPlan,
) -> dict[str, Any]:
    return {
        "provider": provider_plan.provider,
        "foreground_plan": asdict(provider_plan.foreground_plan),
        "profile_names": list(provider_plan.profile_names),
        "startup_timeout_seconds": provider_plan.startup_timeout_seconds,
    }


def _fixture_payload(fixture: DgxEmbeddingWorkerSmokeFixture | None) -> dict[str, Any] | None:
    if fixture is None:
        return None
    return {
        "smoke_run_key": fixture.smoke_run_key,
        "file_id": fixture.file_id,
        "document_id": fixture.document_id,
        "chunk_id": fixture.chunk_id,
        "job_ids_by_profile": dict(fixture.job_ids_by_profile),
    }


def _preflight_result_payload(result: DgxProviderRoutePreflightProfileResult) -> dict[str, Any]:
    return {**asdict(result), "passed": result.passed}


def _profile_result_payload(result: DgxEmbeddingWorkerProfileSmokeResult) -> dict[str, Any]:
    return {**asdict(result), "passed": result.passed}


def _provider_result_payload(result: DgxRemoteEmbeddingWorkerProviderResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["passed"] = result.passed
    payload["preflight_results"] = [
        _preflight_result_payload(preflight) for preflight in result.preflight_results
    ]
    payload["profile_results"] = [
        _profile_result_payload(profile) for profile in result.profile_results
    ]
    return payload


def _report_payload(report: DgxRemoteEmbeddingWorkerSmokeReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "plan": _plan_payload(report.plan),
        "fixture": _fixture_payload(report.fixture),
        "results": [_provider_result_payload(result) for result in report.results],
        "cleanup_attempted": report.cleanup_attempted,
        "cleanup_confirmed": report.cleanup_confirmed,
        "total_elapsed_seconds": report.total_elapsed_seconds,
    }


def _print_human_report(report: DgxRemoteEmbeddingWorkerSmokeReport) -> None:
    status = "PASS" if report.passed else "FAIL"
    print(f"DGX remote embedding worker E2E smoke: {status}")
    print(f"- providers: {len(report.results)}/{len(report.plan.providers)} executed")
    print(f"- cleanup_confirmed: {report.cleanup_confirmed}")
    print(f"- total_elapsed_seconds: {report.total_elapsed_seconds:.2f}")
    for result in report.results:
        print(
            f"- {result.provider}: "
            f"passed={result.passed} "
            f"health_ok={result.health_ok} "
            f"profiles={len(result.profile_results)}/{len(result.profile_names)}"
        )
        for profile in result.profile_results:
            print(
                f"  - {profile.profile_name}: "
                f"passed={profile.passed} "
                f"job={profile.job_id} "
                f"table={profile.vector_table_name} "
                f"dimension={profile.vector_dimension}"
            )
        if result.error:
            print(f"  - error: {result.error}")


def write_markdown_report(
    report: DgxRemoteEmbeddingWorkerSmokeReport,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_markdown_report(report), encoding="utf-8")


def _markdown_report(report: DgxRemoteEmbeddingWorkerSmokeReport) -> str:
    lines = [
        "# DGX Remote Embedding Worker E2E Smoke Result",
        "",
        f"- `passed`: `{str(report.passed).lower()}`",
        f"- `database_url`: `{_redact_database_url(report.plan.database_url)}`",
        f"- `preflight_before_worker`: `{str(report.plan.preflight_before_worker).lower()}`",
        f"- `cleanup_attempted`: `{str(report.cleanup_attempted).lower()}`",
        f"- `cleanup_confirmed`: `{str(report.cleanup_confirmed).lower()}`",
        f"- `total_elapsed_seconds`: `{report.total_elapsed_seconds:.2f}`",
        f"- providers executed: `{len(report.results)}`",
        "",
        "## Fixture",
        "",
    ]
    if report.fixture is None:
        lines.append("- fixture was not created")
    else:
        lines.extend(
            [
                f"- `smoke_run_key`: `{report.fixture.smoke_run_key}`",
                f"- `file_id`: `{report.fixture.file_id}`",
                f"- `document_id`: `{report.fixture.document_id}`",
                f"- `chunk_id`: `{report.fixture.chunk_id}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Provider Results",
            "",
            "| Provider | Passed | Base URL | Health | Profiles | Error |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for result in report.results:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{result.provider}`",
                    f"`{str(result.passed).lower()}`",
                    f"`{result.base_url}`",
                    f"`{str(result.health_ok).lower()}`",
                    f"`{len(result.profile_results)}/{len(result.profile_names)}`",
                    f"`{result.error or ''}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Worker Profile Results",
            "",
            (
                "| Provider | Profile | Passed | Job | Status | Vector Table | Dimension | "
                "Route | Health Snapshot | Contract Snapshot | Provider Model | "
                "Provider ms | Worker ms | Error |"
            ),
            (
                "| --- | --- | --- | ---: | --- | --- | ---: | --- | ---: | ---: | "
                "--- | ---: | ---: | --- |"
            ),
        ]
    )
    for result in report.results:
        for profile in result.profile_results:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{result.provider}`",
                        f"`{profile.profile_name}`",
                        f"`{str(profile.passed).lower()}`",
                        f"`{profile.job_id or ''}`",
                        f"`{profile.job_status or ''}`",
                        f"`{profile.vector_table_name or ''}`",
                        f"`{profile.vector_dimension or ''}`",
                        f"`{profile.provider_route_id or ''}`",
                        f"`{profile.readiness_health_snapshot_id or ''}`",
                        f"`{profile.readiness_contract_snapshot_id or ''}`",
                        f"`{profile.provider_model_id or ''}`",
                        f"`{profile.provider_elapsed_ms or ''}`",
                        f"`{profile.elapsed_ms or ''}`",
                        f"`{profile.error or ''}`",
                    ]
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    preset_names = [preset.preset_name for preset in list_embedding_provider_presets()]
    parser = argparse.ArgumentParser(
        description=(
            "Launch DGX remote providers sequentially and verify that route-aware "
            "embedding workers persist vectors for a smoke fixture chunk."
        ),
    )
    parser.add_argument("--provider", choices=preset_names, action="append", default=[])
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--host", default=DEFAULT_GPU_HOST)
    parser.add_argument("--ssh-user", default=DEFAULT_GPU_USER)
    parser.add_argument("--workdir", default=DEFAULT_GPU_WORKDIR)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--python-bin", default=None)
    parser.add_argument("--provider-host", default="0.0.0.0")
    parser.add_argument("--route-host", default=None)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=DEFAULT_STARTUP_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--health-timeout-seconds",
        type=float,
        default=DEFAULT_HEALTH_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--shutdown-timeout-seconds",
        type=float,
        default=DEFAULT_SHUTDOWN_TIMEOUT_SECONDS,
    )
    parser.add_argument("--fixture-text", default=DEFAULT_FIXTURE_TEXT)
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip profile-scoped route preflight before processing worker jobs.",
    )
    parser.add_argument(
        "--include-inactive-preflight",
        action="store_true",
        help="Include inactive routes when preflight is enabled.",
    )
    parser.add_argument(
        "--keep-fixture",
        action="store_true",
        help="Leave the smoke fixture rows in the database after the run.",
    )
    parser.add_argument("--worker-name-prefix", default="dgx-worker-e2e")
    parser.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    parser.add_argument(
        "--remote-timeout-seconds",
        type=float,
        default=DEFAULT_REMOTE_TIMEOUT_SECONDS,
    )
    parser.add_argument("--readiness-gate-defer-seconds", type=int, default=300)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown-output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    settings = get_settings()
    database_url = args.database_url or settings.database_url
    if not database_url:
        parser.error("--database-url or NEX_PCX_DATABASE_URL is required")

    try:
        plan = build_dgx_remote_embedding_worker_smoke_plan(
            provider_names=tuple(args.provider) or None,
            database_url=database_url,
            host=args.host,
            ssh_user=args.ssh_user,
            workdir=args.workdir,
            models_dir=args.models_dir,
            python_bin=args.python_bin,
            provider_host=args.provider_host,
            route_host=args.route_host,
            device=args.device,
            startup_timeout_seconds=args.startup_timeout_seconds,
            health_timeout_seconds=args.health_timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
            shutdown_timeout_seconds=args.shutdown_timeout_seconds,
            fixture_text=args.fixture_text,
            preflight_before_worker=not args.skip_preflight,
            active_only_preflight=not args.include_inactive_preflight,
            cleanup_fixture=not args.keep_fixture,
            worker_name_prefix=args.worker_name_prefix,
            lease_seconds=args.lease_seconds,
            remote_timeout_seconds=args.remote_timeout_seconds,
            readiness_gate_defer_seconds=args.readiness_gate_defer_seconds,
            fail_fast=args.fail_fast,
        )
    except (InvalidEmbeddingProviderPresetError, ValueError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        payload = {"dry_run": True, "plan": _plan_payload(plan)}
        print(json.dumps(payload, ensure_ascii=False, indent=None if args.json else 2))
        return 0

    report = run_dgx_remote_embedding_worker_smoke(plan)
    if args.markdown_output:
        write_markdown_report(report, Path(args.markdown_output))

    if args.json:
        print(json.dumps(_report_payload(report), ensure_ascii=False))
    else:
        _print_human_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
