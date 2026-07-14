"""Run a DGX-backed small corpus ingestion benchmark across embedding profiles."""

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
from app.core.dgx_ingestion_benchmarks import (  # noqa: E402
    DgxIngestionBenchmarkDetail,
    DgxIngestionBenchmarkInput,
    DgxIngestionBenchmarkJobInput,
    DgxIngestionBenchmarkProfileInput,
    record_dgx_ingestion_benchmark,
)
from app.core.embedding_jobs import EmbeddingJobInput, create_embedding_job  # noqa: E402
from app.core.embedding_provider_presets import (  # noqa: E402
    InvalidEmbeddingProviderPresetError,
    get_embedding_provider_preset,
    list_embedding_provider_presets,
)
from app.core.embedding_providers import EmbeddingProviderRuntimeConfig  # noqa: E402
from app.core.embedding_vectors import get_chunk_embedding  # noqa: E402
from app.core.embedding_worker import process_next_embedding_job_with_provider_routes  # noqa: E402
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
from scripts.run_dgx_remote_embedding_worker_e2e_smoke import (  # noqa: E402
    DEFAULT_REMOTE_TIMEOUT_SECONDS,
    DEFAULT_STARTUP_TIMEOUT_SECONDS,
    SMOKE_JOB_PRIORITY_TIMESTAMP,
    DgxEmbeddingWorkerProfileSmokeResult,
    _normalize_provider_names,
    _profile_result_from_worker_result,
    _worker_profile_error,
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

DEFAULT_CHUNK_COUNT = 3
MAX_CHUNK_COUNT = 50
DEFAULT_WORKER_NAME_PREFIX = "dgx-small-corpus-benchmark"


@dataclass(frozen=True)
class DgxSmallCorpusBenchmarkProviderPlan:
    provider: str
    foreground_plan: RemoteProviderForegroundSmokePlan
    profile_names: tuple[str, ...]
    startup_timeout_seconds: float


@dataclass(frozen=True)
class DgxSmallCorpusBenchmarkPlan:
    database_url: str
    providers: tuple[DgxSmallCorpusBenchmarkProviderPlan, ...]
    corpus_texts: tuple[str, ...]
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

    @property
    def chunk_count(self) -> int:
        return len(self.corpus_texts)


@dataclass(frozen=True)
class DgxSmallCorpusJobTarget:
    chunk_index: int
    chunk_id: int
    job_id: int


@dataclass(frozen=True)
class DgxSmallCorpusBenchmarkFixture:
    benchmark_run_key: str
    file_id: int
    document_id: int
    chunk_ids: tuple[int, ...]
    job_targets_by_profile: dict[str, tuple[DgxSmallCorpusJobTarget, ...]]

    @property
    def job_count(self) -> int:
        return sum(len(targets) for targets in self.job_targets_by_profile.values())


@dataclass(frozen=True)
class DgxSmallCorpusProfileBenchmarkSummary:
    profile_name: str
    expected_job_count: int
    processed_count: int
    succeeded_count: int
    failed_count: int
    vector_count: int
    vector_table_name: str | None
    vector_dimension: int | None
    vector_storage_type: str | None
    provider_route_id: int | None
    provider_route_name: str | None
    provider_runtime_base_url: str | None
    provider_model_id: str | None
    provider_type: str | None
    readiness_status: str | None
    readiness_health_snapshot_id: int | None
    readiness_contract_snapshot_id: int | None
    total_provider_elapsed_ms: int | None
    avg_provider_elapsed_ms: float | None
    max_provider_elapsed_ms: int | None
    total_worker_elapsed_ms: int | None
    avg_worker_elapsed_ms: float | None
    max_worker_elapsed_ms: int | None
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            self.expected_job_count > 0
            and self.processed_count == self.expected_job_count
            and self.succeeded_count == self.expected_job_count
            and self.failed_count == 0
            and self.vector_count == self.expected_job_count
            and self.provider_type == "remote"
            and not self.errors
        )


@dataclass(frozen=True)
class DgxSmallCorpusProviderBenchmarkResult:
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
    profile_summaries: tuple[DgxSmallCorpusProfileBenchmarkSummary, ...]
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
            and bool(self.profile_summaries)
            and all(summary.passed for summary in self.profile_summaries)
            and self.stopped
            and self.stop_confirmed
            and (not self.remote_stop_attempted or self.remote_stop_exit_code == 0)
            and not self.post_stop_health_reachable
            and self.error is None
        )


@dataclass(frozen=True)
class DgxSmallCorpusBenchmarkReport:
    plan: DgxSmallCorpusBenchmarkPlan
    fixture: DgxSmallCorpusBenchmarkFixture | None
    results: tuple[DgxSmallCorpusProviderBenchmarkResult, ...]
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


def build_dgx_small_corpus_benchmark_plan(
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
    chunk_count: int = DEFAULT_CHUNK_COUNT,
    corpus_texts: tuple[str, ...] | None = None,
    preflight_before_worker: bool = True,
    active_only_preflight: bool = True,
    cleanup_fixture: bool = True,
    worker_name_prefix: str = DEFAULT_WORKER_NAME_PREFIX,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    remote_timeout_seconds: float = DEFAULT_REMOTE_TIMEOUT_SECONDS,
    readiness_gate_defer_seconds: int = 300,
    fail_fast: bool = False,
) -> DgxSmallCorpusBenchmarkPlan:
    if not database_url.strip():
        raise ValueError("database_url is required")
    if not worker_name_prefix.strip():
        raise ValueError("worker_name_prefix is required")
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be greater than 0")
    if readiness_gate_defer_seconds <= 0:
        raise ValueError("readiness_gate_defer_seconds must be greater than 0")

    texts = corpus_texts or _default_corpus_texts(chunk_count)
    _validate_corpus_texts(texts)

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

    provider_plans: list[DgxSmallCorpusBenchmarkProviderPlan] = []
    for provider_name in _normalize_provider_names(provider_names):
        foreground_plan = build_foreground_smoke_plan(
            get_embedding_provider_preset(provider_name),
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
            DgxSmallCorpusBenchmarkProviderPlan(
                provider=provider_name,
                foreground_plan=foreground_plan,
                profile_names=tuple(foreground_plan.profile_names),
                startup_timeout_seconds=startup_timeout_seconds,
            )
        )

    return DgxSmallCorpusBenchmarkPlan(
        database_url=database_url,
        providers=tuple(provider_plans),
        corpus_texts=tuple(text.strip() for text in texts),
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


def run_dgx_small_corpus_benchmark(
    plan: DgxSmallCorpusBenchmarkPlan,
) -> DgxSmallCorpusBenchmarkReport:
    started_at = time.monotonic()
    fixture: DgxSmallCorpusBenchmarkFixture | None = None
    cleanup_attempted = False
    cleanup_confirmed = False
    results: list[DgxSmallCorpusProviderBenchmarkResult] = []
    try:
        fixture = create_small_corpus_fixture(
            plan.database_url,
            corpus_texts=plan.corpus_texts,
            profile_names=_plan_profile_names(plan),
        )
        for provider_plan in plan.providers:
            result = run_provider_small_corpus_benchmark_session(
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
            cleanup_confirmed = cleanup_small_corpus_fixture(
                plan.database_url,
                fixture.file_id,
            )

    return DgxSmallCorpusBenchmarkReport(
        plan=plan,
        fixture=fixture,
        results=tuple(results),
        cleanup_attempted=cleanup_attempted,
        cleanup_confirmed=cleanup_confirmed,
        total_elapsed_seconds=max(0.0, time.monotonic() - started_at),
    )


def run_provider_small_corpus_benchmark_session(
    provider_plan: DgxSmallCorpusBenchmarkProviderPlan,
    *,
    database_url: str,
    fixture: DgxSmallCorpusBenchmarkFixture,
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
) -> DgxSmallCorpusProviderBenchmarkResult:
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
    profile_summaries: list[DgxSmallCorpusProfileBenchmarkSummary] = []
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
            profile_summaries=(),
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
                    profile_summaries=(),
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
                    error = "One or more route preflight checks failed before benchmark."

            if health_observation.ok and not health_mismatches and error is None:
                for profile_name in provider_plan.profile_names:
                    summary, results = _run_worker_profile_benchmark(
                        database_url,
                        fixture=fixture,
                        profile_name=profile_name,
                        worker_name_prefix=worker_name_prefix,
                        lease_seconds=lease_seconds,
                        remote_timeout_seconds=remote_timeout_seconds,
                        readiness_gate_defer_seconds=readiness_gate_defer_seconds,
                        fail_fast=fail_fast,
                    )
                    profile_summaries.append(summary)
                    profile_results.extend(results)
                    if fail_fast and not summary.passed:
                        break
                if any(not summary.passed for summary in profile_summaries) and error is None:
                    error = "One or more small corpus benchmark profiles failed."

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
                profile_summaries=tuple(profile_summaries),
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


def create_small_corpus_fixture(
    database_url: str,
    *,
    corpus_texts: tuple[str, ...],
    profile_names: tuple[str, ...],
) -> DgxSmallCorpusBenchmarkFixture:
    _validate_corpus_texts(corpus_texts)
    benchmark_run_key = f"dgx-small-corpus-{int(time.time() * 1000)}"
    chunk_ids: list[int] = []
    total_size_bytes = sum(len(text.encode("utf-8")) for text in corpus_texts)
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
                    f"{benchmark_run_key}.md",
                    f"{benchmark_run_key}.stored.md",
                    total_size_bytes,
                    benchmark_run_key,
                    f"/tmp/nex_pcx/{benchmark_run_key}.md",
                ),
            )
            file_id = int(cursor.fetchone()["file_id"])
            cursor.execute(
                """
                INSERT INTO documents (file_id, document_title)
                VALUES (%s, %s)
                RETURNING document_id
                """,
                (file_id, f"DGX small corpus benchmark {benchmark_run_key}"),
            )
            document_id = int(cursor.fetchone()["document_id"])
            for chunk_index, chunk_text in enumerate(corpus_texts):
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
                    VALUES (%s, %s, %s, %s, 'heading_512_64', %s, %s, %s)
                    RETURNING chunk_id
                    """,
                    (
                        document_id,
                        chunk_index,
                        chunk_text,
                        f"chunk-{benchmark_run_key}-{chunk_index}",
                        len(chunk_text),
                        max(1, len(chunk_text.split())),
                        Json(
                            {
                                "script": ("run_dgx_small_corpus_embedding_benchmark.py"),
                                "benchmark_run_key": benchmark_run_key,
                                "chunk_index": chunk_index,
                            }
                        ),
                    ),
                )
                chunk_ids.append(int(cursor.fetchone()["chunk_id"]))

    job_targets_by_profile: dict[str, tuple[DgxSmallCorpusJobTarget, ...]] = {}
    job_ids: list[int] = []
    for profile_name in profile_names:
        targets: list[DgxSmallCorpusJobTarget] = []
        for chunk_index, chunk_id in enumerate(chunk_ids):
            result = create_embedding_job(
                database_url,
                EmbeddingJobInput(
                    chunk_id=chunk_id,
                    profile_name=profile_name,
                    runtime_metadata={
                        "script": "run_dgx_small_corpus_embedding_benchmark.py",
                        "benchmark_run_key": benchmark_run_key,
                        "chunk_index": chunk_index,
                    },
                ),
            )
            target = DgxSmallCorpusJobTarget(
                chunk_index=chunk_index,
                chunk_id=chunk_id,
                job_id=result.job.job_id,
            )
            targets.append(target)
            job_ids.append(target.job_id)
        job_targets_by_profile[profile_name] = tuple(targets)

    _prioritize_benchmark_jobs(database_url, tuple(job_ids))
    return DgxSmallCorpusBenchmarkFixture(
        benchmark_run_key=benchmark_run_key,
        file_id=file_id,
        document_id=document_id,
        chunk_ids=tuple(chunk_ids),
        job_targets_by_profile=job_targets_by_profile,
    )


def cleanup_small_corpus_fixture(database_url: str, file_id: int) -> bool:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))
            cursor.execute("SELECT 1 FROM files WHERE file_id = %s", (file_id,))
            return cursor.fetchone() is None


def _prioritize_benchmark_jobs(database_url: str, job_ids: tuple[int, ...]) -> None:
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


def _run_worker_profile_benchmark(
    database_url: str,
    *,
    fixture: DgxSmallCorpusBenchmarkFixture,
    profile_name: str,
    worker_name_prefix: str,
    lease_seconds: int,
    remote_timeout_seconds: float,
    readiness_gate_defer_seconds: int,
    fail_fast: bool,
) -> tuple[
    DgxSmallCorpusProfileBenchmarkSummary,
    tuple[DgxEmbeddingWorkerProfileSmokeResult, ...],
]:
    targets = fixture.job_targets_by_profile.get(profile_name, ())
    results: list[DgxEmbeddingWorkerProfileSmokeResult] = []
    for target in targets:
        result = _run_worker_profile_target(
            database_url,
            target=target,
            profile_name=profile_name,
            worker_name=(f"{worker_name_prefix}-{profile_name}-chunk-{target.chunk_index + 1}"),
            lease_seconds=lease_seconds,
            remote_timeout_seconds=remote_timeout_seconds,
            readiness_gate_defer_seconds=readiness_gate_defer_seconds,
        )
        results.append(result)
        if fail_fast and not result.passed:
            break
    return (
        _summarize_profile_benchmark(profile_name, targets=targets, results=tuple(results)),
        tuple(results),
    )


def _run_worker_profile_target(
    database_url: str,
    *,
    target: DgxSmallCorpusJobTarget,
    profile_name: str,
    worker_name: str,
    lease_seconds: int,
    remote_timeout_seconds: float,
    readiness_gate_defer_seconds: int,
) -> DgxEmbeddingWorkerProfileSmokeResult:
    fallback_runtime_config = EmbeddingProviderRuntimeConfig(
        mode="remote",
        remote_base_url="http://127.0.0.1:1",
        remote_timeout_seconds=min(remote_timeout_seconds, 1.0),
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
            job_id=target.job_id,
            chunk_id=target.chunk_id,
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

    vector = (
        get_chunk_embedding(
            database_url,
            profile_name=profile_name,
            chunk_id=target.chunk_id,
        )
        if result.job is not None
        else None
    )
    error = _worker_profile_error(
        result,
        expected_job_id=target.job_id,
        expected_chunk_id=target.chunk_id,
        vector=vector,
    )
    return _profile_result_from_worker_result(
        profile_name,
        result,
        vector=vector,
        error=error,
    )


def _summarize_profile_benchmark(
    profile_name: str,
    *,
    targets: tuple[DgxSmallCorpusJobTarget, ...],
    results: tuple[DgxEmbeddingWorkerProfileSmokeResult, ...],
) -> DgxSmallCorpusProfileBenchmarkSummary:
    expected_job_count = len(targets)
    processed_count = sum(1 for result in results if result.processed)
    succeeded_count = sum(1 for result in results if result.job_status == "succeeded")
    failed_count = sum(1 for result in results if result.job_status == "failed")
    vector_count = sum(1 for result in results if result.vector_table_name is not None)
    provider_elapsed_values = tuple(
        value for value in (result.provider_elapsed_ms for result in results) if value is not None
    )
    worker_elapsed_values = tuple(
        value for value in (result.elapsed_ms for result in results) if value is not None
    )
    first_evidence = next(
        (
            result
            for result in results
            if result.vector_table_name is not None or result.provider_runtime_base_url
        ),
        None,
    )
    errors = tuple(result.error for result in results if result.error)
    if expected_job_count == 0:
        errors = (*errors, f"No benchmark jobs were created for {profile_name}.")
    if len(results) < expected_job_count:
        errors = (
            *errors,
            f"Only {len(results)} of {expected_job_count} benchmark jobs were processed.",
        )
    if vector_count < succeeded_count:
        errors = (
            *errors,
            f"Only {vector_count} vectors were found for {succeeded_count} succeeded jobs.",
        )

    return DgxSmallCorpusProfileBenchmarkSummary(
        profile_name=profile_name,
        expected_job_count=expected_job_count,
        processed_count=processed_count,
        succeeded_count=succeeded_count,
        failed_count=failed_count,
        vector_count=vector_count,
        vector_table_name=first_evidence.vector_table_name if first_evidence else None,
        vector_dimension=first_evidence.vector_dimension if first_evidence else None,
        vector_storage_type=first_evidence.vector_storage_type if first_evidence else None,
        provider_route_id=first_evidence.provider_route_id if first_evidence else None,
        provider_route_name=first_evidence.provider_route_name if first_evidence else None,
        provider_runtime_base_url=(
            first_evidence.provider_runtime_base_url if first_evidence else None
        ),
        provider_model_id=first_evidence.provider_model_id if first_evidence else None,
        provider_type=first_evidence.provider_type if first_evidence else None,
        readiness_status=first_evidence.readiness_status if first_evidence else None,
        readiness_health_snapshot_id=(
            first_evidence.readiness_health_snapshot_id if first_evidence else None
        ),
        readiness_contract_snapshot_id=(
            first_evidence.readiness_contract_snapshot_id if first_evidence else None
        ),
        total_provider_elapsed_ms=_sum_or_none(provider_elapsed_values),
        avg_provider_elapsed_ms=_avg_or_none(provider_elapsed_values),
        max_provider_elapsed_ms=max(provider_elapsed_values) if provider_elapsed_values else None,
        total_worker_elapsed_ms=_sum_or_none(worker_elapsed_values),
        avg_worker_elapsed_ms=_avg_or_none(worker_elapsed_values),
        max_worker_elapsed_ms=max(worker_elapsed_values) if worker_elapsed_values else None,
        errors=errors,
    )


def _provider_result(
    provider_plan: DgxSmallCorpusBenchmarkProviderPlan,
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
    profile_summaries: tuple[DgxSmallCorpusProfileBenchmarkSummary, ...],
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
) -> DgxSmallCorpusProviderBenchmarkResult:
    foreground_plan = provider_plan.foreground_plan
    return DgxSmallCorpusProviderBenchmarkResult(
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
        profile_summaries=profile_summaries,
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


def _default_corpus_texts(chunk_count: int) -> tuple[str, ...]:
    if chunk_count <= 0:
        raise ValueError("chunk_count must be greater than 0")
    if chunk_count > MAX_CHUNK_COUNT:
        raise ValueError(f"chunk_count must be less than or equal to {MAX_CHUNK_COUNT}")
    base_texts = (
        (
            "NeX-PCX benchmark chunk about route-aware embedding ingestion, "
            "provider readiness, and vector persistence."
        ),
        (
            "NeX-PCX benchmark chunk about Korean document search experiments, "
            "permission scope, and repeatable golden question evaluation."
        ),
        (
            "NeX-PCX benchmark chunk about GPU remote providers, elapsed metadata, "
            "and operational queue observability."
        ),
        (
            "NeX-PCX benchmark chunk about chunk policy choices, overlap, storage cost, "
            "and retrieval quality comparison."
        ),
    )
    return tuple(
        f"{base_texts[index % len(base_texts)]} Corpus item {index + 1}."
        for index in range(chunk_count)
    )


def _validate_corpus_texts(corpus_texts: tuple[str, ...]) -> None:
    if not corpus_texts:
        raise ValueError("at least one corpus text is required")
    if len(corpus_texts) > MAX_CHUNK_COUNT:
        raise ValueError(f"chunk_count must be less than or equal to {MAX_CHUNK_COUNT}")
    if any(not text.strip() for text in corpus_texts):
        raise ValueError("corpus texts must not be blank")


def _plan_profile_names(plan: DgxSmallCorpusBenchmarkPlan) -> tuple[str, ...]:
    profile_names: list[str] = []
    for provider in plan.providers:
        profile_names.extend(provider.profile_names)
    return tuple(profile_names)


def _sum_or_none(values: tuple[int, ...]) -> int | None:
    return sum(values) if values else None


def _avg_or_none(values: tuple[int, ...]) -> float | None:
    return sum(values) / len(values) if values else None


def _float_or_empty(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else ""


def _int_or_empty(value: int | None) -> str:
    return str(value) if value is not None else ""


def _plan_payload(plan: DgxSmallCorpusBenchmarkPlan) -> dict[str, Any]:
    return {
        "database_url": _redact_database_url(plan.database_url),
        "providers": [_provider_plan_payload(provider) for provider in plan.providers],
        "chunk_count": plan.chunk_count,
        "corpus_texts": ["<corpus_text>" for _ in plan.corpus_texts],
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
    provider_plan: DgxSmallCorpusBenchmarkProviderPlan,
) -> dict[str, Any]:
    return {
        "provider": provider_plan.provider,
        "foreground_plan": asdict(provider_plan.foreground_plan),
        "profile_names": list(provider_plan.profile_names),
        "startup_timeout_seconds": provider_plan.startup_timeout_seconds,
    }


def _fixture_payload(
    fixture: DgxSmallCorpusBenchmarkFixture | None,
) -> dict[str, Any] | None:
    if fixture is None:
        return None
    return {
        "benchmark_run_key": fixture.benchmark_run_key,
        "file_id": fixture.file_id,
        "document_id": fixture.document_id,
        "chunk_ids": list(fixture.chunk_ids),
        "job_targets_by_profile": {
            profile_name: [asdict(target) for target in targets]
            for profile_name, targets in fixture.job_targets_by_profile.items()
        },
        "job_count": fixture.job_count,
    }


def _preflight_result_payload(result: DgxProviderRoutePreflightProfileResult) -> dict[str, Any]:
    return {**asdict(result), "passed": result.passed}


def _profile_result_payload(result: DgxEmbeddingWorkerProfileSmokeResult) -> dict[str, Any]:
    return {**asdict(result), "passed": result.passed}


def _profile_summary_payload(
    summary: DgxSmallCorpusProfileBenchmarkSummary,
) -> dict[str, Any]:
    return {**asdict(summary), "passed": summary.passed}


def _provider_result_payload(result: DgxSmallCorpusProviderBenchmarkResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["passed"] = result.passed
    payload["preflight_results"] = [
        _preflight_result_payload(preflight) for preflight in result.preflight_results
    ]
    payload["profile_summaries"] = [
        _profile_summary_payload(summary) for summary in result.profile_summaries
    ]
    payload["profile_results"] = [
        _profile_result_payload(profile) for profile in result.profile_results
    ]
    return payload


def _report_payload(report: DgxSmallCorpusBenchmarkReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "plan": _plan_payload(report.plan),
        "fixture": _fixture_payload(report.fixture),
        "results": [_provider_result_payload(result) for result in report.results],
        "cleanup_attempted": report.cleanup_attempted,
        "cleanup_confirmed": report.cleanup_confirmed,
        "total_elapsed_seconds": report.total_elapsed_seconds,
    }


def persist_dgx_small_corpus_benchmark_report(
    database_url: str,
    report: DgxSmallCorpusBenchmarkReport,
    *,
    created_by: str | None = None,
    created_by_user_id: int | None = None,
) -> DgxIngestionBenchmarkDetail:
    benchmark_input = build_dgx_ingestion_benchmark_input_from_report(
        report,
        created_by=created_by,
        created_by_user_id=created_by_user_id,
    )
    return record_dgx_ingestion_benchmark(database_url, benchmark_input)


def build_dgx_ingestion_benchmark_input_from_report(
    report: DgxSmallCorpusBenchmarkReport,
    *,
    created_by: str | None = None,
    created_by_user_id: int | None = None,
) -> DgxIngestionBenchmarkInput:
    profile_inputs: list[DgxIngestionBenchmarkProfileInput] = []
    for provider_result in report.results:
        job_results_by_profile: dict[str, list[DgxEmbeddingWorkerProfileSmokeResult]] = {}
        for job_result in provider_result.profile_results:
            job_results_by_profile.setdefault(job_result.profile_name, []).append(job_result)
        for summary in provider_result.profile_summaries:
            job_inputs = tuple(
                _benchmark_job_input_from_profile_result(
                    provider_result.provider,
                    job_result,
                )
                for job_result in job_results_by_profile.get(summary.profile_name, [])
            )
            profile_inputs.append(
                DgxIngestionBenchmarkProfileInput(
                    provider=provider_result.provider,
                    profile_name=summary.profile_name,
                    expected_job_count=summary.expected_job_count,
                    processed_count=summary.processed_count,
                    succeeded_count=summary.succeeded_count,
                    failed_count=summary.failed_count,
                    vector_count=summary.vector_count,
                    passed=summary.passed,
                    vector_table_name=summary.vector_table_name,
                    vector_dimension=summary.vector_dimension,
                    vector_storage_type=summary.vector_storage_type,
                    provider_route_id=summary.provider_route_id,
                    provider_route_name=summary.provider_route_name,
                    provider_runtime_base_url=summary.provider_runtime_base_url,
                    provider_model_id=summary.provider_model_id,
                    provider_type=summary.provider_type,
                    readiness_status=summary.readiness_status,
                    readiness_health_snapshot_id=summary.readiness_health_snapshot_id,
                    readiness_contract_snapshot_id=summary.readiness_contract_snapshot_id,
                    total_provider_elapsed_ms=summary.total_provider_elapsed_ms,
                    avg_provider_elapsed_ms=summary.avg_provider_elapsed_ms,
                    max_provider_elapsed_ms=summary.max_provider_elapsed_ms,
                    total_worker_elapsed_ms=summary.total_worker_elapsed_ms,
                    avg_worker_elapsed_ms=summary.avg_worker_elapsed_ms,
                    max_worker_elapsed_ms=summary.max_worker_elapsed_ms,
                    errors=summary.errors,
                    jobs=job_inputs,
                )
            )

    provider_names = tuple(provider.provider for provider in report.plan.providers)
    profile_names = _plan_profile_names(report.plan)
    fixture_payload = _fixture_payload(report.fixture) or {}
    return DgxIngestionBenchmarkInput(
        benchmark_run_key=(
            report.fixture.benchmark_run_key
            if report.fixture is not None
            else f"dgx-small-corpus-report-{int(time.time() * 1000)}"
        ),
        script_name=Path(__file__).name,
        provider_names=provider_names,
        profile_names=profile_names,
        chunk_count=report.plan.chunk_count,
        expected_job_count=sum(profile.expected_job_count for profile in profile_inputs),
        processed_count=sum(profile.processed_count for profile in profile_inputs),
        succeeded_count=sum(profile.succeeded_count for profile in profile_inputs),
        failed_count=sum(profile.failed_count for profile in profile_inputs),
        vector_count=sum(profile.vector_count for profile in profile_inputs),
        passed=report.passed,
        preflight_before_worker=report.plan.preflight_before_worker,
        active_only_preflight=report.plan.active_only_preflight,
        cleanup_attempted=report.cleanup_attempted,
        cleanup_confirmed=report.cleanup_confirmed,
        total_elapsed_seconds=report.total_elapsed_seconds,
        total_provider_elapsed_ms=_sum_optional_ints(
            profile.total_provider_elapsed_ms for profile in profile_inputs
        ),
        total_worker_elapsed_ms=_sum_optional_ints(
            profile.total_worker_elapsed_ms for profile in profile_inputs
        ),
        fixture_file_id=report.fixture.file_id if report.fixture is not None else None,
        fixture_document_id=report.fixture.document_id if report.fixture is not None else None,
        fixture_chunk_ids=report.fixture.chunk_ids if report.fixture is not None else (),
        plan_payload=_plan_payload(report.plan),
        fixture_payload=fixture_payload,
        report_payload=_report_payload(report),
        created_by=created_by,
        created_by_user_id=created_by_user_id,
        profiles=tuple(profile_inputs),
    )


def _benchmark_job_input_from_profile_result(
    provider: str,
    result: DgxEmbeddingWorkerProfileSmokeResult,
) -> DgxIngestionBenchmarkJobInput:
    return DgxIngestionBenchmarkJobInput(
        provider=provider,
        profile_name=result.profile_name,
        source_job_id=result.job_id,
        source_chunk_id=result.chunk_id,
        processed=result.processed,
        job_status=result.job_status,
        vector_table_name=result.vector_table_name,
        vector_dimension=result.vector_dimension,
        vector_storage_type=result.vector_storage_type,
        provider_route_id=result.provider_route_id,
        provider_route_name=result.provider_route_name,
        provider_runtime_base_url=result.provider_runtime_base_url,
        provider_model_id=result.provider_model_id,
        provider_type=result.provider_type,
        provider_elapsed_ms=result.provider_elapsed_ms,
        worker_elapsed_ms=result.elapsed_ms,
        readiness_status=result.readiness_status,
        readiness_health_snapshot_id=result.readiness_health_snapshot_id,
        readiness_contract_snapshot_id=result.readiness_contract_snapshot_id,
        message=result.message,
        error=result.error,
        passed=result.passed,
    )


def _sum_optional_ints(values: Any) -> int | None:
    normalized_values = tuple(value for value in values if value is not None)
    return sum(normalized_values) if normalized_values else None


def _persistence_payload(detail: DgxIngestionBenchmarkDetail | None) -> dict[str, Any] | None:
    if detail is None:
        return None
    return {
        "benchmark_run_id": detail.run.benchmark_run_id,
        "benchmark_run_key": detail.run.benchmark_run_key,
        "profile_count": len(detail.profiles),
        "job_result_count": len(detail.jobs),
    }


def _print_human_report(report: DgxSmallCorpusBenchmarkReport) -> None:
    status = "PASS" if report.passed else "FAIL"
    print(f"DGX small corpus ingestion benchmark: {status}")
    print(f"- chunks: {report.plan.chunk_count}")
    print(f"- providers: {len(report.results)}/{len(report.plan.providers)} executed")
    print(f"- cleanup_confirmed: {report.cleanup_confirmed}")
    print(f"- total_elapsed_seconds: {report.total_elapsed_seconds:.2f}")
    for result in report.results:
        print(
            f"- {result.provider}: "
            f"passed={result.passed} "
            f"health_ok={result.health_ok} "
            f"profiles={len(result.profile_summaries)}/{len(result.profile_names)}"
        )
        for summary in result.profile_summaries:
            print(
                f"  - {summary.profile_name}: "
                f"passed={summary.passed} "
                f"jobs={summary.succeeded_count}/{summary.expected_job_count} "
                f"vectors={summary.vector_count} "
                f"avg_provider_ms={_float_or_empty(summary.avg_provider_elapsed_ms)} "
                f"avg_worker_ms={_float_or_empty(summary.avg_worker_elapsed_ms)}"
            )
        if result.error:
            print(f"  - error: {result.error}")


def write_markdown_report(
    report: DgxSmallCorpusBenchmarkReport,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_markdown_report(report), encoding="utf-8")


def _markdown_report(report: DgxSmallCorpusBenchmarkReport) -> str:
    total_expected_jobs = (
        report.fixture.job_count if report.fixture is not None else report.plan.chunk_count
    )
    lines = [
        "# DGX Small Corpus 4-Profile Ingestion Benchmark Result",
        "",
        f"- `passed`: `{str(report.passed).lower()}`",
        f"- `database_url`: `{_redact_database_url(report.plan.database_url)}`",
        f"- `chunk_count`: `{report.plan.chunk_count}`",
        f"- `expected_job_count`: `{total_expected_jobs}`",
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
                f"- `benchmark_run_key`: `{report.fixture.benchmark_run_key}`",
                f"- `file_id`: `{report.fixture.file_id}`",
                f"- `document_id`: `{report.fixture.document_id}`",
                f"- `chunk_ids`: `{', '.join(str(value) for value in report.fixture.chunk_ids)}`",
                f"- `job_count`: `{report.fixture.job_count}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Provider Results",
            "",
            "| Provider | Passed | Base URL | Health | Profiles | Jobs | Error |",
            "| --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for result in report.results:
        provider_job_count = sum(summary.expected_job_count for summary in result.profile_summaries)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{result.provider}`",
                    f"`{str(result.passed).lower()}`",
                    f"`{result.base_url}`",
                    f"`{str(result.health_ok).lower()}`",
                    f"`{len(result.profile_summaries)}/{len(result.profile_names)}`",
                    f"`{provider_job_count}`",
                    f"`{result.error or ''}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Profile Benchmark Summary",
            "",
            (
                "| Provider | Profile | Passed | Jobs | Vectors | Vector Table | Dimension | "
                "Route | Avg Provider ms | Avg Worker ms | Max Worker ms | Error Count |"
            ),
            ("| --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | ---: | " "---: | ---: |"),
        ]
    )
    for result in report.results:
        for summary in result.profile_summaries:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{result.provider}`",
                        f"`{summary.profile_name}`",
                        f"`{str(summary.passed).lower()}`",
                        f"`{summary.succeeded_count}/{summary.expected_job_count}`",
                        f"`{summary.vector_count}`",
                        f"`{summary.vector_table_name or ''}`",
                        f"`{summary.vector_dimension or ''}`",
                        f"`{summary.provider_route_id or ''}`",
                        f"`{_float_or_empty(summary.avg_provider_elapsed_ms)}`",
                        f"`{_float_or_empty(summary.avg_worker_elapsed_ms)}`",
                        f"`{_int_or_empty(summary.max_worker_elapsed_ms)}`",
                        f"`{len(summary.errors)}`",
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Job-Level Evidence",
            "",
            (
                "| Provider | Profile | Chunk | Job | Status | Dimension | Provider Model | "
                "Provider ms | Worker ms | Error |"
            ),
            "| --- | --- | ---: | ---: | --- | ---: | --- | ---: | ---: | --- |",
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
                        f"`{profile.chunk_id or ''}`",
                        f"`{profile.job_id or ''}`",
                        f"`{profile.job_status or ''}`",
                        f"`{profile.vector_dimension or ''}`",
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
            "Launch DGX remote providers sequentially and ingest a temporary small "
            "corpus through all selected embedding profiles."
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
    parser.add_argument("--chunk-count", type=int, default=DEFAULT_CHUNK_COUNT)
    parser.add_argument(
        "--corpus-text",
        action="append",
        default=[],
        help="Add one corpus chunk text. Repeating this option overrides --chunk-count.",
    )
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
        help="Leave benchmark fixture rows in the database after the run.",
    )
    parser.add_argument("--worker-name-prefix", default=DEFAULT_WORKER_NAME_PREFIX)
    parser.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    parser.add_argument(
        "--remote-timeout-seconds",
        type=float,
        default=DEFAULT_REMOTE_TIMEOUT_SECONDS,
    )
    parser.add_argument("--readiness-gate-defer-seconds", type=int, default=300)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--persist-result",
        action="store_true",
        help="Persist the benchmark run/profile/job evidence to the database.",
    )
    parser.add_argument(
        "--created-by",
        default=None,
        help="Optional actor label stored with --persist-result.",
    )
    parser.add_argument(
        "--created-by-user-id",
        type=int,
        default=None,
        help="Optional app_users.user_id stored with --persist-result.",
    )
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

    corpus_texts = tuple(args.corpus_text) if args.corpus_text else None
    chunk_count = len(corpus_texts) if corpus_texts else args.chunk_count
    try:
        plan = build_dgx_small_corpus_benchmark_plan(
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
            chunk_count=chunk_count,
            corpus_texts=corpus_texts,
            preflight_before_worker=not args.skip_preflight,
            active_only_preflight=not args.include_inactive_preflight,
            cleanup_fixture=not args.keep_fixture,
            worker_name_prefix=args.worker_name_prefix,
            lease_seconds=args.lease_seconds,
            remote_timeout_seconds=args.remote_timeout_seconds,
            readiness_gate_defer_seconds=args.readiness_gate_defer_seconds,
            fail_fast=args.fail_fast,
        )
    except (ValueError, InvalidEmbeddingProviderPresetError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        payload = {"dry_run": True, "plan": _plan_payload(plan)}
        print(json.dumps(payload, ensure_ascii=False, indent=None if args.json else 2))
        return 0

    report = run_dgx_small_corpus_benchmark(plan)
    persistence_detail = None
    if args.persist_result:
        persistence_detail = persist_dgx_small_corpus_benchmark_report(
            database_url,
            report,
            created_by=args.created_by,
            created_by_user_id=args.created_by_user_id,
        )
    if args.markdown_output:
        write_markdown_report(report, Path(args.markdown_output))

    if args.json:
        payload = _report_payload(report)
        payload["persistence"] = _persistence_payload(persistence_detail)
        print(json.dumps(payload, ensure_ascii=False))
    else:
        _print_human_report(report)
        if persistence_detail is not None:
            print("- persisted_benchmark_run_id: " f"{persistence_detail.run.benchmark_run_id}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
