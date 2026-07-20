"""Bounded foreground worker runner with provider resource guard evidence."""

from __future__ import annotations

import json
import shlex
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from urllib.parse import urljoin

from app.core.database import connect
from app.core.embedding_provider_routes import (
    EmbeddingProviderRouteRecord,
    list_embedding_provider_routes,
)
from app.core.embedding_providers import (
    MOCK_EMBEDDING_PROVIDER_TYPE,
    REMOTE_EMBEDDING_PROVIDER_HEALTH_PATH,
    REMOTE_EMBEDDING_PROVIDER_TYPE,
)
from app.core.pipeline_jobs import DEFAULT_LEASE_SECONDS

FOREGROUND_WORKER_RUNNER_VERSION = 1

WORKER_RUN_STATUS_PLANNED = "planned"
WORKER_RUN_STATUS_COMPLETED = "completed"
WORKER_RUN_STATUS_GUARDED = "guarded"
WORKER_RUN_STATUS_PARTIAL = "partial"
WORKER_RUN_STATUS_BLOCKED = "blocked"
WORKER_RUN_STATUS_FAILED = "failed"

GUARD_DECISION_ALLOWED = "allowed"
GUARD_DECISION_SKIPPED = "skipped"
GUARD_DECISION_IDLE = "idle"

DEFAULT_PIPELINE_LIMIT = 1
DEFAULT_EMBEDDING_LIMIT_PER_PROFILE = 5
DEFAULT_HEALTH_TIMEOUT_SECONDS = 5.0
DEFAULT_MAX_HEALTH_ELAPSED_MS = 5_000
DEFAULT_QWEN_TOKEN_LIMIT = 1_200
DEFAULT_QWEN_GUARDED_PROFILES = ("qwen3_4b_1000", "qwen3_4b_2560")
DEFAULT_WORKER_NAME_PREFIX = "foreground-guarded"


@dataclass(frozen=True)
class PendingEmbeddingProfileSummary:
    profile_name: str
    pending_count: int
    max_token_count: int | None
    max_char_count: int | None
    oldest_job_id: int | None
    newest_job_id: int | None


@dataclass(frozen=True)
class ProviderHealthProbe:
    checked: bool
    ready: bool
    status: str
    elapsed_ms: int | None
    provider_model_id: str | None = None
    provider_type: str | None = None
    model_key: str | None = None
    profile_names: tuple[str, ...] = ()
    runtime_metadata: dict[str, object] = field(default_factory=dict)
    error_message: str | None = None


@dataclass(frozen=True)
class ProviderResourceGuardDecision:
    profile_name: str
    decision: str
    reason: str
    pending_count: int
    max_token_count: int | None
    max_char_count: int | None
    token_limit: int | None
    route_id: int | None = None
    provider_name: str | None = None
    provider_mode: str | None = None
    provider_base_url: str | None = None
    health: ProviderHealthProbe | None = None

    @property
    def allowed(self) -> bool:
        return self.decision == GUARD_DECISION_ALLOWED


@dataclass(frozen=True)
class ForegroundWorkerRunnerPlan:
    status: str
    generated_at: datetime
    workdir: str
    pipeline_limit: int
    embedding_limit_per_profile: int
    lease_seconds: int
    worker_name_prefix: str
    health_timeout_seconds: float
    max_health_elapsed_ms: int
    profile_token_limits: dict[str, int]
    excluded_profiles: tuple[str, ...]
    pending_profiles: tuple[PendingEmbeddingProfileSummary, ...]
    guard_decisions: tuple[ProviderResourceGuardDecision, ...]

    @property
    def allowed_profiles(self) -> tuple[str, ...]:
        return tuple(decision.profile_name for decision in self.guard_decisions if decision.allowed)

    @property
    def skipped_profiles(self) -> tuple[str, ...]:
        return tuple(
            decision.profile_name
            for decision in self.guard_decisions
            if decision.decision == GUARD_DECISION_SKIPPED
        )


@dataclass(frozen=True)
class WorkerCommandResult:
    code: str
    command: tuple[str, ...]
    exit_code: int
    elapsed_ms: int
    payload: dict[str, object] | None = None
    stdout: str = ""
    stderr: str = ""
    error_message: str | None = None

    @property
    def shell_command(self) -> str:
        return _quote_command(self.command)

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


@dataclass(frozen=True)
class ForegroundWorkerRunnerEvidence:
    status: str
    dry_run: bool
    generated_at: datetime
    plan: ForegroundWorkerRunnerPlan
    pipeline_results: tuple[WorkerCommandResult, ...] = ()
    embedding_results: tuple[WorkerCommandResult, ...] = ()
    message: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def command_count(self) -> int:
        return len(self.pipeline_results) + len(self.embedding_results)

    @property
    def failed_command_count(self) -> int:
        return sum(
            1
            for result in (*self.pipeline_results, *self.embedding_results)
            if not result.succeeded
        )


HealthProbeFunc = Callable[
    [EmbeddingProviderRouteRecord, float],
    ProviderHealthProbe,
]


def build_foreground_worker_runner_plan(
    database_url: str,
    *,
    workdir: str | Path = ".",
    pipeline_limit: int = DEFAULT_PIPELINE_LIMIT,
    embedding_limit_per_profile: int = DEFAULT_EMBEDDING_LIMIT_PER_PROFILE,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    worker_name_prefix: str = DEFAULT_WORKER_NAME_PREFIX,
    health_timeout_seconds: float = DEFAULT_HEALTH_TIMEOUT_SECONDS,
    max_health_elapsed_ms: int = DEFAULT_MAX_HEALTH_ELAPSED_MS,
    profile_token_limits: dict[str, int] | None = None,
    excluded_profiles: Sequence[str] = (),
    generated_at: datetime | None = None,
    health_probe: HealthProbeFunc | None = None,
) -> ForegroundWorkerRunnerPlan:
    selected_database_url = _require_non_empty(database_url, name="database_url")
    selected_workdir = _require_non_empty(str(workdir), name="workdir")
    selected_pipeline_limit = _validate_non_negative_int(pipeline_limit, name="pipeline_limit")
    selected_embedding_limit = _validate_positive_int(
        embedding_limit_per_profile,
        name="embedding_limit_per_profile",
    )
    selected_lease_seconds = _validate_positive_int(lease_seconds, name="lease_seconds")
    selected_worker_prefix = _require_non_empty(worker_name_prefix, name="worker_name_prefix")
    selected_health_timeout = _validate_positive_float(
        health_timeout_seconds,
        name="health_timeout_seconds",
    )
    selected_max_health_elapsed_ms = _validate_positive_int(
        max_health_elapsed_ms,
        name="max_health_elapsed_ms",
    )
    selected_token_limits = _validate_profile_token_limits(
        profile_token_limits if profile_token_limits is not None else default_profile_token_limits()
    )
    selected_excluded_profiles = _validate_profile_names(excluded_profiles)
    pending_profiles = list_pending_embedding_profile_summaries(selected_database_url)
    routes = list_embedding_provider_routes(selected_database_url, active_only=True)
    decisions = build_provider_resource_guard_decisions(
        pending_profiles,
        routes,
        profile_token_limits=selected_token_limits,
        excluded_profiles=selected_excluded_profiles,
        health_timeout_seconds=selected_health_timeout,
        max_health_elapsed_ms=selected_max_health_elapsed_ms,
        health_probe=health_probe,
    )
    return ForegroundWorkerRunnerPlan(
        status=_plan_status(decisions),
        generated_at=generated_at or datetime.now(UTC),
        workdir=str(Path(selected_workdir)),
        pipeline_limit=selected_pipeline_limit,
        embedding_limit_per_profile=selected_embedding_limit,
        lease_seconds=selected_lease_seconds,
        worker_name_prefix=selected_worker_prefix,
        health_timeout_seconds=selected_health_timeout,
        max_health_elapsed_ms=selected_max_health_elapsed_ms,
        profile_token_limits=selected_token_limits,
        excluded_profiles=selected_excluded_profiles,
        pending_profiles=tuple(pending_profiles),
        guard_decisions=decisions,
    )


def build_provider_resource_guard_decisions(
    pending_profiles: Sequence[PendingEmbeddingProfileSummary],
    routes: Sequence[EmbeddingProviderRouteRecord],
    *,
    profile_token_limits: dict[str, int],
    excluded_profiles: Sequence[str] = (),
    health_timeout_seconds: float = DEFAULT_HEALTH_TIMEOUT_SECONDS,
    max_health_elapsed_ms: int = DEFAULT_MAX_HEALTH_ELAPSED_MS,
    health_probe: HealthProbeFunc | None = None,
) -> tuple[ProviderResourceGuardDecision, ...]:
    route_by_profile = _first_route_by_profile(routes)
    excluded = set(_validate_profile_names(excluded_profiles))
    probe = health_probe or probe_provider_route_health
    decisions = []
    for pending in pending_profiles:
        token_limit = profile_token_limits.get(pending.profile_name)
        route = route_by_profile.get(pending.profile_name)
        if pending.profile_name in excluded:
            decisions.append(
                _skipped_decision(
                    pending,
                    route=route,
                    token_limit=token_limit,
                    reason="Profile is explicitly excluded by the provider resource guard.",
                )
            )
            continue
        if token_limit is not None and pending.max_token_count is not None:
            if pending.max_token_count > token_limit:
                decisions.append(
                    _skipped_decision(
                        pending,
                        route=route,
                        token_limit=token_limit,
                        reason=(
                            f"Pending chunk token count {pending.max_token_count} exceeds "
                            f"guard limit {token_limit}."
                        ),
                    )
                )
                continue
        if route is None:
            decisions.append(
                _skipped_decision(
                    pending,
                    route=None,
                    token_limit=token_limit,
                    reason="No active provider route is registered for this profile.",
                )
            )
            continue
        if route.provider_mode == MOCK_EMBEDDING_PROVIDER_TYPE:
            health = ProviderHealthProbe(
                checked=True,
                ready=False,
                status="mock_blocked",
                elapsed_ms=0,
                provider_type=MOCK_EMBEDDING_PROVIDER_TYPE,
                provider_model_id="mock-provider",
                profile_names=(pending.profile_name,),
                runtime_metadata={"provider": MOCK_EMBEDDING_PROVIDER_TYPE},
                error_message="Mock provider routes are blocked for foreground operation.",
            )
            decisions.append(
                _skipped_decision(
                    pending,
                    route=route,
                    token_limit=token_limit,
                    health=health,
                    reason="Mock provider route is not allowed for foreground operation.",
                )
            )
            continue
        health = probe(route, health_timeout_seconds)
        if not health.ready:
            decisions.append(
                _skipped_decision(
                    pending,
                    route=route,
                    token_limit=token_limit,
                    health=health,
                    reason=f"Provider health is not ready: {health.status}.",
                )
            )
            continue
        if health.elapsed_ms is not None and health.elapsed_ms > max_health_elapsed_ms:
            decisions.append(
                _skipped_decision(
                    pending,
                    route=route,
                    token_limit=token_limit,
                    health=health,
                    reason=(
                        f"Provider health latency {health.elapsed_ms}ms exceeds "
                        f"guard limit {max_health_elapsed_ms}ms."
                    ),
                )
            )
            continue
        decisions.append(
            ProviderResourceGuardDecision(
                profile_name=pending.profile_name,
                decision=GUARD_DECISION_ALLOWED,
                reason="Provider route passed foreground resource guard.",
                pending_count=pending.pending_count,
                max_token_count=pending.max_token_count,
                max_char_count=pending.max_char_count,
                token_limit=token_limit,
                route_id=route.route_id,
                provider_name=route.provider_name,
                provider_mode=route.provider_mode,
                provider_base_url=route.provider_base_url,
                health=health,
            )
        )
    return tuple(decisions)


def list_pending_embedding_profile_summaries(
    database_url: str,
) -> tuple[PendingEmbeddingProfileSummary, ...]:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT
                    ej.profile_name,
                    count(*)::int AS pending_count,
                    max(c.token_count)::int AS max_token_count,
                    max(c.char_count)::int AS max_char_count,
                    min(ej.job_id)::bigint AS oldest_job_id,
                    max(ej.job_id)::bigint AS newest_job_id
                FROM embedding_jobs ej
                JOIN chunks c
                  ON c.chunk_id = ej.chunk_id
                WHERE ej.status = 'pending'
                GROUP BY ej.profile_name
                ORDER BY ej.profile_name ASC
                """)
            rows = cursor.fetchall()
    return tuple(
        PendingEmbeddingProfileSummary(
            profile_name=str(row["profile_name"]),
            pending_count=int(row["pending_count"]),
            max_token_count=_int_or_none(row["max_token_count"]),
            max_char_count=_int_or_none(row["max_char_count"]),
            oldest_job_id=_int_or_none(row["oldest_job_id"]),
            newest_job_id=_int_or_none(row["newest_job_id"]),
        )
        for row in rows
    )


def probe_provider_route_health(
    route: EmbeddingProviderRouteRecord,
    timeout_seconds: float,
) -> ProviderHealthProbe:
    if route.provider_mode == MOCK_EMBEDDING_PROVIDER_TYPE:
        return ProviderHealthProbe(
            checked=True,
            ready=False,
            status="mock_blocked",
            elapsed_ms=0,
            provider_type=MOCK_EMBEDDING_PROVIDER_TYPE,
            provider_model_id="mock-provider",
            profile_names=(route.profile_name,),
            runtime_metadata={"provider": MOCK_EMBEDDING_PROVIDER_TYPE},
            error_message="Mock provider route is not allowed.",
        )
    if route.provider_mode != REMOTE_EMBEDDING_PROVIDER_TYPE:
        return ProviderHealthProbe(
            checked=True,
            ready=False,
            status="unsupported",
            elapsed_ms=0,
            provider_type=route.provider_mode,
            error_message=f"Unsupported provider_mode: {route.provider_mode}",
        )
    if not route.provider_base_url:
        return ProviderHealthProbe(
            checked=True,
            ready=False,
            status="missing_base_url",
            elapsed_ms=0,
            provider_type=REMOTE_EMBEDDING_PROVIDER_TYPE,
            error_message="Remote provider route has no provider_base_url.",
        )
    started_at = perf_counter()
    try:
        with urllib.request.urlopen(
            urljoin(
                f"{route.provider_base_url.rstrip('/')}/",
                REMOTE_EMBEDDING_PROVIDER_HEALTH_PATH.lstrip("/"),
            ),
            timeout=timeout_seconds,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return ProviderHealthProbe(
            checked=True,
            ready=False,
            status="unreachable",
            elapsed_ms=max(0, int((perf_counter() - started_at) * 1000)),
            provider_type=REMOTE_EMBEDDING_PROVIDER_TYPE,
            error_message=str(exc),
        )
    elapsed_ms = max(0, int((perf_counter() - started_at) * 1000))
    if not isinstance(payload, dict):
        return ProviderHealthProbe(
            checked=True,
            ready=False,
            status="invalid_response",
            elapsed_ms=elapsed_ms,
            provider_type=REMOTE_EMBEDDING_PROVIDER_TYPE,
            error_message="Provider health response was not a JSON object.",
        )
    profile_names = tuple(str(name) for name in payload.get("profile_names", ()) if str(name))
    ready = bool(payload.get("ready")) and (
        not profile_names or route.profile_name in profile_names
    )
    if ready:
        status = "ready"
        error_message = None
    elif profile_names and route.profile_name not in profile_names:
        status = "profile_mismatch"
        error_message = f"Profile {route.profile_name} is missing from provider health."
    else:
        status = "not_ready"
        error_message = None
    return ProviderHealthProbe(
        checked=True,
        ready=ready,
        status=status,
        elapsed_ms=elapsed_ms,
        provider_model_id=_string_or_none(payload.get("provider_model_id")),
        provider_type=_string_or_none(payload.get("provider_type")),
        model_key=_string_or_none(payload.get("model_key")),
        profile_names=profile_names,
        runtime_metadata=dict(payload.get("runtime_metadata") or {}),
        error_message=error_message,
    )


def build_foreground_worker_runner_evidence(
    plan: ForegroundWorkerRunnerPlan,
    *,
    status: str,
    dry_run: bool,
    pipeline_results: Sequence[WorkerCommandResult] = (),
    embedding_results: Sequence[WorkerCommandResult] = (),
    message: str = "",
    metadata: dict[str, object] | None = None,
    generated_at: datetime | None = None,
) -> ForegroundWorkerRunnerEvidence:
    selected_status = status.strip()
    if selected_status not in {
        WORKER_RUN_STATUS_PLANNED,
        WORKER_RUN_STATUS_COMPLETED,
        WORKER_RUN_STATUS_GUARDED,
        WORKER_RUN_STATUS_PARTIAL,
        WORKER_RUN_STATUS_BLOCKED,
        WORKER_RUN_STATUS_FAILED,
    }:
        raise ValueError("unsupported foreground worker runner status")
    return ForegroundWorkerRunnerEvidence(
        status=selected_status,
        dry_run=dry_run,
        generated_at=generated_at or datetime.now(UTC),
        plan=plan,
        pipeline_results=tuple(pipeline_results),
        embedding_results=tuple(embedding_results),
        message=message,
        metadata=dict(metadata or {}),
    )


def foreground_worker_runner_evidence_payload(
    evidence: ForegroundWorkerRunnerEvidence,
) -> dict[str, object]:
    return {
        "version": FOREGROUND_WORKER_RUNNER_VERSION,
        "status": evidence.status,
        "dry_run": evidence.dry_run,
        "generated_at": evidence.generated_at.isoformat(),
        "generated_at_label": evidence.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        "message": evidence.message,
        "command_count": evidence.command_count,
        "failed_command_count": evidence.failed_command_count,
        "metadata": dict(evidence.metadata),
        "plan": foreground_worker_runner_plan_payload(evidence.plan),
        "pipeline_results": [
            worker_command_result_payload(result) for result in evidence.pipeline_results
        ],
        "embedding_results": [
            worker_command_result_payload(result) for result in evidence.embedding_results
        ],
    }


def foreground_worker_runner_plan_payload(
    plan: ForegroundWorkerRunnerPlan,
) -> dict[str, object]:
    return {
        "version": FOREGROUND_WORKER_RUNNER_VERSION,
        "status": plan.status,
        "generated_at": plan.generated_at.isoformat(),
        "generated_at_label": plan.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        "workdir": plan.workdir,
        "pipeline_limit": plan.pipeline_limit,
        "embedding_limit_per_profile": plan.embedding_limit_per_profile,
        "lease_seconds": plan.lease_seconds,
        "worker_name_prefix": plan.worker_name_prefix,
        "health_timeout_seconds": plan.health_timeout_seconds,
        "max_health_elapsed_ms": plan.max_health_elapsed_ms,
        "profile_token_limits": dict(plan.profile_token_limits),
        "excluded_profiles": list(plan.excluded_profiles),
        "allowed_profiles": list(plan.allowed_profiles),
        "skipped_profiles": list(plan.skipped_profiles),
        "pending_profiles": [
            {
                "profile_name": summary.profile_name,
                "pending_count": summary.pending_count,
                "max_token_count": summary.max_token_count,
                "max_char_count": summary.max_char_count,
                "oldest_job_id": summary.oldest_job_id,
                "newest_job_id": summary.newest_job_id,
            }
            for summary in plan.pending_profiles
        ],
        "guard_decisions": [
            provider_resource_guard_decision_payload(decision) for decision in plan.guard_decisions
        ],
    }


def provider_resource_guard_decision_payload(
    decision: ProviderResourceGuardDecision,
) -> dict[str, object]:
    return {
        "profile_name": decision.profile_name,
        "decision": decision.decision,
        "reason": decision.reason,
        "pending_count": decision.pending_count,
        "max_token_count": decision.max_token_count,
        "max_char_count": decision.max_char_count,
        "token_limit": decision.token_limit,
        "route_id": decision.route_id,
        "provider_name": decision.provider_name,
        "provider_mode": decision.provider_mode,
        "provider_base_url": decision.provider_base_url,
        "health": provider_health_probe_payload(decision.health),
    }


def provider_health_probe_payload(
    health: ProviderHealthProbe | None,
) -> dict[str, object] | None:
    if health is None:
        return None
    return {
        "checked": health.checked,
        "ready": health.ready,
        "status": health.status,
        "elapsed_ms": health.elapsed_ms,
        "provider_model_id": health.provider_model_id,
        "provider_type": health.provider_type,
        "model_key": health.model_key,
        "profile_names": list(health.profile_names),
        "runtime_metadata": dict(health.runtime_metadata),
        "error_message": health.error_message,
    }


def worker_command_result_payload(result: WorkerCommandResult) -> dict[str, object]:
    return {
        "code": result.code,
        "command": list(result.command),
        "shell_command": result.shell_command,
        "exit_code": result.exit_code,
        "elapsed_ms": result.elapsed_ms,
        "succeeded": result.succeeded,
        "payload": result.payload,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error_message": result.error_message,
    }


def render_foreground_worker_runner_markdown(payload: dict[str, object]) -> str:
    plan = _dict(payload.get("plan"))
    lines = [
        "# Foreground Worker Runner Evidence",
        "",
        f"- Generated At: {_text(payload.get('generated_at_label'))}",
        f"- Status: `{_text(payload.get('status'))}`",
        f"- Dry Run: {_text(payload.get('dry_run'))}",
        f"- Message: {_text(payload.get('message'))}",
        f"- Commands: {_text(payload.get('command_count'))}",
        f"- Failed Commands: {_text(payload.get('failed_command_count'))}",
        "",
        "## Resource Guard",
        "",
        f"- Plan Status: `{_text(plan.get('status'))}`",
        f"- Health Timeout Seconds: {_text(plan.get('health_timeout_seconds'))}",
        f"- Max Health Elapsed Ms: {_text(plan.get('max_health_elapsed_ms'))}",
        f"- Allowed Profiles: {_text(', '.join(_strings(plan.get('allowed_profiles'))) or '-')}",
        f"- Skipped Profiles: {_text(', '.join(_strings(plan.get('skipped_profiles'))) or '-')}",
        "",
        "| Profile | Decision | Pending | Max Tokens | Token Limit | Health | Reason |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for decision in _list(plan.get("guard_decisions")):
        decision_payload = _dict(decision)
        health = _dict(decision_payload.get("health"))
        health_status = health.get("status") if health else ""
        if health.get("elapsed_ms") is not None:
            health_status = f"{health_status} ({health.get('elapsed_ms')}ms)"
        lines.append(
            "| "
            f"{_md_cell(decision_payload.get('profile_name'))} | "
            f"{_md_cell(decision_payload.get('decision'))} | "
            f"{_md_cell(decision_payload.get('pending_count'))} | "
            f"{_md_cell(decision_payload.get('max_token_count'))} | "
            f"{_md_cell(decision_payload.get('token_limit'))} | "
            f"{_md_cell(health_status)} | "
            f"{_md_cell(decision_payload.get('reason'))} |"
        )
    lines.extend(
        [
            "",
            "## Command Results",
            "",
            "| Code | Exit | Elapsed Ms | Command |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for result in [
        *_list(payload.get("pipeline_results")),
        *_list(payload.get("embedding_results")),
    ]:
        result_payload = _dict(result)
        lines.append(
            "| "
            f"{_md_cell(result_payload.get('code'))} | "
            f"{_md_cell(result_payload.get('exit_code'))} | "
            f"{_md_cell(result_payload.get('elapsed_ms'))} | "
            f"`{_md_cell(result_payload.get('shell_command'))}` |"
        )
    lines.extend(
        [
            "",
            "## Operator Notes",
            "",
            "- This runner is bounded and intended for supervised foreground operation.",
            "- Skipped profiles remain queued for a later run after resource pressure is resolved.",
            "- Mock provider routes are blocked by the guard.",
            "- Keep DGX memory and swap visible while processing Qwen profiles.",
        ]
    )
    return "\n".join(lines) + "\n"


def payload_to_json(payload: dict[str, object], *, pretty: bool = False) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)


def default_profile_token_limits() -> dict[str, int]:
    return {profile: DEFAULT_QWEN_TOKEN_LIMIT for profile in DEFAULT_QWEN_GUARDED_PROFILES}


def parse_profile_token_limits(values: Sequence[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values:
        selected_profile, token_limit = _parse_profile_token_limit(value)
        if token_limit > 0:
            parsed[selected_profile] = token_limit
    return parsed


def merge_profile_token_limits(
    overrides: Sequence[str],
    *,
    include_defaults: bool = True,
) -> dict[str, int]:
    limits = default_profile_token_limits() if include_defaults else {}
    for value in overrides:
        selected_profile, token_limit = _parse_profile_token_limit(value)
        if token_limit <= 0:
            limits.pop(selected_profile, None)
        else:
            limits[selected_profile] = token_limit
    return _validate_profile_token_limits(limits)


def _parse_profile_token_limit(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise ValueError("profile token limit must use PROFILE=LIMIT format")
    profile_name, raw_limit = value.split("=", 1)
    selected_profile = _require_non_empty(profile_name, name="profile_name")
    try:
        return selected_profile, int(raw_limit)
    except ValueError as exc:
        raise ValueError("profile token limit must be an integer") from exc


def _first_route_by_profile(
    routes: Sequence[EmbeddingProviderRouteRecord],
) -> dict[str, EmbeddingProviderRouteRecord]:
    selected: dict[str, EmbeddingProviderRouteRecord] = {}
    for route in sorted(routes, key=lambda item: (item.profile_name, item.priority, item.route_id)):
        selected.setdefault(route.profile_name, route)
    return selected


def _skipped_decision(
    pending: PendingEmbeddingProfileSummary,
    *,
    route: EmbeddingProviderRouteRecord | None,
    token_limit: int | None,
    reason: str,
    health: ProviderHealthProbe | None = None,
) -> ProviderResourceGuardDecision:
    return ProviderResourceGuardDecision(
        profile_name=pending.profile_name,
        decision=GUARD_DECISION_SKIPPED,
        reason=reason,
        pending_count=pending.pending_count,
        max_token_count=pending.max_token_count,
        max_char_count=pending.max_char_count,
        token_limit=token_limit,
        route_id=route.route_id if route else None,
        provider_name=route.provider_name if route else None,
        provider_mode=route.provider_mode if route else None,
        provider_base_url=route.provider_base_url if route else None,
        health=health,
    )


def _plan_status(decisions: Sequence[ProviderResourceGuardDecision]) -> str:
    if not decisions:
        return "idle"
    if any(decision.allowed for decision in decisions):
        return (
            WORKER_RUN_STATUS_GUARDED
            if any(not decision.allowed for decision in decisions)
            else "ready"
        )
    return WORKER_RUN_STATUS_BLOCKED


def _validate_profile_token_limits(values: dict[str, int]) -> dict[str, int]:
    validated = {}
    for profile_name, token_limit in values.items():
        profile = _require_non_empty(profile_name, name="profile_name")
        validated[profile] = _validate_positive_int(token_limit, name="token_limit")
    return validated


def _validate_profile_names(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(_require_non_empty(value, name="profile_name") for value in values)


def _validate_positive_int(value: int, *, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _validate_non_negative_int(value: int, *, name: str) -> int:
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to zero")
    return value


def _validate_positive_float(value: float, *, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _require_non_empty(value: str, *, name: str) -> str:
    selected = value.strip()
    if not selected:
        raise ValueError(f"{name} is required")
    return selected


def _quote_command(command: tuple[str, ...]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _int_or_none(value: object) -> int | None:
    return int(value) if value is not None else None


def _string_or_none(value: object) -> str | None:
    return str(value) if value is not None else None


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _md_cell(value: object) -> str:
    return _text(value).replace("\n", " ").replace("|", "\\|")
