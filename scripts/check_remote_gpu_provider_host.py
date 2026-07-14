"""Check whether a remote GPU host is ready to run embedding providers."""

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.embedding_provider_presets import (  # noqa: E402
    EmbeddingProviderPreset,
    get_embedding_provider_preset,
    list_embedding_provider_presets,
)

DEFAULT_GPU_HOST = "192.168.20.243"
DEFAULT_GPU_USER = "nexpcx"
DEFAULT_GPU_WORKDIR = "/home/nexpcx/2607_nex_pcx"
DEFAULT_TIMEOUT_SECONDS = 12
OUTPUT_LIMIT = 4000


class CommandRunner(Protocol):
    def __call__(
        self,
        args: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]: ...


@dataclass(frozen=True)
class RemoteReadinessCheckPlan:
    name: str
    description: str
    remote_command: str
    required: bool = True


@dataclass(frozen=True)
class RemoteReadinessPlan:
    host: str
    ssh_user: str
    ssh_target: str
    workdir: str
    python_bin: str
    models_dir: str
    route_host: str
    timeout_seconds: int
    providers: tuple[str, ...]
    checks: tuple[RemoteReadinessCheckPlan, ...]


@dataclass(frozen=True)
class RemoteReadinessCheckResult:
    name: str
    description: str
    required: bool
    status: str
    exit_code: int | None
    remote_command: str
    ssh_command: tuple[str, ...]
    stdout: str
    stderr: str


@dataclass(frozen=True)
class RemoteReadinessReport:
    host: str
    ssh_user: str
    ssh_target: str
    workdir: str
    python_bin: str
    models_dir: str
    route_host: str
    providers: tuple[str, ...]
    ready: bool
    required_passed: int
    required_failed: int
    optional_failed: int
    checks: tuple[RemoteReadinessCheckResult, ...]


def _quote(value: str) -> str:
    return shlex.quote(value)


def _select_presets(provider: str) -> tuple[EmbeddingProviderPreset, ...]:
    if provider == "all":
        return list_embedding_provider_presets()
    return (get_embedding_provider_preset(provider),)


def _expected_model_keys(presets: tuple[EmbeddingProviderPreset, ...]) -> tuple[str, ...]:
    seen = []
    for preset in presets:
        if preset.model_key not in seen:
            seen.append(preset.model_key)
    return tuple(seen)


def _port_snapshot_command(presets: tuple[EmbeddingProviderPreset, ...]) -> str:
    ports = "|".join(str(preset.default_port) for preset in presets)
    pattern = f":({ports})$"
    return (
        "if command -v ss >/dev/null 2>&1; then "
        f"ss -ltnH | awk '{{print $4}}' | grep -E {_quote(pattern)} || true; "
        "else echo 'ss command not available'; fi"
    )


def build_readiness_plan(
    *,
    host: str = DEFAULT_GPU_HOST,
    ssh_user: str = DEFAULT_GPU_USER,
    workdir: str = DEFAULT_GPU_WORKDIR,
    models_dir: str | None = None,
    python_bin: str | None = None,
    route_host: str | None = None,
    provider: str = "all",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> RemoteReadinessPlan:
    selected_host = host.strip()
    selected_user = ssh_user.strip()
    selected_workdir = workdir.strip()
    if not selected_host:
        raise ValueError("host is required")
    if not selected_user:
        raise ValueError("ssh_user is required")
    if not selected_workdir:
        raise ValueError("workdir is required")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than 0")

    presets = _select_presets(provider)
    selected_models_dir = (models_dir or f"{selected_workdir}/models").strip()
    selected_python_bin = (python_bin or f"{selected_workdir}/.venv/bin/python").strip()
    selected_route_host = (route_host or selected_host).strip()
    if not selected_models_dir:
        raise ValueError("models_dir is required")
    if not selected_python_bin:
        raise ValueError("python_bin is required")
    if not selected_route_host:
        raise ValueError("route_host is required")

    provider_import_code = (
        "import fastapi, uvicorn; "
        "import app.embedding_provider_service; "
        "print('provider_import_ok')"
    )

    checks: list[RemoteReadinessCheckPlan] = [
        RemoteReadinessCheckPlan(
            name="ssh_identity",
            description="Confirm the SSH session can identify the remote user and host.",
            remote_command="id -un && hostname",
        ),
        RemoteReadinessCheckPlan(
            name="workdir_exists",
            description="Confirm the NeX_PCX work directory exists.",
            remote_command=f"test -d {_quote(selected_workdir)} && echo {_quote(selected_workdir)}",
        ),
        RemoteReadinessCheckPlan(
            name="python_version",
            description="Confirm the virtualenv Python executable is available.",
            remote_command=f"{_quote(selected_python_bin)} --version",
        ),
        RemoteReadinessCheckPlan(
            name="provider_runtime_import",
            description="Confirm provider runtime dependencies can import.",
            remote_command=(
                f"cd {_quote(selected_workdir)} && {_quote(selected_python_bin)} -c "
                f"{_quote(provider_import_code)}"
            ),
        ),
        RemoteReadinessCheckPlan(
            name="nvidia_smi",
            description="Confirm NVIDIA GPU runtime is visible.",
            remote_command=(
                "nvidia-smi "
                "--query-gpu=name,driver_version,memory.total "
                "--format=csv,noheader"
            ),
        ),
        RemoteReadinessCheckPlan(
            name="setup_script_exists",
            description="Confirm the remote deployment setup script exists.",
            remote_command=(
                f"test -f {_quote(selected_workdir)}/scripts/setup_remote_gpu_provider.py "
                "&& echo setup_remote_gpu_provider.py"
            ),
        ),
        RemoteReadinessCheckPlan(
            name="port_listener_snapshot",
            description="Read-only snapshot of configured provider ports.",
            remote_command=_port_snapshot_command(presets),
            required=False,
        ),
    ]

    for model_key in _expected_model_keys(presets):
        model_dir = f"{selected_models_dir}/{model_key}"
        model_dir_command = (
            f"test -d {_quote(model_dir)} && find {_quote(model_dir)} -maxdepth 1 | head"
        )
        checks.append(
            RemoteReadinessCheckPlan(
                name=f"model_dir_{model_key}",
                description=f"Confirm model directory exists for {model_key}.",
                remote_command=model_dir_command,
            )
        )

    for preset in presets:
        checks.append(
            RemoteReadinessCheckPlan(
                name=f"setup_dry_run_{preset.preset_name}",
                description=f"Confirm setup dry-run works for {preset.preset_name}.",
                remote_command=(
                    f"cd {_quote(selected_workdir)} && {_quote(selected_python_bin)} "
                    "scripts/setup_remote_gpu_provider.py "
                    f"--provider {_quote(preset.preset_name)} "
                    f"--route-host {_quote(selected_route_host)} "
                    "--json"
                ),
            )
        )

    return RemoteReadinessPlan(
        host=selected_host,
        ssh_user=selected_user,
        ssh_target=f"{selected_user}@{selected_host}",
        workdir=selected_workdir,
        python_bin=selected_python_bin,
        models_dir=selected_models_dir,
        route_host=selected_route_host,
        timeout_seconds=timeout_seconds,
        providers=tuple(preset.preset_name for preset in presets),
        checks=tuple(checks),
    )


def build_ssh_command(plan: RemoteReadinessPlan, remote_command: str) -> tuple[str, ...]:
    return (
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={plan.timeout_seconds}",
        plan.ssh_target,
        remote_command,
    )


def _limit_output(value: str) -> str:
    if len(value) <= OUTPUT_LIMIT:
        return value
    return value[:OUTPUT_LIMIT] + "\n...[truncated]"


def run_readiness_check(
    plan: RemoteReadinessPlan,
    check: RemoteReadinessCheckPlan,
    *,
    runner: CommandRunner = subprocess.run,
) -> RemoteReadinessCheckResult:
    ssh_command = build_ssh_command(plan, check.remote_command)
    completed = runner(
        list(ssh_command),
        capture_output=True,
        text=True,
        timeout=plan.timeout_seconds + 5,
        check=False,
    )
    status = "passed" if completed.returncode == 0 else "failed"
    return RemoteReadinessCheckResult(
        name=check.name,
        description=check.description,
        required=check.required,
        status=status,
        exit_code=completed.returncode,
        remote_command=check.remote_command,
        ssh_command=ssh_command,
        stdout=_limit_output(completed.stdout.strip()),
        stderr=_limit_output(completed.stderr.strip()),
    )


def build_dry_run_report(plan: RemoteReadinessPlan) -> RemoteReadinessReport:
    checks = tuple(
        RemoteReadinessCheckResult(
            name=check.name,
            description=check.description,
            required=check.required,
            status="planned",
            exit_code=None,
            remote_command=check.remote_command,
            ssh_command=build_ssh_command(plan, check.remote_command),
            stdout="",
            stderr="",
        )
        for check in plan.checks
    )
    return RemoteReadinessReport(
        host=plan.host,
        ssh_user=plan.ssh_user,
        ssh_target=plan.ssh_target,
        workdir=plan.workdir,
        python_bin=plan.python_bin,
        models_dir=plan.models_dir,
        route_host=plan.route_host,
        providers=plan.providers,
        ready=False,
        required_passed=0,
        required_failed=0,
        optional_failed=0,
        checks=checks,
    )


def run_readiness_plan(
    plan: RemoteReadinessPlan,
    *,
    runner: CommandRunner = subprocess.run,
) -> RemoteReadinessReport:
    results = tuple(run_readiness_check(plan, check, runner=runner) for check in plan.checks)
    required_passed = sum(1 for result in results if result.required and result.status == "passed")
    required_failed = sum(1 for result in results if result.required and result.status == "failed")
    optional_failed = sum(
        1 for result in results if not result.required and result.status == "failed"
    )
    return RemoteReadinessReport(
        host=plan.host,
        ssh_user=plan.ssh_user,
        ssh_target=plan.ssh_target,
        workdir=plan.workdir,
        python_bin=plan.python_bin,
        models_dir=plan.models_dir,
        route_host=plan.route_host,
        providers=plan.providers,
        ready=required_failed == 0,
        required_passed=required_passed,
        required_failed=required_failed,
        optional_failed=optional_failed,
        checks=results,
    )


def report_payload(report: RemoteReadinessReport) -> dict[str, object]:
    return {
        **asdict(report),
        "checks": [
            {
                **asdict(check),
                "ssh_command": list(check.ssh_command),
                "ssh_shell_command": " ".join(_quote(part) for part in check.ssh_command),
            }
            for check in report.checks
        ],
    }


def _print_human_report(report: RemoteReadinessReport, *, dry_run: bool) -> None:
    title = (
        "Remote GPU provider host readiness plan"
        if dry_run
        else "Remote GPU provider host readiness report"
    )
    print(title)
    print(f"- ssh_target: {report.ssh_target}")
    print(f"- workdir: {report.workdir}")
    print(f"- models_dir: {report.models_dir}")
    print(f"- providers: {', '.join(report.providers)}")
    if not dry_run:
        print(f"- ready: {report.ready}")
        print(f"- required_passed: {report.required_passed}")
        print(f"- required_failed: {report.required_failed}")
        print(f"- optional_failed: {report.optional_failed}")
    for check in report.checks:
        marker = "required" if check.required else "optional"
        print(f"[{check.status}] {check.name} ({marker})")
        if check.stdout:
            print(f"  stdout: {check.stdout}")
        if check.stderr:
            print(f"  stderr: {check.stderr}")


def main(argv: list[str] | None = None) -> int:
    preset_names = [preset.preset_name for preset in list_embedding_provider_presets()]
    parser = argparse.ArgumentParser(
        description="Run read-only readiness checks against a remote GPU provider host.",
    )
    parser.add_argument("--host", default=DEFAULT_GPU_HOST)
    parser.add_argument("--ssh-user", default=DEFAULT_GPU_USER)
    parser.add_argument("--workdir", default=DEFAULT_GPU_WORKDIR)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--python-bin", default=None)
    parser.add_argument("--route-host", default=None)
    parser.add_argument("--provider", choices=[*preset_names, "all"], default="all")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        plan = build_readiness_plan(
            host=args.host,
            ssh_user=args.ssh_user,
            workdir=args.workdir,
            models_dir=args.models_dir,
            python_bin=args.python_bin,
            route_host=args.route_host,
            provider=args.provider,
            timeout_seconds=args.timeout_seconds,
        )
    except ValueError as exc:
        parser.error(str(exc))

    report = build_dry_run_report(plan) if args.dry_run else run_readiness_plan(plan)
    if args.json:
        print(json.dumps(report_payload(report), ensure_ascii=False))
    else:
        _print_human_report(report, dry_run=args.dry_run)

    return 0 if args.dry_run or report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
