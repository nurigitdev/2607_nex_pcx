"""Build a foreground launch smoke plan for the remote reranker provider."""

import argparse
import json
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.rerankers import (  # noqa: E402
    DEFAULT_RERANKER_MODEL_ID,
    DEFAULT_RERANKER_PROFILE_NAME,
)
from app.reranker_provider_service import (  # noqa: E402
    DEFAULT_RERANKER_MODEL_DIR_NAME,
    RERANKER_PROVIDER_BACKEND_QWEN,
)
from scripts.plan_remote_provider_foreground_smoke import (  # noqa: E402
    DEFAULT_DEVICE,
    DEFAULT_GPU_HOST,
    DEFAULT_GPU_USER,
    DEFAULT_GPU_WORKDIR,
)
from scripts.run_reranker_provider import (  # noqa: E402
    DEFAULT_RERANKER_PROVIDER_NAME,
    DEFAULT_RERANKER_PROVIDER_PORT,
    RerankerProviderLaunchPlan,
    build_launch_plan,
)


@dataclass(frozen=True)
class RemoteRerankerForegroundSmokePlan:
    provider_name: str
    backend: str
    reranker_profile_name: str
    ssh_target: str
    workdir: str
    python_bin: str
    models_dir: str
    model_dir_name: str
    host: str
    port: int
    route_host: str
    base_url: str
    health_url: str
    device: str
    provider_model_id: str
    launch_plan: RerankerProviderLaunchPlan
    readiness_command: tuple[str, ...]
    remote_launch_command: str
    ssh_launch_command: tuple[str, ...]
    remote_port_check_command: tuple[str, ...]
    health_check_command: tuple[str, ...]
    stop_instruction: str


def build_reranker_foreground_smoke_plan(
    *,
    host: str = DEFAULT_GPU_HOST,
    ssh_user: str = DEFAULT_GPU_USER,
    workdir: str = DEFAULT_GPU_WORKDIR,
    models_dir: str | None = None,
    model_dir_name: str = DEFAULT_RERANKER_MODEL_DIR_NAME,
    python_bin: str | None = None,
    provider_host: str = "0.0.0.0",
    port: int = DEFAULT_RERANKER_PROVIDER_PORT,
    route_host: str | None = None,
    device: str = DEFAULT_DEVICE,
    provider_name: str = DEFAULT_RERANKER_PROVIDER_NAME,
    provider_model_id: str = DEFAULT_RERANKER_MODEL_ID,
    reranker_profile_name: str = DEFAULT_RERANKER_PROFILE_NAME,
    backend: str = RERANKER_PROVIDER_BACKEND_QWEN,
) -> RemoteRerankerForegroundSmokePlan:
    selected_host = _require_nonblank(host, "host")
    selected_user = _require_nonblank(ssh_user, "ssh_user")
    selected_workdir = _require_nonblank(workdir, "workdir")
    selected_models_dir = _require_nonblank(
        models_dir or f"{selected_workdir}/models",
        "models_dir",
    )
    selected_model_dir_name = _require_nonblank(model_dir_name, "model_dir_name")
    selected_python_bin = _require_nonblank(
        python_bin or f"{selected_workdir}/.venv/bin/python",
        "python_bin",
    )
    selected_provider_host = _require_nonblank(provider_host, "provider_host")
    selected_route_host = _require_nonblank(route_host or selected_host, "route_host")
    selected_device = _require_nonblank(device, "device")
    selected_provider_name = _require_nonblank(provider_name, "provider_name")
    selected_provider_model_id = _require_nonblank(provider_model_id, "provider_model_id")
    selected_reranker_profile_name = _require_nonblank(
        reranker_profile_name,
        "reranker_profile_name",
    )

    launch_plan = build_launch_plan(
        python_bin=selected_python_bin,
        provider_name=selected_provider_name,
        backend=backend,
        host=selected_provider_host,
        port=port,
        device=selected_device,
        models_dir=selected_models_dir,
        model_dir_name=selected_model_dir_name,
        provider_model_id=selected_provider_model_id,
        reranker_profile_name=selected_reranker_profile_name,
    )
    ssh_target = f"{selected_user}@{selected_host}"
    base_url = f"http://{selected_route_host}:{launch_plan.port}"
    remote_launch_command = f"cd {shlex.quote(selected_workdir)} && {launch_plan.shell_command}"
    readiness_command = (
        "ssh",
        ssh_target,
        (
            f"cd {shlex.quote(selected_workdir)} && "
            "test -x .venv/bin/python && "
            "test -f app/reranker_provider_service.py && "
            f"test -d {shlex.quote(selected_models_dir + '/' + selected_model_dir_name)}"
        ),
    )
    ssh_launch_command = ("ssh", "-t", ssh_target, remote_launch_command)
    remote_port_check_command = (
        "ssh",
        ssh_target,
        (
            "if command -v ss >/dev/null 2>&1; then "
            f"ss -ltnH | grep -E {shlex.quote(':' + str(launch_plan.port) + '$')} || true; "
            "else echo 'ss command not available'; fi"
        ),
    )
    health_check_command = ("curl", "-fsS", f"{base_url}/healthz")
    return RemoteRerankerForegroundSmokePlan(
        provider_name=selected_provider_name,
        backend=launch_plan.backend,
        reranker_profile_name=launch_plan.reranker_profile_name,
        ssh_target=ssh_target,
        workdir=selected_workdir,
        python_bin=selected_python_bin,
        models_dir=selected_models_dir,
        model_dir_name=selected_model_dir_name,
        host=launch_plan.host,
        port=launch_plan.port,
        route_host=selected_route_host,
        base_url=base_url,
        health_url=f"{base_url}/healthz",
        device=launch_plan.device,
        provider_model_id=launch_plan.provider_model_id,
        launch_plan=launch_plan,
        readiness_command=readiness_command,
        remote_launch_command=remote_launch_command,
        ssh_launch_command=ssh_launch_command,
        remote_port_check_command=remote_port_check_command,
        health_check_command=health_check_command,
        stop_instruction=(
            "Press Ctrl-C in the foreground SSH session to stop the reranker provider."
        ),
    )


def _require_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _quote_command(command: tuple[str, ...] | list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _plan_payload(plan: RemoteRerankerForegroundSmokePlan) -> dict[str, object]:
    return {
        **asdict(plan),
        "launch_command": list(plan.launch_plan.command),
        "launch_shell_command": plan.launch_plan.shell_command,
        "readiness_shell_command": _quote_command(plan.readiness_command),
        "ssh_launch_shell_command": _quote_command(plan.ssh_launch_command),
        "remote_port_check_shell_command": _quote_command(plan.remote_port_check_command),
        "health_check_shell_command": _quote_command(plan.health_check_command),
    }


def _print_human_plan(plan: RemoteRerankerForegroundSmokePlan) -> None:
    print("Remote reranker foreground smoke plan")
    print(f"- ssh_target: {plan.ssh_target}")
    print(f"- provider_name: {plan.provider_name}")
    print(f"- backend: {plan.backend}")
    print(f"- provider_base_url: {plan.base_url}")
    print(f"- health_url: {plan.health_url}")
    print(f"- readiness: {_quote_command(plan.readiness_command)}")
    print(f"- launch: {_quote_command(plan.ssh_launch_command)}")
    print(f"- port_check: {_quote_command(plan.remote_port_check_command)}")
    print(f"- health_check: {_quote_command(plan.health_check_command)}")
    print(f"- stop: {plan.stop_instruction}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a foreground launch smoke plan for the remote reranker provider.",
    )
    parser.add_argument("--host", default=DEFAULT_GPU_HOST)
    parser.add_argument("--ssh-user", default=DEFAULT_GPU_USER)
    parser.add_argument("--workdir", default=DEFAULT_GPU_WORKDIR)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--model-dir-name", default=DEFAULT_RERANKER_MODEL_DIR_NAME)
    parser.add_argument("--python-bin", default=None)
    parser.add_argument("--provider-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_RERANKER_PROVIDER_PORT)
    parser.add_argument("--route-host", default=None)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--provider-name", default=DEFAULT_RERANKER_PROVIDER_NAME)
    parser.add_argument("--provider-model-id", default=DEFAULT_RERANKER_MODEL_ID)
    parser.add_argument("--profile-name", default=DEFAULT_RERANKER_PROFILE_NAME)
    parser.add_argument("--backend", default=RERANKER_PROVIDER_BACKEND_QWEN)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_reranker_foreground_smoke_plan(
            host=args.host,
            ssh_user=args.ssh_user,
            workdir=args.workdir,
            models_dir=args.models_dir,
            model_dir_name=args.model_dir_name,
            python_bin=args.python_bin,
            provider_host=args.provider_host,
            port=args.port,
            route_host=args.route_host,
            device=args.device,
            provider_name=args.provider_name,
            provider_model_id=args.provider_model_id,
            reranker_profile_name=args.profile_name,
            backend=args.backend,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps({"plan": _plan_payload(plan)}, ensure_ascii=False))
    else:
        _print_human_plan(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
