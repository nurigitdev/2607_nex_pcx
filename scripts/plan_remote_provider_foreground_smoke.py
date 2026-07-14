"""Build a foreground launch smoke plan for a remote embedding provider."""

import argparse
import json
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.embedding_provider_presets import (  # noqa: E402
    EmbeddingProviderLaunchPlan,
    EmbeddingProviderPreset,
    InvalidEmbeddingProviderPresetError,
    build_embedding_provider_launch_plan,
    get_embedding_provider_preset,
    list_embedding_provider_presets,
)

DEFAULT_GPU_HOST = "192.168.20.243"
DEFAULT_GPU_USER = "nexpcx"
DEFAULT_GPU_WORKDIR = "/home/nexpcx/2607_nex_pcx"
DEFAULT_DEVICE = "cuda:0"


@dataclass(frozen=True)
class RemoteProviderForegroundSmokePlan:
    provider: str
    provider_name: str
    backend: str
    model_key: str
    profile_names: tuple[str, ...]
    ssh_target: str
    workdir: str
    python_bin: str
    models_dir: str
    host: str
    port: int
    route_host: str
    base_url: str
    health_url: str
    device: str
    provider_model_id: str
    launch_plan: EmbeddingProviderLaunchPlan
    readiness_command: tuple[str, ...]
    remote_launch_command: str
    ssh_launch_command: tuple[str, ...]
    remote_port_check_command: tuple[str, ...]
    health_check_command: tuple[str, ...]
    stop_instruction: str


def _quote_command(command: tuple[str, ...] | list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _normalize_required(value: str | None, *, default: str, name: str) -> str:
    selected_value = (value or default).strip()
    if not selected_value:
        raise ValueError(f"{name} is required")
    return selected_value


def build_foreground_smoke_plan(
    preset: EmbeddingProviderPreset,
    *,
    host: str = DEFAULT_GPU_HOST,
    ssh_user: str = DEFAULT_GPU_USER,
    workdir: str = DEFAULT_GPU_WORKDIR,
    models_dir: str | None = None,
    python_bin: str | None = None,
    provider_host: str = "0.0.0.0",
    port: int | None = None,
    route_host: str | None = None,
    device: str = DEFAULT_DEVICE,
    provider_model_id: str | None = None,
    readiness_timeout_seconds: int = 12,
) -> RemoteProviderForegroundSmokePlan:
    selected_host = _normalize_required(host, default=DEFAULT_GPU_HOST, name="host")
    selected_user = _normalize_required(ssh_user, default=DEFAULT_GPU_USER, name="ssh_user")
    selected_workdir = _normalize_required(
        workdir,
        default=DEFAULT_GPU_WORKDIR,
        name="workdir",
    )
    selected_models_dir = _normalize_required(
        models_dir,
        default=f"{selected_workdir}/models",
        name="models_dir",
    )
    selected_python_bin = _normalize_required(
        python_bin,
        default=f"{selected_workdir}/.venv/bin/python",
        name="python_bin",
    )
    selected_provider_host = _normalize_required(
        provider_host,
        default="0.0.0.0",
        name="provider_host",
    )
    selected_route_host = _normalize_required(
        route_host,
        default=selected_host,
        name="route_host",
    )
    selected_device = _normalize_required(device, default=DEFAULT_DEVICE, name="device")
    if readiness_timeout_seconds <= 0:
        raise ValueError("readiness_timeout_seconds must be greater than 0")

    launch_plan = build_embedding_provider_launch_plan(
        preset,
        python_bin=selected_python_bin,
        host=selected_provider_host,
        port=port,
        device=selected_device,
        models_dir=selected_models_dir,
        provider_model_id=provider_model_id,
    )
    ssh_target = f"{selected_user}@{selected_host}"
    base_url = f"http://{selected_route_host}:{launch_plan.port}"
    remote_launch_command = f"cd {shlex.quote(selected_workdir)} && {launch_plan.shell_command}"
    readiness_command = (
        "./.venv/bin/python",
        "scripts/check_remote_gpu_provider_host.py",
        "--host",
        selected_host,
        "--ssh-user",
        selected_user,
        "--provider",
        preset.preset_name,
        "--timeout-seconds",
        str(readiness_timeout_seconds),
    )
    ssh_launch_command = (
        "ssh",
        "-t",
        ssh_target,
        remote_launch_command,
    )
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
    return RemoteProviderForegroundSmokePlan(
        provider=preset.preset_name,
        provider_name=preset.provider_name,
        backend=preset.backend,
        model_key=preset.model_key,
        profile_names=preset.profile_names,
        ssh_target=ssh_target,
        workdir=selected_workdir,
        python_bin=selected_python_bin,
        models_dir=selected_models_dir,
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
        stop_instruction="Press Ctrl-C in the foreground SSH session to stop the provider.",
    )


def _plan_payload(plan: RemoteProviderForegroundSmokePlan) -> dict[str, object]:
    return {
        **asdict(plan),
        "launch_command": list(plan.launch_plan.command),
        "launch_shell_command": plan.launch_plan.shell_command,
        "readiness_shell_command": _quote_command(plan.readiness_command),
        "ssh_launch_shell_command": _quote_command(plan.ssh_launch_command),
        "remote_port_check_shell_command": _quote_command(plan.remote_port_check_command),
        "health_check_shell_command": _quote_command(plan.health_check_command),
    }


def _print_human_plan(plan: RemoteProviderForegroundSmokePlan) -> None:
    print(f"Remote provider foreground smoke plan: {plan.provider}")
    print(f"- ssh_target: {plan.ssh_target}")
    print(f"- provider_base_url: {plan.base_url}")
    print(f"- health_url: {plan.health_url}")
    print(f"- profiles: {', '.join(plan.profile_names)}")
    print(f"- readiness: {_quote_command(plan.readiness_command)}")
    print(f"- launch: {_quote_command(plan.ssh_launch_command)}")
    print(f"- port_check: {_quote_command(plan.remote_port_check_command)}")
    print(f"- health_check: {_quote_command(plan.health_check_command)}")
    print(f"- stop: {plan.stop_instruction}")


def main(argv: list[str] | None = None) -> int:
    preset_names = [preset.preset_name for preset in list_embedding_provider_presets()]
    parser = argparse.ArgumentParser(
        description="Build a foreground launch smoke plan for a remote embedding provider.",
    )
    parser.add_argument("--provider", choices=preset_names, default="kure")
    parser.add_argument("--host", default=DEFAULT_GPU_HOST)
    parser.add_argument("--ssh-user", default=DEFAULT_GPU_USER)
    parser.add_argument("--workdir", default=DEFAULT_GPU_WORKDIR)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--python-bin", default=None)
    parser.add_argument("--provider-host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--route-host", default=None)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--provider-model-id", default=None)
    parser.add_argument("--readiness-timeout-seconds", type=int, default=12)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        plan = build_foreground_smoke_plan(
            get_embedding_provider_preset(args.provider),
            host=args.host,
            ssh_user=args.ssh_user,
            workdir=args.workdir,
            models_dir=args.models_dir,
            python_bin=args.python_bin,
            provider_host=args.provider_host,
            port=args.port,
            route_host=args.route_host,
            device=args.device,
            provider_model_id=args.provider_model_id,
            readiness_timeout_seconds=args.readiness_timeout_seconds,
        )
    except (InvalidEmbeddingProviderPresetError, ValueError) as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps({"plan": _plan_payload(plan)}, ensure_ascii=False))
    else:
        _print_human_plan(plan)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
