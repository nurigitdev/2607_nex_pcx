"""Prepare deployment files for a remote GPU embedding provider host."""

import argparse
import json
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

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

DEFAULT_GPU_WORKDIR = "/home/nexpcx/2607_nex_pcx"
DEFAULT_GPU_USER = "nexpcx"
DEFAULT_DEVICE = "cuda:0"


@dataclass(frozen=True)
class RemoteGpuProviderSetupPlan:
    provider: str
    service_name: str
    user: str
    group: str
    workdir: str
    python_bin: str
    models_dir: str
    env_file: str
    systemd_unit_file: str
    route_base_url: str
    health_url: str
    launch_plan: EmbeddingProviderLaunchPlan
    route_registration_command: tuple[str, ...]
    preflight_command: tuple[str, ...]
    create_directory_commands: tuple[str, ...]

    @property
    def systemd_unit_name(self) -> str:
        return Path(self.systemd_unit_file).name


def _normalize_path(path: str | None, *, default: str) -> str:
    selected_path = (path or default).strip()
    if not selected_path:
        raise ValueError("path value is required")
    return str(Path(selected_path).expanduser())


def _validate_service_name(service_name: str) -> str:
    selected_name = service_name.strip()
    if not selected_name:
        raise ValueError("service_name is required")
    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-@")
    if any(char not in allowed_chars for char in selected_name):
        raise ValueError("service_name may only contain letters, numbers, '.', '-', '_', or '@'")
    if selected_name.endswith(".service"):
        selected_name = selected_name[: -len(".service")]
    if not selected_name:
        raise ValueError("service_name is required")
    return selected_name


def _resolve_route_base_url(
    *,
    route_base_url: str | None,
    route_host: str | None,
    fallback_host: str,
    port: int,
) -> str:
    if route_base_url is not None:
        normalized_url = route_base_url.strip().rstrip("/")
        if not normalized_url:
            raise ValueError("route_base_url is required")
        parsed_url = urlparse(normalized_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("route_base_url must be an absolute http(s) URL")
        return normalized_url

    selected_host = (route_host or fallback_host).strip()
    if not selected_host:
        raise ValueError("route_host is required")
    if selected_host in {"0.0.0.0", "::"}:
        selected_host = "127.0.0.1"
    return f"http://{selected_host}:{port}"


def _quote_command(command: tuple[str, ...]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _provider_default_service_name(preset: EmbeddingProviderPreset) -> str:
    return f"nex-pcx-embedding-provider-{preset.preset_name}"


def build_setup_plan(
    preset: EmbeddingProviderPreset,
    *,
    workdir: str = DEFAULT_GPU_WORKDIR,
    models_dir: str | None = None,
    python_bin: str | None = None,
    host: str = "0.0.0.0",
    port: int | None = None,
    route_host: str | None = None,
    route_base_url: str | None = None,
    device: str = DEFAULT_DEVICE,
    provider_model_id: str | None = None,
    service_name: str | None = None,
    user: str = DEFAULT_GPU_USER,
    group: str | None = None,
    env_dir: str | None = None,
    systemd_dir: str | None = None,
) -> RemoteGpuProviderSetupPlan:
    selected_workdir = _normalize_path(workdir, default=DEFAULT_GPU_WORKDIR)
    selected_models_dir = _normalize_path(
        models_dir,
        default=str(Path(selected_workdir) / "models"),
    )
    selected_python_bin = _normalize_path(
        python_bin,
        default=str(Path(selected_workdir) / ".venv" / "bin" / "python"),
    )
    selected_env_dir = _normalize_path(
        env_dir,
        default=str(Path(selected_workdir) / "deployment" / "env"),
    )
    selected_systemd_dir = _normalize_path(
        systemd_dir,
        default=str(Path(selected_workdir) / "deployment" / "systemd"),
    )
    selected_service_name = _validate_service_name(
        service_name or _provider_default_service_name(preset)
    )
    selected_user = user.strip()
    if not selected_user:
        raise ValueError("user is required")
    selected_group = (group or selected_user).strip()
    if not selected_group:
        raise ValueError("group is required")

    launch_plan = build_embedding_provider_launch_plan(
        preset,
        python_bin=selected_python_bin,
        host=host,
        port=port,
        device=device,
        models_dir=selected_models_dir,
        provider_model_id=provider_model_id,
    )
    selected_route_base_url = _resolve_route_base_url(
        route_base_url=route_base_url,
        route_host=route_host,
        fallback_host=launch_plan.host,
        port=launch_plan.port,
    )
    route_registration_command = (
        "./.venv/bin/python",
        "scripts/register_embedding_provider_routes.py",
        "--provider",
        preset.preset_name,
        "--base-url",
        selected_route_base_url,
        "--database-url",
        "$NEX_PCX_DATABASE_URL",
    )
    preflight_command = (
        "./.venv/bin/python",
        "scripts/preflight_provider_routes.py",
        "--database-url",
        "$NEX_PCX_DATABASE_URL",
    )
    return RemoteGpuProviderSetupPlan(
        provider=preset.preset_name,
        service_name=selected_service_name,
        user=selected_user,
        group=selected_group,
        workdir=selected_workdir,
        python_bin=selected_python_bin,
        models_dir=selected_models_dir,
        env_file=str(Path(selected_env_dir) / f"{selected_service_name}.env"),
        systemd_unit_file=str(Path(selected_systemd_dir) / f"{selected_service_name}.service"),
        route_base_url=selected_route_base_url,
        health_url=f"{selected_route_base_url}/healthz",
        launch_plan=launch_plan,
        route_registration_command=route_registration_command,
        preflight_command=preflight_command,
        create_directory_commands=(
            f"mkdir -p {shlex.quote(selected_env_dir)}",
            f"mkdir -p {shlex.quote(selected_systemd_dir)}",
        ),
    )


def render_env_file(plan: RemoteGpuProviderSetupPlan) -> str:
    lines = [
        "# Generated by scripts/setup_remote_gpu_provider.py",
        "# This file intentionally contains provider runtime metadata only.",
        "PYTHONUNBUFFERED=1",
    ]
    for key, value in sorted(plan.launch_plan.environment.items()):
        lines.append(f"{key}={shlex.quote(value)}")
    return "\n".join(lines) + "\n"


def render_systemd_unit(plan: RemoteGpuProviderSetupPlan, *, user_systemd: bool = False) -> str:
    unit_dependencies = (
        "" if user_systemd else "After=network-online.target\nWants=network-online.target\n"
    )
    service_identity = "" if user_systemd else (f"User={plan.user}\n" f"Group={plan.group}\n")
    service_hardening = "" if user_systemd else "NoNewPrivileges=true\nPrivateTmp=true\n"
    install_target = "default.target" if user_systemd else "multi-user.target"
    return (
        "[Unit]\n"
        f"Description=NeX-PCX embedding provider ({plan.provider})\n"
        f"{unit_dependencies}"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"{service_identity}"
        f"WorkingDirectory={plan.workdir}\n"
        f"EnvironmentFile={plan.env_file}\n"
        f"ExecStart={_quote_command(plan.launch_plan.command)}\n"
        "Restart=on-failure\n"
        "RestartSec=5\n"
        f"{service_hardening}"
        "\n"
        "[Install]\n"
        f"WantedBy={install_target}\n"
    )


def write_plan_files(
    plan: RemoteGpuProviderSetupPlan,
    *,
    user_systemd: bool = False,
) -> tuple[Path, Path]:
    env_file = Path(plan.env_file)
    systemd_unit_file = Path(plan.systemd_unit_file)
    env_file.parent.mkdir(parents=True, exist_ok=True)
    systemd_unit_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text(render_env_file(plan), encoding="utf-8")
    systemd_unit_file.write_text(
        render_systemd_unit(plan, user_systemd=user_systemd), encoding="utf-8"
    )
    return env_file, systemd_unit_file


def _plan_payload(plan: RemoteGpuProviderSetupPlan) -> dict[str, object]:
    return {
        **asdict(plan),
        "systemd_unit_name": plan.systemd_unit_name,
        "launch_command": list(plan.launch_plan.command),
        "launch_shell_command": plan.launch_plan.shell_command,
        "route_registration_shell_command": _quote_command(plan.route_registration_command),
        "preflight_shell_command": _quote_command(plan.preflight_command),
    }


def _print_human_plan(plan: RemoteGpuProviderSetupPlan, *, wrote_files: bool) -> None:
    print(f"Remote GPU provider setup plan: {plan.provider}")
    print(f"- workdir: {plan.workdir}")
    print(f"- service_name: {plan.service_name}")
    print(f"- env_file: {plan.env_file}")
    print(f"- systemd_unit_file: {plan.systemd_unit_file}")
    print(f"- provider_base_url: {plan.route_base_url}")
    print(f"- health_url: {plan.health_url}")
    print(f"- launch_command: {_quote_command(plan.launch_plan.command)}")
    print(f"- route_registration: {_quote_command(plan.route_registration_command)}")
    print(f"- preflight: {_quote_command(plan.preflight_command)}")
    print(f"- wrote_files: {wrote_files}")


def main(argv: list[str] | None = None) -> int:
    preset_names = [preset.preset_name for preset in list_embedding_provider_presets()]
    parser = argparse.ArgumentParser(
        description=(
            "Generate env and systemd unit files for a remote GPU embedding provider host."
        ),
    )
    parser.add_argument("--provider", choices=preset_names, required=True)
    parser.add_argument("--workdir", default=DEFAULT_GPU_WORKDIR)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--python-bin", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--route-host", default=None)
    parser.add_argument("--route-base-url", default=None)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--provider-model-id", default=None)
    parser.add_argument("--service-name", default=None)
    parser.add_argument("--user", default=DEFAULT_GPU_USER)
    parser.add_argument("--group", default=None)
    parser.add_argument("--env-dir", default=None)
    parser.add_argument("--systemd-dir", default=None)
    parser.add_argument(
        "--user-systemd",
        action="store_true",
        help="Render a unit suitable for systemctl --user instead of system-level systemd.",
    )
    parser.add_argument("--write-files", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        plan = build_setup_plan(
            get_embedding_provider_preset(args.provider),
            workdir=args.workdir,
            models_dir=args.models_dir,
            python_bin=args.python_bin,
            host=args.host,
            port=args.port,
            route_host=args.route_host,
            route_base_url=args.route_base_url,
            device=args.device,
            provider_model_id=args.provider_model_id,
            service_name=args.service_name,
            user=args.user,
            group=args.group,
            env_dir=args.env_dir,
            systemd_dir=args.systemd_dir,
        )
    except (InvalidEmbeddingProviderPresetError, ValueError) as exc:
        parser.error(str(exc))

    wrote_files = False
    if args.write_files:
        write_plan_files(plan, user_systemd=args.user_systemd)
        wrote_files = True

    if args.json:
        print(
            json.dumps(
                {"wrote_files": wrote_files, "plan": _plan_payload(plan)},
                ensure_ascii=False,
            )
        )
    else:
        _print_human_plan(plan, wrote_files=wrote_files)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
