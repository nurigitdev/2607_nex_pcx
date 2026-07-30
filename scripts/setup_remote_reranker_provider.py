"""Generate env and systemd files for the remote DGX reranker provider."""

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

from app.core.rerankers import (  # noqa: E402
    DEFAULT_RERANKER_MODEL_ID,
    DEFAULT_RERANKER_PROFILE_NAME,
)
from app.reranker_provider_service import (  # noqa: E402
    DEFAULT_RERANKER_MODEL_DIR_NAME,
    RERANKER_PROVIDER_BACKEND_QWEN,
)
from scripts.run_reranker_provider import (  # noqa: E402
    DEFAULT_RERANKER_PROVIDER_NAME,
    DEFAULT_RERANKER_PROVIDER_PORT,
    RerankerProviderLaunchPlan,
    build_launch_plan,
)

DEFAULT_GPU_WORKDIR = "/home/nexpcx/2607_nex_pcx"
DEFAULT_GPU_USER = "nexpcx"
DEFAULT_GPU_HOST = "192.168.20.243"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_RERANKER_SERVICE_NAME = "nex-pcx-reranker-provider"


@dataclass(frozen=True)
class RemoteRerankerProviderSetupPlan:
    service_name: str
    user: str
    group: str
    workdir: str
    python_bin: str
    models_dir: str
    model_dir_name: str
    env_file: str
    systemd_unit_file: str
    route_base_url: str
    health_url: str
    request_smoke_command: tuple[str, ...]
    operations_status_command: tuple[str, ...]
    launch_plan: RerankerProviderLaunchPlan
    create_directory_commands: tuple[str, ...]

    @property
    def systemd_unit_name(self) -> str:
        return Path(self.systemd_unit_file).name


def build_setup_plan(
    *,
    workdir: str = DEFAULT_GPU_WORKDIR,
    models_dir: str | None = None,
    model_dir_name: str = DEFAULT_RERANKER_MODEL_DIR_NAME,
    python_bin: str | None = None,
    host: str = "0.0.0.0",
    port: int = DEFAULT_RERANKER_PROVIDER_PORT,
    route_host: str | None = None,
    route_base_url: str | None = None,
    device: str = DEFAULT_DEVICE,
    provider_name: str = DEFAULT_RERANKER_PROVIDER_NAME,
    provider_model_id: str = DEFAULT_RERANKER_MODEL_ID,
    profile_name: str = DEFAULT_RERANKER_PROFILE_NAME,
    backend: str = RERANKER_PROVIDER_BACKEND_QWEN,
    service_name: str = DEFAULT_RERANKER_SERVICE_NAME,
    user: str = DEFAULT_GPU_USER,
    group: str | None = None,
    env_dir: str | None = None,
    systemd_dir: str | None = None,
) -> RemoteRerankerProviderSetupPlan:
    selected_workdir = _normalize_path(workdir, default=DEFAULT_GPU_WORKDIR)
    selected_models_dir = _normalize_path(
        models_dir,
        default=str(Path(selected_workdir) / "models"),
    )
    selected_model_dir_name = _require_nonblank(model_dir_name, "model_dir_name")
    selected_python_bin = _normalize_path(
        python_bin,
        default=str(Path(selected_workdir) / ".venv" / "bin" / "python"),
    )
    selected_service_name = _validate_service_name(service_name)
    selected_user = _require_nonblank(user, "user")
    selected_group = _require_nonblank(group or selected_user, "group")
    selected_env_dir = _normalize_path(
        env_dir,
        default=str(Path(selected_workdir) / "deployment" / "env"),
    )
    selected_systemd_dir = _normalize_path(
        systemd_dir,
        default=str(Path(selected_workdir) / "deployment" / "systemd"),
    )
    launch_plan = build_launch_plan(
        python_bin=selected_python_bin,
        provider_name=provider_name,
        backend=backend,
        host=host,
        port=port,
        device=device,
        models_dir=selected_models_dir,
        model_dir_name=selected_model_dir_name,
        provider_model_id=provider_model_id,
        reranker_profile_name=profile_name,
    )
    selected_route_base_url = _resolve_route_base_url(
        route_base_url=route_base_url,
        route_host=route_host,
        fallback_host=DEFAULT_GPU_HOST,
        port=launch_plan.port,
    )
    parsed_route_url = urlparse(selected_route_base_url)
    selected_route_host = parsed_route_url.hostname or DEFAULT_GPU_HOST
    selected_route_port = parsed_route_url.port or _default_port_for_scheme(parsed_route_url.scheme)
    return RemoteRerankerProviderSetupPlan(
        service_name=selected_service_name,
        user=selected_user,
        group=selected_group,
        workdir=selected_workdir,
        python_bin=selected_python_bin,
        models_dir=selected_models_dir,
        model_dir_name=selected_model_dir_name,
        env_file=str(Path(selected_env_dir) / f"{selected_service_name}.env"),
        systemd_unit_file=str(Path(selected_systemd_dir) / f"{selected_service_name}.service"),
        route_base_url=selected_route_base_url,
        health_url=f"{selected_route_base_url}/healthz",
        request_smoke_command=(
            "./.venv/bin/python",
            "scripts/run_remote_reranker_request_smoke.py",
            "--base-url",
            selected_route_base_url,
            "--markdown-output",
            "artifacts/remote_reranker_request_smoke.md",
        ),
        operations_status_command=(
            "./.venv/bin/python",
            "scripts/run_remote_reranker_background.py",
            "smoke",
            "--route-host",
            selected_route_host,
            "--port",
            str(selected_route_port),
            "--markdown-output",
            "artifacts/remote_reranker_systemd_smoke.md",
        ),
        launch_plan=launch_plan,
        create_directory_commands=(
            f"mkdir -p {shlex.quote(selected_env_dir)}",
            f"mkdir -p {shlex.quote(selected_systemd_dir)}",
        ),
    )


def render_env_file(plan: RemoteRerankerProviderSetupPlan) -> str:
    lines = [
        "# Generated by scripts/setup_remote_reranker_provider.py",
        "# This file intentionally contains reranker runtime metadata only.",
        "PYTHONUNBUFFERED=1",
    ]
    for key, value in sorted(plan.launch_plan.environment.items()):
        lines.append(f"{key}={shlex.quote(value)}")
    return "\n".join(lines) + "\n"


def render_systemd_unit(
    plan: RemoteRerankerProviderSetupPlan,
    *,
    user_systemd: bool = False,
) -> str:
    unit_dependencies = (
        "" if user_systemd else "After=network-online.target\nWants=network-online.target\n"
    )
    service_identity = "" if user_systemd else f"User={plan.user}\nGroup={plan.group}\n"
    service_hardening = "" if user_systemd else "NoNewPrivileges=true\nPrivateTmp=true\n"
    install_target = "default.target" if user_systemd else "multi-user.target"
    return (
        "[Unit]\n"
        f"Description=NeX-PCX reranker provider ({plan.launch_plan.provider_name})\n"
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
        "KillSignal=SIGTERM\n"
        "TimeoutStopSec=120\n"
        f"{service_hardening}"
        "\n"
        "[Install]\n"
        f"WantedBy={install_target}\n"
    )


def render_operator_readme(
    plan: RemoteRerankerProviderSetupPlan,
    *,
    user_systemd: bool = False,
) -> str:
    systemctl = "systemctl --user" if user_systemd else "sudo systemctl"
    journalctl = "journalctl --user" if user_systemd else "sudo journalctl"
    lines = [
        "# Remote Reranker Provider systemd Setup",
        "",
        "Generated files are templates for the DGX reranker provider host.",
        "",
        "## Files",
        "",
        f"- Environment file: `{plan.env_file}`",
        f"- Unit file: `{plan.systemd_unit_file}`",
        f"- Unit name: `{plan.systemd_unit_name}`",
        f"- Scope: `{'user' if user_systemd else 'system'}`",
        "",
        "## Start",
        "",
        "```bash",
        f"{systemctl} daemon-reload",
        f"{systemctl} enable --now {plan.systemd_unit_name}",
        f"{systemctl} status {plan.systemd_unit_name} --no-pager",
        f"{journalctl} -u {plan.systemd_unit_name} -n 100 --no-pager",
        "```",
        "",
        "## Health Evidence",
        "",
        "```bash",
        f"curl -fsS {plan.health_url}",
        _quote_command(plan.request_smoke_command),
        "```",
        "",
    ]
    if user_systemd:
        lines.extend(
            [
                "## User Manager Hardening",
                "",
                "Confirm that the user manager survives logout/reboot before production use:",
                "",
                "```bash",
                f"loginctl show-user {plan.user} -p Linger",
                f"sudo loginctl enable-linger {plan.user}",
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def write_plan_files(
    plan: RemoteRerankerProviderSetupPlan,
    *,
    user_systemd: bool = False,
) -> tuple[Path, Path, Path]:
    env_file = Path(plan.env_file)
    systemd_unit_file = Path(plan.systemd_unit_file)
    readme_file = systemd_unit_file.parent / f"{plan.service_name}.README.md"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    systemd_unit_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text(render_env_file(plan), encoding="utf-8")
    systemd_unit_file.write_text(
        render_systemd_unit(plan, user_systemd=user_systemd),
        encoding="utf-8",
    )
    readme_file.write_text(
        render_operator_readme(plan, user_systemd=user_systemd),
        encoding="utf-8",
    )
    return env_file, systemd_unit_file, readme_file


def _plan_payload(plan: RemoteRerankerProviderSetupPlan) -> dict[str, object]:
    return {
        **asdict(plan),
        "systemd_unit_name": plan.systemd_unit_name,
        "launch_command": list(plan.launch_plan.command),
        "launch_shell_command": plan.launch_plan.shell_command,
        "request_smoke_shell_command": _quote_command(plan.request_smoke_command),
        "operations_status_shell_command": _quote_command(plan.operations_status_command),
    }


def _print_human_plan(
    plan: RemoteRerankerProviderSetupPlan,
    *,
    wrote_files: bool,
    user_systemd: bool,
) -> None:
    print("Remote reranker systemd setup plan")
    print(f"- service_name: {plan.service_name}")
    print(f"- scope: {'user' if user_systemd else 'system'}")
    print(f"- workdir: {plan.workdir}")
    print(f"- env_file: {plan.env_file}")
    print(f"- systemd_unit_file: {plan.systemd_unit_file}")
    print(f"- provider_base_url: {plan.route_base_url}")
    print(f"- health_url: {plan.health_url}")
    print(f"- launch_command: {_quote_command(plan.launch_plan.command)}")
    print(f"- request_smoke: {_quote_command(plan.request_smoke_command)}")
    print(f"- wrote_files: {wrote_files}")


def _normalize_path(path: str | None, *, default: str) -> str:
    selected_path = (path or default).strip()
    if not selected_path:
        raise ValueError("path value is required")
    return str(Path(selected_path).expanduser())


def _require_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _validate_service_name(service_name: str) -> str:
    selected_name = _require_nonblank(service_name, "service_name")
    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-@")
    if any(char not in allowed_chars for char in selected_name):
        raise ValueError("service_name may only contain letters, numbers, '.', '-', '_', or '@'")
    if selected_name.endswith(".service"):
        selected_name = selected_name[: -len(".service")]
    return _require_nonblank(selected_name, "service_name")


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
    selected_host = _require_nonblank(route_host or fallback_host, "route_host")
    if selected_host in {"0.0.0.0", "::"}:
        selected_host = "127.0.0.1"
    return f"http://{selected_host}:{port}"


def _default_port_for_scheme(scheme: str) -> int:
    if scheme == "https":
        return 443
    return 80


def _quote_command(command: tuple[str, ...] | list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate env and systemd unit files for a remote DGX reranker provider.",
    )
    parser.add_argument("--workdir", default=DEFAULT_GPU_WORKDIR)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--model-dir-name", default=DEFAULT_RERANKER_MODEL_DIR_NAME)
    parser.add_argument("--python-bin", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_RERANKER_PROVIDER_PORT)
    parser.add_argument("--route-host", default=None)
    parser.add_argument("--route-base-url", default=None)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--provider-name", default=DEFAULT_RERANKER_PROVIDER_NAME)
    parser.add_argument("--provider-model-id", default=DEFAULT_RERANKER_MODEL_ID)
    parser.add_argument("--profile-name", default=DEFAULT_RERANKER_PROFILE_NAME)
    parser.add_argument("--backend", default=RERANKER_PROVIDER_BACKEND_QWEN)
    parser.add_argument("--service-name", default=DEFAULT_RERANKER_SERVICE_NAME)
    parser.add_argument("--user", default=DEFAULT_GPU_USER)
    parser.add_argument("--group", default=None)
    parser.add_argument("--env-dir", default=None)
    parser.add_argument("--systemd-dir", default=None)
    parser.add_argument(
        "--user-systemd",
        action="store_true",
        help="Render a unit suitable for systemctl --user.",
    )
    parser.add_argument("--write-files", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_setup_plan(
            workdir=args.workdir,
            models_dir=args.models_dir,
            model_dir_name=args.model_dir_name,
            python_bin=args.python_bin,
            host=args.host,
            port=args.port,
            route_host=args.route_host,
            route_base_url=args.route_base_url,
            device=args.device,
            provider_name=args.provider_name,
            provider_model_id=args.provider_model_id,
            profile_name=args.profile_name,
            backend=args.backend,
            service_name=args.service_name,
            user=args.user,
            group=args.group,
            env_dir=args.env_dir,
            systemd_dir=args.systemd_dir,
        )
    except ValueError as exc:
        parser.error(str(exc))

    wrote_files = False
    if args.write_files:
        write_plan_files(plan, user_systemd=args.user_systemd)
        wrote_files = True

    if args.json:
        print(
            json.dumps(
                {
                    "wrote_files": wrote_files,
                    "user_systemd": args.user_systemd,
                    "plan": _plan_payload(plan),
                },
                ensure_ascii=False,
            )
        )
    else:
        _print_human_plan(plan, wrote_files=wrote_files, user_systemd=args.user_systemd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
