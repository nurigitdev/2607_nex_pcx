"""Service startup template generation for NeX_PCX operations."""

import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_APP_SERVICE_NAME = "nex-pcx-web"
DEFAULT_PIPELINE_WORKER_SERVICE_NAME = "nex-pcx-pipeline-worker"
DEFAULT_EMBEDDING_WORKER_SERVICE_NAME = "nex-pcx-embedding-worker"
DEFAULT_WEB_HOST = "0.0.0.0"
DEFAULT_WEB_PORT = 8000
DEFAULT_RESTART_SECONDS = 5
DEFAULT_CHUNK_POLICY_NAMES = (
    "heading_512_64",
    "heading_1000_200",
    "heading_1500_200",
)


@dataclass(frozen=True)
class ServiceTemplate:
    service_name: str
    description: str
    command: tuple[str, ...]
    restart: str
    restart_seconds: int
    unit_file: str

    @property
    def unit_name(self) -> str:
        return Path(self.unit_file).name

    @property
    def shell_command(self) -> str:
        return _quote_command(self.command)


@dataclass(frozen=True)
class ServiceStartupTemplatePlan:
    workdir: str
    user: str
    group: str
    python_bin: str
    env_file: str
    systemd_dir: str
    log_dir: str
    user_systemd: bool
    environment: dict[str, str]
    services: tuple[ServiceTemplate, ...]


def build_service_startup_template_plan(
    *,
    workdir: str,
    user: str,
    group: str | None = None,
    python_bin: str | None = None,
    output_dir: str | None = None,
    web_host: str = DEFAULT_WEB_HOST,
    web_port: int = DEFAULT_WEB_PORT,
    database_url_placeholder: str = "postgresql://<user>:<password>@127.0.0.1:5432/nex_pcx_app",
    upload_storage_dir: str | None = None,
    models_dir: str | None = None,
    environment_name: str = "production",
    restart_seconds: int = DEFAULT_RESTART_SECONDS,
    chunk_policy_names: tuple[str, ...] = DEFAULT_CHUNK_POLICY_NAMES,
    user_systemd: bool = False,
) -> ServiceStartupTemplatePlan:
    selected_workdir = _normalize_path(workdir, name="workdir")
    selected_output_dir = _normalize_path(
        output_dir,
        name="output_dir",
        default=str(Path(selected_workdir) / "deployment"),
    )
    selected_python_bin = _normalize_path(
        python_bin,
        name="python_bin",
        default=str(Path(selected_workdir) / ".venv" / "bin" / "python"),
    )
    selected_user = _require_non_empty(user, name="user")
    selected_group = _require_non_empty(group or selected_user, name="group")
    selected_restart_seconds = _validate_positive_int(
        restart_seconds,
        name="restart_seconds",
    )
    selected_web_port = _validate_positive_int(web_port, name="web_port")
    selected_web_host = _require_non_empty(web_host, name="web_host")
    selected_chunk_policy_names = _validate_chunk_policy_names(chunk_policy_names)

    env_file = str(Path(selected_output_dir) / "env" / "nex-pcx.env")
    systemd_dir = str(Path(selected_output_dir) / "systemd")
    log_dir = str(Path(selected_output_dir) / "logs")
    environment = {
        "NEX_PCX_ENV": environment_name,
        "NEX_PCX_DATABASE_URL": database_url_placeholder,
        "NEX_PCX_UPLOAD_STORAGE_DIR": upload_storage_dir
        or str(Path(selected_workdir) / "storage" / "uploads"),
        "NEX_PCX_MODELS_DIR": models_dir or str(Path(selected_workdir) / "models"),
        "NEX_PCX_EMBEDDING_PROVIDER_MODE": "remote",
        "NEX_PCX_EMBEDDING_REQUIRE_ROUTE_READINESS": "true",
        "NEX_PCX_EMBEDDING_ROUTE_READINESS_FAILURE_MODE": "defer",
        "NEX_PCX_EMBEDDING_ROUTE_READINESS_DEFER_SECONDS": "300",
        "PYTHONUNBUFFERED": "1",
    }
    services = (
        ServiceTemplate(
            service_name=DEFAULT_APP_SERVICE_NAME,
            description="NeX-PCX FastAPI web application",
            command=(
                selected_python_bin,
                "-m",
                "uvicorn",
                "app.main:create_app",
                "--factory",
                "--host",
                selected_web_host,
                "--port",
                str(selected_web_port),
            ),
            restart="on-failure",
            restart_seconds=selected_restart_seconds,
            unit_file=str(Path(systemd_dir) / f"{DEFAULT_APP_SERVICE_NAME}.service"),
        ),
        ServiceTemplate(
            service_name=DEFAULT_PIPELINE_WORKER_SERVICE_NAME,
            description="NeX-PCX pipeline queue worker",
            command=(
                selected_python_bin,
                "scripts/process_pipeline_job.py",
                "--chunk-policy-names",
                *selected_chunk_policy_names,
            ),
            restart="always",
            restart_seconds=selected_restart_seconds,
            unit_file=str(Path(systemd_dir) / f"{DEFAULT_PIPELINE_WORKER_SERVICE_NAME}.service"),
        ),
        ServiceTemplate(
            service_name=DEFAULT_EMBEDDING_WORKER_SERVICE_NAME,
            description="NeX-PCX route-aware embedding worker",
            command=(
                selected_python_bin,
                "scripts/process_embedding_job.py",
                "--provider-source",
                "route",
                "--require-route-readiness",
                "--limit",
                "20",
            ),
            restart="always",
            restart_seconds=selected_restart_seconds,
            unit_file=str(Path(systemd_dir) / f"{DEFAULT_EMBEDDING_WORKER_SERVICE_NAME}.service"),
        ),
    )
    return ServiceStartupTemplatePlan(
        workdir=selected_workdir,
        user=selected_user,
        group=selected_group,
        python_bin=selected_python_bin,
        env_file=env_file,
        systemd_dir=systemd_dir,
        log_dir=log_dir,
        user_systemd=user_systemd,
        environment=environment,
        services=services,
    )


def render_env_file(plan: ServiceStartupTemplatePlan) -> str:
    lines = [
        "# Generated by scripts/render_service_startup_templates.py",
        "# Replace placeholders before installing these units.",
    ]
    for key, value in sorted(plan.environment.items()):
        lines.append(f"{key}={shlex.quote(value)}")
    return "\n".join(lines) + "\n"


def render_systemd_unit(plan: ServiceStartupTemplatePlan, service: ServiceTemplate) -> str:
    unit_lines = [
        "[Unit]",
        f"Description={service.description}",
    ]
    if not plan.user_systemd:
        unit_lines.extend(
            [
                "After=network-online.target postgresql.service",
                "Wants=network-online.target",
            ]
        )
    service_lines = [
        "[Service]",
        "Type=simple",
    ]
    if not plan.user_systemd:
        service_lines.extend(
            [
                f"User={plan.user}",
                f"Group={plan.group}",
            ]
        )
    service_lines.extend(
        [
            f"WorkingDirectory={plan.workdir}",
            f"EnvironmentFile={plan.env_file}",
            f"ExecStart={service.shell_command}",
            f"Restart={service.restart}",
            f"RestartSec={service.restart_seconds}",
            "KillSignal=SIGTERM",
            "TimeoutStopSec=60",
        ]
    )
    if not plan.user_systemd:
        service_lines.extend(
            [
                "NoNewPrivileges=true",
                "PrivateTmp=true",
            ]
        )
    install_target = "default.target" if plan.user_systemd else "multi-user.target"
    return (
        "\n".join(unit_lines)
        + "\n\n"
        + "\n".join(service_lines)
        + "\n\n"
        + "[Install]\n"
        + f"WantedBy={install_target}\n"
    )


def render_operator_readme(plan: ServiceStartupTemplatePlan) -> str:
    lines = [
        "# NeX-PCX Service Startup Templates",
        "",
        "Generated files are templates. Review paths and replace secrets before use.",
        "",
        "## Files",
        "",
        f"- Environment file: `{plan.env_file}`",
        f"- Systemd directory: `{plan.systemd_dir}`",
        f"- Suggested log directory: `{plan.log_dir}`",
        f"- Systemd scope: `{'user' if plan.user_systemd else 'system'}`",
        "",
        "## Services",
        "",
        "| Service | Restart | Command |",
        "| --- | --- | --- |",
    ]
    for service in plan.services:
        lines.append(
            "| "
            f"{_md_cell(service.unit_name)} | "
            f"{_md_cell(service.restart)} | "
            f"{_md_cell(service.shell_command)} |"
        )
    lines.extend(
        [
            "",
            "## Install Flow",
            "",
            "1. Replace placeholder values in the environment file.",
            "2. Copy unit files into the systemd unit directory used by the host.",
            "3. Run `systemctl --user daemon-reload` for user units or "
            "`systemctl daemon-reload` for system units.",
            "4. Start the web service first, then pipeline and embedding workers.",
            "5. Run `scripts/validate_operations_startup.py --strict` after start.",
            "",
        ]
    )
    return "\n".join(lines)


def write_service_startup_templates(plan: ServiceStartupTemplatePlan) -> tuple[Path, ...]:
    env_file = Path(plan.env_file)
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text(render_env_file(plan), encoding="utf-8")

    unit_paths = []
    for service in plan.services:
        unit_file = Path(service.unit_file)
        unit_file.parent.mkdir(parents=True, exist_ok=True)
        unit_file.write_text(render_systemd_unit(plan, service), encoding="utf-8")
        unit_paths.append(unit_file)

    readme_file = Path(plan.systemd_dir).parent / "README.md"
    readme_file.write_text(render_operator_readme(plan), encoding="utf-8")
    return (env_file, *unit_paths, readme_file)


def service_startup_template_plan_payload(
    plan: ServiceStartupTemplatePlan,
) -> dict[str, object]:
    return {
        **asdict(plan),
        "environment": {
            key: _mask_if_secret(key, value) for key, value in plan.environment.items()
        },
        "services": [
            {
                **asdict(service),
                "unit_name": service.unit_name,
                "shell_command": service.shell_command,
            }
            for service in plan.services
        ],
        "env_preview": render_env_file(plan),
        "systemd_previews": {
            service.unit_name: render_systemd_unit(plan, service) for service in plan.services
        },
        "operator_readme_preview": render_operator_readme(plan),
    }


def render_service_startup_template_plan_json(
    plan: ServiceStartupTemplatePlan,
    *,
    pretty: bool = False,
) -> str:
    return json.dumps(
        service_startup_template_plan_payload(plan),
        ensure_ascii=False,
        indent=2 if pretty else None,
    )


def _normalize_path(
    path: str | None,
    *,
    name: str,
    default: str | None = None,
) -> str:
    selected_path = (path or default or "").strip()
    if not selected_path:
        raise ValueError(f"{name} is required")
    return str(Path(selected_path).expanduser())


def _require_non_empty(value: str, *, name: str) -> str:
    selected_value = value.strip()
    if not selected_value:
        raise ValueError(f"{name} is required")
    return selected_value


def _validate_positive_int(value: int, *, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _validate_chunk_policy_names(chunk_policy_names: tuple[str, ...]) -> tuple[str, ...]:
    selected_names = tuple(name.strip() for name in chunk_policy_names if name.strip())
    if not selected_names:
        raise ValueError("at least one chunk policy name is required")
    return selected_names


def _quote_command(command: tuple[str, ...]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _mask_if_secret(key: str, value: str) -> str:
    if "PASSWORD" in key or key.endswith("_URL"):
        return "***" if value else ""
    return value


def _md_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")
