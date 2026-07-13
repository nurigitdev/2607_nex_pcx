"""Launch a local embedding provider process from a known provider preset."""

import argparse
import json
import os
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.embedding_provider_presets import (  # noqa: E402
    EmbeddingProviderPreset,
    get_embedding_provider_preset,
    list_embedding_provider_presets,
)


@dataclass(frozen=True)
class EmbeddingProviderLaunchPlan:
    preset_name: str
    provider_name: str
    backend: str
    model_key: str
    profile_names: tuple[str, ...]
    provider_model_id: str
    host: str
    port: int
    device: str
    models_dir: str
    command: tuple[str, ...]
    environment: dict[str, str]

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def shell_command(self) -> str:
        env_parts = [
            f"{key}={shlex.quote(value)}" for key, value in sorted(self.environment.items())
        ]
        command = " ".join(shlex.quote(part) for part in self.command)
        return " ".join([*env_parts, command])


def build_launch_plan(
    preset: EmbeddingProviderPreset,
    *,
    python_bin: str,
    host: str | None = None,
    port: int | None = None,
    device: str = "cpu",
    models_dir: Path | str | None = None,
    provider_model_id: str | None = None,
    reload: bool = False,
) -> EmbeddingProviderLaunchPlan:
    selected_host = host or preset.default_host
    selected_port = port or preset.default_port
    selected_models_dir = str(models_dir or get_settings().embedding_models_dir)
    selected_provider_model_id = provider_model_id or preset.provider_model_id
    command = [
        python_bin,
        "-m",
        "uvicorn",
        "app.embedding_provider_service:app",
        "--host",
        selected_host,
        "--port",
        str(selected_port),
    ]
    if reload:
        command.append("--reload")

    return EmbeddingProviderLaunchPlan(
        preset_name=preset.preset_name,
        provider_name=preset.provider_name,
        backend=preset.backend,
        model_key=preset.model_key,
        profile_names=preset.profile_names,
        provider_model_id=selected_provider_model_id,
        host=selected_host,
        port=selected_port,
        device=device,
        models_dir=selected_models_dir,
        command=tuple(command),
        environment={
            "NEX_PCX_PROVIDER_BACKEND": preset.backend,
            "NEX_PCX_PROVIDER_MODEL_KEY": preset.model_key,
            "NEX_PCX_PROVIDER_PROFILE_NAMES": ",".join(preset.profile_names),
            "NEX_PCX_PROVIDER_MODEL_ID": selected_provider_model_id,
            "NEX_PCX_PROVIDER_DEVICE": device,
            "NEX_PCX_PROVIDER_MODELS_DIR": selected_models_dir,
        },
    )


def launch_provider(plan: EmbeddingProviderLaunchPlan) -> None:
    environment = os.environ.copy()
    environment.update(plan.environment)
    os.execvpe(plan.command[0], list(plan.command), environment)


def _plan_payload(plan: EmbeddingProviderLaunchPlan) -> dict[str, object]:
    return {
        **asdict(plan),
        "profile_names": list(plan.profile_names),
        "command": list(plan.command),
        "base_url": plan.base_url,
        "shell_command": plan.shell_command,
    }


def _print_human_plan(plan: EmbeddingProviderLaunchPlan, *, dry_run: bool) -> None:
    action = "Dry-run launch plan" if dry_run else "Launching embedding provider"
    print(f"{action}: {plan.preset_name}")
    print(f"- provider_name: {plan.provider_name}")
    print(f"- backend: {plan.backend}")
    print(f"- model_key: {plan.model_key}")
    print(f"- profiles: {', '.join(plan.profile_names)}")
    print(f"- base_url: {plan.base_url}")
    print(f"- provider_model_id: {plan.provider_model_id}")
    print(f"- models_dir: {plan.models_dir}")
    print(f"- command: {plan.shell_command}")


def main() -> int:
    preset_names = [preset.preset_name for preset in list_embedding_provider_presets()]
    parser = argparse.ArgumentParser(
        description="Launch a local embedding provider from a NeX_PCX provider preset.",
    )
    parser.add_argument("--provider", choices=preset_names, required=True)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--provider-model-id", default=None)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    preset = get_embedding_provider_preset(args.provider)
    plan = build_launch_plan(
        preset,
        python_bin=args.python_bin,
        host=args.host,
        port=args.port,
        device=args.device,
        models_dir=args.models_dir,
        provider_model_id=args.provider_model_id,
        reload=args.reload,
    )

    if args.json:
        print(
            json.dumps({"dry_run": args.dry_run, "plan": _plan_payload(plan)}, ensure_ascii=False)
        )
    else:
        _print_human_plan(plan, dry_run=args.dry_run)

    if args.dry_run:
        return 0

    launch_provider(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
