"""Launch the standalone Qwen reranker provider process."""

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
from app.core.rerankers import (  # noqa: E402
    DEFAULT_RERANKER_MODEL_ID,
    DEFAULT_RERANKER_PROFILE_NAME,
)
from app.reranker_provider_service import (  # noqa: E402
    DEFAULT_RERANKER_MODEL_DIR_NAME,
    RERANKER_PROVIDER_BACKEND_MOCK,
    RERANKER_PROVIDER_BACKEND_QWEN,
)

DEFAULT_RERANKER_PROVIDER_NAME = "qwen-reranker-primary"
DEFAULT_RERANKER_PROVIDER_HOST = "127.0.0.1"
DEFAULT_RERANKER_PROVIDER_PORT = 9104
DEFAULT_RERANKER_PROVIDER_DEVICE = "cuda:0"
DEFAULT_RERANKER_PROVIDER_TORCH_DTYPE = "bfloat16"
RERANKER_PROVIDER_BACKENDS = (
    RERANKER_PROVIDER_BACKEND_QWEN,
    RERANKER_PROVIDER_BACKEND_MOCK,
)


@dataclass(frozen=True)
class RerankerProviderLaunchPlan:
    provider_name: str
    backend: str
    reranker_profile_name: str
    provider_model_id: str
    host: str
    port: int
    device: str
    torch_dtype: str | None
    models_dir: str
    model_dir_name: str
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
    *,
    python_bin: str,
    provider_name: str = DEFAULT_RERANKER_PROVIDER_NAME,
    backend: str = RERANKER_PROVIDER_BACKEND_QWEN,
    host: str = DEFAULT_RERANKER_PROVIDER_HOST,
    port: int = DEFAULT_RERANKER_PROVIDER_PORT,
    device: str = DEFAULT_RERANKER_PROVIDER_DEVICE,
    torch_dtype: str | None = DEFAULT_RERANKER_PROVIDER_TORCH_DTYPE,
    models_dir: Path | str | None = None,
    model_dir_name: str = DEFAULT_RERANKER_MODEL_DIR_NAME,
    provider_model_id: str = DEFAULT_RERANKER_MODEL_ID,
    reranker_profile_name: str = DEFAULT_RERANKER_PROFILE_NAME,
    reload: bool = False,
) -> RerankerProviderLaunchPlan:
    selected_python_bin = _require_nonblank(python_bin, "python_bin")
    selected_provider_name = _require_nonblank(provider_name, "provider_name")
    selected_backend = _require_nonblank(backend, "backend").lower()
    if selected_backend not in RERANKER_PROVIDER_BACKENDS:
        raise ValueError(f"Unsupported reranker provider backend: {backend}")
    selected_host = _require_nonblank(host, "host")
    if port <= 0 or port > 65535:
        raise ValueError("port must be between 1 and 65535")
    selected_device = _require_nonblank(device, "device")
    selected_torch_dtype = (torch_dtype or "").strip() or None
    selected_models_dir = _require_nonblank(
        str(models_dir or get_settings().embedding_models_dir),
        "models_dir",
    )
    selected_model_dir_name = _require_nonblank(model_dir_name, "model_dir_name")
    selected_provider_model_id = _require_nonblank(provider_model_id, "provider_model_id")
    selected_reranker_profile_name = _require_nonblank(
        reranker_profile_name,
        "reranker_profile_name",
    )

    command = [
        selected_python_bin,
        "-m",
        "uvicorn",
        "app.reranker_provider_service:app",
        "--host",
        selected_host,
        "--port",
        str(port),
    ]
    if reload:
        command.append("--reload")

    return RerankerProviderLaunchPlan(
        provider_name=selected_provider_name,
        backend=selected_backend,
        reranker_profile_name=selected_reranker_profile_name,
        provider_model_id=selected_provider_model_id,
        host=selected_host,
        port=port,
        device=selected_device,
        torch_dtype=selected_torch_dtype,
        models_dir=selected_models_dir,
        model_dir_name=selected_model_dir_name,
        command=tuple(command),
        environment={
            "NEX_PCX_RERANKER_PROVIDER_BACKEND": selected_backend,
            "NEX_PCX_RERANKER_PROVIDER_MODEL_ID": selected_provider_model_id,
            "NEX_PCX_RERANKER_PROVIDER_PROFILE_NAME": selected_reranker_profile_name,
            "NEX_PCX_RERANKER_PROVIDER_DEVICE": selected_device,
            **(
                {"NEX_PCX_RERANKER_PROVIDER_TORCH_DTYPE": selected_torch_dtype}
                if selected_torch_dtype is not None
                else {}
            ),
            "NEX_PCX_RERANKER_PROVIDER_MODELS_DIR": selected_models_dir,
            "NEX_PCX_RERANKER_PROVIDER_MODEL_DIR_NAME": selected_model_dir_name,
        },
    )


def launch_provider(plan: RerankerProviderLaunchPlan) -> None:
    environment = os.environ.copy()
    environment.update(plan.environment)
    os.execvpe(plan.command[0], list(plan.command), environment)


def _require_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _plan_payload(plan: RerankerProviderLaunchPlan) -> dict[str, object]:
    return {
        **asdict(plan),
        "command": list(plan.command),
        "base_url": plan.base_url,
        "shell_command": plan.shell_command,
    }


def _print_human_plan(plan: RerankerProviderLaunchPlan, *, dry_run: bool) -> None:
    action = "Dry-run launch plan" if dry_run else "Launching reranker provider"
    print(f"{action}: {plan.provider_name}")
    print(f"- backend: {plan.backend}")
    print(f"- profile: {plan.reranker_profile_name}")
    print(f"- base_url: {plan.base_url}")
    print(f"- provider_model_id: {plan.provider_model_id}")
    print(f"- torch_dtype: {plan.torch_dtype}")
    print(f"- models_dir: {plan.models_dir}")
    print(f"- model_dir_name: {plan.model_dir_name}")
    print(f"- command: {plan.shell_command}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the standalone Qwen reranker provider process.",
    )
    parser.add_argument("--provider-name", default=DEFAULT_RERANKER_PROVIDER_NAME)
    parser.add_argument(
        "--backend",
        choices=RERANKER_PROVIDER_BACKENDS,
        default=RERANKER_PROVIDER_BACKEND_QWEN,
    )
    parser.add_argument("--host", default=DEFAULT_RERANKER_PROVIDER_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_RERANKER_PROVIDER_PORT)
    parser.add_argument("--device", default=DEFAULT_RERANKER_PROVIDER_DEVICE)
    parser.add_argument("--torch-dtype", default=DEFAULT_RERANKER_PROVIDER_TORCH_DTYPE)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--model-dir-name", default=DEFAULT_RERANKER_MODEL_DIR_NAME)
    parser.add_argument("--provider-model-id", default=DEFAULT_RERANKER_MODEL_ID)
    parser.add_argument("--profile-name", default=DEFAULT_RERANKER_PROFILE_NAME)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        plan = build_launch_plan(
            python_bin=args.python_bin,
            provider_name=args.provider_name,
            backend=args.backend,
            host=args.host,
            port=args.port,
            device=args.device,
            torch_dtype=args.torch_dtype,
            models_dir=args.models_dir,
            model_dir_name=args.model_dir_name,
            provider_model_id=args.provider_model_id,
            reranker_profile_name=args.profile_name,
            reload=args.reload,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.json:
        print(
            json.dumps(
                {"dry_run": args.dry_run, "plan": _plan_payload(plan)},
                ensure_ascii=False,
            )
        )
    else:
        _print_human_plan(plan, dry_run=args.dry_run)

    if args.dry_run:
        return 0

    launch_provider(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
