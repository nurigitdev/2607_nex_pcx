"""Download embedding model bundles into the local models directory."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.embedding_model_distribution import (  # noqa: E402
    EmbeddingModelDistribution,
    get_embedding_model_distribution,
    list_embedding_model_distributions,
    resolve_embedding_model_dir,
)


def _select_distributions(model_keys: list[str]) -> tuple[EmbeddingModelDistribution, ...]:
    if not model_keys:
        return list_embedding_model_distributions()
    return tuple(get_embedding_model_distribution(model_key) for model_key in model_keys)


def _distribution_payload(
    distribution: EmbeddingModelDistribution,
    *,
    models_dir: Path,
    revision: str,
) -> dict[str, object]:
    return {
        "model_key": distribution.model_key,
        "repo_id": distribution.repo_id,
        "revision": revision,
        "local_dir": str(resolve_embedding_model_dir(distribution, models_dir)),
        "profile_names": list(distribution.profile_names),
        "adapter_name": distribution.adapter_name,
        "note": distribution.note,
    }


def _print_plan(
    distributions: tuple[EmbeddingModelDistribution, ...],
    *,
    models_dir: Path,
    revision_override: str | None,
    output_json: bool,
) -> None:
    payload = [
        _distribution_payload(
            distribution,
            models_dir=models_dir,
            revision=revision_override or distribution.default_revision,
        )
        for distribution in distributions
    ]
    if output_json:
        print(json.dumps({"models_dir": str(models_dir), "models": payload}, ensure_ascii=False))
        return

    print(f"models_dir={models_dir}")
    for item in payload:
        profiles = ", ".join(str(profile) for profile in item["profile_names"])
        print(
            f"- {item['model_key']}: {item['repo_id']}@{item['revision']} "
            f"-> {item['local_dir']} [{profiles}]"
        )


def _snapshot_download(
    *,
    repo_id: str,
    revision: str,
    local_dir: Path,
) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install it with "
            '`./.venv/bin/pip install -e ".[models]"`.'
        ) from exc

    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download NeX_PCX embedding models into local model directories."
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model key to download. Repeat to download multiple keys. Defaults to all.",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Target model bundle root. Defaults to NEX_PCX_MODELS_DIR or models.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Override the manifest revision for all selected models.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the download plan without contacting Hugging Face.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the plan/result as JSON.",
    )
    args = parser.parse_args()

    settings = get_settings()
    models_dir = Path(args.models_dir) if args.models_dir else settings.embedding_models_dir
    distributions = _select_distributions(args.model)

    if args.dry_run:
        _print_plan(
            distributions,
            models_dir=models_dir,
            revision_override=args.revision,
            output_json=args.json,
        )
        return 0

    downloaded = []
    for distribution in distributions:
        revision = args.revision or distribution.default_revision
        local_dir = resolve_embedding_model_dir(distribution, models_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = _snapshot_download(
            repo_id=distribution.repo_id,
            revision=revision,
            local_dir=local_dir,
        )
        downloaded.append(
            {
                **_distribution_payload(
                    distribution,
                    models_dir=models_dir,
                    revision=revision,
                ),
                "snapshot_path": snapshot_path,
            }
        )

    if args.json:
        print(
            json.dumps(
                {"models_dir": str(models_dir), "downloaded": downloaded}, ensure_ascii=False
            )
        )
        return 0

    print(f"Downloaded {len(downloaded)} embedding model bundle(s) into {models_dir}")
    for item in downloaded:
        print(f"- {item['model_key']}: {item['snapshot_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
