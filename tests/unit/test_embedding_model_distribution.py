import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.embedding_model_distribution import (
    InvalidEmbeddingModelDistributionError,
    get_embedding_model_distribution,
    list_embedding_model_distributions,
    resolve_embedding_model_dir,
)


def test_embedding_model_distribution_manifest_maps_profiles_to_three_models() -> None:
    distributions = list_embedding_model_distributions()

    assert [distribution.model_key for distribution in distributions] == [
        "kure_v1",
        "bge_m3",
        "qwen3_embedding_4b",
    ]
    assert [distribution.repo_id for distribution in distributions] == [
        "nlpai-lab/KURE-v1",
        "BAAI/bge-m3",
        "Qwen/Qwen3-Embedding-4B",
    ]
    assert distributions[2].profile_names == ("qwen3_4b_1000", "qwen3_4b_2560")


def test_embedding_model_distribution_resolves_local_model_directories() -> None:
    models_dir = Path("models")
    distribution = get_embedding_model_distribution("bge_m3")

    assert resolve_embedding_model_dir(distribution, models_dir).as_posix() == "models/bge_m3"


def test_embedding_model_distribution_rejects_unknown_model_key() -> None:
    with pytest.raises(InvalidEmbeddingModelDistributionError, match="Unsupported"):
        get_embedding_model_distribution("missing")


def test_download_embedding_models_script_prints_json_dry_run_plan() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/download_embedding_models.py",
            "--model",
            "qwen3_embedding_4b",
            "--models-dir",
            "/tmp/nex_pcx_models",
            "--revision",
            "test-revision",
            "--dry-run",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert payload["models_dir"] == "/tmp/nex_pcx_models"
    assert payload["models"] == [
        {
            "model_key": "qwen3_embedding_4b",
            "repo_id": "Qwen/Qwen3-Embedding-4B",
            "revision": "test-revision",
            "local_dir": "/tmp/nex_pcx_models/qwen3_embedding_4b",
            "profile_names": ["qwen3_4b_1000", "qwen3_4b_2560"],
            "adapter_name": "qwen_embedding",
            "note": "Shared local model for both Qwen output-dimension profiles.",
        }
    ]
