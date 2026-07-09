import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.embedding_model_distribution import (
    InvalidEmbeddingModelDistributionError,
    audit_embedding_model_readiness,
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


def test_embedding_model_readiness_audit_reports_missing_and_ready_models(tmp_path: Path) -> None:
    kure_dir = tmp_path / "kure_v1"
    kure_dir.mkdir()
    (kure_dir / "config.json").write_text("{}", encoding="utf-8")
    (kure_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (kure_dir / "model.safetensors").write_bytes(b"weights")

    bge_dir = tmp_path / "bge_m3"
    bge_dir.mkdir()
    (bge_dir / "config.json").write_text("{}", encoding="utf-8")

    readiness = {
        item.distribution.model_key: item for item in audit_embedding_model_readiness(tmp_path)
    }

    assert readiness["kure_v1"].ready is True
    assert readiness["kure_v1"].has_tokenizer is True
    assert readiness["kure_v1"].file_count == 3
    assert readiness["kure_v1"].total_size_bytes > 0
    assert readiness["bge_m3"].exists is True
    assert readiness["bge_m3"].ready is False
    assert readiness["qwen3_embedding_4b"].exists is False
    assert readiness["qwen3_embedding_4b"].ready is False


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
