import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.embedding_model_distribution import (
    InvalidEmbeddingModelDistributionError,
    audit_embedding_model_readiness,
    get_embedding_model_distribution,
    get_embedding_model_distribution_for_profile,
    list_embedding_model_distributions,
    resolve_embedding_model_dir,
)


def _load_check_embedding_models_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "check_embedding_models.py"
    spec = importlib.util.spec_from_file_location("check_embedding_models_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_embedding_models = _load_check_embedding_models_module()


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


def test_embedding_model_distribution_resolves_distribution_by_profile_name() -> None:
    assert (
        get_embedding_model_distribution_for_profile("qwen3_4b_2560").model_key
        == "qwen3_embedding_4b"
    )

    with pytest.raises(InvalidEmbeddingModelDistributionError, match="Unsupported"):
        get_embedding_model_distribution_for_profile("missing_profile")


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


def test_check_embedding_models_script_dry_run_defaults_to_sentence_transformers_models(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_embedding_models.py",
            "--models-dir",
            str(tmp_path),
            "--dry-run",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert payload["models_dir"] == str(tmp_path)
    assert payload["dry_run"] is True
    assert [model["model_key"] for model in payload["models"]] == ["kure_v1", "bge_m3"]
    assert payload["models"][0]["ready"] is False


def test_embedding_model_smoke_uses_local_sentence_transformer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "kure_v1"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    calls = {}

    class FakeSentenceTransformer:
        def __init__(self, model_source, **kwargs) -> None:
            calls["model_source"] = model_source
            calls["kwargs"] = kwargs

        def encode(self, texts, **kwargs):
            calls["texts"] = texts
            calls["encode_kwargs"] = kwargs
            return [[1.0 for _ in range(1024)] for _ in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    result = check_embedding_models.run_embedding_model_smoke(
        get_embedding_model_distribution("kure_v1"),
        models_dir=tmp_path,
        device="cpu",
        texts=("local smoke text",),
    )

    assert result.ok is True
    assert result.status == "passed"
    assert result.input_count == 1
    assert result.dimension == 1024
    assert result.local_dir == str(model_dir)
    assert calls["model_source"] == str(model_dir)
    assert calls["kwargs"] == {"device": "cpu"}
    assert calls["texts"] == ["local smoke text"]
