"""Embedding model distribution manifest and local path helpers."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EmbeddingModelDistribution:
    model_key: str
    repo_id: str
    local_dir_name: str
    profile_names: tuple[str, ...]
    default_revision: str = "main"
    adapter_name: str = "sentence_transformers"
    note: str = ""


EMBEDDING_MODEL_DISTRIBUTIONS: tuple[EmbeddingModelDistribution, ...] = (
    EmbeddingModelDistribution(
        model_key="kure_v1",
        repo_id="nlpai-lab/KURE-v1",
        local_dir_name="kure_v1",
        profile_names=("kure_v1_1024",),
        adapter_name="sentence_transformers",
        note="Korean retrieval baseline.",
    ),
    EmbeddingModelDistribution(
        model_key="bge_m3",
        repo_id="BAAI/bge-m3",
        local_dir_name="bge_m3",
        profile_names=("bge_m3_1024",),
        adapter_name="sentence_transformers",
        note="Multilingual dense retrieval baseline.",
    ),
    EmbeddingModelDistribution(
        model_key="qwen3_embedding_4b",
        repo_id="Qwen/Qwen3-Embedding-4B",
        local_dir_name="qwen3_embedding_4b",
        profile_names=("qwen3_4b_1000", "qwen3_4b_2560"),
        adapter_name="qwen_embedding",
        note="Shared local model for both Qwen output-dimension profiles.",
    ),
)


class InvalidEmbeddingModelDistributionError(ValueError):
    """Raised when an embedding model distribution lookup is invalid."""


def list_embedding_model_distributions() -> tuple[EmbeddingModelDistribution, ...]:
    return EMBEDDING_MODEL_DISTRIBUTIONS


def get_embedding_model_distribution(model_key: str) -> EmbeddingModelDistribution:
    normalized_key = model_key.strip()
    if not normalized_key:
        raise InvalidEmbeddingModelDistributionError("model_key is required")

    for distribution in EMBEDDING_MODEL_DISTRIBUTIONS:
        if distribution.model_key == normalized_key:
            return distribution

    raise InvalidEmbeddingModelDistributionError(f"Unsupported embedding model: {model_key}")


def resolve_embedding_model_dir(
    distribution: EmbeddingModelDistribution,
    models_dir: Path,
) -> Path:
    return models_dir / distribution.local_dir_name
