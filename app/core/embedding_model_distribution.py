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


@dataclass(frozen=True)
class EmbeddingModelReadiness:
    distribution: EmbeddingModelDistribution
    local_dir: Path
    exists: bool
    ready: bool
    has_config: bool
    has_tokenizer: bool
    has_model_weights: bool
    file_count: int
    total_size_bytes: int


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

TOKENIZER_FILE_NAMES = frozenset(
    {
        "tokenizer.json",
        "tokenizer_config.json",
        "sentencepiece.bpe.model",
        "spiece.model",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
    }
)
MODEL_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")
MODEL_WEIGHT_FILE_NAMES = frozenset(
    {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
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


def audit_embedding_model_readiness(
    models_dir: Path,
) -> tuple[EmbeddingModelReadiness, ...]:
    return tuple(
        audit_single_embedding_model_readiness(distribution, models_dir)
        for distribution in EMBEDDING_MODEL_DISTRIBUTIONS
    )


def audit_single_embedding_model_readiness(
    distribution: EmbeddingModelDistribution,
    models_dir: Path,
) -> EmbeddingModelReadiness:
    local_dir = resolve_embedding_model_dir(distribution, models_dir)
    exists = local_dir.is_dir()
    file_count = 0
    total_size_bytes = 0
    has_config = False
    has_tokenizer = False
    has_model_weights = False

    if exists:
        for path in local_dir.rglob("*"):
            if not path.is_file():
                continue
            file_count += 1
            try:
                total_size_bytes += path.stat().st_size
            except OSError:
                pass
            file_name = path.name
            if file_name == "config.json":
                has_config = True
            if file_name in TOKENIZER_FILE_NAMES:
                has_tokenizer = True
            if file_name in MODEL_WEIGHT_FILE_NAMES or file_name.endswith(MODEL_WEIGHT_SUFFIXES):
                has_model_weights = True

    return EmbeddingModelReadiness(
        distribution=distribution,
        local_dir=local_dir,
        exists=exists,
        ready=exists and has_config and has_model_weights,
        has_config=has_config,
        has_tokenizer=has_tokenizer,
        has_model_weights=has_model_weights,
        file_count=file_count,
        total_size_bytes=total_size_bytes,
    )
