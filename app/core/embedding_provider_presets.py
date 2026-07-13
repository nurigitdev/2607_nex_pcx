"""Local launch and route registration presets for embedding providers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingProviderPreset:
    preset_name: str
    provider_name: str
    backend: str
    model_key: str
    provider_model_id: str
    profile_names: tuple[str, ...]
    default_host: str
    default_port: int

    @property
    def default_base_url(self) -> str:
        return f"http://{self.default_host}:{self.default_port}"


EMBEDDING_PROVIDER_PRESETS: tuple[EmbeddingProviderPreset, ...] = (
    EmbeddingProviderPreset(
        preset_name="kure",
        provider_name="kure-primary",
        backend="sentence_transformers",
        model_key="kure_v1",
        provider_model_id="local-kure-v1",
        profile_names=("kure_v1_1024",),
        default_host="127.0.0.1",
        default_port=9101,
    ),
    EmbeddingProviderPreset(
        preset_name="bge",
        provider_name="bge-primary",
        backend="sentence_transformers",
        model_key="bge_m3",
        provider_model_id="local-bge-m3",
        profile_names=("bge_m3_1024",),
        default_host="127.0.0.1",
        default_port=9102,
    ),
    EmbeddingProviderPreset(
        preset_name="qwen",
        provider_name="qwen-primary",
        backend="qwen_embedding",
        model_key="qwen3_embedding_4b",
        provider_model_id="local-qwen3-embedding-4b",
        profile_names=("qwen3_4b_1000", "qwen3_4b_2560"),
        default_host="127.0.0.1",
        default_port=9103,
    ),
)


class InvalidEmbeddingProviderPresetError(ValueError):
    """Raised when a provider preset is not known."""


def list_embedding_provider_presets() -> tuple[EmbeddingProviderPreset, ...]:
    return EMBEDDING_PROVIDER_PRESETS


def get_embedding_provider_preset(preset_name: str) -> EmbeddingProviderPreset:
    normalized_name = preset_name.strip().lower()
    if not normalized_name:
        raise InvalidEmbeddingProviderPresetError("preset_name is required")

    for preset in EMBEDDING_PROVIDER_PRESETS:
        if preset.preset_name == normalized_name:
            return preset

    raise InvalidEmbeddingProviderPresetError(
        f"Unsupported embedding provider preset: {preset_name}"
    )
