"""Local launch and route registration presets for embedding providers."""

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from app.core.embedding_provider_routes import EmbeddingProviderRouteInput


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


@dataclass(frozen=True)
class EmbeddingProviderPresetRoutePlan:
    preset_name: str
    profile_name: str
    provider_name: str
    provider_mode: str
    provider_base_url: str
    provider_port: int | None
    timeout_seconds: float
    priority: int
    is_active: bool
    health_check_enabled: bool
    runtime_metadata: dict[str, Any]

    def to_route_input(self) -> EmbeddingProviderRouteInput:
        return EmbeddingProviderRouteInput(
            profile_name=self.profile_name,
            provider_name=self.provider_name,
            provider_mode=self.provider_mode,
            provider_base_url=self.provider_base_url,
            timeout_seconds=self.timeout_seconds,
            priority=self.priority,
            is_active=self.is_active,
            health_check_enabled=self.health_check_enabled,
            runtime_metadata=self.runtime_metadata,
        )


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


def build_embedding_provider_preset_route_plans(
    preset: EmbeddingProviderPreset,
    *,
    host: str | None = None,
    port: int | None = None,
    base_url: str | None = None,
    provider_name: str | None = None,
    timeout_seconds: float = 30.0,
    priority: int = 100,
    is_active: bool = True,
    health_check_enabled: bool = True,
    runtime_metadata: dict[str, Any] | None = None,
    metadata_source: str = "preset_registration",
) -> tuple[EmbeddingProviderPresetRoutePlan, ...]:
    selected_base_url = _resolve_preset_base_url(preset, host=host, port=port, base_url=base_url)
    selected_port = urlparse(selected_base_url).port or port or preset.default_port
    selected_provider_name = (provider_name or preset.provider_name).strip()
    if not selected_provider_name:
        raise InvalidEmbeddingProviderPresetError("provider_name is required")
    if timeout_seconds <= 0:
        raise InvalidEmbeddingProviderPresetError("timeout_seconds must be greater than 0")
    if priority < 0:
        raise InvalidEmbeddingProviderPresetError("priority must be greater than or equal to 0")

    shared_metadata = {
        "preset_name": preset.preset_name,
        "backend": preset.backend,
        "model_key": preset.model_key,
        "provider_model_id": preset.provider_model_id,
        "provider_port": selected_port,
        "source": metadata_source,
        **dict(runtime_metadata or {}),
    }
    return tuple(
        EmbeddingProviderPresetRoutePlan(
            preset_name=preset.preset_name,
            profile_name=profile_name,
            provider_name=selected_provider_name,
            provider_mode="remote",
            provider_base_url=selected_base_url,
            provider_port=selected_port,
            timeout_seconds=timeout_seconds,
            priority=priority,
            is_active=is_active,
            health_check_enabled=health_check_enabled,
            runtime_metadata={
                **shared_metadata,
                "profile_name": profile_name,
                "profile_names": list(preset.profile_names),
            },
        )
        for profile_name in preset.profile_names
    )


def _resolve_preset_base_url(
    preset: EmbeddingProviderPreset,
    *,
    host: str | None = None,
    port: int | None = None,
    base_url: str | None = None,
) -> str:
    if base_url:
        normalized_base_url = base_url.strip().rstrip("/")
        if not normalized_base_url:
            raise InvalidEmbeddingProviderPresetError("base_url is required")
        parsed = urlparse(normalized_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise InvalidEmbeddingProviderPresetError("base_url must be an absolute http(s) URL")
        return normalized_base_url

    selected_host = (host or preset.default_host).strip()
    if not selected_host:
        raise InvalidEmbeddingProviderPresetError("host is required")
    selected_port = port or preset.default_port
    if selected_port <= 0 or selected_port > 65535:
        raise InvalidEmbeddingProviderPresetError("port must be between 1 and 65535")
    return f"http://{selected_host}:{selected_port}"
