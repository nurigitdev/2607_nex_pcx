"""Standalone embedding provider service skeleton."""

from dataclasses import dataclass
from os import getenv
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.embedding_adapters import (
    QWEN_EMBEDDING_ADAPTER_NAME,
    SENTENCE_TRANSFORMERS_ADAPTER_NAME,
    EmbeddingAdapter,
    EmbeddingModelProfile,
    QwenEmbeddingAdapter,
    SentenceTransformersEmbeddingAdapter,
)
from app.core.embedding_model_distribution import (
    get_embedding_model_distribution,
    resolve_embedding_model_dir,
)
from app.core.embedding_providers import (
    REMOTE_EMBEDDING_PROVIDER_TYPE,
    EmbeddingProviderRequest,
    EmbeddingProviderResponse,
    validate_embedding_provider_request,
    validate_embedding_provider_response,
)
from app.core.embedding_vectors import generate_mock_embedding, get_embedding_vector_table
from app.core.model_runtime_dtypes import normalize_torch_dtype_name

PROVIDER_BACKEND_MOCK = "mock"
PROVIDER_BACKEND_SENTENCE_TRANSFORMERS = "sentence_transformers"
PROVIDER_BACKEND_QWEN_EMBEDDING = QWEN_EMBEDDING_ADAPTER_NAME


@dataclass(frozen=True)
class EmbeddingProviderServiceSettings:
    backend: str = PROVIDER_BACKEND_MOCK
    provider_model_id: str = "skeleton-embedding-provider"
    model_key: str = "kure_v1"
    profile_names: tuple[str, ...] = ("kure_v1_1024",)
    dimension: int = 1024
    device: str = "cpu"
    torch_dtype: str | None = None
    ready: bool = True
    models_dir: Path = Path("models")


class EmbeddingRequestBody(BaseModel):
    profile_name: str
    model_key: str
    input_type: str
    texts: list[str]
    output_dimension: int
    normalize_embeddings: bool = True
    trace_id: str | None = None
    runtime_metadata: dict[str, Any] = Field(default_factory=dict)


class SkeletonEmbeddingProvider:
    def __init__(self, settings: EmbeddingProviderServiceSettings) -> None:
        self.settings = settings

    def embed(self, request: EmbeddingProviderRequest) -> EmbeddingProviderResponse:
        validated = validate_embedding_provider_request(request)
        if validated.model_key != self.settings.model_key:
            raise ValueError(f"Unsupported model_key: {validated.model_key}")
        if validated.profile_name not in self.settings.profile_names:
            raise ValueError(f"Unsupported profile_name: {validated.profile_name}")
        if validated.output_dimension != self.settings.dimension:
            raise ValueError(
                f"Unsupported output_dimension: {validated.output_dimension}; "
                f"expected {self.settings.dimension}"
            )

        started_at = perf_counter()
        embeddings = tuple(
            generate_mock_embedding(
                text,
                profile_name=validated.profile_name,
                dimension=validated.output_dimension,
            )
            for text in validated.texts
        )
        response = EmbeddingProviderResponse(
            embeddings=embeddings,
            dimension=validated.output_dimension,
            provider_model_id=self.settings.provider_model_id,
            provider_type=REMOTE_EMBEDDING_PROVIDER_TYPE,
            elapsed_ms=max(0, int((perf_counter() - started_at) * 1000)),
            input_count=len(validated.texts),
            runtime_metadata={
                "service": "nex_pcx_embedding_provider_skeleton",
                "device": self.settings.device,
                "model_key": self.settings.model_key,
                "trace_id": validated.trace_id,
            },
        )
        return validate_embedding_provider_response(response, validated)


class LocalSentenceTransformersProvider:
    def __init__(self, settings: EmbeddingProviderServiceSettings) -> None:
        self.settings = settings
        self.distribution = get_embedding_model_distribution(settings.model_key)
        if self.distribution.adapter_name != SENTENCE_TRANSFORMERS_ADAPTER_NAME:
            raise ValueError(
                f"Model {settings.model_key} does not use the sentence_transformers adapter"
            )
        self.local_model_dir = resolve_embedding_model_dir(self.distribution, settings.models_dir)
        self._adapters: dict[str, EmbeddingAdapter] = {
            profile_name: SentenceTransformersEmbeddingAdapter(
                _embedding_model_profile_for_provider(
                    profile_name,
                    settings=settings,
                    local_model_dir=self.local_model_dir,
                    adapter_name=SENTENCE_TRANSFORMERS_ADAPTER_NAME,
                )
            )
            for profile_name in settings.profile_names
        }

    def runtime_metadata(self) -> dict[str, object]:
        return _provider_adapter_runtime_metadata(self._adapters)

    def embed(self, request: EmbeddingProviderRequest) -> EmbeddingProviderResponse:
        validated = validate_embedding_provider_request(request)
        if validated.model_key != self.settings.model_key:
            raise ValueError(f"Unsupported model_key: {validated.model_key}")
        try:
            adapter = self._adapters[validated.profile_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported profile_name: {validated.profile_name}") from exc

        if validated.output_dimension != adapter.profile.dimension:
            raise ValueError(
                f"Unsupported output_dimension: {validated.output_dimension}; "
                f"expected {adapter.profile.dimension}"
            )

        started_at = perf_counter()
        if validated.input_type == "query":
            embeddings = tuple(adapter.embed_query(text) for text in validated.texts)
        else:
            embeddings = tuple(adapter.embed_documents(validated.texts))
        response = EmbeddingProviderResponse(
            embeddings=embeddings,
            dimension=validated.output_dimension,
            provider_model_id=self.settings.provider_model_id,
            provider_type=REMOTE_EMBEDDING_PROVIDER_TYPE,
            elapsed_ms=max(0, int((perf_counter() - started_at) * 1000)),
            input_count=len(validated.texts),
            runtime_metadata={
                **adapter.runtime_metadata(),
                "service": "nex_pcx_embedding_provider_service",
                "backend": PROVIDER_BACKEND_SENTENCE_TRANSFORMERS,
                "device": self.settings.device,
                "model_key": self.settings.model_key,
                "trace_id": validated.trace_id,
            },
        )
        return validate_embedding_provider_response(response, validated)


class LocalQwenEmbeddingProvider:
    def __init__(self, settings: EmbeddingProviderServiceSettings) -> None:
        self.settings = settings
        self.distribution = get_embedding_model_distribution(settings.model_key)
        if self.distribution.adapter_name != QWEN_EMBEDDING_ADAPTER_NAME:
            raise ValueError(f"Model {settings.model_key} does not use the qwen_embedding adapter")
        self.local_model_dir = resolve_embedding_model_dir(self.distribution, settings.models_dir)
        self._adapters: dict[str, QwenEmbeddingAdapter] = {
            profile_name: QwenEmbeddingAdapter(
                _embedding_model_profile_for_provider(
                    profile_name,
                    settings=settings,
                    local_model_dir=self.local_model_dir,
                    adapter_name=QWEN_EMBEDDING_ADAPTER_NAME,
                )
            )
            for profile_name in settings.profile_names
        }

    def runtime_metadata(self) -> dict[str, object]:
        return _provider_adapter_runtime_metadata(self._adapters)

    @property
    def profile_dimensions(self) -> dict[str, int]:
        return {
            profile_name: adapter.profile.dimension
            for profile_name, adapter in self._adapters.items()
        }

    def embed(self, request: EmbeddingProviderRequest) -> EmbeddingProviderResponse:
        validated = validate_embedding_provider_request(request)
        if validated.model_key != self.settings.model_key:
            raise ValueError(f"Unsupported model_key: {validated.model_key}")
        try:
            adapter = self._adapters[validated.profile_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported profile_name: {validated.profile_name}") from exc

        if validated.output_dimension != adapter.profile.dimension:
            raise ValueError(
                f"Unsupported output_dimension: {validated.output_dimension}; "
                f"expected {adapter.profile.dimension}"
            )

        started_at = perf_counter()
        if validated.input_type == "query":
            embeddings = tuple(adapter.embed_query(text) for text in validated.texts)
        else:
            embeddings = tuple(adapter.embed_documents(validated.texts))
        response = EmbeddingProviderResponse(
            embeddings=embeddings,
            dimension=validated.output_dimension,
            provider_model_id=self.settings.provider_model_id,
            provider_type=REMOTE_EMBEDDING_PROVIDER_TYPE,
            elapsed_ms=max(0, int((perf_counter() - started_at) * 1000)),
            input_count=len(validated.texts),
            runtime_metadata={
                **adapter.runtime_metadata(),
                "service": "nex_pcx_embedding_provider_service",
                "backend": PROVIDER_BACKEND_QWEN_EMBEDDING,
                "device": self.settings.device,
                "model_key": self.settings.model_key,
                "trace_id": validated.trace_id,
            },
        )
        return validate_embedding_provider_response(response, validated)


def get_embedding_provider_service_settings() -> EmbeddingProviderServiceSettings:
    app_settings = get_settings()
    return EmbeddingProviderServiceSettings(
        backend=getenv("NEX_PCX_PROVIDER_BACKEND", PROVIDER_BACKEND_MOCK).strip().lower(),
        provider_model_id=getenv(
            "NEX_PCX_PROVIDER_MODEL_ID",
            "skeleton-embedding-provider",
        ),
        model_key=getenv("NEX_PCX_PROVIDER_MODEL_KEY", "kure_v1"),
        profile_names=_split_profile_names(getenv("NEX_PCX_PROVIDER_PROFILE_NAMES")),
        dimension=int(getenv("NEX_PCX_PROVIDER_DIMENSION", "1024")),
        device=getenv("NEX_PCX_PROVIDER_DEVICE", "cpu"),
        torch_dtype=getenv("NEX_PCX_PROVIDER_TORCH_DTYPE"),
        ready=_parse_bool(getenv("NEX_PCX_PROVIDER_READY", "true")),
        models_dir=Path(
            getenv("NEX_PCX_PROVIDER_MODELS_DIR", str(app_settings.embedding_models_dir))
        ),
    )


def create_app(
    settings: EmbeddingProviderServiceSettings | None = None,
    *,
    provider: object | None = None,
) -> FastAPI:
    provider_settings = settings or get_embedding_provider_service_settings()
    embedding_provider = provider or _build_provider(provider_settings)
    app_settings = get_settings()
    app = FastAPI(
        title=f"{app_settings.app_name} Embedding Provider",
        version=app_settings.app_version,
    )

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        runtime_metadata = {
            "service": "nex_pcx_embedding_provider_skeleton",
            "backend": provider_settings.backend,
            "models_dir": str(provider_settings.models_dir),
            "requested_torch_dtype": normalize_torch_dtype_name(provider_settings.torch_dtype),
        }
        dimension: int | None = provider_settings.dimension
        adapter_runtime_metadata = getattr(embedding_provider, "runtime_metadata", None)
        if callable(adapter_runtime_metadata):
            runtime_metadata.update(adapter_runtime_metadata())
        profile_dimensions = getattr(embedding_provider, "profile_dimensions", None)
        if profile_dimensions:
            normalized_profile_dimensions = dict(profile_dimensions)
            runtime_metadata["profile_dimensions"] = normalized_profile_dimensions
            unique_dimensions = set(normalized_profile_dimensions.values())
            dimension = next(iter(unique_dimensions)) if len(unique_dimensions) == 1 else None

        return {
            "ready": provider_settings.ready,
            "provider_type": REMOTE_EMBEDDING_PROVIDER_TYPE,
            "provider_model_id": provider_settings.provider_model_id,
            "model_key": provider_settings.model_key,
            "profile_names": list(provider_settings.profile_names),
            "dimension": dimension,
            "device": provider_settings.device,
            "runtime_metadata": runtime_metadata,
        }

    @app.post("/v1/embeddings")
    def create_embeddings(payload: EmbeddingRequestBody) -> dict[str, object]:
        if not provider_settings.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Embedding provider is not ready.",
            )
        try:
            response = embedding_provider.embed(
                EmbeddingProviderRequest(
                    profile_name=payload.profile_name,
                    model_key=payload.model_key,
                    input_type=payload.input_type,
                    texts=tuple(payload.texts),
                    output_dimension=payload.output_dimension,
                    normalize_embeddings=payload.normalize_embeddings,
                    trace_id=payload.trace_id,
                    runtime_metadata=payload.runtime_metadata,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return {
            "embeddings": [list(embedding) for embedding in response.embeddings],
            "dimension": response.dimension,
            "provider_model_id": response.provider_model_id,
            "provider_type": response.provider_type,
            "elapsed_ms": response.elapsed_ms,
            "input_count": response.input_count,
            "runtime_metadata": dict(response.runtime_metadata),
        }

    return app


def _split_profile_names(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ("kure_v1_1024",)
    profile_names = tuple(profile.strip() for profile in value.split(",") if profile.strip())
    return profile_names or ("kure_v1_1024",)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_provider(settings: EmbeddingProviderServiceSettings) -> object:
    if settings.backend == PROVIDER_BACKEND_MOCK:
        return SkeletonEmbeddingProvider(settings)
    if settings.backend == PROVIDER_BACKEND_SENTENCE_TRANSFORMERS:
        return LocalSentenceTransformersProvider(settings)
    if settings.backend == PROVIDER_BACKEND_QWEN_EMBEDDING:
        return LocalQwenEmbeddingProvider(settings)
    raise ValueError(f"Unsupported provider backend: {settings.backend}")


def _embedding_model_profile_for_provider(
    profile_name: str,
    *,
    settings: EmbeddingProviderServiceSettings,
    local_model_dir: Path,
    adapter_name: str,
) -> EmbeddingModelProfile:
    table = get_embedding_vector_table(profile_name)
    return EmbeddingModelProfile(
        profile_name=profile_name,
        model_name=settings.model_key,
        dimension=table.dimension,
        storage_type=table.storage_type,
        adapter_name=adapter_name,
        local_model_path=str(local_model_dir),
        device=settings.device,
        dtype=settings.torch_dtype,
    )


def _provider_adapter_runtime_metadata(
    adapters: dict[str, EmbeddingAdapter],
) -> dict[str, object]:
    adapter_metadata = {
        profile_name: adapter.runtime_metadata() for profile_name, adapter in adapters.items()
    }
    loaded_parameter_dtypes = sorted(
        {
            str(metadata["loaded_parameter_dtype"])
            for metadata in adapter_metadata.values()
            if metadata.get("loaded_parameter_dtype") is not None
        }
    )
    requested_torch_dtypes = sorted(
        {
            str(metadata["requested_torch_dtype"])
            for metadata in adapter_metadata.values()
            if metadata.get("requested_torch_dtype") is not None
        }
    )
    return {
        "adapter_runtime_metadata": adapter_metadata,
        "loaded_parameter_dtypes": loaded_parameter_dtypes,
        "requested_torch_dtypes": requested_torch_dtypes,
    }


app = create_app()
