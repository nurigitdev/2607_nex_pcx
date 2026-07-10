"""Standalone embedding provider service skeleton."""

from dataclasses import dataclass
from os import getenv
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.embedding_providers import (
    REMOTE_EMBEDDING_PROVIDER_TYPE,
    EmbeddingProviderRequest,
    EmbeddingProviderResponse,
    validate_embedding_provider_request,
    validate_embedding_provider_response,
)
from app.core.embedding_vectors import generate_mock_embedding


@dataclass(frozen=True)
class EmbeddingProviderServiceSettings:
    provider_model_id: str = "skeleton-embedding-provider"
    model_key: str = "kure_v1"
    profile_names: tuple[str, ...] = ("kure_v1_1024",)
    dimension: int = 1024
    device: str = "cpu"
    ready: bool = True


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


def get_embedding_provider_service_settings() -> EmbeddingProviderServiceSettings:
    return EmbeddingProviderServiceSettings(
        provider_model_id=getenv(
            "NEX_PCX_PROVIDER_MODEL_ID",
            "skeleton-embedding-provider",
        ),
        model_key=getenv("NEX_PCX_PROVIDER_MODEL_KEY", "kure_v1"),
        profile_names=_split_profile_names(getenv("NEX_PCX_PROVIDER_PROFILE_NAMES")),
        dimension=int(getenv("NEX_PCX_PROVIDER_DIMENSION", "1024")),
        device=getenv("NEX_PCX_PROVIDER_DEVICE", "cpu"),
        ready=_parse_bool(getenv("NEX_PCX_PROVIDER_READY", "true")),
    )


def create_app(
    settings: EmbeddingProviderServiceSettings | None = None,
    *,
    provider: SkeletonEmbeddingProvider | None = None,
) -> FastAPI:
    provider_settings = settings or get_embedding_provider_service_settings()
    embedding_provider = provider or SkeletonEmbeddingProvider(provider_settings)
    app_settings = get_settings()
    app = FastAPI(
        title=f"{app_settings.app_name} Embedding Provider",
        version=app_settings.app_version,
    )

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {
            "ready": provider_settings.ready,
            "provider_type": REMOTE_EMBEDDING_PROVIDER_TYPE,
            "provider_model_id": provider_settings.provider_model_id,
            "model_key": provider_settings.model_key,
            "profile_names": list(provider_settings.profile_names),
            "dimension": provider_settings.dimension,
            "device": provider_settings.device,
            "runtime_metadata": {
                "service": "nex_pcx_embedding_provider_skeleton",
            },
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


app = create_app()
