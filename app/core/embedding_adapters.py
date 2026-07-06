"""Embedding adapter interface and process-local adapter cache."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from app.core.embedding_jobs import EmbeddingProfileRecord
from app.core.embedding_vectors import EmbeddingVectorTable, generate_mock_embedding

DEFAULT_ADAPTER_NAME = "mock"


@dataclass(frozen=True)
class EmbeddingModelProfile:
    profile_name: str
    model_name: str
    dimension: int
    storage_type: str
    adapter_name: str = DEFAULT_ADAPTER_NAME
    normalize_embeddings: bool = True
    pooling_strategy: str | None = None
    query_instruction: str | None = None
    document_instruction: str | None = None
    dtype: str | None = None

    @classmethod
    def from_embedding_profile_record(
        cls,
        record: EmbeddingProfileRecord,
        *,
        adapter_name: str | None = None,
    ) -> "EmbeddingModelProfile":
        return cls(
            profile_name=record.profile_name,
            model_name=record.model_name,
            dimension=record.dimension,
            storage_type=record.storage_type,
            adapter_name=adapter_name or record.adapter_name or DEFAULT_ADAPTER_NAME,
            normalize_embeddings=record.normalize_embeddings,
            pooling_strategy=record.pooling_strategy,
            query_instruction=record.query_instruction,
            document_instruction=record.document_instruction,
            dtype=record.dtype,
        )

    @classmethod
    def from_vector_table(
        cls,
        table: EmbeddingVectorTable,
        *,
        adapter_name: str = DEFAULT_ADAPTER_NAME,
    ) -> "EmbeddingModelProfile":
        return cls(
            profile_name=table.profile_name,
            model_name=table.profile_name,
            dimension=table.dimension,
            storage_type=table.storage_type,
            adapter_name=adapter_name,
        )


class EmbeddingAdapter(Protocol):
    profile: EmbeddingModelProfile

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Embed document-side text batches."""

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Embed query-side text."""

    def runtime_metadata(self) -> dict[str, object]:
        """Return adapter/runtime metadata for reproducibility logs."""


class InvalidEmbeddingAdapterError(ValueError):
    """Raised when an embedding adapter or model cache operation is invalid."""


def _validate_text(text: str, field_name: str) -> str:
    normalized = text.strip()
    if not normalized:
        raise InvalidEmbeddingAdapterError(f"{field_name} must not be blank")
    return text


def _validate_profile(profile: EmbeddingModelProfile) -> None:
    if not profile.profile_name.strip():
        raise InvalidEmbeddingAdapterError("profile_name is required")
    if not profile.model_name.strip():
        raise InvalidEmbeddingAdapterError("model_name is required")
    if profile.dimension <= 0:
        raise InvalidEmbeddingAdapterError("dimension must be greater than 0")
    if profile.storage_type not in {"vector", "halfvec"}:
        raise InvalidEmbeddingAdapterError(f"Unsupported storage_type: {profile.storage_type}")
    if not profile.adapter_name.strip():
        raise InvalidEmbeddingAdapterError("adapter_name is required")


@dataclass(frozen=True)
class MockEmbeddingAdapter:
    profile: EmbeddingModelProfile

    def __post_init__(self) -> None:
        _validate_profile(self.profile)

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [
            generate_mock_embedding(
                _validate_text(text, "document text"),
                profile_name=self.profile.profile_name,
                dimension=self.profile.dimension,
            )
            for text in texts
        ]

    def embed_query(self, text: str) -> tuple[float, ...]:
        return generate_mock_embedding(
            _validate_text(text, "query text"),
            profile_name=self.profile.profile_name,
            dimension=self.profile.dimension,
        )

    def runtime_metadata(self) -> dict[str, object]:
        return {
            "adapter": DEFAULT_ADAPTER_NAME,
            "profile_name": self.profile.profile_name,
            "model_name": self.profile.model_name,
            "dimension": self.profile.dimension,
            "storage_type": self.profile.storage_type,
            "normalize_embeddings": self.profile.normalize_embeddings,
            "pooling_strategy": self.profile.pooling_strategy,
            "dtype": self.profile.dtype,
        }


AdapterFactory = Callable[[EmbeddingModelProfile], EmbeddingAdapter]


class EmbeddingAdapterCache:
    def __init__(
        self,
        adapter_factories: dict[str, AdapterFactory] | None = None,
    ) -> None:
        self._adapter_factories = adapter_factories or {
            DEFAULT_ADAPTER_NAME: MockEmbeddingAdapter,
        }
        self._adapters: dict[str, EmbeddingAdapter] = {}

    def get_adapter(
        self,
        profile: EmbeddingModelProfile,
        *,
        adapter_name: str | None = None,
    ) -> EmbeddingAdapter:
        selected_adapter_name = (adapter_name or profile.adapter_name).strip()
        if not selected_adapter_name:
            raise InvalidEmbeddingAdapterError("adapter_name is required")
        try:
            factory = self._adapter_factories[selected_adapter_name]
        except KeyError as exc:
            raise InvalidEmbeddingAdapterError(
                f"Unsupported embedding adapter: {selected_adapter_name}"
            ) from exc

        normalized_profile = EmbeddingModelProfile(
            profile_name=profile.profile_name,
            model_name=profile.model_name,
            dimension=profile.dimension,
            storage_type=profile.storage_type,
            adapter_name=selected_adapter_name,
            normalize_embeddings=profile.normalize_embeddings,
            pooling_strategy=profile.pooling_strategy,
            query_instruction=profile.query_instruction,
            document_instruction=profile.document_instruction,
            dtype=profile.dtype,
        )
        _validate_profile(normalized_profile)
        cache_key = self.cache_key(normalized_profile)
        if cache_key not in self._adapters:
            self._adapters[cache_key] = factory(normalized_profile)
        return self._adapters[cache_key]

    def preload(
        self,
        profiles: Sequence[EmbeddingModelProfile],
        *,
        adapter_name: str | None = None,
    ) -> list[EmbeddingAdapter]:
        return [self.get_adapter(profile, adapter_name=adapter_name) for profile in profiles]

    def clear(self) -> None:
        self._adapters.clear()

    def loaded_cache_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    @staticmethod
    def cache_key(profile: EmbeddingModelProfile) -> str:
        return (
            f"{profile.adapter_name}:{profile.profile_name}:"
            f"{profile.dimension}:{profile.storage_type}"
        )


_GLOBAL_EMBEDDING_ADAPTER_CACHE = EmbeddingAdapterCache()


def get_global_embedding_adapter_cache() -> EmbeddingAdapterCache:
    return _GLOBAL_EMBEDDING_ADAPTER_CACHE
