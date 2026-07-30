"""Embedding adapter interface and process-local adapter cache."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from app.core.embedding_jobs import EmbeddingProfileRecord
from app.core.embedding_vectors import EmbeddingVectorTable, generate_mock_embedding
from app.core.model_runtime_dtypes import (
    model_first_parameter_dtype_name,
    normalize_torch_dtype_name,
    torch_dtype_from_name,
)

DEFAULT_ADAPTER_NAME = "mock"
SENTENCE_TRANSFORMERS_ADAPTER_NAME = "sentence_transformers"
QWEN_EMBEDDING_ADAPTER_NAME = "qwen_embedding"


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
    local_model_path: str | None = None
    device: str | None = None

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


class SentenceTransformersEmbeddingAdapter:
    def __init__(self, profile: EmbeddingModelProfile) -> None:
        _validate_profile(profile)
        self.profile = profile
        self.model_source = profile.local_model_path or profile.model_name
        self.requested_torch_dtype = normalize_torch_dtype_name(profile.dtype)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise InvalidEmbeddingAdapterError(
                "sentence-transformers is required. Install it with "
                '`./.venv/bin/pip install -e ".[models]"`.'
            ) from exc

        model_kwargs = {}
        if profile.device:
            model_kwargs["device"] = profile.device
        torch_dtype = torch_dtype_from_name(profile.dtype)
        if torch_dtype is not None:
            model_kwargs["model_kwargs"] = {"torch_dtype": torch_dtype}
        self._model = SentenceTransformer(self.model_source, **model_kwargs)

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        validated_texts = [
            _apply_instruction(
                _validate_text(text, "document text"), self.profile.document_instruction
            )
            for text in texts
        ]
        return self._encode_batch(validated_texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        validated_text = _apply_instruction(
            _validate_text(text, "query text"),
            self.profile.query_instruction,
        )
        return self._encode_batch([validated_text])[0]

    def runtime_metadata(self) -> dict[str, object]:
        return {
            "adapter": SENTENCE_TRANSFORMERS_ADAPTER_NAME,
            "profile_name": self.profile.profile_name,
            "model_name": self.profile.model_name,
            "model_source": self.model_source,
            "dimension": self.profile.dimension,
            "storage_type": self.profile.storage_type,
            "normalize_embeddings": self.profile.normalize_embeddings,
            "pooling_strategy": self.profile.pooling_strategy,
            "query_instruction": self.profile.query_instruction,
            "document_instruction": self.profile.document_instruction,
            "dtype": self.profile.dtype,
            "requested_torch_dtype": self.requested_torch_dtype,
            "loaded_parameter_dtype": model_first_parameter_dtype_name(self._model),
            "device": self.profile.device,
        }

    def _encode_batch(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        encoded = self._model.encode(
            list(texts),
            normalize_embeddings=self.profile.normalize_embeddings,
            convert_to_numpy=False,
        )
        return [_coerce_vector(vector, self.profile.dimension) for vector in encoded]


class QwenEmbeddingAdapter:
    _shared_models: dict[tuple[str, str, str], object] = {}

    def __init__(self, profile: EmbeddingModelProfile) -> None:
        _validate_profile(profile)
        self.profile = profile
        self.model_source = profile.local_model_path or profile.model_name
        self.requested_torch_dtype = normalize_torch_dtype_name(profile.dtype)
        self._cache_key = (
            self.model_source,
            profile.device or "",
            self.requested_torch_dtype or "",
        )
        self._model = self._load_shared_model()

    def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        validated_texts = [
            _apply_instruction(
                _validate_text(text, "document text"), self.profile.document_instruction
            )
            for text in texts
        ]
        return self._encode_batch(validated_texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        validated_text = _apply_instruction(
            _validate_text(text, "query text"),
            self.profile.query_instruction,
        )
        return self._encode_batch([validated_text])[0]

    def runtime_metadata(self) -> dict[str, object]:
        return {
            "adapter": QWEN_EMBEDDING_ADAPTER_NAME,
            "profile_name": self.profile.profile_name,
            "model_name": self.profile.model_name,
            "model_source": self.model_source,
            "dimension": self.profile.dimension,
            "storage_type": self.profile.storage_type,
            "normalize_embeddings": self.profile.normalize_embeddings,
            "pooling_strategy": "sentence-transformers-truncate-dim",
            "query_instruction": self.profile.query_instruction,
            "document_instruction": self.profile.document_instruction,
            "dtype": self.profile.dtype,
            "requested_torch_dtype": self.requested_torch_dtype,
            "loaded_parameter_dtype": model_first_parameter_dtype_name(self._model),
            "device": self.profile.device,
            "shared_model_cache_key": ":".join(self._cache_key),
        }

    @classmethod
    def clear_shared_model_cache(cls) -> None:
        cls._shared_models.clear()

    def _load_shared_model(self) -> object:
        if self._cache_key in self._shared_models:
            return self._shared_models[self._cache_key]

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise InvalidEmbeddingAdapterError(
                "sentence-transformers is required for qwen_embedding. Install it with "
                '`./.venv/bin/pip install -e ".[models]"`.'
            ) from exc

        model_kwargs = {}
        if self.profile.device:
            model_kwargs["device"] = self.profile.device
        torch_dtype = torch_dtype_from_name(self.profile.dtype)
        if torch_dtype is not None:
            model_kwargs["model_kwargs"] = {"torch_dtype": torch_dtype}
        model = SentenceTransformer(self.model_source, **model_kwargs)
        self._shared_models[self._cache_key] = model
        return model

    def _encode_batch(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        encoded = self._model.encode(
            list(texts),
            normalize_embeddings=self.profile.normalize_embeddings,
            convert_to_numpy=False,
            truncate_dim=self.profile.dimension,
        )
        return [_coerce_vector(vector, self.profile.dimension) for vector in encoded]


def _apply_instruction(text: str, instruction: str | None) -> str:
    if instruction is None or not instruction.strip():
        return text
    return f"{instruction}{text}"


def _coerce_vector(vector: object, expected_dimension: int) -> tuple[float, ...]:
    values = vector.tolist() if hasattr(vector, "tolist") else vector
    try:
        coerced = tuple(float(value) for value in values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise InvalidEmbeddingAdapterError("embedding vector must be iterable") from exc

    if len(coerced) != expected_dimension:
        raise InvalidEmbeddingAdapterError(
            f"embedding dimension mismatch: expected {expected_dimension}, got {len(coerced)}"
        )
    if not all(isfinite(value) for value in coerced):
        raise InvalidEmbeddingAdapterError("embedding values must be finite")
    return coerced


AdapterFactory = Callable[[EmbeddingModelProfile], EmbeddingAdapter]


class EmbeddingAdapterCache:
    def __init__(
        self,
        adapter_factories: dict[str, AdapterFactory] | None = None,
    ) -> None:
        self._adapter_factories = adapter_factories or {
            DEFAULT_ADAPTER_NAME: MockEmbeddingAdapter,
            SENTENCE_TRANSFORMERS_ADAPTER_NAME: SentenceTransformersEmbeddingAdapter,
            QWEN_EMBEDDING_ADAPTER_NAME: QwenEmbeddingAdapter,
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
            local_model_path=profile.local_model_path,
            device=profile.device,
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
            f"{profile.dimension}:{profile.storage_type}:"
            f"{profile.local_model_path or profile.model_name}:"
            f"{profile.device or ''}:{normalize_torch_dtype_name(profile.dtype) or ''}"
        )


_GLOBAL_EMBEDDING_ADAPTER_CACHE = EmbeddingAdapterCache()


def get_global_embedding_adapter_cache() -> EmbeddingAdapterCache:
    return _GLOBAL_EMBEDDING_ADAPTER_CACHE
