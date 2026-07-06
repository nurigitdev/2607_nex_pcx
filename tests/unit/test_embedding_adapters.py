import pytest

from app.core.embedding_adapters import (
    DEFAULT_ADAPTER_NAME,
    EmbeddingAdapterCache,
    EmbeddingModelProfile,
    InvalidEmbeddingAdapterError,
    MockEmbeddingAdapter,
)


def make_profile(**overrides) -> EmbeddingModelProfile:
    values = {
        "profile_name": "kure_v1_1024",
        "model_name": "nlpai-lab/KURE-v1",
        "dimension": 1024,
        "storage_type": "vector",
        "adapter_name": DEFAULT_ADAPTER_NAME,
        "normalize_embeddings": True,
        "pooling_strategy": "sentence-transformers-default",
        "dtype": "float32",
    }
    values.update(overrides)
    return EmbeddingModelProfile(**values)


def test_mock_embedding_adapter_embeds_documents_and_query_deterministically() -> None:
    adapter = MockEmbeddingAdapter(make_profile(dimension=8))

    documents = adapter.embed_documents(["hello", "hello"])
    query = adapter.embed_query("hello")

    assert documents == [query, query]
    assert len(query) == 8
    assert adapter.runtime_metadata() == {
        "adapter": "mock",
        "profile_name": "kure_v1_1024",
        "model_name": "nlpai-lab/KURE-v1",
        "dimension": 8,
        "storage_type": "vector",
        "normalize_embeddings": True,
        "pooling_strategy": "sentence-transformers-default",
        "dtype": "float32",
    }


def test_mock_embedding_adapter_rejects_blank_text() -> None:
    adapter = MockEmbeddingAdapter(make_profile(dimension=8))

    with pytest.raises(InvalidEmbeddingAdapterError, match="document text"):
        adapter.embed_documents(["  "])

    with pytest.raises(InvalidEmbeddingAdapterError, match="query text"):
        adapter.embed_query("")


def test_embedding_adapter_cache_reuses_loaded_adapter() -> None:
    cache = EmbeddingAdapterCache()
    profile = make_profile(dimension=8)

    first = cache.get_adapter(profile)
    second = cache.get_adapter(profile)

    assert first is second
    assert cache.loaded_cache_keys() == ("mock:kure_v1_1024:8:vector",)


def test_embedding_adapter_cache_preloads_and_clears_profiles() -> None:
    cache = EmbeddingAdapterCache()
    profiles = [
        make_profile(profile_name="kure_v1_1024", dimension=8),
        make_profile(profile_name="qwen3_4b_2560", dimension=16, storage_type="halfvec"),
    ]

    adapters = cache.preload(profiles)

    assert len(adapters) == 2
    assert cache.loaded_cache_keys() == (
        "mock:kure_v1_1024:8:vector",
        "mock:qwen3_4b_2560:16:halfvec",
    )
    cache.clear()
    assert cache.loaded_cache_keys() == ()


def test_embedding_adapter_cache_rejects_unknown_adapter() -> None:
    cache = EmbeddingAdapterCache()

    with pytest.raises(InvalidEmbeddingAdapterError, match="Unsupported embedding adapter"):
        cache.get_adapter(make_profile(adapter_name="missing"))


def test_embedding_model_profile_validation_rejects_invalid_values() -> None:
    with pytest.raises(InvalidEmbeddingAdapterError, match="dimension"):
        MockEmbeddingAdapter(make_profile(dimension=0))

    with pytest.raises(InvalidEmbeddingAdapterError, match="storage_type"):
        MockEmbeddingAdapter(make_profile(storage_type="bits"))
