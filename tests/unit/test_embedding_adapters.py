import sys
from types import SimpleNamespace

import pytest

from app.core.embedding_adapters import (
    DEFAULT_ADAPTER_NAME,
    QWEN_EMBEDDING_ADAPTER_NAME,
    SENTENCE_TRANSFORMERS_ADAPTER_NAME,
    EmbeddingAdapterCache,
    EmbeddingModelProfile,
    InvalidEmbeddingAdapterError,
    MockEmbeddingAdapter,
    QwenEmbeddingAdapter,
    SentenceTransformersEmbeddingAdapter,
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
    assert cache.loaded_cache_keys() == ("mock:kure_v1_1024:8:vector:nlpai-lab/KURE-v1:",)


def test_embedding_adapter_cache_preloads_and_clears_profiles() -> None:
    cache = EmbeddingAdapterCache()
    profiles = [
        make_profile(profile_name="kure_v1_1024", dimension=8),
        make_profile(profile_name="qwen3_4b_2560", dimension=16, storage_type="halfvec"),
    ]

    adapters = cache.preload(profiles)

    assert len(adapters) == 2
    assert cache.loaded_cache_keys() == (
        "mock:kure_v1_1024:8:vector:nlpai-lab/KURE-v1:",
        "mock:qwen3_4b_2560:16:halfvec:nlpai-lab/KURE-v1:",
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


def test_sentence_transformers_adapter_embeds_with_local_model_path_and_instructions(
    monkeypatch,
) -> None:
    calls = {}

    class FakeSentenceTransformer:
        def __init__(self, model_source, **kwargs) -> None:
            calls["model_source"] = model_source
            calls["kwargs"] = kwargs

        def encode(self, texts, **kwargs):
            calls["texts"] = texts
            calls["encode_kwargs"] = kwargs
            return [[float(len(text)), 2.0, 3.0] for text in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    profile = make_profile(
        dimension=3,
        adapter_name=SENTENCE_TRANSFORMERS_ADAPTER_NAME,
        local_model_path="models/kure_v1",
        device="cpu",
        query_instruction="query: ",
        document_instruction="passage: ",
    )

    adapter = SentenceTransformersEmbeddingAdapter(profile)

    documents = adapter.embed_documents(["alpha"])
    query = adapter.embed_query("beta")

    assert calls["model_source"] == "models/kure_v1"
    assert calls["kwargs"] == {"device": "cpu"}
    assert documents == [(14.0, 2.0, 3.0)]
    assert query == (11.0, 2.0, 3.0)
    assert calls["texts"] == ["query: beta"]
    assert calls["encode_kwargs"] == {
        "normalize_embeddings": True,
        "convert_to_numpy": False,
    }
    assert adapter.runtime_metadata()["adapter"] == SENTENCE_TRANSFORMERS_ADAPTER_NAME
    assert adapter.runtime_metadata()["model_source"] == "models/kure_v1"


def test_sentence_transformers_adapter_rejects_dimension_mismatch(monkeypatch) -> None:
    class FakeSentenceTransformer:
        def __init__(self, model_source, **kwargs) -> None:
            pass

        def encode(self, texts, **kwargs):
            return [[1.0, 2.0]]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    adapter = SentenceTransformersEmbeddingAdapter(
        make_profile(
            dimension=3,
            adapter_name=SENTENCE_TRANSFORMERS_ADAPTER_NAME,
        )
    )

    with pytest.raises(InvalidEmbeddingAdapterError, match="dimension mismatch"):
        adapter.embed_query("hello")


def test_qwen_embedding_adapter_shares_model_and_truncates_by_profile(monkeypatch) -> None:
    calls = {"init": [], "encode": []}

    class FakeSentenceTransformer:
        def __init__(self, model_source, **kwargs) -> None:
            calls["init"].append((model_source, kwargs))

        def encode(self, texts, **kwargs):
            calls["encode"].append((list(texts), kwargs))
            dimension = kwargs["truncate_dim"]
            return [[float(index + 1) for _ in range(dimension)] for index, _ in enumerate(texts)]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    QwenEmbeddingAdapter.clear_shared_model_cache()
    adapter_1000 = QwenEmbeddingAdapter(
        make_profile(
            profile_name="qwen3_4b_1000",
            model_name="qwen3_embedding_4b",
            dimension=3,
            adapter_name=QWEN_EMBEDDING_ADAPTER_NAME,
            local_model_path="models/qwen3_embedding_4b",
            device="cpu",
        )
    )
    adapter_2560 = QwenEmbeddingAdapter(
        make_profile(
            profile_name="qwen3_4b_2560",
            model_name="qwen3_embedding_4b",
            dimension=5,
            storage_type="halfvec",
            adapter_name=QWEN_EMBEDDING_ADAPTER_NAME,
            local_model_path="models/qwen3_embedding_4b",
            device="cpu",
        )
    )

    documents = adapter_1000.embed_documents(["alpha", "beta"])
    query = adapter_2560.embed_query("gamma")

    assert calls["init"] == [("models/qwen3_embedding_4b", {"device": "cpu"})]
    assert calls["encode"][0][1]["truncate_dim"] == 3
    assert calls["encode"][1][1]["truncate_dim"] == 5
    assert len(documents) == 2
    assert len(documents[0]) == 3
    assert len(query) == 5
    assert adapter_2560.runtime_metadata()["adapter"] == QWEN_EMBEDDING_ADAPTER_NAME
    assert adapter_2560.runtime_metadata()["model_source"] == "models/qwen3_embedding_4b"
    QwenEmbeddingAdapter.clear_shared_model_cache()


def test_embedding_adapter_cache_supports_sentence_transformers(monkeypatch) -> None:
    class FakeSentenceTransformer:
        def __init__(self, model_source, **kwargs) -> None:
            pass

        def encode(self, texts, **kwargs):
            return [[1.0, 2.0, 3.0] for _ in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    cache = EmbeddingAdapterCache()
    profile = make_profile(
        dimension=3,
        adapter_name=SENTENCE_TRANSFORMERS_ADAPTER_NAME,
        local_model_path="models/bge_m3",
    )

    adapter = cache.get_adapter(profile)

    assert adapter.embed_query("hello") == (1.0, 2.0, 3.0)
    assert cache.loaded_cache_keys() == (
        "sentence_transformers:kure_v1_1024:3:vector:models/bge_m3:",
    )


def test_embedding_adapter_cache_supports_qwen_embedding(monkeypatch) -> None:
    class FakeSentenceTransformer:
        def __init__(self, model_source, **kwargs) -> None:
            pass

        def encode(self, texts, **kwargs):
            return [[1.0, 2.0, 3.0] for _ in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    QwenEmbeddingAdapter.clear_shared_model_cache()
    cache = EmbeddingAdapterCache()
    profile = make_profile(
        profile_name="qwen3_4b_1000",
        model_name="qwen3_embedding_4b",
        dimension=3,
        adapter_name=QWEN_EMBEDDING_ADAPTER_NAME,
        local_model_path="models/qwen3_embedding_4b",
    )

    adapter = cache.get_adapter(profile)

    assert adapter.embed_query("hello") == (1.0, 2.0, 3.0)
    assert cache.loaded_cache_keys() == (
        "qwen_embedding:qwen3_4b_1000:3:vector:models/qwen3_embedding_4b:",
    )
    QwenEmbeddingAdapter.clear_shared_model_cache()
