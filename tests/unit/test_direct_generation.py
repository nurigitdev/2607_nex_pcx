from types import SimpleNamespace

import pytest

from app.core.direct_generation import (
    DirectGenerationInput,
    InvalidDirectGenerationError,
    run_direct_generation_query,
)
from app.core.generation_runs import (
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
)


def test_run_direct_generation_query_orchestrates_mock_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    search_result = SimpleNamespace(search_log_id=42)
    retrieval_package = object()
    generation_report = object()

    def fake_run_search_compare(database_url, search_input, **kwargs):
        calls["database_url"] = database_url
        calls["search_input"] = search_input
        calls["search_kwargs"] = kwargs
        return search_result

    def fake_build_retrieval_context_package(database_url, context_input):
        calls["context_database_url"] = database_url
        calls["context_input"] = context_input
        return retrieval_package

    def fake_execute_mock_generation_run(database_url, package, **kwargs):
        calls["mock_database_url"] = database_url
        calls["mock_package"] = package
        calls["mock_kwargs"] = kwargs
        return generation_report

    monkeypatch.setattr("app.core.direct_generation.run_search_compare", fake_run_search_compare)
    monkeypatch.setattr(
        "app.core.direct_generation.build_retrieval_context_package",
        fake_build_retrieval_context_package,
    )
    monkeypatch.setattr(
        "app.core.direct_generation.execute_mock_generation_run",
        fake_execute_mock_generation_run,
    )

    result = run_direct_generation_query(
        "postgresql://example",
        DirectGenerationInput(
            query_text="  직접 생성 테스트  ",
            actor_user_id=7,
            requested_search_scope="company",
            provider_mode=GENERATION_PROVIDER_MODE_MOCK,
            profiles=("bm25_keyword",),
            max_context_chars=2000,
            include_neighbors=False,
            max_items=3,
        ),
    )

    search_input = calls["search_input"]
    context_input = calls["context_input"]
    assert result.search_result is search_result
    assert result.retrieval_package is retrieval_package
    assert result.generation_report is generation_report
    assert search_input.query_text == "직접 생성 테스트"
    assert search_input.actor_user_id == 7
    assert search_input.profiles == ("bm25_keyword",)
    assert context_input.search_log_id == 42
    assert context_input.max_context_chars == 2000
    assert context_input.include_neighbors is False
    assert context_input.max_items == 3
    assert calls["mock_package"] is retrieval_package
    assert calls["mock_kwargs"] == {
        "created_by": "api_direct_generation",
        "created_by_user_id": 7,
    }


def test_run_direct_generation_query_passes_remote_runtime_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieval_package = object()
    fake_provider_client = object()
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "app.core.direct_generation.run_search_compare",
        lambda *_args, **_kwargs: SimpleNamespace(search_log_id=11),
    )
    monkeypatch.setattr(
        "app.core.direct_generation.build_retrieval_context_package",
        lambda *_args, **_kwargs: retrieval_package,
    )

    def fake_execute_remote_generation_run(database_url, package, **kwargs):
        calls["database_url"] = database_url
        calls["package"] = package
        calls["kwargs"] = kwargs
        return "remote-report"

    monkeypatch.setattr(
        "app.core.direct_generation.execute_remote_generation_run",
        fake_execute_remote_generation_run,
    )

    result = run_direct_generation_query(
        "postgresql://example",
        DirectGenerationInput(
            query_text="remote query",
            actor_user_id=8,
            provider_mode=GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE,
        ),
        api_key="secret",
        generation_provider_client=fake_provider_client,  # type: ignore[arg-type]
    )

    assert result.generation_report == "remote-report"
    assert calls["package"] is retrieval_package
    assert calls["kwargs"] == {
        "provider_client": fake_provider_client,
        "api_key": "secret",
        "created_by": "api_direct_generation",
        "created_by_user_id": 8,
    }


def test_run_direct_generation_query_passes_search_builders_and_normalized_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_query_builder():
        return None

    def fake_reranker_builder():
        return None

    def fake_run_search_compare(database_url, search_input, **kwargs):
        calls["search_input"] = search_input
        calls["search_kwargs"] = kwargs
        return SimpleNamespace(search_log_id=77)

    monkeypatch.setattr("app.core.direct_generation.run_search_compare", fake_run_search_compare)
    monkeypatch.setattr(
        "app.core.direct_generation.build_retrieval_context_package",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "app.core.direct_generation.execute_mock_generation_run",
        lambda *_args, **_kwargs: object(),
    )

    run_direct_generation_query(
        "postgresql://example",
        DirectGenerationInput(
            query_text="query",
            actor_user_id=9,
            provider_mode=GENERATION_PROVIDER_MODE_MOCK,
            profiles=None,
            chunk_policy_name=" heading_512_64 ",
            document_group=" docs ",
            file_type=" .md ",
            hybrid_vector_profile_name=" qwen3_4b_2560 ",
            reranked_vector_profile_name=" bge_m3_1024 ",
        ),
        query_embedding_provider_builder=fake_query_builder,  # type: ignore[arg-type]
        reranker_provider_builder=fake_reranker_builder,  # type: ignore[arg-type]
    )

    search_input = calls["search_input"]
    assert search_input.profiles is None
    assert search_input.chunk_policy_name == "heading_512_64"
    assert search_input.document_group == "docs"
    assert search_input.file_type == ".md"
    assert search_input.hybrid_vector_profile_name == "qwen3_4b_2560"
    assert search_input.reranked_vector_profile_name == "bge_m3_1024"
    assert calls["search_kwargs"]["query_embedding_provider_builder"] is fake_query_builder
    assert calls["search_kwargs"]["reranker_provider_builder"] is fake_reranker_builder


def test_run_direct_generation_query_reports_missing_retrieval_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.core.direct_generation.run_search_compare",
        lambda *_args, **_kwargs: SimpleNamespace(search_log_id=99),
    )
    monkeypatch.setattr(
        "app.core.direct_generation.build_retrieval_context_package",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "app.core.direct_generation.execute_mock_generation_run",
        lambda *_args, **_kwargs: pytest.fail("generation should not run"),
    )

    with pytest.raises(
        InvalidDirectGenerationError,
        match="retrieval context package was not created",
    ):
        run_direct_generation_query(
            "postgresql://example",
            DirectGenerationInput(
                query_text="query",
                actor_user_id=1,
                provider_mode=GENERATION_PROVIDER_MODE_MOCK,
            ),
        )


@pytest.mark.parametrize(
    ("direct_input", "message"),
    (
        (
            DirectGenerationInput(query_text="", actor_user_id=1),
            "query_text must not be blank",
        ),
        (
            DirectGenerationInput(query_text="query", actor_user_id=0),
            "actor_user_id must be greater than 0",
        ),
        (
            DirectGenerationInput(query_text="query", actor_user_id=1, top_k=0),
            "top_k must be greater than 0",
        ),
        (
            DirectGenerationInput(query_text="query", actor_user_id=1, provider_mode="local"),
            "provider_mode is not supported",
        ),
        (
            DirectGenerationInput(query_text="query", actor_user_id=1, profiles=(" ",)),
            "profiles must not contain blank values",
        ),
        (
            DirectGenerationInput(
                query_text="query",
                actor_user_id=1,
                profiles=("bm25_keyword", "bm25_keyword"),
            ),
            "profiles must be unique",
        ),
    ),
)
def test_run_direct_generation_query_validates_input(
    monkeypatch: pytest.MonkeyPatch,
    direct_input: DirectGenerationInput,
    message: str,
) -> None:
    monkeypatch.setattr(
        "app.core.direct_generation.run_search_compare",
        lambda *_args, **_kwargs: pytest.fail("search should not run"),
    )

    with pytest.raises(InvalidDirectGenerationError, match=message):
        run_direct_generation_query("postgresql://example", direct_input)
