from decimal import Decimal

import pytest

from app.core.permissions import PermissionSearchFilter
from app.core.query_embeddings import InvalidQueryEmbeddingError, QueryEmbeddingResult
from app.core.search_compare import (
    SEARCH_COMPARE_PROFILE_ERROR_QUERY_EMBEDDING_FAILED,
    SEARCH_COMPARE_PROFILE_STATUS_FAILED,
    SEARCH_COMPARE_PROFILE_STATUS_SUCCEEDED,
    InvalidSearchCompareError,
    SearchCompareInput,
    SearchCompareReadinessInput,
    SearchCompareReadinessProfile,
    SearchCompareReadinessResult,
    SearchPermissionMatrixEntryInput,
    SearchPermissionMatrixInput,
    _coverage_percent,
    _readiness_status,
    _validate_readiness_input,
    run_permission_search_matrix,
    run_search_compare,
)
from app.core.vector_search import VectorSearchResult


def _permission_filter() -> PermissionSearchFilter:
    return PermissionSearchFilter(
        actor_user_id=1,
        requested_search_scope="company",
        effective_search_scope="company",
        where_sql="TRUE",
        params=(),
        metadata={
            "actor_user_id": 1,
            "login_id": "alice.member",
            "display_name": "Alice",
            "role_name": "member",
            "primary_org_unit_id": 1,
            "primary_org_unit_name": "NeX Company",
            "ancestor_org_unit_ids": (),
            "managed_org_unit_ids": (),
            "includes_company_documents": True,
            "filter_clause_count": 1,
            "permission_explainability": {"visible_document_count": 1},
        },
    )


def _query_embedding_result(profile_name: str) -> QueryEmbeddingResult:
    return QueryEmbeddingResult(
        profile_name=profile_name,
        embedding=(0.1, 0.2, 0.3),
        dimension=3,
        provider_type="mock",
        provider_model_id="unit-provider",
        provider_elapsed_ms=4,
        total_elapsed_ms=5,
        runtime_source="fallback_runtime_config",
        runtime_metadata={"query_embedding_bridge": True},
    )


def _vector_search_result(profile_name: str) -> VectorSearchResult:
    return VectorSearchResult(
        profile_name=profile_name,
        rank=1,
        chunk_id=10,
        document_id=20,
        file_id=30,
        distance=0.1,
        score=0.9,
        chunk_text="policy chunk",
        chunk_preview="policy chunk",
        content_hash="hash",
        chunk_policy_name="heading_512_64",
        heading_path=("Policy",),
        page_no=None,
        slide_no=None,
        sheet_name=None,
        cell_range=None,
        document_title="Policy",
        document_group="unit",
        original_file_name="policy.md",
        file_ext=".md",
        embedding_elapsed_ms=3,
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"chunk_count": 0}, "not_chunked"),
        ({"chunk_count": 2, "embedded_chunk_count": 2}, "ready"),
        ({"chunk_count": 2, "running_count": 1}, "running"),
        ({"chunk_count": 2, "failed_count": 1}, "failed"),
        ({"chunk_count": 2, "failed_count": 1, "pending_count": 1}, "pending"),
        ({"chunk_count": 2, "pending_count": 1}, "pending"),
        ({"chunk_count": 2, "embedded_chunk_count": 1}, "partial"),
        ({"chunk_count": 2, "succeeded_job_count": 1}, "partial"),
        ({"chunk_count": 2}, "missing"),
    ],
)
def test_readiness_status_classifies_profile_policy_state(
    kwargs: dict[str, int],
    expected: str,
) -> None:
    defaults = {
        "chunk_count": 1,
        "pending_count": 0,
        "running_count": 0,
        "failed_count": 0,
        "succeeded_job_count": 0,
        "embedded_chunk_count": 0,
    }
    assert _readiness_status(**{**defaults, **kwargs}) == expected


@pytest.mark.parametrize(
    ("readiness_input", "message"),
    [
        (
            SearchCompareReadinessInput(
                actor_user_id=0,
                requested_search_scope="company",
            ),
            "actor_user_id",
        ),
        (
            SearchCompareReadinessInput(
                actor_user_id=1,
                requested_search_scope="company",
                profiles=(),
            ),
            "profiles must not be empty",
        ),
        (
            SearchCompareReadinessInput(
                actor_user_id=1,
                requested_search_scope="company",
                profiles=("kure_v1_1024", "kure_v1_1024"),
            ),
            "profiles must be unique",
        ),
        (
            SearchCompareReadinessInput(
                actor_user_id=1,
                requested_search_scope="company",
                profiles=(" ",),
            ),
            "profile_name",
        ),
        (
            SearchCompareReadinessInput(
                actor_user_id=1,
                requested_search_scope="company",
                chunk_policy_names=(),
            ),
            "chunk_policy_names must not be empty",
        ),
        (
            SearchCompareReadinessInput(
                actor_user_id=1,
                requested_search_scope="company",
                chunk_policy_names=("heading_512_64", "heading_512_64"),
            ),
            "chunk_policy_names must be unique",
        ),
        (
            SearchCompareReadinessInput(
                actor_user_id=1,
                requested_search_scope="company",
                chunk_policy_name="heading_512_64",
                chunk_policy_names=("heading_1000_200",),
            ),
            "cannot be used together",
        ),
        (
            SearchCompareReadinessInput(
                actor_user_id=1,
                requested_search_scope="company",
                document_group=" ",
            ),
            "document_group",
        ),
    ],
)
def test_validate_readiness_input_rejects_invalid_values(
    readiness_input: SearchCompareReadinessInput,
    message: str,
) -> None:
    with pytest.raises(InvalidSearchCompareError, match=message):
        _validate_readiness_input(readiness_input)


def test_validate_readiness_input_normalizes_optional_values() -> None:
    validated = _validate_readiness_input(
        SearchCompareReadinessInput(
            actor_user_id=1,
            requested_search_scope=" company ",
            profiles=(" kure_v1_1024 ",),
            chunk_policy_names=(" heading_512_64 ",),
            file_type=" .md ",
        )
    )

    assert validated.requested_search_scope == "company"
    assert validated.profiles == ("kure_v1_1024",)
    assert validated.chunk_policy_names == ("heading_512_64",)
    assert validated.file_type == ".md"


def test_search_compare_readiness_result_summarizes_profiles() -> None:
    ready_profile = SearchCompareReadinessProfile(
        profile_name="kure_v1_1024",
        chunk_policy_name="heading_512_64",
        chunk_count=2,
        job_count=2,
        pending_count=0,
        running_count=0,
        failed_count=0,
        succeeded_job_count=2,
        skipped_count=0,
        embedded_chunk_count=2,
        coverage_percent=Decimal("100.00"),
        status="ready",
        latest_job_updated_at=None,
        latest_embedding_at=None,
        average_embedding_elapsed_ms=None,
    )
    failed_profile = SearchCompareReadinessProfile(
        profile_name="bge_m3_1024",
        chunk_policy_name=None,
        chunk_count=2,
        job_count=1,
        pending_count=0,
        running_count=0,
        failed_count=1,
        succeeded_job_count=0,
        skipped_count=0,
        embedded_chunk_count=1,
        coverage_percent=Decimal("50.00"),
        status="failed",
        latest_job_updated_at=None,
        latest_embedding_at=None,
        average_embedding_elapsed_ms=None,
    )
    result = SearchCompareReadinessResult(
        actor_user_id=1,
        requested_search_scope="company",
        effective_search_scope="company",
        document_group=None,
        file_type=None,
        chunk_policy_names=("heading_512_64", "all"),
        profiles=(ready_profile, failed_profile),
    )
    empty_result = SearchCompareReadinessResult(
        actor_user_id=1,
        requested_search_scope="company",
        effective_search_scope="company",
        document_group=None,
        file_type=None,
        chunk_policy_names=(),
        profiles=(),
    )

    assert ready_profile.ready is True
    assert failed_profile.ready is False
    assert failed_profile.missing_embedding_count == 1
    assert result.profile_count == 2
    assert result.policy_count == 2
    assert result.expected_embedding_count == 4
    assert result.embedded_chunk_count == 3
    assert result.attention_count == 1
    assert result.ready is False
    assert result.coverage_percent == Decimal("75.00")
    assert empty_result.ready is False
    assert empty_result.coverage_percent == Decimal("0.00")
    assert _coverage_percent(chunk_count=0, embedded_chunk_count=0) == Decimal("0.00")
    assert _coverage_percent(chunk_count=4, embedded_chunk_count=1) == Decimal("25.00")


@pytest.mark.parametrize(
    ("matrix_input", "message"),
    [
        (
            SearchPermissionMatrixInput(
                query_text="hello",
                entries=(),
            ),
            "entries must not be empty",
        ),
        (
            SearchPermissionMatrixInput(
                query_text="hello",
                entries=tuple(
                    SearchPermissionMatrixEntryInput(
                        actor_user_id=index + 1,
                        requested_search_scope="company",
                    )
                    for index in range(13)
                ),
            ),
            "entries must be 12 or fewer",
        ),
        (
            SearchPermissionMatrixInput(
                query_text="hello",
                entries=(
                    SearchPermissionMatrixEntryInput(
                        actor_user_id=1,
                        requested_search_scope="company",
                    ),
                    SearchPermissionMatrixEntryInput(
                        actor_user_id=1,
                        requested_search_scope="company",
                    ),
                ),
            ),
            "entries must be unique",
        ),
        (
            SearchPermissionMatrixInput(
                query_text="hello",
                entries=(
                    SearchPermissionMatrixEntryInput(
                        actor_user_id=0,
                        requested_search_scope="company",
                    ),
                ),
            ),
            "actor_user_id",
        ),
        (
            SearchPermissionMatrixInput(
                query_text=" ",
                entries=(
                    SearchPermissionMatrixEntryInput(
                        actor_user_id=1,
                        requested_search_scope="company",
                    ),
                ),
            ),
            "query_text",
        ),
    ],
)
def test_run_permission_search_matrix_rejects_invalid_input_before_db(
    matrix_input: SearchPermissionMatrixInput,
    message: str,
) -> None:
    with pytest.raises(InvalidSearchCompareError, match=message):
        run_permission_search_matrix("postgresql://unused", matrix_input)


def test_run_search_compare_returns_profile_failure_without_aborting(monkeypatch) -> None:
    captured_search_log_input = None
    captured_allow_mock_fallback: list[bool] = []

    def fake_embed_query_for_profile(*args, profile_name: str, **kwargs):
        captured_allow_mock_fallback.append(kwargs["allow_mock_fallback"])
        if profile_name == "bge_m3_1024":
            raise InvalidQueryEmbeddingError("provider unavailable")
        return _query_embedding_result(profile_name)

    def fake_search_similar_chunks(_database_url, query_input):
        assert tuple(query_input.query_embedding) == (0.1, 0.2, 0.3)
        return [_vector_search_result(query_input.profile_name)]

    def fake_create_search_log(_database_url, search_log_input):
        nonlocal captured_search_log_input
        captured_search_log_input = search_log_input

        class SearchLog:
            search_log_id = 42

        return SearchLog()

    def fake_create_search_log_results(_database_url, result_inputs):
        assert len(result_inputs) == 1

        class SearchLogResult:
            search_log_result_id = 100

        return [SearchLogResult()]

    monkeypatch.setattr(
        "app.core.search_compare.resolve_permission_search_filter",
        lambda *args, **kwargs: _permission_filter(),
    )
    monkeypatch.setattr(
        "app.core.search_compare._with_permission_explainability",
        lambda _database_url, _search_input, permission_filter: permission_filter,
    )
    monkeypatch.setattr(
        "app.core.search_compare.embed_query_for_profile",
        fake_embed_query_for_profile,
    )
    monkeypatch.setattr(
        "app.core.search_compare.search_similar_chunks",
        fake_search_similar_chunks,
    )
    monkeypatch.setattr("app.core.search_compare.create_search_log", fake_create_search_log)
    monkeypatch.setattr(
        "app.core.search_compare.create_search_log_results",
        fake_create_search_log_results,
    )

    result = run_search_compare(
        "postgresql://unused",
        SearchCompareInput(
            query_text="policy",
            actor_user_id=1,
            requested_search_scope="company",
            profiles=("kure_v1_1024", "bge_m3_1024"),
            allow_mock_fallback=False,
        ),
    )

    assert [profile.status for profile in result.profiles] == [
        SEARCH_COMPARE_PROFILE_STATUS_SUCCEEDED,
        SEARCH_COMPARE_PROFILE_STATUS_FAILED,
    ]
    assert len(result.profiles[0].results) == 1
    assert result.profiles[1].results == ()
    assert result.profiles[1].error_code == SEARCH_COMPARE_PROFILE_ERROR_QUERY_EMBEDDING_FAILED
    assert result.profiles[1].error_message == "provider unavailable"
    assert captured_search_log_input is not None
    metadata = captured_search_log_input.query_runtime_metadata
    assert metadata["selected_profile_count"] == 2
    assert metadata["allow_mock_fallback"] is False
    assert metadata["real_provider_required"] is True
    assert metadata["query_embedding_success_count"] == 1
    assert metadata["profile_status_counts"] == {"succeeded": 1, "failed": 1}
    assert metadata["profile_failure_count"] == 1
    assert set(metadata["profile_query_embeddings"]) == {"kure_v1_1024"}
    assert metadata["profile_failures"]["bge_m3_1024"]["error_message"] == (
        "provider unavailable"
    )
    assert captured_allow_mock_fallback == [False, False]
