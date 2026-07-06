import pytest

from app.core.vector_search import (
    InvalidVectorSearchError,
    VectorSearchInput,
    validate_vector_search_input,
)


def test_validate_vector_search_input_generates_mock_query_embedding() -> None:
    table, embedding = validate_vector_search_input(
        VectorSearchInput(
            query_text="search query",
            profile_name="kure_v1_1024",
            top_k=3,
        )
    )

    assert table.profile_name == "kure_v1_1024"
    assert len(embedding) == 1024


@pytest.mark.parametrize(
    ("query_input", "message"),
    [
        (
            VectorSearchInput(query_text=" ", profile_name="kure_v1_1024"),
            "query_text",
        ),
        (
            VectorSearchInput(query_text="hello", profile_name="kure_v1_1024", top_k=0),
            "top_k",
        ),
        (
            VectorSearchInput(query_text="hello", profile_name="kure_v1_1024", top_k=101),
            "top_k",
        ),
        (
            VectorSearchInput(
                query_text="hello",
                profile_name="kure_v1_1024",
                similarity_metric="l2",
            ),
            "Unsupported similarity metric",
        ),
        (
            VectorSearchInput(
                query_text="hello",
                profile_name="kure_v1_1024",
                chunk_policy_name=" ",
            ),
            "chunk_policy_name",
        ),
        (
            VectorSearchInput(query_text="hello", profile_name="missing_profile"),
            "Unsupported embedding profile",
        ),
        (
            VectorSearchInput(
                query_text="hello",
                profile_name="kure_v1_1024",
                query_embedding=(0.1, 0.2),
            ),
            "1024 dimensions",
        ),
    ],
)
def test_validate_vector_search_input_rejects_invalid_values(
    query_input: VectorSearchInput,
    message: str,
) -> None:
    with pytest.raises(InvalidVectorSearchError, match=message):
        validate_vector_search_input(query_input)
