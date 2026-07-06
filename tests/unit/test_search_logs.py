import math

import pytest

from app.core.search_logs import (
    InvalidSearchLogError,
    SearchLogInput,
    SearchLogResultInput,
    SearchResultFeedbackInput,
    validate_search_log_input,
    validate_search_log_result_input,
    validate_search_result_feedback_input,
)


def test_validate_search_log_input_normalizes_values() -> None:
    validated = validate_search_log_input(
        SearchLogInput(
            query_text="  hello  ",
            normalized_query_text="hello",
            actor_user_id=1,
            requested_search_scope="mine",
            effective_search_scope="team",
            permission_filter_metadata={"scope": "team"},
            document_group="policy",
            file_type=".md",
            chunk_policy_name="heading_512_64",
            top_k=5,
            profiles=("kure_v1_1024", "bge_m3_1024"),
            query_runtime_metadata={"adapter": "mock"},
            total_elapsed_ms=10,
            created_by="tester",
            created_by_user_id=2,
        )
    )

    assert validated.query_text == "hello"
    assert validated.profiles == ("kure_v1_1024", "bge_m3_1024")
    assert validated.similarity_metric == "cosine"


@pytest.mark.parametrize(
    ("search_log_input", "message"),
    [
        (SearchLogInput(query_text="", top_k=5, profiles=("kure_v1_1024",)), "query_text"),
        (SearchLogInput(query_text="hello", top_k=0, profiles=("kure_v1_1024",)), "top_k"),
        (SearchLogInput(query_text="hello", top_k=5, profiles=()), "profiles"),
        (
            SearchLogInput(
                query_text="hello",
                top_k=5,
                profiles=("kure_v1_1024", "kure_v1_1024"),
            ),
            "unique",
        ),
        (
            SearchLogInput(
                query_text="hello",
                top_k=5,
                profiles=("kure_v1_1024",),
                requested_search_scope="all",
            ),
            "requested_search_scope",
        ),
        (
            SearchLogInput(
                query_text="hello",
                top_k=5,
                profiles=("kure_v1_1024",),
                similarity_metric="jaccard",
            ),
            "similarity metric",
        ),
        (
            SearchLogInput(
                query_text="hello",
                top_k=5,
                profiles=("kure_v1_1024",),
                total_elapsed_ms=-1,
            ),
            "total_elapsed_ms",
        ),
    ],
)
def test_validate_search_log_input_rejects_invalid_values(
    search_log_input: SearchLogInput,
    message: str,
) -> None:
    with pytest.raises(InvalidSearchLogError, match=message):
        validate_search_log_input(search_log_input)


def test_validate_search_log_result_input_rejects_invalid_values() -> None:
    with pytest.raises(InvalidSearchLogError, match="rank"):
        validate_search_log_result_input(
            SearchLogResultInput(
                search_log_id=1,
                profile_name="kure_v1_1024",
                rank=0,
                chunk_id=1,
            )
        )

    with pytest.raises(InvalidSearchLogError, match="distance"):
        validate_search_log_result_input(
            SearchLogResultInput(
                search_log_id=1,
                profile_name="kure_v1_1024",
                rank=1,
                chunk_id=1,
                distance=math.inf,
            )
        )


def test_validate_search_result_feedback_input_rejects_invalid_values() -> None:
    with pytest.raises(InvalidSearchLogError, match="relevance_label"):
        validate_search_result_feedback_input(
            SearchResultFeedbackInput(search_log_result_id=1, relevance_label="maybe")
        )

    with pytest.raises(InvalidSearchLogError, match="comment"):
        validate_search_result_feedback_input(
            SearchResultFeedbackInput(
                search_log_result_id=1,
                relevance_label="correct",
                comment=" ",
            )
        )
