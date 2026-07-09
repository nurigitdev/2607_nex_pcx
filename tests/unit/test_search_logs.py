import math

import pytest

from app.core.search_logs import (
    InvalidSearchLogError,
    SearchLogInput,
    SearchLogResultInput,
    SearchLogRetentionSettings,
    SearchLogRetentionSettingsInput,
    SearchResultFeedbackInput,
    search_log_retention_settings_from_rows,
    validate_search_log_input,
    validate_search_log_result_input,
    validate_search_log_retention_settings_input,
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


def test_search_log_retention_settings_from_rows_uses_defaults_and_db_values() -> None:
    settings = search_log_retention_settings_from_rows(
        [
            {"setting_name": "search_log_retention_enabled", "setting_value": "false"},
            {"setting_name": "search_log_retention_days", "setting_value": "45"},
            {"setting_name": "search_log_cleanup_batch_size", "setting_value": "250"},
        ]
    )

    assert settings == SearchLogRetentionSettings(
        enabled=False,
        retention_days=45,
        cleanup_batch_size=250,
    )


def test_search_log_retention_settings_from_rows_falls_back_for_invalid_values() -> None:
    settings = search_log_retention_settings_from_rows(
        [
            {"setting_name": "search_log_retention_enabled", "setting_value": "maybe"},
            {"setting_name": "search_log_retention_days", "setting_value": "-1"},
            {"setting_name": "search_log_cleanup_batch_size", "setting_value": "nope"},
        ]
    )

    assert settings == SearchLogRetentionSettings()


@pytest.mark.parametrize(
    ("settings_input", "message"),
    [
        (
            SearchLogRetentionSettingsInput(retention_days=0),
            "retention_days",
        ),
        (
            SearchLogRetentionSettingsInput(retention_days=3651),
            "retention_days",
        ),
        (
            SearchLogRetentionSettingsInput(cleanup_batch_size=0),
            "cleanup_batch_size",
        ),
        (
            SearchLogRetentionSettingsInput(cleanup_batch_size=100001),
            "cleanup_batch_size",
        ),
    ],
)
def test_validate_search_log_retention_settings_input_rejects_invalid_values(
    settings_input: SearchLogRetentionSettingsInput,
    message: str,
) -> None:
    with pytest.raises(InvalidSearchLogError, match=message):
        validate_search_log_retention_settings_input(settings_input)
