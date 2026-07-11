import pytest

from app.core.embedding_worker_batch_run_retention import (
    EmbeddingBatchRunRetentionSettingsInput,
    InvalidEmbeddingBatchRunRetentionError,
    embedding_batch_run_retention_settings_from_rows,
    validate_embedding_batch_run_retention_settings_input,
)


def test_embedding_batch_run_retention_settings_parse_defaults_and_rows() -> None:
    defaults = embedding_batch_run_retention_settings_from_rows([])
    parsed = embedding_batch_run_retention_settings_from_rows(
        [
            {
                "setting_name": "embedding_batch_run_retention_enabled",
                "setting_value": "false",
            },
            {
                "setting_name": "embedding_batch_run_retention_days",
                "setting_value": "14",
            },
            {
                "setting_name": "embedding_batch_run_cleanup_batch_size",
                "setting_value": "250",
            },
        ]
    )

    assert defaults.enabled is True
    assert defaults.retention_days == 30
    assert defaults.cleanup_batch_size == 1000
    assert parsed.enabled is False
    assert parsed.retention_days == 14
    assert parsed.cleanup_batch_size == 250


@pytest.mark.parametrize(
    ("settings_input", "message"),
    [
        (
            EmbeddingBatchRunRetentionSettingsInput(retention_days=0),
            "retention_days",
        ),
        (
            EmbeddingBatchRunRetentionSettingsInput(cleanup_batch_size=0),
            "cleanup_batch_size",
        ),
    ],
)
def test_embedding_batch_run_retention_settings_reject_invalid_values(
    settings_input: EmbeddingBatchRunRetentionSettingsInput,
    message: str,
) -> None:
    with pytest.raises(InvalidEmbeddingBatchRunRetentionError, match=message):
        validate_embedding_batch_run_retention_settings_input(settings_input)
