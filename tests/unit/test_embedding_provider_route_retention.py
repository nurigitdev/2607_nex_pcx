import pytest

from app.core.embedding_provider_route_retention import (
    InvalidProviderRouteRetentionError,
    ProviderRouteRetentionSettingsInput,
    provider_route_retention_settings_from_rows,
    validate_provider_route_retention_settings_input,
)


def test_provider_route_retention_settings_from_rows_uses_db_values() -> None:
    settings = provider_route_retention_settings_from_rows(
        [
            {"setting_name": "provider_route_retention_enabled", "setting_value": "false"},
            {"setting_name": "provider_route_retention_days", "setting_value": "45"},
            {"setting_name": "provider_route_cleanup_batch_size", "setting_value": "250"},
        ]
    )

    assert settings.enabled is False
    assert settings.retention_days == 45
    assert settings.cleanup_batch_size == 250


def test_provider_route_retention_settings_from_rows_falls_back_for_invalid_values() -> None:
    settings = provider_route_retention_settings_from_rows(
        [
            {"setting_name": "provider_route_retention_enabled", "setting_value": "maybe"},
            {"setting_name": "provider_route_retention_days", "setting_value": "-1"},
            {"setting_name": "provider_route_cleanup_batch_size", "setting_value": "nope"},
        ]
    )

    assert settings.enabled is True
    assert settings.retention_days == 30
    assert settings.cleanup_batch_size == 1000


@pytest.mark.parametrize(
    ("settings_input", "message"),
    [
        (
            ProviderRouteRetentionSettingsInput(retention_days=0),
            "retention_days",
        ),
        (
            ProviderRouteRetentionSettingsInput(retention_days=3651),
            "retention_days",
        ),
        (
            ProviderRouteRetentionSettingsInput(cleanup_batch_size=0),
            "cleanup_batch_size",
        ),
        (
            ProviderRouteRetentionSettingsInput(cleanup_batch_size=100001),
            "cleanup_batch_size",
        ),
    ],
)
def test_validate_provider_route_retention_settings_rejects_invalid_values(
    settings_input: ProviderRouteRetentionSettingsInput,
    message: str,
) -> None:
    with pytest.raises(InvalidProviderRouteRetentionError, match=message):
        validate_provider_route_retention_settings_input(settings_input)
