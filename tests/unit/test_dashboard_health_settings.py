import pytest

from app.core.dashboard_health_settings import (
    DashboardHealthThresholdSettingsInput,
    InvalidDashboardHealthThresholdSettingsError,
    dashboard_health_threshold_settings_from_rows,
    validate_dashboard_health_threshold_settings_input,
)


def test_dashboard_health_threshold_settings_parse_defaults_and_rows() -> None:
    defaults = dashboard_health_threshold_settings_from_rows([])
    parsed = dashboard_health_threshold_settings_from_rows(
        [
            {
                "setting_name": "dashboard_pipeline_retryable_warning_threshold",
                "setting_value": "3",
            },
            {
                "setting_name": "dashboard_app_error_warning_threshold",
                "setting_value": "bad-value",
            },
        ]
    )

    assert defaults.thresholds["pipeline_retryable"] == 1
    assert parsed.thresholds["pipeline_retryable"] == 3
    assert parsed.thresholds["app_error"] == 1


def test_validate_dashboard_health_threshold_settings_rejects_unknown_codes() -> None:
    with pytest.raises(InvalidDashboardHealthThresholdSettingsError, match="unknown"):
        validate_dashboard_health_threshold_settings_input(
            DashboardHealthThresholdSettingsInput(thresholds={"unknown": 1})
        )


@pytest.mark.parametrize("value", [0, -1, 100001])
def test_validate_dashboard_health_threshold_settings_rejects_invalid_values(
    value: int,
) -> None:
    with pytest.raises(InvalidDashboardHealthThresholdSettingsError, match="threshold"):
        validate_dashboard_health_threshold_settings_input(
            DashboardHealthThresholdSettingsInput(
                thresholds={"pipeline_retryable": value}
            )
        )
