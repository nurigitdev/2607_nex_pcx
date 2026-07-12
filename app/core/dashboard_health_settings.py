"""Dashboard operational health threshold settings."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from app.core.database import connect

DASHBOARD_HEALTH_THRESHOLD_SETTING_DEFINITIONS: tuple[tuple[str, str, str], ...] = (
    (
        "pipeline_stale",
        "dashboard_pipeline_stale_critical_threshold",
        "Pipeline stale lease count needed for a critical dashboard signal",
    ),
    (
        "pipeline_exhausted",
        "dashboard_pipeline_exhausted_critical_threshold",
        "Pipeline exhausted failure count needed for a critical dashboard signal",
    ),
    (
        "pipeline_retryable",
        "dashboard_pipeline_retryable_warning_threshold",
        "Pipeline retryable failure count needed for a warning dashboard signal",
    ),
    (
        "embedding_stale",
        "dashboard_embedding_stale_critical_threshold",
        "Embedding stale lease count needed for a critical dashboard signal",
    ),
    (
        "embedding_exhausted",
        "dashboard_embedding_exhausted_critical_threshold",
        "Embedding exhausted failure count needed for a critical dashboard signal",
    ),
    (
        "embedding_retryable",
        "dashboard_embedding_retryable_warning_threshold",
        "Embedding retryable failure count needed for a warning dashboard signal",
    ),
    (
        "provider_alert",
        "dashboard_provider_alert_warning_threshold",
        "Provider alert count needed for a warning dashboard signal",
    ),
    (
        "app_error",
        "dashboard_app_error_warning_threshold",
        "App error log count needed for a warning dashboard signal",
    ),
    (
        "parsing_failure",
        "dashboard_parsing_failure_warning_threshold",
        "Parsing failure count needed for a warning dashboard signal",
    ),
)

SETTING_NAME_BY_SIGNAL_CODE = {
    signal_code: setting_name
    for signal_code, setting_name, _ in DASHBOARD_HEALTH_THRESHOLD_SETTING_DEFINITIONS
}
SIGNAL_CODE_BY_SETTING_NAME = {
    setting_name: signal_code
    for signal_code, setting_name, _ in DASHBOARD_HEALTH_THRESHOLD_SETTING_DEFINITIONS
}
DEFAULT_DASHBOARD_HEALTH_THRESHOLDS = {
    signal_code: 1
    for signal_code, _, _ in DASHBOARD_HEALTH_THRESHOLD_SETTING_DEFINITIONS
}


@dataclass(frozen=True)
class DashboardHealthThresholdSettings:
    thresholds: dict[str, int]


@dataclass(frozen=True)
class DashboardHealthThresholdSettingsInput:
    thresholds: dict[str, int]


class InvalidDashboardHealthThresholdSettingsError(ValueError):
    """Raised when dashboard health threshold settings are invalid."""


def dashboard_health_threshold_settings_from_rows(
    rows: Iterable[dict[str, Any]],
) -> DashboardHealthThresholdSettings:
    thresholds = dict(DEFAULT_DASHBOARD_HEALTH_THRESHOLDS)
    for row in rows:
        signal_code = SIGNAL_CODE_BY_SETTING_NAME.get(str(row["setting_name"]))
        if signal_code is None:
            continue
        thresholds[signal_code] = _parse_positive_int(
            row["setting_value"],
            DEFAULT_DASHBOARD_HEALTH_THRESHOLDS[signal_code],
        )
    return DashboardHealthThresholdSettings(thresholds=thresholds)


def validate_dashboard_health_threshold_settings_input(
    settings_input: DashboardHealthThresholdSettingsInput,
) -> DashboardHealthThresholdSettingsInput:
    thresholds = dict(DEFAULT_DASHBOARD_HEALTH_THRESHOLDS)
    unknown_codes = sorted(set(settings_input.thresholds) - set(thresholds))
    if unknown_codes:
        raise InvalidDashboardHealthThresholdSettingsError(
            f"unknown dashboard health threshold code: {unknown_codes[0]}"
        )
    for signal_code, value in settings_input.thresholds.items():
        if value <= 0 or value > 100000:
            raise InvalidDashboardHealthThresholdSettingsError(
                "threshold values must be between 1 and 100000"
            )
        thresholds[signal_code] = int(value)
    return DashboardHealthThresholdSettingsInput(thresholds=thresholds)


def load_dashboard_health_threshold_settings(
    database_url: str,
) -> DashboardHealthThresholdSettings:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT setting_name, setting_value
                FROM app_log_settings
                WHERE setting_name = ANY(%s)
                """,
                (list(SIGNAL_CODE_BY_SETTING_NAME),),
            )
            rows = cursor.fetchall()
    return dashboard_health_threshold_settings_from_rows(dict(row) for row in rows)


def update_dashboard_health_threshold_settings(
    database_url: str,
    settings_input: DashboardHealthThresholdSettingsInput,
) -> DashboardHealthThresholdSettings:
    validated = validate_dashboard_health_threshold_settings_input(settings_input)
    rows = []
    for signal_code, setting_name, description in (
        DASHBOARD_HEALTH_THRESHOLD_SETTING_DEFINITIONS
    ):
        rows.append(
            (
                setting_name,
                str(validated.thresholds[signal_code]),
                "int",
                description,
            )
        )
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO app_log_settings (
                    setting_name,
                    setting_value,
                    value_type,
                    description,
                    updated_at
                )
                VALUES (%s, %s, %s, %s, now())
                ON CONFLICT (setting_name) DO UPDATE
                SET setting_value = EXCLUDED.setting_value,
                    value_type = EXCLUDED.value_type,
                    description = EXCLUDED.description,
                    updated_at = now()
                """,
                rows,
            )
    return DashboardHealthThresholdSettings(thresholds=validated.thresholds)


def reset_dashboard_health_threshold_settings(
    database_url: str,
) -> DashboardHealthThresholdSettings:
    return update_dashboard_health_threshold_settings(
        database_url,
        DashboardHealthThresholdSettingsInput(
            thresholds=dict(DEFAULT_DASHBOARD_HEALTH_THRESHOLDS)
        ),
    )


def _parse_positive_int(raw_value: object, default: int) -> int:
    try:
        value = int(str(raw_value))
    except (TypeError, ValueError):
        return default
    if value <= 0:
        return default
    return value
