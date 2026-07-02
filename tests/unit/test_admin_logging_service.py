import pytest

from app.core import admin_logging
from app.core.admin_logging import LogSettings


def test_settings_from_rows_uses_defaults_and_db_values() -> None:
    settings = admin_logging.settings_from_rows(
        [
            {"setting_name": "logging_enabled", "setting_value": "yes"},
            {"setting_name": "min_log_level", "setting_value": "ERROR"},
            {"setting_name": "log_retention_days", "setting_value": "14"},
            {"setting_name": "admin_log_page_size", "setting_value": "25"},
        ]
    )

    assert settings == LogSettings(
        enabled=True,
        min_level="ERROR",
        retention_days=14,
        page_size=25,
    )


def test_settings_from_rows_falls_back_for_invalid_values() -> None:
    settings = admin_logging.settings_from_rows(
        [
            {"setting_name": "min_log_level", "setting_value": "NOPE"},
            {"setting_name": "log_retention_days", "setting_value": "-1"},
            {"setting_name": "admin_log_page_size", "setting_value": "not-int"},
        ]
    )

    assert settings == LogSettings()


def test_should_store_log_respects_enabled_and_level() -> None:
    assert admin_logging.should_store_log("ERROR", LogSettings(enabled=True, min_level="WARNING"))
    assert not admin_logging.should_store_log(
        "INFO",
        LogSettings(enabled=True, min_level="WARNING"),
    )
    assert not admin_logging.should_store_log("CRITICAL", LogSettings(enabled=False))


def test_normalize_level_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="Unsupported log level"):
        admin_logging.normalize_level("notice")


def test_log_event_returns_none_when_disabled(monkeypatch) -> None:
    class Cursor:
        rowcount = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def execute(self, query, params=None):
            self.query = query
            self.params = params

        def fetchall(self):
            return [{"setting_name": "logging_enabled", "setting_value": "false"}]

    class Connection:
        def cursor(self):
            return Cursor()

    class ConnectionManager:
        def __enter__(self):
            return Connection()

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(admin_logging, "connect", lambda database_url: ConnectionManager())

    assert (
        admin_logging.log_event(
            "postgresql://example/test",
            level="ERROR",
            event_type="unit",
            message="disabled",
        )
        is None
    )
