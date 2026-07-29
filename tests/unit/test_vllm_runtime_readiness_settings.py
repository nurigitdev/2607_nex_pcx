import pytest

from app.core.vllm_runtime_readiness import DEFAULT_VLLM_RUNTIME_READINESS_THRESHOLDS
from app.core.vllm_runtime_readiness_settings import (
    InvalidVLLMRuntimeReadinessThresholdSettingsError,
    VLLMRuntimeReadinessThresholdSettingsInput,
    reset_vllm_runtime_readiness_threshold_settings,
    validate_vllm_runtime_readiness_threshold_settings_input,
    vllm_runtime_readiness_threshold_settings_from_rows,
    vllm_runtime_readiness_threshold_settings_payload,
    vllm_runtime_readiness_threshold_ui_rows,
)


def test_vllm_readiness_threshold_settings_parse_defaults_and_rows() -> None:
    defaults = vllm_runtime_readiness_threshold_settings_from_rows([])
    parsed = vllm_runtime_readiness_threshold_settings_from_rows(
        [
            {
                "setting_name": "vllm_runtime_kv_cache_warning_percent",
                "setting_value": "75.5",
            },
            {
                "setting_name": "vllm_runtime_waiting_requests_critical",
                "setting_value": "8",
            },
            {
                "setting_name": "vllm_runtime_ttft_warning_seconds",
                "setting_value": "bad-value",
            },
            {
                "setting_name": "unrelated_setting",
                "setting_value": "1000",
            },
        ]
    )

    assert defaults.thresholds == DEFAULT_VLLM_RUNTIME_READINESS_THRESHOLDS
    assert parsed.thresholds.kv_cache_warning_percent == 75.5
    assert parsed.thresholds.waiting_requests_critical == 8
    assert parsed.thresholds.ttft_warning_seconds == (
        DEFAULT_VLLM_RUNTIME_READINESS_THRESHOLDS.ttft_warning_seconds
    )


def test_vllm_readiness_threshold_settings_payload_and_ui_rows() -> None:
    settings = vllm_runtime_readiness_threshold_settings_from_rows([])
    payload = vllm_runtime_readiness_threshold_settings_payload(settings)
    rows = vllm_runtime_readiness_threshold_ui_rows(settings)

    assert payload["thresholds"]["kv_cache_warning_percent"] == 80.0
    assert len(rows) == 14
    assert rows[0]["code"] == "stale_snapshot_warning_minutes"
    assert rows[0]["step"] == "0.001"
    assert rows[4]["code"] == "waiting_requests_warning"
    assert rows[4]["step"] == "1"


def test_vllm_readiness_threshold_settings_validate_accepts_partial_updates() -> None:
    validated = validate_vllm_runtime_readiness_threshold_settings_input(
        VLLMRuntimeReadinessThresholdSettingsInput(
            thresholds={
                "kv_cache_warning_percent": 70,
                "kv_cache_critical_percent": 95,
                "waiting_requests_warning": 2.0,
            }
        )
    )

    assert validated.thresholds["kv_cache_warning_percent"] == 70.0
    assert validated.thresholds["kv_cache_critical_percent"] == 95.0
    assert validated.thresholds["waiting_requests_warning"] == 2
    assert validated.thresholds["waiting_requests_critical"] == 5


def test_vllm_readiness_threshold_settings_rejects_unknown_codes() -> None:
    with pytest.raises(InvalidVLLMRuntimeReadinessThresholdSettingsError, match="unknown"):
        validate_vllm_runtime_readiness_threshold_settings_input(
            VLLMRuntimeReadinessThresholdSettingsInput(thresholds={"unknown": 1})
        )


@pytest.mark.parametrize(
    ("code", "value", "message"),
    [
        ("kv_cache_warning_percent", 0, "between"),
        ("kv_cache_warning_percent", 101, "between"),
        ("waiting_requests_warning", 1.5, "integer"),
        ("waiting_requests_warning", "bad", "number"),
    ],
)
def test_vllm_readiness_threshold_settings_rejects_invalid_values(
    code: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(InvalidVLLMRuntimeReadinessThresholdSettingsError, match=message):
        validate_vllm_runtime_readiness_threshold_settings_input(
            VLLMRuntimeReadinessThresholdSettingsInput(thresholds={code: value})
        )


def test_vllm_readiness_threshold_settings_rejects_warning_above_critical() -> None:
    with pytest.raises(
        InvalidVLLMRuntimeReadinessThresholdSettingsError,
        match="kv_cache_warning_percent",
    ):
        validate_vllm_runtime_readiness_threshold_settings_input(
            VLLMRuntimeReadinessThresholdSettingsInput(
                thresholds={
                    "kv_cache_warning_percent": 95,
                    "kv_cache_critical_percent": 90,
                }
            )
        )


def test_reset_vllm_readiness_threshold_settings_uses_default_update(monkeypatch) -> None:
    captured = {}

    def fake_update(database_url, settings_input):
        captured["database_url"] = database_url
        captured["thresholds"] = settings_input.thresholds
        return "updated"

    monkeypatch.setattr(
        "app.core.vllm_runtime_readiness_settings."
        "update_vllm_runtime_readiness_threshold_settings",
        fake_update,
    )

    result = reset_vllm_runtime_readiness_threshold_settings("postgresql://example/db")

    assert result == "updated"
    assert captured["database_url"] == "postgresql://example/db"
    assert captured["thresholds"]["kv_cache_warning_percent"] == 80.0
