"""DB-backed vLLM runtime readiness threshold settings."""

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Literal

from app.core.database import connect
from app.core.vllm_runtime_readiness import (
    DEFAULT_VLLM_RUNTIME_READINESS_THRESHOLDS,
    VLLMRuntimeReadinessThresholds,
)

ThresholdValueType = Literal["float", "int"]


@dataclass(frozen=True)
class VLLMRuntimeReadinessThresholdDefinition:
    code: str
    setting_name: str
    value_type: ThresholdValueType
    min_value: float
    max_value: float
    unit: str
    description: str


VLLM_RUNTIME_READINESS_THRESHOLD_DEFINITIONS: tuple[
    VLLMRuntimeReadinessThresholdDefinition, ...
] = (
    VLLMRuntimeReadinessThresholdDefinition(
        code="stale_snapshot_warning_minutes",
        setting_name="vllm_runtime_stale_snapshot_warning_minutes",
        value_type="float",
        min_value=0.1,
        max_value=1440.0,
        unit="minutes",
        description="vLLM runtime snapshot age in minutes that triggers a warning",
    ),
    VLLMRuntimeReadinessThresholdDefinition(
        code="stale_snapshot_critical_minutes",
        setting_name="vllm_runtime_stale_snapshot_critical_minutes",
        value_type="float",
        min_value=0.1,
        max_value=10080.0,
        unit="minutes",
        description="vLLM runtime snapshot age in minutes that triggers a critical signal",
    ),
    VLLMRuntimeReadinessThresholdDefinition(
        code="kv_cache_warning_percent",
        setting_name="vllm_runtime_kv_cache_warning_percent",
        value_type="float",
        min_value=0.1,
        max_value=100.0,
        unit="%",
        description="vLLM KV cache usage percent that triggers a warning",
    ),
    VLLMRuntimeReadinessThresholdDefinition(
        code="kv_cache_critical_percent",
        setting_name="vllm_runtime_kv_cache_critical_percent",
        value_type="float",
        min_value=0.1,
        max_value=100.0,
        unit="%",
        description="vLLM KV cache usage percent that triggers a critical signal",
    ),
    VLLMRuntimeReadinessThresholdDefinition(
        code="waiting_requests_warning",
        setting_name="vllm_runtime_waiting_requests_warning",
        value_type="int",
        min_value=1.0,
        max_value=100000.0,
        unit="requests",
        description="Waiting request count that triggers a vLLM runtime warning",
    ),
    VLLMRuntimeReadinessThresholdDefinition(
        code="waiting_requests_critical",
        setting_name="vllm_runtime_waiting_requests_critical",
        value_type="int",
        min_value=1.0,
        max_value=100000.0,
        unit="requests",
        description="Waiting request count that triggers a vLLM runtime critical signal",
    ),
    VLLMRuntimeReadinessThresholdDefinition(
        code="swapped_requests_warning",
        setting_name="vllm_runtime_swapped_requests_warning",
        value_type="int",
        min_value=1.0,
        max_value=100000.0,
        unit="requests",
        description="Swapped request count that triggers a vLLM runtime warning",
    ),
    VLLMRuntimeReadinessThresholdDefinition(
        code="swapped_requests_critical",
        setting_name="vllm_runtime_swapped_requests_critical",
        value_type="int",
        min_value=1.0,
        max_value=100000.0,
        unit="requests",
        description="Swapped request count that triggers a vLLM runtime critical signal",
    ),
    VLLMRuntimeReadinessThresholdDefinition(
        code="preemptions_warning_total",
        setting_name="vllm_runtime_preemptions_warning_total",
        value_type="int",
        min_value=1.0,
        max_value=1000000.0,
        unit="count",
        description="Preemption total that triggers a vLLM runtime warning",
    ),
    VLLMRuntimeReadinessThresholdDefinition(
        code="preemptions_critical_total",
        setting_name="vllm_runtime_preemptions_critical_total",
        value_type="int",
        min_value=1.0,
        max_value=1000000.0,
        unit="count",
        description="Preemption total that triggers a vLLM runtime critical signal",
    ),
    VLLMRuntimeReadinessThresholdDefinition(
        code="ttft_warning_seconds",
        setting_name="vllm_runtime_ttft_warning_seconds",
        value_type="float",
        min_value=0.001,
        max_value=3600.0,
        unit="seconds",
        description="Average time to first token in seconds that triggers a warning",
    ),
    VLLMRuntimeReadinessThresholdDefinition(
        code="ttft_critical_seconds",
        setting_name="vllm_runtime_ttft_critical_seconds",
        value_type="float",
        min_value=0.001,
        max_value=3600.0,
        unit="seconds",
        description="Average time to first token in seconds that triggers a critical signal",
    ),
    VLLMRuntimeReadinessThresholdDefinition(
        code="e2e_latency_warning_seconds",
        setting_name="vllm_runtime_e2e_latency_warning_seconds",
        value_type="float",
        min_value=0.001,
        max_value=3600.0,
        unit="seconds",
        description="Average end-to-end request latency in seconds that triggers a warning",
    ),
    VLLMRuntimeReadinessThresholdDefinition(
        code="e2e_latency_critical_seconds",
        setting_name="vllm_runtime_e2e_latency_critical_seconds",
        value_type="float",
        min_value=0.001,
        max_value=3600.0,
        unit="seconds",
        description="Average end-to-end request latency in seconds that triggers a critical signal",
    ),
)

VLLM_RUNTIME_READINESS_THRESHOLD_DEFINITION_BY_CODE = {
    definition.code: definition for definition in VLLM_RUNTIME_READINESS_THRESHOLD_DEFINITIONS
}
VLLM_RUNTIME_READINESS_THRESHOLD_DEFINITION_BY_SETTING = {
    definition.setting_name: definition
    for definition in VLLM_RUNTIME_READINESS_THRESHOLD_DEFINITIONS
}
VLLM_RUNTIME_READINESS_THRESHOLD_PAIR_CODES = (
    ("stale_snapshot_warning_minutes", "stale_snapshot_critical_minutes"),
    ("kv_cache_warning_percent", "kv_cache_critical_percent"),
    ("waiting_requests_warning", "waiting_requests_critical"),
    ("swapped_requests_warning", "swapped_requests_critical"),
    ("preemptions_warning_total", "preemptions_critical_total"),
    ("ttft_warning_seconds", "ttft_critical_seconds"),
    ("e2e_latency_warning_seconds", "e2e_latency_critical_seconds"),
)


@dataclass(frozen=True)
class VLLMRuntimeReadinessThresholdSettings:
    thresholds: VLLMRuntimeReadinessThresholds


@dataclass(frozen=True)
class VLLMRuntimeReadinessThresholdSettingsInput:
    thresholds: dict[str, float | int]


class InvalidVLLMRuntimeReadinessThresholdSettingsError(ValueError):
    """Raised when vLLM runtime readiness threshold settings are invalid."""


def vllm_runtime_readiness_threshold_settings_from_rows(
    rows: Iterable[dict[str, Any]],
) -> VLLMRuntimeReadinessThresholdSettings:
    values = asdict(DEFAULT_VLLM_RUNTIME_READINESS_THRESHOLDS)
    for row in rows:
        definition = VLLM_RUNTIME_READINESS_THRESHOLD_DEFINITION_BY_SETTING.get(
            str(row["setting_name"])
        )
        if definition is None:
            continue
        values[definition.code] = _parse_setting_value(
            row.get("setting_value"),
            definition,
            values[definition.code],
        )
    return VLLMRuntimeReadinessThresholdSettings(
        thresholds=VLLMRuntimeReadinessThresholds(**values)
    )


def validate_vllm_runtime_readiness_threshold_settings_input(
    settings_input: VLLMRuntimeReadinessThresholdSettingsInput,
) -> VLLMRuntimeReadinessThresholdSettingsInput:
    values = asdict(DEFAULT_VLLM_RUNTIME_READINESS_THRESHOLDS)
    unknown_codes = sorted(
        set(settings_input.thresholds) - set(VLLM_RUNTIME_READINESS_THRESHOLD_DEFINITION_BY_CODE)
    )
    if unknown_codes:
        raise InvalidVLLMRuntimeReadinessThresholdSettingsError(
            f"unknown vLLM runtime readiness threshold code: {unknown_codes[0]}"
        )
    for code, raw_value in settings_input.thresholds.items():
        definition = VLLM_RUNTIME_READINESS_THRESHOLD_DEFINITION_BY_CODE[code]
        values[code] = _normalize_input_value(raw_value, definition)
    _validate_threshold_pairs(values)
    return VLLMRuntimeReadinessThresholdSettingsInput(thresholds=values)


def load_vllm_runtime_readiness_threshold_settings(
    database_url: str,
) -> VLLMRuntimeReadinessThresholdSettings:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT setting_name, setting_value
                FROM app_log_settings
                WHERE setting_name = ANY(%s)
                """,
                (list(VLLM_RUNTIME_READINESS_THRESHOLD_DEFINITION_BY_SETTING),),
            )
            rows = cursor.fetchall()
    return vllm_runtime_readiness_threshold_settings_from_rows(dict(row) for row in rows)


def update_vllm_runtime_readiness_threshold_settings(
    database_url: str,
    settings_input: VLLMRuntimeReadinessThresholdSettingsInput,
) -> VLLMRuntimeReadinessThresholdSettings:
    validated = validate_vllm_runtime_readiness_threshold_settings_input(settings_input)
    rows = []
    for definition in VLLM_RUNTIME_READINESS_THRESHOLD_DEFINITIONS:
        value = validated.thresholds[definition.code]
        rows.append(
            (
                definition.setting_name,
                _setting_value_label(value, definition),
                _db_value_type(definition),
                definition.description,
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
    return VLLMRuntimeReadinessThresholdSettings(
        thresholds=VLLMRuntimeReadinessThresholds(**validated.thresholds)
    )


def reset_vllm_runtime_readiness_threshold_settings(
    database_url: str,
) -> VLLMRuntimeReadinessThresholdSettings:
    return update_vllm_runtime_readiness_threshold_settings(
        database_url,
        VLLMRuntimeReadinessThresholdSettingsInput(
            thresholds=asdict(DEFAULT_VLLM_RUNTIME_READINESS_THRESHOLDS)
        ),
    )


def vllm_runtime_readiness_threshold_settings_payload(
    settings: VLLMRuntimeReadinessThresholdSettings,
) -> dict[str, object]:
    return {"thresholds": asdict(settings.thresholds)}


def vllm_runtime_readiness_threshold_ui_rows(
    settings: VLLMRuntimeReadinessThresholdSettings,
) -> list[dict[str, object]]:
    values = asdict(settings.thresholds)
    return [
        {
            "code": definition.code,
            "value": values[definition.code],
            "value_type": definition.value_type,
            "step": "1" if definition.value_type == "int" else "0.001",
            "min": definition.min_value,
            "max": definition.max_value,
            "unit": definition.unit,
        }
        for definition in VLLM_RUNTIME_READINESS_THRESHOLD_DEFINITIONS
    ]


def _parse_setting_value(
    raw_value: object,
    definition: VLLMRuntimeReadinessThresholdDefinition,
    default: float | int,
) -> float | int:
    try:
        return _normalize_input_value(raw_value, definition)
    except InvalidVLLMRuntimeReadinessThresholdSettingsError:
        return default


def _normalize_input_value(
    raw_value: object,
    definition: VLLMRuntimeReadinessThresholdDefinition,
) -> float | int:
    try:
        value = float(str(raw_value))
    except (TypeError, ValueError) as exc:
        raise InvalidVLLMRuntimeReadinessThresholdSettingsError(
            f"{definition.code} must be a number"
        ) from exc
    if value < definition.min_value or value > definition.max_value:
        raise InvalidVLLMRuntimeReadinessThresholdSettingsError(
            f"{definition.code} must be between {definition.min_value:g} "
            f"and {definition.max_value:g}"
        )
    if definition.value_type == "int":
        if not value.is_integer():
            raise InvalidVLLMRuntimeReadinessThresholdSettingsError(
                f"{definition.code} must be an integer"
            )
        return int(value)
    return float(value)


def _validate_threshold_pairs(values: dict[str, float | int]) -> None:
    for warning_code, critical_code in VLLM_RUNTIME_READINESS_THRESHOLD_PAIR_CODES:
        if float(values[warning_code]) > float(values[critical_code]):
            raise InvalidVLLMRuntimeReadinessThresholdSettingsError(
                f"{warning_code} must be less than or equal to {critical_code}"
            )


def _setting_value_label(
    value: float | int,
    definition: VLLMRuntimeReadinessThresholdDefinition,
) -> str:
    if definition.value_type == "int":
        return str(int(value))
    return f"{float(value):g}"


def _db_value_type(definition: VLLMRuntimeReadinessThresholdDefinition) -> str:
    return "int" if definition.value_type == "int" else "text"
