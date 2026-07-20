"""Retention cleanup helpers for embedding worker batch run history."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.core.database import connect


@dataclass(frozen=True)
class EmbeddingBatchRunRetentionSettings:
    enabled: bool = True
    retention_days: int = 30
    cleanup_batch_size: int = 1000


@dataclass(frozen=True)
class EmbeddingBatchRunRetentionSettingsInput:
    enabled: bool = True
    retention_days: int = 30
    cleanup_batch_size: int = 1000


@dataclass(frozen=True)
class EmbeddingBatchRunCleanupResult:
    enabled: bool
    dry_run: bool
    retention_days: int
    cleanup_batch_size: int
    expired_batch_run_count: int
    deleted_batch_run_count: int
    cutoff_at: datetime

    @property
    def expired_count(self) -> int:
        return self.expired_batch_run_count

    @property
    def deleted_count(self) -> int:
        return self.deleted_batch_run_count


class InvalidEmbeddingBatchRunRetentionError(ValueError):
    """Raised when embedding batch run retention settings are invalid."""


def embedding_batch_run_retention_settings_from_rows(
    rows: list[dict[str, Any]],
) -> EmbeddingBatchRunRetentionSettings:
    values = {row["setting_name"]: row["setting_value"] for row in rows}
    defaults = EmbeddingBatchRunRetentionSettings()
    return EmbeddingBatchRunRetentionSettings(
        enabled=_parse_bool(
            values.get("embedding_batch_run_retention_enabled", str(defaults.enabled)),
            defaults.enabled,
        ),
        retention_days=_parse_positive_int(
            values.get(
                "embedding_batch_run_retention_days",
                str(defaults.retention_days),
            ),
            defaults.retention_days,
        ),
        cleanup_batch_size=_parse_positive_int(
            values.get(
                "embedding_batch_run_cleanup_batch_size",
                str(defaults.cleanup_batch_size),
            ),
            defaults.cleanup_batch_size,
        ),
    )


def validate_embedding_batch_run_retention_settings_input(
    settings_input: EmbeddingBatchRunRetentionSettingsInput,
) -> EmbeddingBatchRunRetentionSettingsInput:
    if settings_input.retention_days <= 0 or settings_input.retention_days > 3650:
        raise InvalidEmbeddingBatchRunRetentionError("retention_days must be between 1 and 3650")
    if settings_input.cleanup_batch_size <= 0 or settings_input.cleanup_batch_size > 100000:
        raise InvalidEmbeddingBatchRunRetentionError(
            "cleanup_batch_size must be between 1 and 100000"
        )
    return EmbeddingBatchRunRetentionSettingsInput(
        enabled=bool(settings_input.enabled),
        retention_days=settings_input.retention_days,
        cleanup_batch_size=settings_input.cleanup_batch_size,
    )


def load_embedding_batch_run_retention_settings(
    database_url: str,
) -> EmbeddingBatchRunRetentionSettings:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT setting_name, setting_value
                FROM app_log_settings
                WHERE setting_name IN (
                    'embedding_batch_run_retention_enabled',
                    'embedding_batch_run_retention_days',
                    'embedding_batch_run_cleanup_batch_size'
                )
                """)
            rows = cursor.fetchall()
    return embedding_batch_run_retention_settings_from_rows([dict(row) for row in rows])


def update_embedding_batch_run_retention_settings(
    database_url: str,
    settings_input: EmbeddingBatchRunRetentionSettingsInput,
) -> EmbeddingBatchRunRetentionSettings:
    validated = validate_embedding_batch_run_retention_settings_input(settings_input)
    rows = (
        (
            "embedding_batch_run_retention_enabled",
            "true" if validated.enabled else "false",
            "bool",
            "Enable embedding batch run retention cleanup actions",
        ),
        (
            "embedding_batch_run_retention_days",
            str(validated.retention_days),
            "int",
            "Number of days to retain embedding worker batch run history",
        ),
        (
            "embedding_batch_run_cleanup_batch_size",
            str(validated.cleanup_batch_size),
            "int",
            "Maximum embedding batch run rows cleaned up in one action",
        ),
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
    return EmbeddingBatchRunRetentionSettings(
        enabled=validated.enabled,
        retention_days=validated.retention_days,
        cleanup_batch_size=validated.cleanup_batch_size,
    )


def cleanup_expired_embedding_batch_run_records(
    database_url: str,
    *,
    dry_run: bool = True,
) -> EmbeddingBatchRunCleanupResult:
    retention_settings = load_embedding_batch_run_retention_settings(database_url)
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT now() - (%s::int * interval '1 day') AS cutoff_at
                """,
                (retention_settings.retention_days,),
            )
            cutoff_at = cursor.fetchone()["cutoff_at"]
            expired_batch_run_count = _count_expired_batch_runs(cursor, cutoff_at)
            deleted_batch_run_count = 0
            if retention_settings.enabled and not dry_run:
                deleted_batch_run_count = _delete_expired_batch_runs(
                    cursor,
                    cutoff_at=cutoff_at,
                    limit=retention_settings.cleanup_batch_size,
                )

    return EmbeddingBatchRunCleanupResult(
        enabled=retention_settings.enabled,
        dry_run=dry_run,
        retention_days=retention_settings.retention_days,
        cleanup_batch_size=retention_settings.cleanup_batch_size,
        expired_batch_run_count=expired_batch_run_count,
        deleted_batch_run_count=deleted_batch_run_count,
        cutoff_at=cutoff_at,
    )


def _count_expired_batch_runs(cursor, cutoff_at: datetime) -> int:
    cursor.execute(
        """
        SELECT count(*) AS expired_count
        FROM embedding_worker_batch_runs
        WHERE completed_at < %s
        """,
        (cutoff_at,),
    )
    return int(cursor.fetchone()["expired_count"] or 0)


def _delete_expired_batch_runs(cursor, *, cutoff_at: datetime, limit: int) -> int:
    cursor.execute(
        """
        WITH doomed AS (
            SELECT batch_run_id
            FROM embedding_worker_batch_runs
            WHERE completed_at < %s
            ORDER BY completed_at ASC, batch_run_id ASC
            LIMIT %s
        )
        DELETE FROM embedding_worker_batch_runs
        WHERE batch_run_id IN (SELECT batch_run_id FROM doomed)
        """,
        (cutoff_at, limit),
    )
    return cursor.rowcount or 0


def _parse_bool(value: str, default: bool) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_positive_int(value: str, default: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default
