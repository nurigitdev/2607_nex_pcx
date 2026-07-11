"""Repository helpers for embedding provider contract sample sets."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from psycopg.types.json import Json

from app.core.database import connect
from app.core.embedding_provider_route_contracts import DEFAULT_CONTRACT_SAMPLE_TEXTS
from app.core.embedding_providers import EMBEDDING_PROVIDER_INPUT_TYPES

DEFAULT_CONTRACT_SAMPLE_SET_NAME = "default_route_contract"


@dataclass(frozen=True)
class EmbeddingProviderContractSampleSetInput:
    sample_set_name: str
    description: str | None = None
    input_type: str = "document"
    sample_texts: tuple[str, ...] = DEFAULT_CONTRACT_SAMPLE_TEXTS
    is_active: bool = True
    is_default: bool = False


@dataclass(frozen=True)
class EmbeddingProviderContractSampleSetRecord:
    sample_set_name: str
    description: str | None
    input_type: str
    sample_texts: tuple[str, ...]
    is_active: bool
    is_default: bool
    created_at: datetime
    updated_at: datetime


class InvalidEmbeddingProviderContractSampleSetError(ValueError):
    """Raised when a contract sample set is invalid or unavailable."""


def get_embedding_provider_contract_sample_set(
    database_url: str,
    sample_set_name: str,
) -> EmbeddingProviderContractSampleSetRecord | None:
    normalized_sample_set_name = _validate_nonblank(sample_set_name, "sample_set_name")
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM embedding_provider_contract_sample_sets
                WHERE sample_set_name = %s
                """,
                (normalized_sample_set_name,),
            )
            row = cursor.fetchone()
    return _row_to_sample_set_record(dict(row)) if row is not None else None


def validate_embedding_provider_contract_sample_set_input(
    sample_input: EmbeddingProviderContractSampleSetInput,
) -> EmbeddingProviderContractSampleSetInput:
    sample_set_name = _validate_nonblank(sample_input.sample_set_name, "sample_set_name")
    input_type = _validate_nonblank(sample_input.input_type, "input_type")
    if input_type not in EMBEDDING_PROVIDER_INPUT_TYPES:
        raise InvalidEmbeddingProviderContractSampleSetError(
            f"Unsupported input_type: {input_type}"
        )
    sample_texts = tuple(
        _validate_nonblank(sample_text, "sample_text") for sample_text in sample_input.sample_texts
    )
    if not sample_texts:
        raise InvalidEmbeddingProviderContractSampleSetError("sample_texts must not be empty")
    if sample_input.is_default and not sample_input.is_active:
        raise InvalidEmbeddingProviderContractSampleSetError(
            "Default contract sample set must be active"
        )
    return EmbeddingProviderContractSampleSetInput(
        sample_set_name=sample_set_name,
        description=sample_input.description.strip() if sample_input.description else None,
        input_type=input_type,
        sample_texts=sample_texts,
        is_active=sample_input.is_active,
        is_default=sample_input.is_default,
    )


def upsert_embedding_provider_contract_sample_set(
    database_url: str,
    sample_input: EmbeddingProviderContractSampleSetInput,
) -> EmbeddingProviderContractSampleSetRecord:
    validated = validate_embedding_provider_contract_sample_set_input(sample_input)
    with connect(database_url) as connection:
        if not validated.is_default:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT is_default
                    FROM embedding_provider_contract_sample_sets
                    WHERE sample_set_name = %s
                    """,
                    (validated.sample_set_name,),
                )
                row = cursor.fetchone()
            if row is not None and bool(row["is_default"]):
                raise InvalidEmbeddingProviderContractSampleSetError(
                    "Default contract sample set cannot be unset directly"
                )
        if validated.is_default:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE embedding_provider_contract_sample_sets
                    SET is_default = false,
                        updated_at = now()
                    WHERE is_default
                      AND sample_set_name <> %s
                    """,
                    (validated.sample_set_name,),
                )
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO embedding_provider_contract_sample_sets (
                    sample_set_name,
                    description,
                    input_type,
                    sample_texts,
                    is_active,
                    is_default
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (sample_set_name)
                DO UPDATE SET
                    description = EXCLUDED.description,
                    input_type = EXCLUDED.input_type,
                    sample_texts = EXCLUDED.sample_texts,
                    is_active = EXCLUDED.is_active,
                    is_default = EXCLUDED.is_default,
                    updated_at = now()
                RETURNING *
                """,
                (
                    validated.sample_set_name,
                    validated.description,
                    validated.input_type,
                    Json(list(validated.sample_texts)),
                    validated.is_active,
                    validated.is_default,
                ),
            )
            return _row_to_sample_set_record(dict(cursor.fetchone()))


def list_embedding_provider_contract_sample_sets(
    database_url: str,
    *,
    active_only: bool = False,
) -> list[EmbeddingProviderContractSampleSetRecord]:
    where_sql = "WHERE is_active" if active_only else ""
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"""
                SELECT *
                FROM embedding_provider_contract_sample_sets
                {where_sql}
                ORDER BY is_default DESC, sample_set_name ASC
                """)
            rows = cursor.fetchall()
    return [_row_to_sample_set_record(dict(row)) for row in rows]


def delete_embedding_provider_contract_sample_set(
    database_url: str,
    sample_set_name: str,
) -> EmbeddingProviderContractSampleSetRecord | None:
    existing = get_embedding_provider_contract_sample_set(database_url, sample_set_name)
    if existing is None:
        return None
    if existing.is_default:
        raise InvalidEmbeddingProviderContractSampleSetError(
            "Default contract sample set cannot be deleted"
        )
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM embedding_provider_contract_sample_sets
                WHERE sample_set_name = %s
                """,
                (existing.sample_set_name,),
            )
    return existing


def get_default_embedding_provider_contract_sample_set(
    database_url: str,
) -> EmbeddingProviderContractSampleSetRecord:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT *
                FROM embedding_provider_contract_sample_sets
                WHERE is_active
                  AND is_default
                ORDER BY sample_set_name ASC
                LIMIT 1
                """)
            row = cursor.fetchone()
    if row is None:
        raise InvalidEmbeddingProviderContractSampleSetError(
            "No active default embedding provider contract sample set is configured"
        )
    return _row_to_sample_set_record(dict(row))


def _validate_nonblank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise InvalidEmbeddingProviderContractSampleSetError(f"{field_name} is required")
    return normalized


def _row_to_sample_set_record(
    row: dict[str, Any],
) -> EmbeddingProviderContractSampleSetRecord:
    return EmbeddingProviderContractSampleSetRecord(
        sample_set_name=str(row["sample_set_name"]),
        description=row["description"],
        input_type=str(row["input_type"]),
        sample_texts=tuple(str(sample_text) for sample_text in row["sample_texts"]),
        is_active=bool(row["is_active"]),
        is_default=bool(row["is_default"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
