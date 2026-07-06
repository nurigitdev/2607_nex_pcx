"""Embedding vector persistence and deterministic mock embedding helpers."""

import math
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from psycopg import Connection

from app.core.database import connect


@dataclass(frozen=True)
class EmbeddingVectorTable:
    profile_name: str
    table_name: str
    dimension: int
    storage_type: str


@dataclass(frozen=True)
class EmbeddingVectorInput:
    chunk_id: int
    profile_name: str
    embedding: tuple[float, ...]
    elapsed_ms: int | None = None


@dataclass(frozen=True)
class EmbeddingVectorRecord:
    chunk_id: int
    profile_name: str
    table_name: str
    dimension: int
    storage_type: str
    embedding_text: str
    elapsed_ms: int | None
    created_at: Any


class InvalidEmbeddingVectorError(ValueError):
    """Raised when embedding vector input is invalid before reaching the DB."""


EMBEDDING_VECTOR_TABLES: dict[str, EmbeddingVectorTable] = {
    "kure_v1_1024": EmbeddingVectorTable(
        profile_name="kure_v1_1024",
        table_name="chunk_embeddings_kure_v1_1024",
        dimension=1024,
        storage_type="vector",
    ),
    "bge_m3_1024": EmbeddingVectorTable(
        profile_name="bge_m3_1024",
        table_name="chunk_embeddings_bge_m3_1024",
        dimension=1024,
        storage_type="vector",
    ),
    "qwen3_4b_1000": EmbeddingVectorTable(
        profile_name="qwen3_4b_1000",
        table_name="chunk_embeddings_qwen3_4b_1000",
        dimension=1000,
        storage_type="vector",
    ),
    "qwen3_4b_2560": EmbeddingVectorTable(
        profile_name="qwen3_4b_2560",
        table_name="chunk_embeddings_qwen3_4b_2560",
        dimension=2560,
        storage_type="halfvec",
    ),
}


def get_embedding_vector_table(profile_name: str) -> EmbeddingVectorTable:
    profile = profile_name.strip()
    if not profile:
        raise InvalidEmbeddingVectorError("profile_name is required")
    try:
        return EMBEDDING_VECTOR_TABLES[profile]
    except KeyError as exc:
        raise InvalidEmbeddingVectorError(f"Unsupported embedding profile: {profile}") from exc


def _require_positive_id(value: int, field_name: str) -> None:
    if value <= 0:
        raise InvalidEmbeddingVectorError(f"{field_name} must be greater than 0")


def _validate_elapsed_ms(elapsed_ms: int | None) -> None:
    if elapsed_ms is not None and elapsed_ms < 0:
        raise InvalidEmbeddingVectorError("elapsed_ms must be greater than or equal to 0")


def _normalize_embedding_values(embedding: tuple[float, ...]) -> tuple[float, ...]:
    normalized_values = tuple(float(value) for value in embedding)
    if any(not math.isfinite(value) for value in normalized_values):
        raise InvalidEmbeddingVectorError("embedding values must be finite")
    return normalized_values


def vector_to_pg_literal(embedding: tuple[float, ...]) -> str:
    values = _normalize_embedding_values(embedding)
    return "[" + ",".join(format(value, ".8g") for value in values) + "]"


def validate_embedding_vector_input(
    vector_input: EmbeddingVectorInput,
) -> EmbeddingVectorTable:
    _require_positive_id(vector_input.chunk_id, "chunk_id")
    table = get_embedding_vector_table(vector_input.profile_name)
    _validate_elapsed_ms(vector_input.elapsed_ms)
    values = _normalize_embedding_values(vector_input.embedding)
    if len(values) != table.dimension:
        raise InvalidEmbeddingVectorError(
            f"{table.profile_name} requires {table.dimension} dimensions"
        )
    return table


def store_chunk_embedding_in_connection(
    connection: Connection,
    vector_input: EmbeddingVectorInput,
) -> EmbeddingVectorRecord:
    table = validate_embedding_vector_input(vector_input)
    cast_type = "halfvec" if table.storage_type == "halfvec" else "vector"
    embedding_literal = vector_to_pg_literal(vector_input.embedding)
    dimension_expression = (
        "vector_dims(embedding::vector)"
        if table.storage_type == "halfvec"
        else "vector_dims(embedding)"
    )

    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            INSERT INTO {table.table_name} (
                chunk_id,
                embedding,
                elapsed_ms
            )
            VALUES (%s, %s::{cast_type}, %s)
            ON CONFLICT (chunk_id) DO UPDATE
            SET embedding = EXCLUDED.embedding,
                elapsed_ms = EXCLUDED.elapsed_ms,
                created_at = now()
            RETURNING
                chunk_id,
                embedding::text AS embedding_text,
                {dimension_expression} AS dimension,
                elapsed_ms,
                created_at
            """,
            (vector_input.chunk_id, embedding_literal, vector_input.elapsed_ms),
        )
        row = dict(cursor.fetchone())

    return EmbeddingVectorRecord(
        chunk_id=int(row["chunk_id"]),
        profile_name=table.profile_name,
        table_name=table.table_name,
        dimension=int(row["dimension"]),
        storage_type=table.storage_type,
        embedding_text=str(row["embedding_text"]),
        elapsed_ms=int(row["elapsed_ms"]) if row["elapsed_ms"] is not None else None,
        created_at=row["created_at"],
    )


def store_chunk_embedding(
    database_url: str,
    vector_input: EmbeddingVectorInput,
) -> EmbeddingVectorRecord:
    with connect(database_url) as connection:
        return store_chunk_embedding_in_connection(connection, vector_input)


def get_chunk_embedding_in_connection(
    connection: Connection,
    *,
    profile_name: str,
    chunk_id: int,
) -> EmbeddingVectorRecord | None:
    _require_positive_id(chunk_id, "chunk_id")
    table = get_embedding_vector_table(profile_name)
    dimension_expression = (
        "vector_dims(embedding::vector)"
        if table.storage_type == "halfvec"
        else "vector_dims(embedding)"
    )
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT
                chunk_id,
                embedding::text AS embedding_text,
                {dimension_expression} AS dimension,
                elapsed_ms,
                created_at
            FROM {table.table_name}
            WHERE chunk_id = %s
            """,
            (chunk_id,),
        )
        row = cursor.fetchone()

    if row is None:
        return None
    payload = dict(row)
    return EmbeddingVectorRecord(
        chunk_id=int(payload["chunk_id"]),
        profile_name=table.profile_name,
        table_name=table.table_name,
        dimension=int(payload["dimension"]),
        storage_type=table.storage_type,
        embedding_text=str(payload["embedding_text"]),
        elapsed_ms=int(payload["elapsed_ms"]) if payload["elapsed_ms"] is not None else None,
        created_at=payload["created_at"],
    )


def get_chunk_embedding(
    database_url: str,
    *,
    profile_name: str,
    chunk_id: int,
) -> EmbeddingVectorRecord | None:
    with connect(database_url) as connection:
        return get_chunk_embedding_in_connection(
            connection,
            profile_name=profile_name,
            chunk_id=chunk_id,
        )


def get_chunk_text_in_connection(connection: Connection, chunk_id: int) -> str | None:
    _require_positive_id(chunk_id, "chunk_id")
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT chunk_text
            FROM chunks
            WHERE chunk_id = %s
            """,
            (chunk_id,),
        )
        row = cursor.fetchone()
    return str(row["chunk_text"]) if row else None


def generate_mock_embedding(
    text: str,
    *,
    profile_name: str,
    dimension: int,
) -> tuple[float, ...]:
    if dimension <= 0:
        raise InvalidEmbeddingVectorError("dimension must be greater than 0")
    seed = f"{profile_name}\n{text}".encode()
    values: list[float] = []
    block_index = 0
    while len(values) < dimension:
        digest = sha256(seed + block_index.to_bytes(4, byteorder="big")).digest()
        for offset in range(0, len(digest), 4):
            integer = int.from_bytes(digest[offset : offset + 4], byteorder="big")
            values.append((integer / 2**32) * 2 - 1)
            if len(values) == dimension:
                break
        block_index += 1

    magnitude = math.sqrt(sum(value * value for value in values))
    if magnitude == 0:
        return tuple(0.0 for _ in values)
    return tuple(value / magnitude for value in values)
