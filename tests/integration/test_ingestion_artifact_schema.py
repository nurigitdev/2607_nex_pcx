from uuid import uuid4

import pytest
from psycopg import errors
from psycopg.types.json import Json

from app.core.database import connect, fetch_one

pytestmark = pytest.mark.integration


def _create_document(database_url: str) -> tuple[int, int]:
    checksum = f"ingestion-artifact-{uuid4()}"
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO files (
                    original_file_name,
                    stored_file_name,
                    file_ext,
                    mime_type,
                    detected_file_type,
                    file_type_confidence,
                    file_size_bytes,
                    sha256_checksum,
                    storage_path,
                    common_metadata,
                    format_metadata
                )
                VALUES (
                    %s,
                    %s,
                    '.md',
                    'text/markdown',
                    'md',
                    99.50,
                    128,
                    %s,
                    %s,
                    '{"source": "integration"}',
                    '{"heading_count": 1}'
                )
                RETURNING file_id
                """,
                (
                    f"{checksum}.md",
                    f"{checksum}.stored.md",
                    checksum,
                    f"/tmp/{checksum}.md",
                ),
            )
            file_id = cursor.fetchone()["file_id"]
            cursor.execute(
                """
                INSERT INTO documents (file_id, document_title)
                VALUES (%s, %s)
                RETURNING document_id
                """,
                (file_id, f"Ingestion artifact fixture {checksum}"),
            )
            document_id = cursor.fetchone()["document_id"]
    return file_id, document_id


def _cleanup_file(database_url: str, file_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def test_ingestion_artifact_tables_and_chunk_source_columns(
    migrated_database_url: str,
) -> None:
    table_count = fetch_one(
        migrated_database_url,
        """
        SELECT count(*) AS count
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name IN (
              'extraction_profiles',
              'extraction_runs',
              'extraction_artifacts',
              'document_blocks',
              'table_artifacts',
              'image_artifacts'
          )
        """,
    )
    files_columns = fetch_one(
        migrated_database_url,
        """
        SELECT count(*) AS count
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'files'
          AND column_name IN (
              'detected_file_type',
              'file_type_confidence',
              'common_metadata',
              'format_metadata'
          )
        """,
    )
    chunks_columns = fetch_one(
        migrated_database_url,
        """
        SELECT count(*) AS count
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'chunks'
          AND column_name IN (
              'artifact_id',
              'block_id',
              'chunk_type',
              'content_markdown',
              'source_anchor',
              'source_char_start',
              'source_char_end'
          )
        """,
    )
    chunk_policy_sequence_constraint = fetch_one(
        migrated_database_url,
        """
        SELECT count(*) AS count
        FROM pg_constraint
        WHERE conname = 'chunks_document_policy_seq_key'
        """,
    )

    assert table_count["count"] == 6
    assert files_columns["count"] == 4
    assert chunks_columns["count"] == 7
    assert chunk_policy_sequence_constraint["count"] == 1


def test_ingestion_artifact_lineage_and_set_null_chunk_references(
    migrated_database_url: str,
) -> None:
    file_id, document_id = _create_document(migrated_database_url)
    profile_name = f"local_md_{uuid4().hex}"
    try:
        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO extraction_profiles (
                        extraction_profile_name,
                        extractor_name,
                        extractor_version,
                        supported_file_types,
                        default_options
                    )
                    VALUES (%s, 'local_markdown', '0.1.0', ARRAY['md'], %s)
                    RETURNING extraction_profile_name
                    """,
                    (profile_name, Json({"normalize_headings": True})),
                )
                cursor.execute(
                    """
                    INSERT INTO extraction_runs (
                        file_id,
                        document_id,
                        extraction_profile_name,
                        status,
                        extractor_name,
                        extractor_version,
                        elapsed_ms,
                        runtime_metadata
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        'succeeded',
                        'local_markdown',
                        '0.1.0',
                        12,
                        %s
                    )
                    RETURNING extraction_run_id
                    """,
                    (
                        file_id,
                        document_id,
                        profile_name,
                        Json({"provider_mode": "local"}),
                    ),
                )
                extraction_run_id = cursor.fetchone()["extraction_run_id"]
                cursor.execute(
                    """
                    INSERT INTO extraction_artifacts (
                        extraction_run_id,
                        file_id,
                        document_id,
                        artifact_type,
                        content_text,
                        content_hash,
                        language,
                        metadata
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        'normalized_markdown',
                        '# Title\n\n| A | B |\n',
                        'artifact-hash-one',
                        'ko',
                        %s
                    )
                    RETURNING artifact_id
                    """,
                    (
                        extraction_run_id,
                        file_id,
                        document_id,
                        Json({"normalizer": "markdown"}),
                    ),
                )
                artifact_id = cursor.fetchone()["artifact_id"]
                cursor.execute(
                    """
                    INSERT INTO document_blocks (
                        artifact_id,
                        document_id,
                        block_seq,
                        block_type,
                        content_text,
                        content_markdown,
                        heading_path,
                        source_anchor,
                        page_no,
                        char_start,
                        char_end,
                        token_count
                    )
                    VALUES (
                        %s,
                        %s,
                        0,
                        'table',
                        'A B',
                        '| A | B |',
                        ARRAY['Title'],
                        %s,
                        1,
                        0,
                        12,
                        4
                    )
                    RETURNING block_id
                    """,
                    (
                        artifact_id,
                        document_id,
                        Json({"page_no": 1, "table_index": 0}),
                    ),
                )
                block_id = cursor.fetchone()["block_id"]
                cursor.execute(
                    """
                    INSERT INTO table_artifacts (
                        block_id,
                        content_markdown,
                        content_json,
                        row_count,
                        column_count,
                        source_anchor
                    )
                    VALUES (%s, '| A | B |', %s, 1, 2, %s)
                    RETURNING table_artifact_id
                    """,
                    (
                        block_id,
                        Json({"rows": [["A", "B"]]}),
                        Json({"page_no": 1, "table_index": 0}),
                    ),
                )
                table_artifact_id = cursor.fetchone()["table_artifact_id"]
                cursor.execute(
                    """
                    INSERT INTO document_blocks (
                        artifact_id,
                        document_id,
                        block_seq,
                        block_type,
                        content_text,
                        source_anchor
                    )
                    VALUES (%s, %s, 1, 'image', 'figure caption', %s)
                    RETURNING block_id
                    """,
                    (
                        artifact_id,
                        document_id,
                        Json({"page_no": 1, "image_index": 0}),
                    ),
                )
                image_block_id = cursor.fetchone()["block_id"]
                cursor.execute(
                    """
                    INSERT INTO image_artifacts (
                        block_id,
                        storage_path,
                        mime_type,
                        width_px,
                        height_px,
                        caption_text,
                        source_anchor
                    )
                    VALUES (%s, '/tmp/figure.png', 'image/png', 320, 200, 'figure caption', %s)
                    RETURNING image_artifact_id
                    """,
                    (
                        image_block_id,
                        Json({"page_no": 1, "image_index": 0}),
                    ),
                )
                image_artifact_id = cursor.fetchone()["image_artifact_id"]
                cursor.execute(
                    """
                    INSERT INTO chunks (
                        document_id,
                        artifact_id,
                        block_id,
                        chunk_seq,
                        chunk_type,
                        chunk_text,
                        content_markdown,
                        content_hash,
                        chunk_policy_name,
                        heading_path,
                        source_anchor,
                        source_char_start,
                        source_char_end,
                        token_count,
                        char_count
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        0,
                        'table',
                        'A B',
                        '| A | B |',
                        'chunk-hash-one',
                        'heading_512_64',
                        ARRAY['Title'],
                        %s,
                        0,
                        12,
                        4,
                        3
                    )
                    RETURNING chunk_id
                    """,
                    (
                        document_id,
                        artifact_id,
                        block_id,
                        Json({"page_no": 1, "table_index": 0}),
                    ),
                )
                chunk_id = cursor.fetchone()["chunk_id"]

                cursor.execute(
                    """
                    SELECT
                        c.chunk_type,
                        c.source_anchor->>'table_index' AS table_index,
                        b.block_type,
                        a.artifact_type
                    FROM chunks c
                    JOIN document_blocks b ON b.block_id = c.block_id
                    JOIN extraction_artifacts a ON a.artifact_id = c.artifact_id
                    WHERE c.chunk_id = %s
                    """,
                    (chunk_id,),
                )
                lineage = cursor.fetchone()
                cursor.execute(
                    "DELETE FROM extraction_artifacts WHERE artifact_id = %s",
                    (artifact_id,),
                )
                cursor.execute(
                    """
                    SELECT artifact_id, block_id
                    FROM chunks
                    WHERE chunk_id = %s
                    """,
                    (chunk_id,),
                )
                null_refs = cursor.fetchone()
                cursor.execute("""
                    SELECT
                        to_regclass('public.table_artifacts') IS NOT NULL AS table_exists,
                        to_regclass('public.image_artifacts') IS NOT NULL AS image_exists
                    """)
                artifact_tables = cursor.fetchone()

        assert table_artifact_id > 0
        assert image_artifact_id > 0
        assert lineage == {
            "chunk_type": "table",
            "table_index": "0",
            "block_type": "table",
            "artifact_type": "normalized_markdown",
        }
        assert null_refs == {"artifact_id": None, "block_id": None}
        assert artifact_tables == {"table_exists": True, "image_exists": True}
    finally:
        _cleanup_file(migrated_database_url, file_id)


def test_ingestion_artifact_check_constraints(migrated_database_url: str) -> None:
    file_id, document_id = _create_document(migrated_database_url)
    try:
        with connect(migrated_database_url) as connection:
            with connection.cursor() as cursor:
                with pytest.raises(errors.CheckViolation):
                    cursor.execute(
                        """
                        INSERT INTO extraction_profiles (
                            extraction_profile_name,
                            extractor_name,
                            extractor_version,
                            provider_mode,
                            supported_file_types
                        )
                        VALUES (%s, 'bad', '0.1.0', 'sidecar', ARRAY['md'])
                        """,
                        (f"bad_profile_{uuid4().hex}",),
                    )
            connection.rollback()

            with connection.cursor() as cursor:
                with pytest.raises(errors.CheckViolation):
                    cursor.execute(
                        """
                        INSERT INTO extraction_artifacts (
                            file_id,
                            document_id,
                            artifact_type
                        )
                        VALUES (%s, %s, 'normalized_markdown')
                        """,
                        (file_id, document_id),
                    )
            connection.rollback()

            with connection.cursor() as cursor:
                with pytest.raises(errors.CheckViolation):
                    cursor.execute(
                        """
                        INSERT INTO chunks (
                            document_id,
                            chunk_seq,
                            chunk_text,
                            content_hash,
                            chunk_policy_name,
                            char_count,
                            source_char_start,
                            source_char_end
                        )
                        VALUES (%s, 0, 'Bad range', 'bad-range', 'heading_512_64', 9, 20, 10)
                        """,
                        (document_id,),
                    )
            connection.rollback()
    finally:
        _cleanup_file(migrated_database_url, file_id)
