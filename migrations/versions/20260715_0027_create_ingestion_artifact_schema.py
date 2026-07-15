"""Create ingestion artifact schema.

Revision ID: 20260715_0027
Revises: 20260715_0026
Create Date: 2026-07-15
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260715_0027"
down_revision: str | None = "20260715_0026"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE files
        ADD COLUMN detected_file_type TEXT,
        ADD COLUMN file_type_confidence NUMERIC(5, 2)
            CHECK (
                file_type_confidence IS NULL OR
                (file_type_confidence >= 0 AND file_type_confidence <= 100)
            ),
        ADD COLUMN common_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
            CHECK (jsonb_typeof(common_metadata) = 'object'),
        ADD COLUMN format_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
            CHECK (jsonb_typeof(format_metadata) = 'object')
        """)
    op.execute("CREATE INDEX idx_files_detected_file_type ON files (detected_file_type)")

    op.execute("""
        CREATE TABLE extraction_profiles (
            extraction_profile_name TEXT PRIMARY KEY
                CHECK (length(btrim(extraction_profile_name)) > 0),
            extractor_name TEXT NOT NULL CHECK (length(btrim(extractor_name)) > 0),
            extractor_version TEXT NOT NULL CHECK (length(btrim(extractor_version)) > 0),
            provider_mode TEXT NOT NULL DEFAULT 'local'
                CHECK (provider_mode IN ('local', 'remote')),
            supported_file_types TEXT[] NOT NULL
                CHECK (cardinality(supported_file_types) > 0),
            default_options JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(default_options) = 'object'),
            is_active BOOLEAN NOT NULL DEFAULT true,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("CREATE INDEX idx_extraction_profiles_active ON extraction_profiles (is_active)")
    op.execute("""
        CREATE INDEX idx_extraction_profiles_supported_file_types
        ON extraction_profiles USING GIN (supported_file_types)
        """)

    op.execute("""
        CREATE TABLE extraction_runs (
            extraction_run_id BIGSERIAL PRIMARY KEY,
            file_id BIGINT NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
            document_id BIGINT REFERENCES documents(document_id) ON DELETE CASCADE,
            extraction_profile_name TEXT REFERENCES extraction_profiles(extraction_profile_name),
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'running', 'succeeded', 'failed', 'skipped')),
            provider_mode TEXT NOT NULL DEFAULT 'local'
                CHECK (provider_mode IN ('local', 'remote')),
            extractor_name TEXT,
            extractor_version TEXT,
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            elapsed_ms INT CHECK (elapsed_ms IS NULL OR elapsed_ms >= 0),
            warning_count INT NOT NULL DEFAULT 0 CHECK (warning_count >= 0),
            error_count INT NOT NULL DEFAULT 0 CHECK (error_count >= 0),
            error_code TEXT,
            error_message TEXT,
            runtime_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(runtime_metadata) = 'object'),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("CREATE INDEX idx_extraction_runs_file ON extraction_runs (file_id)")
    op.execute("CREATE INDEX idx_extraction_runs_document ON extraction_runs (document_id)")
    op.execute("""
        CREATE INDEX idx_extraction_runs_status_created
        ON extraction_runs (status, created_at DESC)
        """)
    op.execute("""
        CREATE INDEX idx_extraction_runs_profile
        ON extraction_runs (extraction_profile_name, created_at DESC)
        """)

    op.execute("""
        CREATE TABLE extraction_artifacts (
            artifact_id BIGSERIAL PRIMARY KEY,
            extraction_run_id BIGINT REFERENCES extraction_runs(extraction_run_id)
                ON DELETE CASCADE,
            file_id BIGINT NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
            document_id BIGINT REFERENCES documents(document_id) ON DELETE CASCADE,
            artifact_type TEXT NOT NULL
                CHECK (
                    artifact_type IN (
                        'normalized_markdown',
                        'plain_text',
                        'parser_metadata',
                        'warning_report',
                        'source_snapshot'
                    )
                ),
            content_text TEXT,
            storage_path TEXT,
            content_hash TEXT,
            size_bytes BIGINT CHECK (size_bytes IS NULL OR size_bytes >= 0),
            language TEXT,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(metadata) = 'object'),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CHECK (content_text IS NOT NULL OR storage_path IS NOT NULL),
            CHECK (content_hash IS NULL OR length(btrim(content_hash)) > 0)
        )
        """)
    op.execute("""
        CREATE INDEX idx_extraction_artifacts_run_type
        ON extraction_artifacts (extraction_run_id, artifact_type)
        """)
    op.execute("CREATE INDEX idx_extraction_artifacts_file ON extraction_artifacts (file_id)")
    op.execute("""
        CREATE INDEX idx_extraction_artifacts_document
        ON extraction_artifacts (document_id, artifact_type)
        """)
    op.execute("""
        CREATE INDEX idx_extraction_artifacts_content_hash
        ON extraction_artifacts (content_hash)
        WHERE content_hash IS NOT NULL
        """)

    op.execute("""
        CREATE TABLE document_blocks (
            block_id BIGSERIAL PRIMARY KEY,
            artifact_id BIGINT NOT NULL REFERENCES extraction_artifacts(artifact_id)
                ON DELETE CASCADE,
            document_id BIGINT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
            parent_block_id BIGINT REFERENCES document_blocks(block_id) ON DELETE CASCADE,
            block_seq INT NOT NULL CHECK (block_seq >= 0),
            block_type TEXT NOT NULL
                CHECK (
                    block_type IN (
                        'document',
                        'heading',
                        'paragraph',
                        'table',
                        'image',
                        'figure',
                        'list',
                        'code',
                        'page',
                        'slide',
                        'sheet'
                    )
                ),
            content_text TEXT,
            content_markdown TEXT,
            heading_path TEXT[],
            source_anchor JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(source_anchor) = 'object'),
            page_no INT CHECK (page_no IS NULL OR page_no > 0),
            slide_no INT CHECK (slide_no IS NULL OR slide_no > 0),
            sheet_name TEXT,
            cell_range TEXT,
            char_start INT CHECK (char_start IS NULL OR char_start >= 0),
            char_end INT CHECK (char_end IS NULL OR char_end >= 0),
            token_count INT CHECK (token_count IS NULL OR token_count >= 0),
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(metadata) = 'object'),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CHECK (char_start IS NULL OR char_end IS NULL OR char_end >= char_start),
            UNIQUE (artifact_id, block_seq)
        )
        """)
    op.execute("""
        CREATE INDEX idx_document_blocks_artifact
        ON document_blocks (artifact_id, block_seq)
        """)
    op.execute("""
        CREATE INDEX idx_document_blocks_document
        ON document_blocks (document_id, block_seq)
        """)
    op.execute("CREATE INDEX idx_document_blocks_type ON document_blocks (block_type)")
    op.execute(
        "CREATE INDEX idx_document_blocks_heading_path ON document_blocks USING GIN (heading_path)"
    )
    op.execute("""
        CREATE INDEX idx_document_blocks_source_anchor
        ON document_blocks USING GIN (source_anchor)
        """)

    op.execute("""
        CREATE TABLE table_artifacts (
            table_artifact_id BIGSERIAL PRIMARY KEY,
            block_id BIGINT NOT NULL REFERENCES document_blocks(block_id) ON DELETE CASCADE,
            content_markdown TEXT,
            content_json JSONB,
            storage_path TEXT,
            row_count INT CHECK (row_count IS NULL OR row_count >= 0),
            column_count INT CHECK (column_count IS NULL OR column_count >= 0),
            source_anchor JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(source_anchor) = 'object'),
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(metadata) = 'object'),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CHECK (
                content_markdown IS NOT NULL OR
                content_json IS NOT NULL OR
                storage_path IS NOT NULL
            )
        )
        """)
    op.execute("CREATE INDEX idx_table_artifacts_block ON table_artifacts (block_id)")
    op.execute("""
        CREATE INDEX idx_table_artifacts_source_anchor
        ON table_artifacts USING GIN (source_anchor)
        """)

    op.execute("""
        CREATE TABLE image_artifacts (
            image_artifact_id BIGSERIAL PRIMARY KEY,
            block_id BIGINT NOT NULL REFERENCES document_blocks(block_id) ON DELETE CASCADE,
            storage_path TEXT NOT NULL CHECK (length(btrim(storage_path)) > 0),
            mime_type TEXT,
            width_px INT CHECK (width_px IS NULL OR width_px > 0),
            height_px INT CHECK (height_px IS NULL OR height_px > 0),
            ocr_text TEXT,
            caption_text TEXT,
            surrounding_text TEXT,
            source_anchor JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(source_anchor) = 'object'),
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(metadata) = 'object'),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("CREATE INDEX idx_image_artifacts_block ON image_artifacts (block_id)")
    op.execute("""
        CREATE INDEX idx_image_artifacts_source_anchor
        ON image_artifacts USING GIN (source_anchor)
        """)

    op.execute("""
        ALTER TABLE chunks
        ADD COLUMN artifact_id BIGINT REFERENCES extraction_artifacts(artifact_id)
            ON DELETE SET NULL,
        ADD COLUMN block_id BIGINT REFERENCES document_blocks(block_id)
            ON DELETE SET NULL,
        ADD COLUMN chunk_type TEXT NOT NULL DEFAULT 'text'
            CHECK (chunk_type IN ('text', 'table', 'image', 'figure', 'code', 'list', 'heading')),
        ADD COLUMN content_markdown TEXT,
        ADD COLUMN source_anchor JSONB NOT NULL DEFAULT '{}'::jsonb
            CHECK (jsonb_typeof(source_anchor) = 'object'),
        ADD COLUMN source_char_start INT
            CHECK (source_char_start IS NULL OR source_char_start >= 0),
        ADD COLUMN source_char_end INT
            CHECK (source_char_end IS NULL OR source_char_end >= 0),
        ADD CONSTRAINT chunks_source_char_range_check
            CHECK (
                source_char_start IS NULL OR
                source_char_end IS NULL OR
                source_char_end >= source_char_start
            )
        """)
    op.execute("""
        ALTER TABLE chunks
        DROP CONSTRAINT IF EXISTS chunks_document_id_chunk_seq_key
        """)
    op.execute("""
        ALTER TABLE chunks
        ADD CONSTRAINT chunks_document_policy_seq_key
        UNIQUE (document_id, chunk_policy_name, chunk_seq)
        """)
    op.execute("CREATE INDEX idx_chunks_artifact ON chunks (artifact_id)")
    op.execute("CREATE INDEX idx_chunks_block ON chunks (block_id)")
    op.execute("CREATE INDEX idx_chunks_chunk_type ON chunks (chunk_type)")
    op.execute("CREATE INDEX idx_chunks_source_anchor ON chunks USING GIN (source_anchor)")
    op.execute("""
        CREATE INDEX idx_chunks_document_policy_seq
        ON chunks (document_id, chunk_policy_name, chunk_seq)
        """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_chunks_document_policy_seq")
    op.execute("DROP INDEX IF EXISTS idx_chunks_source_anchor")
    op.execute("DROP INDEX IF EXISTS idx_chunks_chunk_type")
    op.execute("DROP INDEX IF EXISTS idx_chunks_block")
    op.execute("DROP INDEX IF EXISTS idx_chunks_artifact")
    op.execute("""
        ALTER TABLE chunks
        DROP CONSTRAINT IF EXISTS chunks_document_policy_seq_key
        """)
    op.execute("""
        ALTER TABLE chunks
        ADD CONSTRAINT chunks_document_id_chunk_seq_key
        UNIQUE (document_id, chunk_seq)
        """)
    op.execute("""
        ALTER TABLE chunks
        DROP CONSTRAINT IF EXISTS chunks_source_char_range_check,
        DROP COLUMN IF EXISTS source_char_end,
        DROP COLUMN IF EXISTS source_char_start,
        DROP COLUMN IF EXISTS source_anchor,
        DROP COLUMN IF EXISTS content_markdown,
        DROP COLUMN IF EXISTS chunk_type,
        DROP COLUMN IF EXISTS block_id,
        DROP COLUMN IF EXISTS artifact_id
        """)

    op.execute("DROP TABLE IF EXISTS image_artifacts")
    op.execute("DROP TABLE IF EXISTS table_artifacts")
    op.execute("DROP TABLE IF EXISTS document_blocks")
    op.execute("DROP TABLE IF EXISTS extraction_artifacts")
    op.execute("DROP TABLE IF EXISTS extraction_runs")
    op.execute("DROP TABLE IF EXISTS extraction_profiles")

    op.execute("DROP INDEX IF EXISTS idx_files_detected_file_type")
    op.execute("""
        ALTER TABLE files
        DROP COLUMN IF EXISTS format_metadata,
        DROP COLUMN IF EXISTS common_metadata,
        DROP COLUMN IF EXISTS file_type_confidence,
        DROP COLUMN IF EXISTS detected_file_type
        """)
