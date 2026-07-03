"""Create core metadata schema.

Revision ID: 20260703_0003
Revises: 20260703_0002
Create Date: 2026-07-03
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260703_0003"
down_revision: str | None = "20260703_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE files (
            file_id BIGSERIAL PRIMARY KEY,
            original_file_name TEXT NOT NULL,
            stored_file_name TEXT NOT NULL,
            file_ext TEXT,
            mime_type TEXT,
            file_size_bytes BIGINT CHECK (file_size_bytes IS NULL OR file_size_bytes >= 0),
            sha256_checksum TEXT UNIQUE,
            storage_path TEXT NOT NULL,
            document_group TEXT NOT NULL DEFAULT 'default',
            security_level TEXT NOT NULL DEFAULT 'internal',
            uploaded_by TEXT,
            uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            parser_name TEXT,
            parser_version TEXT,
            parse_status TEXT NOT NULL DEFAULT 'pending'
                CHECK (parse_status IN ('pending', 'running', 'succeeded', 'failed')),
            parse_error_message TEXT,
            page_count INT CHECK (page_count IS NULL OR page_count >= 0),
            slide_count INT CHECK (slide_count IS NULL OR slide_count >= 0),
            sheet_count INT CHECK (sheet_count IS NULL OR sheet_count >= 0),
            extracted_text_size BIGINT CHECK (
                extracted_text_size IS NULL OR extracted_text_size >= 0
            ),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX idx_files_uploaded_at ON files (uploaded_at DESC)")
    op.execute("CREATE INDEX idx_files_parse_status ON files (parse_status)")

    op.execute(
        """
        CREATE TABLE documents (
            document_id BIGSERIAL PRIMARY KEY,
            file_id BIGINT NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
            document_title TEXT,
            document_group TEXT NOT NULL DEFAULT 'default',
            security_level TEXT NOT NULL DEFAULT 'internal',
            document_status TEXT NOT NULL DEFAULT 'active'
                CHECK (document_status IN ('active', 'archived', 'deleted')),
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX idx_documents_file_id ON documents (file_id)")
    op.execute("CREATE INDEX idx_documents_status ON documents (document_status)")

    op.execute(
        """
        CREATE TABLE chunk_policies (
            chunk_policy_name TEXT PRIMARY KEY,
            target_token_size INT NOT NULL CHECK (target_token_size > 0),
            overlap_token_size INT NOT NULL CHECK (
                overlap_token_size >= 0 AND overlap_token_size < target_token_size
            ),
            split_strategy TEXT NOT NULL,
            preserve_table BOOLEAN NOT NULL DEFAULT true,
            preserve_code_block BOOLEAN NOT NULL DEFAULT true,
            description TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        """
        CREATE TABLE embedding_profiles (
            profile_name TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            dimension INT NOT NULL CHECK (dimension > 0),
            storage_type TEXT NOT NULL CHECK (storage_type IN ('vector', 'halfvec')),
            max_sequence_length INT CHECK (
                max_sequence_length IS NULL OR max_sequence_length > 0
            ),
            mvp_max_input_tokens INT CHECK (
                mvp_max_input_tokens IS NULL OR mvp_max_input_tokens > 0
            ),
            normalize_embeddings BOOLEAN NOT NULL DEFAULT true,
            pooling_strategy TEXT,
            query_instruction TEXT,
            document_instruction TEXT,
            dtype TEXT,
            adapter_name TEXT,
            is_active BOOLEAN NOT NULL DEFAULT true,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX idx_embedding_profiles_active ON embedding_profiles (is_active)")

    op.execute(
        """
        INSERT INTO chunk_policies (
            chunk_policy_name,
            target_token_size,
            overlap_token_size,
            split_strategy,
            preserve_table,
            preserve_code_block,
            description
        )
        VALUES (
            'heading_512_64',
            512,
            64,
            'heading-aware',
            true,
            true,
            'MVP default heading-aware chunk policy'
        )
        ON CONFLICT (chunk_policy_name) DO NOTHING
        """
    )

    op.execute(
        """
        INSERT INTO embedding_profiles (
            profile_name,
            model_name,
            dimension,
            storage_type,
            max_sequence_length,
            mvp_max_input_tokens,
            normalize_embeddings,
            pooling_strategy,
            query_instruction,
            document_instruction,
            dtype,
            adapter_name,
            is_active
        )
        VALUES
            (
                'kure_v1_1024',
                'nlpai-lab/KURE-v1',
                1024,
                'vector',
                8192,
                8192,
                true,
                'sentence-transformers-default',
                NULL,
                NULL,
                'float32',
                'sentence_transformers',
                true
            ),
            (
                'bge_m3_1024',
                'BAAI/bge-m3',
                1024,
                'vector',
                8192,
                8192,
                true,
                'dense',
                NULL,
                NULL,
                'float32',
                'sentence_transformers',
                true
            ),
            (
                'qwen3_4b_1000',
                'Qwen/Qwen3-Embedding-4B',
                1000,
                'vector',
                32768,
                8192,
                true,
                'last-token',
                NULL,
                NULL,
                'float32',
                'qwen_embedding',
                true
            ),
            (
                'qwen3_4b_2560',
                'Qwen/Qwen3-Embedding-4B',
                2560,
                'halfvec',
                32768,
                8192,
                true,
                'last-token',
                NULL,
                NULL,
                'float32',
                'qwen_embedding',
                true
            )
        ON CONFLICT (profile_name) DO NOTHING
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS embedding_profiles")
    op.execute("DROP TABLE IF EXISTS chunk_policies")
    op.execute("DROP TABLE IF EXISTS documents")
    op.execute("DROP TABLE IF EXISTS files")
