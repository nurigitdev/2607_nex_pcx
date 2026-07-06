"""Create chunks schema.

Revision ID: 20260706_0006
Revises: 20260706_0005
Create Date: 2026-07-06
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260706_0006"
down_revision: str | None = "20260706_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO chunk_policies (
            chunk_policy_name,
            target_token_size,
            overlap_token_size,
            split_strategy,
            preserve_table,
            preserve_code_block,
            description
        )
        VALUES
            (
                'heading_1000_200',
                1000,
                200,
                'heading-aware',
                true,
                true,
                'Practical default candidate balancing context and storage cost'
            ),
            (
                'heading_1500_200',
                1500,
                200,
                'heading-aware',
                true,
                true,
                'Long-context candidate for policy and report style documents'
            )
        ON CONFLICT (chunk_policy_name) DO NOTHING
        """)

    op.execute("""
        CREATE TABLE chunks (
            chunk_id BIGSERIAL PRIMARY KEY,
            document_id BIGINT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
            chunk_seq INT NOT NULL CHECK (chunk_seq >= 0),
            chunk_text TEXT NOT NULL CHECK (length(chunk_text) > 0),
            content_hash TEXT NOT NULL,
            chunk_policy_name TEXT NOT NULL REFERENCES chunk_policies(chunk_policy_name),
            parser_name TEXT,
            parser_version TEXT,
            heading_path TEXT[],
            page_no INT CHECK (page_no IS NULL OR page_no > 0),
            slide_no INT CHECK (slide_no IS NULL OR slide_no > 0),
            sheet_name TEXT,
            cell_range TEXT,
            token_count INT CHECK (token_count IS NULL OR token_count >= 0),
            char_count INT NOT NULL CHECK (char_count >= 0),
            prev_chunk_id BIGINT REFERENCES chunks(chunk_id),
            next_chunk_id BIGINT REFERENCES chunks(chunk_id),
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (document_id, chunk_seq),
            UNIQUE (document_id, content_hash, chunk_policy_name)
        )
        """)
    op.execute("CREATE INDEX idx_chunks_document ON chunks (document_id, chunk_seq)")
    op.execute("CREATE INDEX idx_chunks_policy ON chunks (chunk_policy_name)")
    op.execute("CREATE INDEX idx_chunks_content_hash ON chunks (content_hash)")
    op.execute("CREATE INDEX idx_chunks_heading_path ON chunks USING GIN (heading_path)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS chunks")
    op.execute("""
        DELETE FROM chunk_policies
        WHERE chunk_policy_name IN ('heading_1000_200', 'heading_1500_200')
        """)
