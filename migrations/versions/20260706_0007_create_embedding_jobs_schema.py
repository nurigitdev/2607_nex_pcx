"""Create embedding jobs and vector storage schema.

Revision ID: 20260706_0007
Revises: 20260706_0006
Create Date: 2026-07-06
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260706_0007"
down_revision: str | None = "20260706_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE embedding_jobs (
            job_id BIGSERIAL PRIMARY KEY,
            chunk_id BIGINT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            profile_name TEXT NOT NULL REFERENCES embedding_profiles(profile_name),
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'running', 'succeeded', 'failed', 'skipped')),
            attempts INT NOT NULL DEFAULT 0 CHECK (attempts >= 0),
            max_attempts INT NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
            lease_owner TEXT,
            lease_expires_at TIMESTAMPTZ,
            error_code TEXT,
            error_message TEXT,
            last_error_at TIMESTAMPTZ,
            runtime_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (chunk_id, profile_name)
        )
        """)
    op.execute("CREATE INDEX idx_embedding_jobs_status ON embedding_jobs (status, created_at)")
    op.execute("CREATE INDEX idx_embedding_jobs_chunk ON embedding_jobs (chunk_id)")
    op.execute("CREATE INDEX idx_embedding_jobs_profile ON embedding_jobs (profile_name)")
    op.execute("""
        CREATE INDEX idx_embedding_jobs_claim
        ON embedding_jobs (status, profile_name, created_at)
        WHERE status IN ('pending', 'running')
        """)
    op.execute("""
        CREATE INDEX idx_embedding_jobs_lease
        ON embedding_jobs (lease_expires_at)
        WHERE status = 'running'
        """)

    op.execute("""
        CREATE TABLE chunk_embeddings_kure_v1_1024 (
            chunk_id BIGINT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            embedding vector(1024) NOT NULL,
            elapsed_ms INT CHECK (elapsed_ms IS NULL OR elapsed_ms >= 0),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE TABLE chunk_embeddings_bge_m3_1024 (
            chunk_id BIGINT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            embedding vector(1024) NOT NULL,
            elapsed_ms INT CHECK (elapsed_ms IS NULL OR elapsed_ms >= 0),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE TABLE chunk_embeddings_qwen3_4b_1000 (
            chunk_id BIGINT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            embedding vector(1000) NOT NULL,
            elapsed_ms INT CHECK (elapsed_ms IS NULL OR elapsed_ms >= 0),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE TABLE chunk_embeddings_qwen3_4b_2560 (
            chunk_id BIGINT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            embedding halfvec(2560) NOT NULL,
            elapsed_ms INT CHECK (elapsed_ms IS NULL OR elapsed_ms >= 0),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS chunk_embeddings_qwen3_4b_2560")
    op.execute("DROP TABLE IF EXISTS chunk_embeddings_qwen3_4b_1000")
    op.execute("DROP TABLE IF EXISTS chunk_embeddings_bge_m3_1024")
    op.execute("DROP TABLE IF EXISTS chunk_embeddings_kure_v1_1024")
    op.execute("DROP TABLE IF EXISTS embedding_jobs")
