"""Create chat session schema.

Revision ID: 20260728_0039
Revises: 20260728_0038
Create Date: 2026-07-28
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260728_0039"
down_revision: str | None = "20260728_0038"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE chat_sessions (
            chat_session_id BIGSERIAL PRIMARY KEY,
            actor_user_id BIGINT REFERENCES app_users(user_id) ON DELETE SET NULL,
            session_title TEXT NOT NULL CHECK (length(btrim(session_title)) > 0),
            status TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('active', 'archived')),
            default_language TEXT NOT NULL DEFAULT 'ko'
                CHECK (length(btrim(default_language)) > 0),
            default_provider_mode TEXT NOT NULL DEFAULT 'mock'
                CHECK (default_provider_mode IN ('mock', 'remote_openai_compatible')),
            default_search_profile_name TEXT,
            default_search_scope TEXT
                CHECK (
                    default_search_scope IS NULL
                    OR default_search_scope IN ('mine', 'team', 'managed_org', 'company')
                ),
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(metadata) = 'object'),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """)
    op.execute("""
        CREATE INDEX idx_chat_sessions_actor_time
        ON chat_sessions (actor_user_id, updated_at DESC, chat_session_id DESC)
        """)
    op.execute("""
        CREATE INDEX idx_chat_sessions_status_time
        ON chat_sessions (status, updated_at DESC, chat_session_id DESC)
        """)

    op.execute("""
        CREATE TABLE chat_messages (
            chat_message_id BIGSERIAL PRIMARY KEY,
            chat_session_id BIGINT NOT NULL
                REFERENCES chat_sessions(chat_session_id) ON DELETE CASCADE,
            parent_message_id BIGINT
                REFERENCES chat_messages(chat_message_id) ON DELETE SET NULL,
            sequence_no INT NOT NULL CHECK (sequence_no > 0),
            role TEXT NOT NULL
                CHECK (role IN ('user', 'assistant', 'system', 'tool')),
            content TEXT NOT NULL,
            intent TEXT
                CHECK (
                    intent IS NULL
                    OR intent IN (
                        'general_answer',
                        'document_search_summary',
                        'grounded_answer',
                        'document_generation',
                        'document_summary'
                    )
                ),
            status TEXT NOT NULL DEFAULT 'completed'
                CHECK (status IN ('pending', 'running', 'completed', 'failed', 'blocked')),
            intent_confidence NUMERIC(5, 4)
                CHECK (
                    intent_confidence IS NULL
                    OR (intent_confidence >= 0 AND intent_confidence <= 1)
                ),
            routing_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(routing_metadata) = 'object'),
            runtime_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(runtime_metadata) = 'object'),
            error_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(error_metadata) = 'object'),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (chat_session_id, sequence_no)
        )
        """)
    op.execute("""
        CREATE INDEX idx_chat_messages_session_sequence
        ON chat_messages (chat_session_id, sequence_no, chat_message_id)
        """)
    op.execute("""
        CREATE INDEX idx_chat_messages_intent_status_time
        ON chat_messages (intent, status, created_at DESC)
        """)

    op.execute("""
        CREATE TABLE chat_message_links (
            chat_message_link_id BIGSERIAL PRIMARY KEY,
            chat_message_id BIGINT NOT NULL
                REFERENCES chat_messages(chat_message_id) ON DELETE CASCADE,
            link_type TEXT NOT NULL
                CHECK (
                    link_type IN (
                        'search_log',
                        'generation_run',
                        'document_summary',
                        'document',
                        'artifact',
                        'download'
                    )
                ),
            target_id BIGINT CHECK (target_id IS NULL OR target_id > 0),
            target_url TEXT,
            label TEXT,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                CHECK (jsonb_typeof(metadata) = 'object'),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CHECK (
                target_id IS NOT NULL
                OR (target_url IS NOT NULL AND length(btrim(target_url)) > 0)
            )
        )
        """)
    op.execute("""
        CREATE INDEX idx_chat_message_links_message
        ON chat_message_links (chat_message_id, link_type, chat_message_link_id)
        """)
    op.execute("""
        CREATE INDEX idx_chat_message_links_target
        ON chat_message_links (link_type, target_id)
        WHERE target_id IS NOT NULL
        """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS chat_message_links")
    op.execute("DROP TABLE IF EXISTS chat_messages")
    op.execute("DROP TABLE IF EXISTS chat_sessions")
