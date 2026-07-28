import pytest
from psycopg import errors

from app.core.database import connect, fetch_one

pytestmark = pytest.mark.integration


def test_chat_session_message_and_link_schema_roundtrip(migrated_database_url: str) -> None:
    with connect(migrated_database_url) as conn:
        session = conn.execute("""
            INSERT INTO chat_sessions (
                session_title,
                default_provider_mode,
                default_search_scope,
                metadata
            )
            VALUES (
                '대화형 UX schema smoke',
                'mock',
                'company',
                '{"slice": 386}'::jsonb
            )
            RETURNING chat_session_id, status, default_language
            """).fetchone()
        assert session is not None
        user_message = conn.execute(
            """
            INSERT INTO chat_messages (
                chat_session_id,
                sequence_no,
                role,
                content,
                intent,
                intent_confidence,
                routing_metadata
            )
            VALUES (
                %s,
                1,
                'user',
                '보고서를 생성해줘',
                'document_generation',
                0.9500,
                '{"routing_reason": "contains_report_request"}'::jsonb
            )
            RETURNING chat_message_id
            """,
            (session["chat_session_id"],),
        ).fetchone()
        assert user_message is not None
        assistant_message = conn.execute(
            """
            INSERT INTO chat_messages (
                chat_session_id,
                parent_message_id,
                sequence_no,
                role,
                content,
                intent,
                status,
                runtime_metadata
            )
            VALUES (
                %s,
                %s,
                2,
                'assistant',
                '문서 생성 artifact를 준비했습니다.',
                'document_generation',
                'completed',
                '{"provider_mode": "mock"}'::jsonb
            )
            RETURNING chat_message_id
            """,
            (session["chat_session_id"], user_message["chat_message_id"]),
        ).fetchone()
        assert assistant_message is not None
        conn.execute(
            """
            INSERT INTO chat_message_links (
                chat_message_id,
                link_type,
                target_id,
                target_url,
                label,
                metadata
            )
            VALUES (
                %s,
                'download',
                NULL,
                '/api/generation/runs/10/export/docx',
                'DOCX 다운로드',
                '{"format": "docx"}'::jsonb
            )
            """,
            (assistant_message["chat_message_id"],),
        )
        conn.commit()

    stored = fetch_one(
        migrated_database_url,
        """
        SELECT
            cs.status,
            cs.default_language,
            cm.intent,
            cm.intent_confidence,
            cml.link_type,
            cml.target_url
        FROM chat_sessions cs
        JOIN chat_messages cm
          ON cm.chat_session_id = cs.chat_session_id
        JOIN chat_message_links cml
          ON cml.chat_message_id = cm.chat_message_id
        WHERE cs.session_title = '대화형 UX schema smoke'
        """,
    )

    assert stored["status"] == "active"
    assert stored["default_language"] == "ko"
    assert stored["intent"] == "document_generation"
    assert stored["intent_confidence"] is None
    assert stored["link_type"] == "download"
    assert stored["target_url"] == "/api/generation/runs/10/export/docx"


def test_chat_message_sequence_is_unique_per_session(migrated_database_url: str) -> None:
    with connect(migrated_database_url) as conn:
        session = conn.execute("""
            INSERT INTO chat_sessions (session_title)
            VALUES ('sequence unique smoke')
            RETURNING chat_session_id
            """).fetchone()
        assert session is not None
        conn.execute(
            """
            INSERT INTO chat_messages (chat_session_id, sequence_no, role, content)
            VALUES (%s, 1, 'user', '첫 번째 메시지')
            """,
            (session["chat_session_id"],),
        )
        with pytest.raises(errors.UniqueViolation):
            conn.execute(
                """
                INSERT INTO chat_messages (chat_session_id, sequence_no, role, content)
                VALUES (%s, 1, 'assistant', '중복 순서 메시지')
                """,
                (session["chat_session_id"],),
            )
        conn.rollback()


@pytest.mark.parametrize(
    ("intent_confidence", "expected_error"),
    [
        (-0.0001, errors.CheckViolation),
        (1.0001, errors.CheckViolation),
    ],
)
def test_chat_message_intent_confidence_is_bounded(
    migrated_database_url: str,
    intent_confidence: float,
    expected_error: type[Exception],
) -> None:
    with connect(migrated_database_url) as conn:
        session = conn.execute("""
            INSERT INTO chat_sessions (session_title)
            VALUES ('confidence bound smoke')
            RETURNING chat_session_id
            """).fetchone()
        assert session is not None
        with pytest.raises(expected_error):
            conn.execute(
                """
                INSERT INTO chat_messages (
                    chat_session_id,
                    sequence_no,
                    role,
                    content,
                    intent,
                    intent_confidence
                )
                VALUES (%s, 1, 'assistant', 'confidence', 'general_answer', %s)
                """,
                (session["chat_session_id"], intent_confidence),
            )
        conn.rollback()


def test_chat_message_link_requires_target_id_or_url(migrated_database_url: str) -> None:
    with connect(migrated_database_url) as conn:
        session = conn.execute("""
            INSERT INTO chat_sessions (session_title)
            VALUES ('link target smoke')
            RETURNING chat_session_id
            """).fetchone()
        assert session is not None
        message = conn.execute(
            """
            INSERT INTO chat_messages (chat_session_id, sequence_no, role, content)
            VALUES (%s, 1, 'assistant', 'link target')
            RETURNING chat_message_id
            """,
            (session["chat_session_id"],),
        ).fetchone()
        assert message is not None
        with pytest.raises(errors.CheckViolation):
            conn.execute(
                """
                INSERT INTO chat_message_links (chat_message_id, link_type)
                VALUES (%s, 'artifact')
                """,
                (message["chat_message_id"],),
            )
        conn.rollback()
