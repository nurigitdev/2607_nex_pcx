import pytest
from fastapi.testclient import TestClient

from app.core.chat import (
    CHAT_INTENT_DOCUMENT_SEARCH_SUMMARY,
    CHAT_INTENT_GROUNDED_ANSWER,
    CHAT_ROLE_ASSISTANT,
    CHAT_ROLE_USER,
)
from app.core.config import Settings
from app.core.database import connect, fetch_one
from app.main import create_app

pytestmark = pytest.mark.integration


def _delete_chat_session(database_url: str, chat_session_id: int) -> None:
    with connect(database_url) as conn:
        conn.execute("DELETE FROM chat_sessions WHERE chat_session_id = %s", (chat_session_id,))
        conn.commit()


def test_chat_api_creates_session_and_routes_message_with_mock_assistant(
    migrated_database_url: str,
) -> None:
    app = create_app(Settings(database_url=migrated_database_url))
    chat_session_id: int | None = None

    try:
        with TestClient(app) as client:
            create_response = client.post(
                "/api/chat/sessions",
                json={
                    "session_title": "대화형 검색 요약",
                    "default_provider_mode": "mock",
                    "default_search_scope": "company",
                    "metadata": {"slice": 388},
                },
            )
            chat_session_id = int(create_response.json()["session"]["chat_session_id"])
            message_response = client.post(
                f"/api/chat/sessions/{chat_session_id}/messages",
                json={
                    "content": "관련 문서를 검색해서 요약해줘",
                    "routing_metadata": {"ui_surface": "chat"},
                },
            )
            thread_response = client.get(f"/api/chat/sessions/{chat_session_id}")
            list_response = client.get("/api/chat/sessions", params={"status": "all", "limit": 5})

        create_body = create_response.json()
        message_body = message_response.json()
        thread_body = thread_response.json()
        session_row = fetch_one(
            migrated_database_url,
            """
            SELECT session_title, default_search_scope, metadata
            FROM chat_sessions
            WHERE chat_session_id = %s
            """,
            (chat_session_id,),
        )

        assert create_response.status_code == 201
        assert create_body["session"]["session_title"] == "대화형 검색 요약"
        assert create_body["session"]["metadata"] == {"slice": 388}
        assert message_response.status_code == 201
        assert message_body["router"]["intent"] == CHAT_INTENT_DOCUMENT_SEARCH_SUMMARY
        assert message_body["user_message"]["role"] == CHAT_ROLE_USER
        assert message_body["user_message"]["routing_metadata"]["ui_surface"] == "chat"
        assert message_body["assistant_message"]["role"] == CHAT_ROLE_ASSISTANT
        assert "검색 결과 요약 의도" in message_body["assistant_message"]["content"]
        assert len(message_body["thread"]["messages"]) == 2
        assert thread_response.status_code == 200
        assert [item["message"]["role"] for item in thread_body["messages"]] == [
            CHAT_ROLE_USER,
            CHAT_ROLE_ASSISTANT,
        ]
        assert list_response.status_code == 200
        assert any(
            item["chat_session_id"] == chat_session_id for item in list_response.json()["sessions"]
        )
        assert session_row["session_title"] == "대화형 검색 요약"
        assert session_row["default_search_scope"] == "company"
        assert session_row["metadata"] == {"slice": 388}
    finally:
        if chat_session_id is not None:
            _delete_chat_session(migrated_database_url, chat_session_id)


def test_chat_api_can_store_user_message_without_mock_assistant(
    migrated_database_url: str,
) -> None:
    app = create_app(Settings(database_url=migrated_database_url))
    chat_session_id: int | None = None

    try:
        with TestClient(app) as client:
            create_response = client.post(
                "/api/chat/sessions",
                json={"session_title": "근거 답변 대화"},
            )
            chat_session_id = int(create_response.json()["session"]["chat_session_id"])
            message_response = client.post(
                f"/api/chat/sessions/{chat_session_id}/messages",
                json={
                    "content": "사규 근거를 찾아서 답변해줘",
                    "run_mock_router": False,
                },
            )

        body = message_response.json()

        assert message_response.status_code == 201
        assert body["router"]["intent"] == CHAT_INTENT_GROUNDED_ANSWER
        assert body["assistant_message"] is None
        assert len(body["thread"]["messages"]) == 1
    finally:
        if chat_session_id is not None:
            _delete_chat_session(migrated_database_url, chat_session_id)


def test_chat_api_returns_expected_errors(migrated_database_url: str) -> None:
    app = create_app(Settings(database_url=migrated_database_url))
    app_without_database = create_app(Settings(database_url=None))

    with TestClient(app) as client:
        missing_response = client.post(
            "/api/chat/sessions/999999999/messages",
            json={"content": "없는 세션입니다."},
        )
        invalid_filter_response = client.get(
            "/api/chat/sessions",
            params={"status": "legacy"},
        )
        invalid_session_response = client.post(
            "/api/chat/sessions",
            json={
                "session_title": "잘못된 provider",
                "default_provider_mode": "local",
            },
        )

    with TestClient(app_without_database) as client:
        no_database_response = client.get("/api/chat/sessions")

    assert missing_response.status_code == 404
    assert invalid_filter_response.status_code == 400
    assert invalid_session_response.status_code == 400
    assert no_database_response.status_code == 503
