import pytest
from fastapi.testclient import TestClient

from app.core.chat import (
    CHAT_INTENT_DOCUMENT_GENERATION,
    CHAT_INTENT_DOCUMENT_SEARCH_SUMMARY,
    CHAT_INTENT_GENERAL_ANSWER,
    CHAT_INTENT_GROUNDED_ANSWER,
    CHAT_LINK_TYPE_GENERATION_RUN,
    CHAT_LINK_TYPE_SEARCH_LOG,
    CHAT_ROLE_ASSISTANT,
    CHAT_ROLE_USER,
    ChatMessageInput,
    ChatSessionInput,
    append_chat_message,
    create_chat_session,
)
from app.core.config import Settings
from app.core.database import connect, fetch_one
from app.main import create_app

pytestmark = pytest.mark.integration


def _delete_chat_session(database_url: str, chat_session_id: int) -> None:
    with connect(database_url) as conn:
        conn.execute("DELETE FROM chat_sessions WHERE chat_session_id = %s", (chat_session_id,))
        conn.commit()


def _delete_search_log(database_url: str, search_log_id: int) -> None:
    with connect(database_url) as conn:
        conn.execute("DELETE FROM search_logs WHERE search_log_id = %s", (search_log_id,))
        conn.commit()


def _delete_generation_run(database_url: str, generation_run_id: int) -> None:
    with connect(database_url) as conn:
        conn.execute(
            "DELETE FROM generation_run_citations WHERE generation_run_id = %s",
            (generation_run_id,),
        )
        conn.execute(
            "DELETE FROM generation_runs WHERE generation_run_id = %s",
            (generation_run_id,),
        )
        conn.commit()


def _seed_actor_user_id(database_url: str) -> int:
    row = fetch_one(
        database_url,
        "SELECT user_id FROM app_users WHERE login_id = 'alice.member'",
    )
    return int(row["user_id"])


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
                    "session_title": "대화형 문서 생성",
                    "default_provider_mode": "mock",
                    "default_search_scope": "company",
                    "metadata": {"slice": 388},
                },
            )
            chat_session_id = int(create_response.json()["session"]["chat_session_id"])
            message_response = client.post(
                f"/api/chat/sessions/{chat_session_id}/messages",
                json={
                    "content": "보고서를 작성해줘",
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
        assert create_body["session"]["session_title"] == "대화형 문서 생성"
        assert create_body["session"]["metadata"] == {"slice": 388}
        assert message_response.status_code == 201
        assert message_body["router"]["intent"] == CHAT_INTENT_DOCUMENT_GENERATION
        assert message_body["user_message"]["role"] == CHAT_ROLE_USER
        assert message_body["user_message"]["routing_metadata"]["ui_surface"] == "chat"
        assert message_body["assistant_message"]["role"] == CHAT_ROLE_ASSISTANT
        assert "문서 생성 의도" in message_body["assistant_message"]["content"]
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
        assert session_row["session_title"] == "대화형 문서 생성"
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


def test_chat_api_executes_general_answer_with_mock_llm_path(
    migrated_database_url: str,
) -> None:
    app = create_app(Settings(database_url=migrated_database_url))
    chat_session_id: int | None = None

    try:
        with TestClient(app) as client:
            create_response = client.post(
                "/api/chat/sessions",
                json={
                    "session_title": "일반 대화 실행",
                    "default_provider_mode": "mock",
                },
            )
            chat_session_id = int(create_response.json()["session"]["chat_session_id"])
            message_response = client.post(
                f"/api/chat/sessions/{chat_session_id}/messages",
                json={
                    "content": "오늘 회의를 준비하는 좋은 방법을 알려줘",
                    "routing_metadata": {"ui_surface": "chat"},
                },
            )

        body = message_response.json()
        assistant = body["assistant_message"]
        general_answer = assistant["runtime_metadata"]["chat_general_answer"]

        assert message_response.status_code == 201
        assert body["router"]["intent"] == CHAT_INTENT_GENERAL_ANSWER
        assert assistant["role"] == CHAT_ROLE_ASSISTANT
        assert "일반 답변 초안입니다." in assistant["content"]
        assert assistant["runtime_metadata"]["execution_mode"] == "general_llm_mock"
        assert general_answer["provider_mode"] == "mock"
        assert general_answer["prompt_version"] == "chat_general_answer_v1"
        assert general_answer["total_token_count"] > 0
        assert general_answer["request_metadata"]["runtime_metadata"]["chat_session_id"] == (
            chat_session_id
        )
        assert general_answer["request_metadata"]["runtime_metadata"]["routing_metadata"] == {
            "ui_surface": "chat"
        }
    finally:
        if chat_session_id is not None:
            _delete_chat_session(migrated_database_url, chat_session_id)


def test_chat_api_executes_search_summary_and_links_search_log(
    migrated_database_url: str,
) -> None:
    app = create_app(Settings(database_url=migrated_database_url))
    actor_user_id = _seed_actor_user_id(migrated_database_url)
    chat_session_id: int | None = None
    search_log_id: int | None = None

    try:
        with TestClient(app) as client:
            create_response = client.post(
                "/api/chat/sessions",
                json={
                    "session_title": "검색 요약 실행",
                    "actor_user_id": actor_user_id,
                    "default_provider_mode": "mock",
                    "default_search_scope": "company",
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

        body = message_response.json()
        assistant = body["assistant_message"]
        chat_search_summary = assistant["runtime_metadata"]["chat_search_summary"]
        search_log_id = int(chat_search_summary["search_log_id"])
        assistant_thread_item = body["thread"]["messages"][1]
        search_log_row = fetch_one(
            migrated_database_url,
            "SELECT query_text, actor_user_id FROM search_logs WHERE search_log_id = %s",
            (search_log_id,),
        )

        assert message_response.status_code == 201
        assert body["router"]["intent"] == CHAT_INTENT_DOCUMENT_SEARCH_SUMMARY
        assert "검색 결과 요약입니다." in assistant["content"]
        assert assistant["runtime_metadata"]["execution_mode"] == "search_compare_summary"
        assert chat_search_summary["search_result"]["profiles"][0]["profile_name"] == (
            "bm25_keyword"
        )
        assert chat_search_summary["request_metadata"]["runtime_metadata"]["chat_session_id"] == (
            chat_session_id
        )
        assert assistant_thread_item["links"][0]["link_type"] == CHAT_LINK_TYPE_SEARCH_LOG
        assert assistant_thread_item["links"][0]["target_id"] == search_log_id
        assert search_log_row["query_text"] == "관련 문서를 검색해서 요약해줘"
        assert int(search_log_row["actor_user_id"]) == actor_user_id
    finally:
        if chat_session_id is not None:
            _delete_chat_session(migrated_database_url, chat_session_id)
        if search_log_id is not None:
            _delete_search_log(migrated_database_url, search_log_id)


def test_chat_api_executes_grounded_answer_and_links_generation_run(
    migrated_database_url: str,
) -> None:
    app = create_app(Settings(database_url=migrated_database_url))
    actor_user_id = _seed_actor_user_id(migrated_database_url)
    chat_session_id: int | None = None
    generation_run_id: int | None = None
    search_log_id: int | None = None

    try:
        with TestClient(app) as client:
            create_response = client.post(
                "/api/chat/sessions",
                json={
                    "session_title": "근거 답변 실행",
                    "actor_user_id": actor_user_id,
                    "default_provider_mode": "mock",
                    "default_search_scope": "company",
                },
            )
            chat_session_id = int(create_response.json()["session"]["chat_session_id"])
            message_response = client.post(
                f"/api/chat/sessions/{chat_session_id}/messages",
                json={
                    "content": "사규 근거를 찾아서 답변해줘",
                    "routing_metadata": {"ui_surface": "chat"},
                },
            )

        body = message_response.json()
        assistant = body["assistant_message"]
        grounded = assistant["runtime_metadata"]["chat_grounded_answer"]
        search_log_id = int(grounded["search_log_id"])
        generation_run_id = int(grounded["generation_run_id"])
        link_types = {link["link_type"] for link in body["thread"]["messages"][1]["links"]}
        generation_row = fetch_one(
            migrated_database_url,
            """
            SELECT search_log_id, provider_mode, created_by_user_id
            FROM generation_runs
            WHERE generation_run_id = %s
            """,
            (generation_run_id,),
        )

        assert message_response.status_code == 201
        assert body["router"]["intent"] == CHAT_INTENT_GROUNDED_ANSWER
        assert assistant["runtime_metadata"]["execution_mode"] == "direct_grounded_generation"
        assert grounded["direct_generation"]["generation_run_id"] == generation_run_id
        assert grounded["direct_generation"]["search_log_id"] == search_log_id
        assert grounded["retrieval_context_included_count"] >= 0
        assert CHAT_LINK_TYPE_SEARCH_LOG in link_types
        assert CHAT_LINK_TYPE_GENERATION_RUN in link_types
        assert int(generation_row["search_log_id"]) == search_log_id
        assert generation_row["provider_mode"] == "mock"
        assert int(generation_row["created_by_user_id"]) == actor_user_id
    finally:
        if chat_session_id is not None:
            _delete_chat_session(migrated_database_url, chat_session_id)
        if generation_run_id is not None:
            _delete_generation_run(migrated_database_url, generation_run_id)
        if search_log_id is not None:
            _delete_search_log(migrated_database_url, search_log_id)


def test_chat_api_returns_expected_errors(migrated_database_url: str) -> None:
    app = create_app(Settings(database_url=migrated_database_url))
    app_without_database = create_app(Settings(database_url=None))
    chat_session_id: int | None = None

    try:
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
            no_actor_session_response = client.post(
                "/api/chat/sessions",
                json={"session_title": "검색 actor 없음"},
            )
            chat_session_id = int(no_actor_session_response.json()["session"]["chat_session_id"])
            missing_actor_response = client.post(
                f"/api/chat/sessions/{chat_session_id}/messages",
                json={"content": "관련 문서를 검색해서 요약해줘"},
            )
    finally:
        if chat_session_id is not None:
            _delete_chat_session(migrated_database_url, chat_session_id)

    with TestClient(app_without_database) as client:
        no_database_response = client.get("/api/chat/sessions")

    assert missing_response.status_code == 404
    assert invalid_filter_response.status_code == 400
    assert invalid_session_response.status_code == 400
    assert missing_actor_response.status_code == 400
    assert "actor_user_id" in missing_actor_response.json()["detail"]
    assert no_database_response.status_code == 503


def test_chat_page_renders_shell_and_existing_thread(
    migrated_database_url: str,
) -> None:
    session = create_chat_session(
        migrated_database_url,
        ChatSessionInput(
            session_title="UI Shell 대화",
            default_provider_mode="mock",
            default_search_scope="company",
        ),
    )
    append_chat_message(
        migrated_database_url,
        ChatMessageInput(
            chat_session_id=session.chat_session_id,
            role=CHAT_ROLE_USER,
            content="관련 문서를 검색해서 요약해줘",
            intent=CHAT_INTENT_DOCUMENT_SEARCH_SUMMARY,
        ),
    )
    app = create_app(Settings(database_url=migrated_database_url))

    try:
        with TestClient(app) as client:
            response = client.get(f"/chat?chat_session_id={session.chat_session_id}")
            no_database_response = TestClient(create_app(Settings(database_url=None))).get("/chat")

        assert response.status_code == 200
        assert "Chat Workspace" in response.text
        assert "새 대화" in response.text
        assert "UI Shell 대화" in response.text
        assert "관련 문서를 검색해서 요약해줘" in response.text
        assert 'href="/chat"' in response.text
        assert 'id="chat-session-form"' in response.text
        assert 'id="chat-message-form"' in response.text
        assert "/api/chat/sessions" in response.text
        assert 'name="run_mock_router"' in response.text
        assert no_database_response.status_code == 200
        assert "NEX_PCX_DATABASE_URL is not configured." in no_database_response.text
    finally:
        _delete_chat_session(migrated_database_url, session.chat_session_id)
