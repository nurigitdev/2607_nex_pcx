from uuid import uuid4

import pytest

from app.core.chat import (
    CHAT_INTENT_DOCUMENT_GENERATION,
    CHAT_INTENT_GROUNDED_ANSWER,
    CHAT_LINK_TYPE_DOCUMENT_SUMMARY,
    CHAT_LINK_TYPE_DOWNLOAD,
    CHAT_LINK_TYPE_GENERATION_RUN,
    CHAT_LINK_TYPE_SEARCH_LOG,
    CHAT_MESSAGE_STATUS_COMPLETED,
    CHAT_ROLE_ASSISTANT,
    CHAT_ROLE_USER,
    ChatMessageInput,
    ChatSessionInput,
    InvalidChatRepositoryError,
    append_chat_message,
    create_chat_session,
    get_chat_message,
    get_chat_session,
    get_chat_thread,
    link_chat_message_to_document_summary,
    link_chat_message_to_download_url,
    link_chat_message_to_generation_run,
    link_chat_message_to_search_log,
    list_chat_message_links,
    list_chat_messages,
    list_chat_sessions,
    update_chat_session,
)
from app.core.citation_readiness import CITATION_READINESS_READY
from app.core.database import connect, fetch_one
from app.core.generation_runs import (
    GENERATION_GUARDRAIL_ALLOWED,
    GENERATION_PROVIDER_MODE_MOCK,
    GENERATION_STATUS_SUCCEEDED,
    GenerationRunInput,
    create_generation_run,
)
from app.core.retrieval_confidence import RETRIEVAL_CONFIDENCE_ANSWERABLE
from app.core.search_logs import SearchLogInput, create_search_log

pytestmark = pytest.mark.integration


def _delete_chat_session(database_url: str, chat_session_id: int) -> None:
    with connect(database_url) as conn:
        conn.execute("DELETE FROM chat_sessions WHERE chat_session_id = %s", (chat_session_id,))
        conn.commit()


def _delete_search_log(database_url: str, search_log_id: int) -> None:
    with connect(database_url) as conn:
        conn.execute("DELETE FROM search_logs WHERE search_log_id = %s", (search_log_id,))
        conn.commit()


def _seed_actor_user_id(database_url: str) -> int:
    row = fetch_one(
        database_url,
        "SELECT user_id FROM app_users WHERE login_id = 'alice.member'",
    )
    return int(row["user_id"])


def _create_generation_fixture(database_url: str) -> tuple[int, int]:
    query = f"chat repository grounded answer {uuid4()}"
    search_log = create_search_log(
        database_url,
        SearchLogInput(
            query_text=query,
            normalized_query_text=query,
            top_k=3,
            profiles=("reranked_hybrid",),
            requested_search_scope="company",
            effective_search_scope="company",
            strategy_name="reranked_hybrid",
            similarity_metric="cosine",
            query_runtime_metadata={"source": "chat_repository_test"},
            total_elapsed_ms=25,
            created_by="chat-repository-test",
        ),
    )
    generation_run = create_generation_run(
        database_url,
        GenerationRunInput(
            search_log_id=search_log.search_log_id,
            retrieval_package_key=f"retrieval-package-{uuid4()}",
            provider_name="mock_qwen35_122b_a10b_nvfp4",
            provider_mode=GENERATION_PROVIDER_MODE_MOCK,
            model_id="mock-generation-model",
            retrieval_confidence_status=RETRIEVAL_CONFIDENCE_ANSWERABLE,
            citation_readiness_status=CITATION_READINESS_READY,
            query_text=query,
            status=GENERATION_STATUS_SUCCEEDED,
            guardrail_status=GENERATION_GUARDRAIL_ALLOWED,
            answer_text="근거 기반 답변입니다.",
            request_metadata={"source": "chat_repository_test"},
            response_metadata={"quality": "fixture"},
            created_by="chat-repository-test",
        ),
    )
    return search_log.search_log_id, generation_run.generation_run_id


def test_chat_repository_persists_thread_and_artifact_links(
    migrated_database_url: str,
) -> None:
    search_log_id, generation_run_id = _create_generation_fixture(migrated_database_url)
    session = create_chat_session(
        migrated_database_url,
        ChatSessionInput(
            session_title="대화형 생성 테스트",
            default_search_profile_name="reranked_hybrid",
            default_search_scope="company",
            metadata={"slice": 387},
        ),
    )
    try:
        user_message = append_chat_message(
            migrated_database_url,
            ChatMessageInput(
                chat_session_id=session.chat_session_id,
                role=CHAT_ROLE_USER,
                content="회사 규정을 바탕으로 보고서를 작성해줘",
                intent=CHAT_INTENT_DOCUMENT_GENERATION,
                intent_confidence=0.91,
                routing_metadata={"detected_template": "report"},
            ),
        )
        assistant_message = append_chat_message(
            migrated_database_url,
            ChatMessageInput(
                chat_session_id=session.chat_session_id,
                parent_message_id=user_message.chat_message_id,
                role=CHAT_ROLE_ASSISTANT,
                content="보고서 초안을 생성했습니다.",
                intent=CHAT_INTENT_GROUNDED_ANSWER,
                status=CHAT_MESSAGE_STATUS_COMPLETED,
                runtime_metadata={"provider_mode": "mock"},
            ),
        )
        search_link = link_chat_message_to_search_log(
            migrated_database_url,
            chat_message_id=assistant_message.chat_message_id,
            search_log_id=search_log_id,
            metadata={"package_key": "retrieval-context"},
        )
        generation_link = link_chat_message_to_generation_run(
            migrated_database_url,
            chat_message_id=assistant_message.chat_message_id,
            generation_run_id=generation_run_id,
        )
        summary_link = link_chat_message_to_document_summary(
            migrated_database_url,
            chat_message_id=assistant_message.chat_message_id,
            generation_run_id=generation_run_id,
            label="요약 실행",
        )
        download_link = link_chat_message_to_download_url(
            migrated_database_url,
            chat_message_id=assistant_message.chat_message_id,
            target_url=f"/api/generation/runs/{generation_run_id}/export/docx",
            metadata={"format": "docx"},
        )

        stored_session = get_chat_session(migrated_database_url, session.chat_session_id)
        stored_messages = list_chat_messages(migrated_database_url, session.chat_session_id)
        stored_assistant = get_chat_message(
            migrated_database_url,
            assistant_message.chat_message_id,
        )
        stored_links = list_chat_message_links(
            migrated_database_url,
            assistant_message.chat_message_id,
        )
        thread = get_chat_thread(migrated_database_url, session.chat_session_id)
        recent_sessions = list_chat_sessions(migrated_database_url, status=None, limit=5)

        assert stored_session is not None
        assert stored_session.default_search_scope == "company"
        assert stored_session.default_search_profile_name == "reranked_hybrid"
        assert stored_session.metadata == {"slice": 387}
        assert stored_messages == (user_message, assistant_message)
        assert stored_messages[0].sequence_no == 1
        assert stored_messages[1].sequence_no == 2
        assert stored_assistant == assistant_message
        assert {link.link_type for link in stored_links} == {
            CHAT_LINK_TYPE_SEARCH_LOG,
            CHAT_LINK_TYPE_GENERATION_RUN,
            CHAT_LINK_TYPE_DOCUMENT_SUMMARY,
            CHAT_LINK_TYPE_DOWNLOAD,
        }
        assert search_link.target_id == search_log_id
        assert generation_link.target_id == generation_run_id
        assert summary_link.label == "요약 실행"
        assert download_link.target_url == f"/api/generation/runs/{generation_run_id}/export/docx"
        assert download_link.metadata == {"format": "docx"}
        assert thread is not None
        assert thread.session == stored_session
        assert [item.message.role for item in thread.messages] == [
            CHAT_ROLE_USER,
            CHAT_ROLE_ASSISTANT,
        ]
        assert thread.messages[0].links == ()
        assert tuple(link.link_type for link in thread.messages[1].links) == tuple(
            link.link_type for link in stored_links
        )
        assert any(item.chat_session_id == session.chat_session_id for item in recent_sessions)
    finally:
        _delete_chat_session(migrated_database_url, session.chat_session_id)
        _delete_search_log(migrated_database_url, search_log_id)


def test_chat_repository_updates_session_runtime_defaults(
    migrated_database_url: str,
) -> None:
    actor_user_id = _seed_actor_user_id(migrated_database_url)
    session = create_chat_session(
        migrated_database_url,
        ChatSessionInput(
            session_title="수정 전 대화",
            default_search_scope="company",
            metadata={"source": "repository_test"},
        ),
    )
    try:
        updated = update_chat_session(
            migrated_database_url,
            session.chat_session_id,
            ChatSessionInput(
                session_title="수정 후 대화",
                actor_user_id=actor_user_id,
                default_provider_mode="remote_openai_compatible",
                default_search_profile_name="reranked_vector_cosine",
                default_search_scope="team",
                metadata={"source": "repository_test", "default_generation_template_key": "report"},
            ),
        )
        stored = get_chat_session(migrated_database_url, session.chat_session_id)

        assert updated is not None
        assert updated.chat_session_id == session.chat_session_id
        assert updated.session_title == "수정 후 대화"
        assert updated.actor_user_id == actor_user_id
        assert updated.default_provider_mode == "remote_openai_compatible"
        assert updated.default_search_profile_name == "reranked_vector_cosine"
        assert updated.default_search_scope == "team"
        assert updated.metadata == {
            "source": "repository_test",
            "default_generation_template_key": "report",
        }
        assert stored == updated
    finally:
        _delete_chat_session(migrated_database_url, session.chat_session_id)


def test_chat_repository_returns_none_and_empty_for_missing_records(
    migrated_database_url: str,
) -> None:
    assert get_chat_session(migrated_database_url, 999999999) is None
    assert (
        update_chat_session(
            migrated_database_url,
            999999999,
            ChatSessionInput(session_title="missing"),
        )
        is None
    )
    assert get_chat_message(migrated_database_url, 999999999) is None
    assert get_chat_thread(migrated_database_url, 999999999) is None
    assert list_chat_messages(migrated_database_url, 999999999) == ()
    assert list_chat_message_links(migrated_database_url, 999999999) == ()


def test_append_chat_message_rejects_parent_from_another_session(
    migrated_database_url: str,
) -> None:
    first_session = create_chat_session(
        migrated_database_url,
        ChatSessionInput(session_title="첫 번째 대화"),
    )
    second_session = create_chat_session(
        migrated_database_url,
        ChatSessionInput(session_title="두 번째 대화"),
    )
    try:
        parent_message = append_chat_message(
            migrated_database_url,
            ChatMessageInput(
                chat_session_id=first_session.chat_session_id,
                role=CHAT_ROLE_USER,
                content="첫 번째 세션 메시지",
            ),
        )

        with pytest.raises(InvalidChatRepositoryError):
            append_chat_message(
                migrated_database_url,
                ChatMessageInput(
                    chat_session_id=second_session.chat_session_id,
                    parent_message_id=parent_message.chat_message_id,
                    role=CHAT_ROLE_ASSISTANT,
                    content="다른 세션에 연결될 수 없습니다.",
                ),
            )
    finally:
        _delete_chat_session(migrated_database_url, first_session.chat_session_id)
        _delete_chat_session(migrated_database_url, second_session.chat_session_id)


def test_append_chat_message_rejects_missing_parent(
    migrated_database_url: str,
) -> None:
    session = create_chat_session(
        migrated_database_url,
        ChatSessionInput(session_title="부모 메시지 누락 대화"),
    )
    try:
        with pytest.raises(InvalidChatRepositoryError):
            append_chat_message(
                migrated_database_url,
                ChatMessageInput(
                    chat_session_id=session.chat_session_id,
                    parent_message_id=999999999,
                    role=CHAT_ROLE_ASSISTANT,
                    content="부모 메시지를 찾을 수 없습니다.",
                ),
            )
    finally:
        _delete_chat_session(migrated_database_url, session.chat_session_id)
