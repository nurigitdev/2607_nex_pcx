from math import nan

import pytest

from app.core.chat import (
    CHAT_INTENT_DOCUMENT_GENERATION,
    CHAT_INTENT_DOCUMENT_SEARCH_SUMMARY,
    CHAT_INTENT_DOCUMENT_SUMMARY,
    CHAT_INTENT_GENERAL_ANSWER,
    CHAT_INTENT_GROUNDED_ANSWER,
    CHAT_LINK_TYPE_DOWNLOAD,
    CHAT_MESSAGE_STATUS_COMPLETED,
    CHAT_ROLE_ASSISTANT,
    CHAT_ROLE_USER,
    ChatMessageInput,
    ChatMessageLinkInput,
    ChatSessionInput,
    InvalidChatRepositoryError,
    route_chat_intent_mock,
    validate_chat_message_input,
    validate_chat_message_link_input,
    validate_chat_session_input,
    validate_chat_session_limit,
)
from app.core.generation_runs import GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE


def test_validate_chat_session_input_normalizes_defaults_and_metadata() -> None:
    validated = validate_chat_session_input(
        ChatSessionInput(
            session_title="  보고서 작성 대화  ",
            default_provider_mode=" REMOTE_OPENAI_COMPATIBLE ",
            default_search_profile_name="  reranked_hybrid  ",
            default_search_scope=" Company ",
            metadata={"source": "unit"},
        )
    )

    assert validated.session_title == "보고서 작성 대화"
    assert validated.default_provider_mode == GENERATION_PROVIDER_MODE_REMOTE_OPENAI_COMPATIBLE
    assert validated.default_search_profile_name == "reranked_hybrid"
    assert validated.default_search_scope == "company"
    assert validated.metadata == {"source": "unit"}


@pytest.mark.parametrize(
    "session_input",
    [
        ChatSessionInput(session_title=" "),
        ChatSessionInput(session_title="valid", actor_user_id=0),
        ChatSessionInput(session_title="valid", default_language=" "),
        ChatSessionInput(session_title="valid", default_provider_mode="local"),
        ChatSessionInput(session_title="valid", default_search_scope="division"),
        ChatSessionInput(session_title="valid", metadata=[]),  # type: ignore[arg-type]
    ],
)
def test_validate_chat_session_input_rejects_invalid_values(
    session_input: ChatSessionInput,
) -> None:
    with pytest.raises(InvalidChatRepositoryError):
        validate_chat_session_input(session_input)


def test_validate_chat_message_input_normalizes_router_metadata() -> None:
    validated = validate_chat_message_input(
        ChatMessageInput(
            chat_session_id=10,
            parent_message_id=3,
            sequence_no=2,
            role=" ASSISTANT ",
            content="  생성 결과입니다.  ",
            intent=" DOCUMENT_GENERATION ",
            status=" COMPLETED ",
            intent_confidence=0.875,
            routing_metadata={"router": "mock"},
            runtime_metadata={"provider_mode": "mock"},
            error_metadata={},
        )
    )

    assert validated.role == CHAT_ROLE_ASSISTANT
    assert validated.content == "생성 결과입니다."
    assert validated.intent == CHAT_INTENT_DOCUMENT_GENERATION
    assert validated.status == CHAT_MESSAGE_STATUS_COMPLETED
    assert validated.intent_confidence == pytest.approx(0.875)
    assert validated.routing_metadata == {"router": "mock"}


@pytest.mark.parametrize(
    "message_input",
    [
        ChatMessageInput(chat_session_id=0, role=CHAT_ROLE_USER, content="질문"),
        ChatMessageInput(chat_session_id=1, role="unknown", content="질문"),
        ChatMessageInput(chat_session_id=1, role=CHAT_ROLE_USER, content=" "),
        ChatMessageInput(chat_session_id=1, role=CHAT_ROLE_USER, content="질문", status="done"),
        ChatMessageInput(chat_session_id=1, role=CHAT_ROLE_USER, content="질문", intent="legacy"),
        ChatMessageInput(
            chat_session_id=1,
            role=CHAT_ROLE_USER,
            content="질문",
            intent_confidence=-0.01,
        ),
        ChatMessageInput(
            chat_session_id=1,
            role=CHAT_ROLE_USER,
            content="질문",
            intent_confidence=1.01,
        ),
        ChatMessageInput(
            chat_session_id=1,
            role=CHAT_ROLE_USER,
            content="질문",
            intent_confidence=nan,
        ),
        ChatMessageInput(
            chat_session_id=1,
            role=CHAT_ROLE_USER,
            content="질문",
            routing_metadata=[],  # type: ignore[arg-type]
        ),
    ],
)
def test_validate_chat_message_input_rejects_invalid_values(
    message_input: ChatMessageInput,
) -> None:
    with pytest.raises(InvalidChatRepositoryError):
        validate_chat_message_input(message_input)


def test_validate_chat_message_link_input_normalizes_url_link() -> None:
    validated = validate_chat_message_link_input(
        ChatMessageLinkInput(
            chat_message_id=5,
            link_type=" DOWNLOAD ",
            target_url="  /api/generation/runs/10/export/docx  ",
            label="  DOCX 다운로드  ",
            metadata={"format": "docx"},
        )
    )

    assert validated.link_type == CHAT_LINK_TYPE_DOWNLOAD
    assert validated.target_url == "/api/generation/runs/10/export/docx"
    assert validated.label == "DOCX 다운로드"
    assert validated.metadata == {"format": "docx"}


@pytest.mark.parametrize(
    "link_input",
    [
        ChatMessageLinkInput(chat_message_id=0, link_type=CHAT_LINK_TYPE_DOWNLOAD, target_url="/x"),
        ChatMessageLinkInput(chat_message_id=1, link_type="legacy", target_url="/x"),
        ChatMessageLinkInput(chat_message_id=1, link_type=CHAT_LINK_TYPE_DOWNLOAD),
        ChatMessageLinkInput(chat_message_id=1, link_type=CHAT_LINK_TYPE_DOWNLOAD, target_id=-1),
        ChatMessageLinkInput(
            chat_message_id=1,
            link_type=CHAT_LINK_TYPE_DOWNLOAD,
            target_url=" ",
        ),
        ChatMessageLinkInput(
            chat_message_id=1,
            link_type=CHAT_LINK_TYPE_DOWNLOAD,
            target_url="/x",
            metadata=[],  # type: ignore[arg-type]
        ),
    ],
)
def test_validate_chat_message_link_input_rejects_invalid_values(
    link_input: ChatMessageLinkInput,
) -> None:
    with pytest.raises(InvalidChatRepositoryError):
        validate_chat_message_link_input(link_input)


def test_validate_chat_session_limit_bounds() -> None:
    assert validate_chat_session_limit(1) == 1
    assert validate_chat_session_limit(200) == 200

    with pytest.raises(InvalidChatRepositoryError):
        validate_chat_session_limit(0)
    with pytest.raises(InvalidChatRepositoryError):
        validate_chat_session_limit(201)


@pytest.mark.parametrize(
    ("content", "expected_intent", "expected_rationale"),
    [
        (
            "사내 규정을 바탕으로 보고서를 작성해줘",
            CHAT_INTENT_DOCUMENT_GENERATION,
            "document_generation_keyword_match",
        ),
        (
            "관련 문서를 검색해서 요약해줘",
            CHAT_INTENT_DOCUMENT_SEARCH_SUMMARY,
            "summary_and_search_keyword_match",
        ),
        (
            "이 문서를 핵심만 요약해줘",
            CHAT_INTENT_DOCUMENT_SUMMARY,
            "summary_keyword_match",
        ),
        (
            "사규 근거를 찾아서 답변해줘",
            CHAT_INTENT_GROUNDED_ANSWER,
            "grounded_or_search_keyword_match",
        ),
        (
            "오늘 회의 준비에 필요한 질문을 만들어줘",
            CHAT_INTENT_GENERAL_ANSWER,
            "fallback_general_answer",
        ),
    ],
)
def test_route_chat_intent_mock_detects_core_intents(
    content: str,
    expected_intent: str,
    expected_rationale: str,
) -> None:
    route = route_chat_intent_mock(content)

    assert route.intent == expected_intent
    assert route.rationale == expected_rationale
    assert 0 <= route.intent_confidence <= 1
    assert route.suggested_action


def test_route_chat_intent_mock_rejects_blank_content() -> None:
    with pytest.raises(InvalidChatRepositoryError):
        route_chat_intent_mock(" ")
