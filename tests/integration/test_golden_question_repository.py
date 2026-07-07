from uuid import uuid4

import pytest

from app.core.database import connect
from app.core.golden_questions import (
    GoldenQuestionExpectedTargetInput,
    GoldenQuestionInput,
    GoldenQuestionSetInput,
    create_expected_target,
    create_golden_question,
    create_golden_question_set,
    get_golden_question,
    get_golden_question_detail,
    get_golden_question_set,
    list_expected_targets,
    list_golden_question_sets,
    list_golden_questions,
)

pytestmark = pytest.mark.integration


def _create_chunk_fixture(database_url: str) -> tuple[int, int, int, str]:
    checksum = f"golden-repository-{uuid4()}"
    document_group = f"slice-034-{uuid4()}"
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT user_id FROM app_users WHERE login_id = 'alice.member'")
            user_id = cursor.fetchone()["user_id"]
            cursor.execute(
                """
                INSERT INTO files (
                    original_file_name,
                    stored_file_name,
                    file_ext,
                    file_size_bytes,
                    sha256_checksum,
                    storage_path,
                    uploaded_by_user_id,
                    document_group
                )
                VALUES (%s, %s, '.md', 1, %s, %s, %s, %s)
                RETURNING file_id
                """,
                (
                    f"{checksum}.md",
                    f"{checksum}.stored.md",
                    checksum,
                    f"/tmp/{checksum}.md",
                    user_id,
                    document_group,
                ),
            )
            file_id = cursor.fetchone()["file_id"]
            cursor.execute(
                """
                INSERT INTO documents (
                    file_id,
                    document_title,
                    document_group,
                    owner_user_id,
                    access_scope
                )
                VALUES (%s, %s, %s, %s, 'company')
                RETURNING document_id
                """,
                (file_id, f"Golden repository fixture {checksum}", document_group, user_id),
            )
            document_id = cursor.fetchone()["document_id"]
            cursor.execute(
                """
                INSERT INTO chunks (
                    document_id,
                    chunk_seq,
                    chunk_text,
                    content_hash,
                    chunk_policy_name,
                    heading_path,
                    char_count
                )
                VALUES (
                    %s,
                    0,
                    'Golden repository expected chunk',
                    %s,
                    'heading_512_64',
                    ARRAY['Policy', 'Scope'],
                    32
                )
                RETURNING chunk_id
                """,
                (document_id, f"chunk-{checksum}"),
            )
            chunk_id = cursor.fetchone()["chunk_id"]
    return file_id, chunk_id, user_id, document_group


def _cleanup_file(database_url: str, file_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def _cleanup_question_sets(database_url: str, *set_names: str) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM golden_question_sets WHERE set_name = ANY(%s)",
                (list(set_names),),
            )


def test_golden_question_repository_persists_question_set_questions_and_targets(
    migrated_database_url: str,
) -> None:
    file_id, chunk_id, user_id, document_group = _create_chunk_fixture(migrated_database_url)
    set_name = f"slice-034-active-{uuid4()}"
    inactive_set_name = f"slice-034-inactive-{uuid4()}"
    try:
        question_set = create_golden_question_set(
            migrated_database_url,
            GoldenQuestionSetInput(
                set_name=f" {set_name} ",
                description=" Slice 034 repository fixture ",
                metadata={"slice": "034"},
                created_by_user_id=user_id,
            ),
        )
        inactive_set = create_golden_question_set(
            migrated_database_url,
            GoldenQuestionSetInput(
                set_name=inactive_set_name,
                is_active=False,
                created_by_user_id=user_id,
            ),
        )
        question = create_golden_question(
            migrated_database_url,
            GoldenQuestionInput(
                question_set_id=question_set.question_set_id,
                question_text=" What does Slice 034 cover? ",
                question_type="single_fact",
                actor_user_id=user_id,
                requested_search_scope="company",
                document_group=document_group,
                file_type=".md",
                chunk_policy_name="heading_512_64",
                top_k=7,
                metadata={"difficulty": "easy"},
                created_by_user_id=user_id,
            ),
        )
        visible_target = create_expected_target(
            migrated_database_url,
            GoldenQuestionExpectedTargetInput(
                question_id=question.question_id,
                chunk_id=chunk_id,
                expectation_type="visible",
                relevance_grade=3,
                notes="primary answer",
            ),
        )
        hidden_target = create_expected_target(
            migrated_database_url,
            GoldenQuestionExpectedTargetInput(
                question_id=question.question_id,
                expected_heading_path=("Policy", "Scope"),
                expectation_type="hidden",
                relevance_grade=0,
                metadata={"permission_case": True},
            ),
        )

        stored_set = get_golden_question_set(
            migrated_database_url,
            question_set.question_set_id,
        )
        active_sets = list_golden_question_sets(migrated_database_url, active_only=True)
        all_sets = list_golden_question_sets(migrated_database_url, active_only=False)
        stored_question = get_golden_question(migrated_database_url, question.question_id)
        questions = list_golden_questions(
            migrated_database_url,
            question_set.question_set_id,
            actor_user_id=user_id,
            requested_search_scope="company",
        )
        targets = list_expected_targets(migrated_database_url, question.question_id)
        detail = get_golden_question_detail(migrated_database_url, question.question_id)

        assert stored_set == question_set
        assert question_set.set_name == set_name
        assert question_set.description == "Slice 034 repository fixture"
        assert question_set.metadata == {"slice": "034"}
        assert set_name in {item.set_name for item in active_sets}
        assert inactive_set_name not in {item.set_name for item in active_sets}
        assert inactive_set_name in {item.set_name for item in all_sets}
        assert inactive_set.is_active is False
        assert stored_question == question
        assert question.normalized_question_text == "what does slice 034 cover?"
        assert question.top_k == 7
        assert question.metadata == {"difficulty": "easy"}
        assert questions == [question]
        assert targets == [visible_target, hidden_target]
        assert visible_target.chunk_id == chunk_id
        assert hidden_target.expected_heading_path == ("Policy", "Scope")
        assert detail is not None
        assert detail.question == question
        assert detail.expected_targets == (visible_target, hidden_target)
    finally:
        _cleanup_question_sets(migrated_database_url, set_name, inactive_set_name)
        _cleanup_file(migrated_database_url, file_id)


def test_golden_question_repository_returns_none_and_empty_results(
    migrated_database_url: str,
) -> None:
    assert get_golden_question_set(migrated_database_url, 999999999) is None
    assert get_golden_question(migrated_database_url, 999999999) is None
    assert get_golden_question_detail(migrated_database_url, 999999999) is None
    assert list_golden_questions(migrated_database_url, 999999999) == []
    assert list_expected_targets(migrated_database_url, 999999999) == []
