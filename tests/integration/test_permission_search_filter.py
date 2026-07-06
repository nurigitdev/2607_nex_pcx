from uuid import uuid4

import pytest

from app.core.database import connect
from app.core.embedding_vectors import (
    EmbeddingVectorInput,
    generate_mock_embedding,
    store_chunk_embedding,
)
from app.core.permissions import (
    InvalidPermissionError,
    resolve_permission_search_filter,
)
from app.core.vector_search import VectorSearchInput, search_similar_chunks

pytestmark = pytest.mark.integration


def _seed_ids(database_url: str) -> dict[str, int]:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT login_id, user_id
                FROM app_users
                WHERE login_id IN (
                    'alice.member',
                    'bob.member',
                    'chloe.teamlead',
                    'dana.grouplead'
                )
                """)
            users = {row["login_id"]: int(row["user_id"]) for row in cursor.fetchall()}
            cursor.execute("""
                SELECT org_unit_name, org_unit_id
                FROM org_units
                WHERE org_unit_name IN (
                    'NeX Company',
                    'Platform Group',
                    'Platform Team',
                    'Business Team'
                )
                """)
            orgs = {row["org_unit_name"]: int(row["org_unit_id"]) for row in cursor.fetchall()}
    return {**users, **orgs}


def _create_permission_chunk(
    database_url: str,
    *,
    title: str,
    owner_user_id: int | None,
    owner_org_unit_id: int,
    access_scope: str,
    chunk_text: str,
) -> tuple[int, int]:
    checksum = f"permission-search-{uuid4()}"
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
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
                VALUES (%s, %s, '.md', 1, %s, %s, %s, 'slice-027')
                RETURNING file_id
                """,
                (
                    f"{checksum}.md",
                    f"{checksum}.stored.md",
                    checksum,
                    f"/tmp/{checksum}.md",
                    owner_user_id,
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
                    owner_org_unit_id,
                    access_scope
                )
                VALUES (%s, %s, 'slice-027', %s, %s, %s)
                RETURNING document_id
                """,
                (file_id, title, owner_user_id, owner_org_unit_id, access_scope),
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
                    char_count
                )
                VALUES (%s, 0, %s, %s, 'heading_512_64', %s)
                RETURNING chunk_id
                """,
                (document_id, chunk_text, f"chunk-{checksum}", len(chunk_text)),
            )
            chunk_id = cursor.fetchone()["chunk_id"]
    return file_id, chunk_id


def _cleanup_files(database_url: str, file_ids: list[int]) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            for file_id in file_ids:
                cursor.execute("DELETE FROM files WHERE file_id = %s", (file_id,))


def _store_anchor_embedding(database_url: str, chunk_id: int, chunk_text: str) -> None:
    store_chunk_embedding(
        database_url,
        EmbeddingVectorInput(
            chunk_id=chunk_id,
            profile_name="kure_v1_1024",
            embedding=generate_mock_embedding(
                chunk_text,
                profile_name="kure_v1_1024",
                dimension=1024,
            ),
            elapsed_ms=3,
        ),
    )


def _create_permission_corpus(
    database_url: str,
) -> tuple[list[int], dict[str, int], dict[str, int]]:
    ids = _seed_ids(database_url)
    anchor_text = "Permission search shared anchor"
    fixtures = [
        (
            "alice_personal",
            ids["alice.member"],
            ids["Platform Team"],
            "personal",
        ),
        (
            "bob_personal",
            ids["bob.member"],
            ids["Business Team"],
            "personal",
        ),
        (
            "platform_team",
            None,
            ids["Platform Team"],
            "team",
        ),
        (
            "business_team",
            None,
            ids["Business Team"],
            "team",
        ),
        (
            "platform_group_tree",
            None,
            ids["Platform Group"],
            "org_tree",
        ),
        (
            "company",
            None,
            ids["NeX Company"],
            "company",
        ),
    ]
    file_ids = []
    chunk_ids = {}
    for name, owner_user_id, owner_org_unit_id, access_scope in fixtures:
        file_id, chunk_id = _create_permission_chunk(
            database_url,
            title=name,
            owner_user_id=owner_user_id,
            owner_org_unit_id=owner_org_unit_id,
            access_scope=access_scope,
            chunk_text=anchor_text,
        )
        file_ids.append(file_id)
        chunk_ids[name] = chunk_id
        _store_anchor_embedding(database_url, chunk_id, anchor_text)
    return file_ids, chunk_ids, ids


def _search_visible_chunk_ids(
    database_url: str,
    *,
    actor_user_id: int,
    requested_search_scope: str,
) -> set[int]:
    permission_filter = resolve_permission_search_filter(
        database_url,
        actor_user_id=actor_user_id,
        requested_search_scope=requested_search_scope,
    )
    results = search_similar_chunks(
        database_url,
        VectorSearchInput(
            query_text="Permission search shared anchor",
            profile_name="kure_v1_1024",
            top_k=10,
            document_group="slice-027",
            permission_filter=permission_filter,
        ),
    )
    return {result.chunk_id for result in results}


def test_permission_resolver_downgrades_member_managed_scope(
    migrated_database_url: str,
) -> None:
    ids = _seed_ids(migrated_database_url)

    permission_filter = resolve_permission_search_filter(
        migrated_database_url,
        actor_user_id=ids["alice.member"],
        requested_search_scope="managed_org",
    )

    assert permission_filter.requested_search_scope == "managed_org"
    assert permission_filter.effective_search_scope == "team"
    assert permission_filter.metadata["role_name"] == "member"
    assert permission_filter.metadata["managed_org_unit_ids"] == []


def test_permission_prefilter_limits_member_company_scope(
    migrated_database_url: str,
) -> None:
    file_ids, chunk_ids, ids = _create_permission_corpus(migrated_database_url)
    try:
        visible = _search_visible_chunk_ids(
            migrated_database_url,
            actor_user_id=ids["alice.member"],
            requested_search_scope="company",
        )

        assert visible == {
            chunk_ids["alice_personal"],
            chunk_ids["platform_team"],
            chunk_ids["platform_group_tree"],
            chunk_ids["company"],
        }
    finally:
        _cleanup_files(migrated_database_url, file_ids)


def test_permission_prefilter_expands_group_lead_managed_scope(
    migrated_database_url: str,
) -> None:
    file_ids, chunk_ids, ids = _create_permission_corpus(migrated_database_url)
    try:
        visible = _search_visible_chunk_ids(
            migrated_database_url,
            actor_user_id=ids["dana.grouplead"],
            requested_search_scope="managed_org",
        )

        assert visible == set(chunk_ids.values())
    finally:
        _cleanup_files(migrated_database_url, file_ids)


def test_permission_resolver_rejects_invalid_actor_and_scope(
    migrated_database_url: str,
) -> None:
    ids = _seed_ids(migrated_database_url)

    with pytest.raises(InvalidPermissionError, match="requested_search_scope"):
        resolve_permission_search_filter(
            migrated_database_url,
            actor_user_id=ids["alice.member"],
            requested_search_scope="all",
        )

    with pytest.raises(InvalidPermissionError, match="actor_user_id"):
        resolve_permission_search_filter(
            migrated_database_url,
            actor_user_id=0,
            requested_search_scope="mine",
        )
