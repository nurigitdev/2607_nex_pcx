from uuid import uuid4

import pytest
from psycopg import errors

from app.core.database import connect, fetch_one
from app.core.embedding_provider_routes import (
    EmbeddingProviderRouteInput,
    list_embedding_provider_routes,
    select_embedding_provider_route,
    upsert_embedding_provider_route,
)

pytestmark = pytest.mark.integration


def _cleanup_routes(database_url: str, provider_names: list[str]) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM embedding_provider_routes WHERE provider_name = ANY(%s)",
                (provider_names,),
            )


def test_embedding_provider_routes_table_exists_and_enforces_remote_url(
    migrated_database_url: str,
) -> None:
    table_name = fetch_one(
        migrated_database_url,
        "SELECT to_regclass('public.embedding_provider_routes') AS table_name",
    )

    assert table_name["table_name"] == "embedding_provider_routes"

    with connect(migrated_database_url) as connection:
        with connection.cursor() as cursor:
            with pytest.raises(errors.CheckViolation):
                cursor.execute(
                    """
                    INSERT INTO embedding_provider_routes (
                        profile_name,
                        provider_name,
                        provider_mode,
                        provider_base_url
                    )
                    VALUES ('kure_v1_1024', %s, 'remote', NULL)
                    """,
                    (f"invalid-{uuid4()}",),
                )
            connection.rollback()


def test_embedding_provider_route_repository_upserts_and_selects_best_route(
    migrated_database_url: str,
) -> None:
    suffix = uuid4().hex
    slow_name = f"gpu-slow-{suffix}"
    fast_name = f"gpu-fast-{suffix}"
    provider_names = [slow_name, fast_name]
    try:
        slow = upsert_embedding_provider_route(
            migrated_database_url,
            EmbeddingProviderRouteInput(
                profile_name="kure_v1_1024",
                provider_name=slow_name,
                provider_base_url="http://slow-provider.local/",
                timeout_seconds=20.0,
                priority=20,
                runtime_metadata={"device": "cuda:1"},
            ),
        )
        fast = upsert_embedding_provider_route(
            migrated_database_url,
            EmbeddingProviderRouteInput(
                profile_name="kure_v1_1024",
                provider_name=fast_name,
                provider_base_url="http://fast-provider.local/",
                timeout_seconds=5.0,
                priority=5,
                runtime_metadata={"device": "cuda:0"},
            ),
        )

        routes = list_embedding_provider_routes(
            migrated_database_url,
            profile_name="kure_v1_1024",
            active_only=True,
        )
        selected = select_embedding_provider_route(migrated_database_url, "kure_v1_1024")

        assert slow.provider_base_url == "http://slow-provider.local"
        assert fast.timeout_seconds == 5.0
        assert fast.runtime_metadata == {"device": "cuda:0"}
        ordered_provider_names = [
            route.provider_name for route in routes if route.provider_name in provider_names
        ]
        assert ordered_provider_names == [
            fast_name,
            slow_name,
        ]
        assert selected is not None
        assert selected.provider_name == fast_name

        updated = upsert_embedding_provider_route(
            migrated_database_url,
            EmbeddingProviderRouteInput(
                profile_name="kure_v1_1024",
                provider_name=fast_name,
                provider_base_url="http://fast-provider-v2.local",
                timeout_seconds=3.0,
                priority=30,
                is_active=False,
            ),
        )

        assert updated.route_id == fast.route_id
        assert updated.provider_base_url == "http://fast-provider-v2.local"
        assert updated.is_active is False
    finally:
        _cleanup_routes(migrated_database_url, provider_names)
