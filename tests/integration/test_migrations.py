import pytest
from alembic.script import ScriptDirectory

from app.core.database import fetch_one
from app.core.migrations import downgrade, make_alembic_config, upgrade

pytestmark = pytest.mark.integration
HEAD_REVISION = "20260712_0023"


def test_alembic_upgrade_head_enables_pgvector(test_database_url: str) -> None:
    downgrade("base", test_database_url)
    upgrade("head", test_database_url)

    revision = fetch_one(test_database_url, "SELECT version_num FROM alembic_version")
    extension = fetch_one(
        test_database_url,
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'",
    )
    dashboard_threshold_settings = fetch_one(
        test_database_url,
        """
        SELECT count(*) AS count
        FROM app_log_settings
        WHERE setting_name LIKE 'dashboard_%%_threshold'
        """,
    )

    assert revision["version_num"] == HEAD_REVISION
    assert extension["extversion"]
    assert dashboard_threshold_settings["count"] == 9


def test_alembic_downgrade_base_clears_revision(test_database_url: str) -> None:
    upgrade("head", test_database_url)
    downgrade("base", test_database_url)
    try:
        revision_count = fetch_one(
            test_database_url,
            "SELECT count(*) AS revision_count FROM alembic_version",
        )
        extension = fetch_one(
            test_database_url,
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'",
        )

        assert revision_count["revision_count"] == 0
        assert extension["extversion"]
        assert (
            fetch_one(test_database_url, "SELECT to_regclass('public.files') AS table_name")[
                "table_name"
            ]
            is None
        )
    finally:
        upgrade("head", test_database_url)


def test_alembic_config_points_at_project_migrations(test_database_url: str) -> None:
    config = make_alembic_config(test_database_url)
    script = ScriptDirectory.from_config(config)

    assert script.get_current_head() == HEAD_REVISION
