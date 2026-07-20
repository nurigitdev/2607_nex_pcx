import pytest
from alembic.script import ScriptDirectory

from app.core.database import fetch_one
from app.core.migrations import downgrade, make_alembic_config, upgrade

pytestmark = pytest.mark.integration
HEAD_REVISION = "20260715_0029"


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
    search_experiment_table = fetch_one(
        test_database_url,
        "SELECT to_regclass('public.search_experiment_runs') AS table_name",
    )
    golden_batch_metric_snapshot_table = fetch_one(
        test_database_url,
        """
        SELECT to_regclass(
            'public.golden_search_experiment_batch_metric_snapshots'
        ) AS table_name
        """,
    )
    dgx_benchmark_run_table = fetch_one(
        test_database_url,
        """
        SELECT to_regclass(
            'public.dgx_ingestion_benchmark_runs'
        ) AS table_name
        """,
    )
    extraction_artifact_table = fetch_one(
        test_database_url,
        """
        SELECT to_regclass(
            'public.extraction_artifacts'
        ) AS table_name
        """,
    )
    extraction_quality_snapshot_table = fetch_one(
        test_database_url,
        """
        SELECT to_regclass(
            'public.extraction_quality_snapshots'
        ) AS table_name
        """,
    )
    local_extraction_profile_count = fetch_one(
        test_database_url,
        """
        SELECT count(*) AS count
        FROM extraction_profiles
        WHERE extraction_profile_name LIKE 'local_%%_default'
        """,
    )

    assert revision["version_num"] == HEAD_REVISION
    assert extension["extversion"]
    assert dashboard_threshold_settings["count"] == 9
    assert search_experiment_table["table_name"] == "search_experiment_runs"
    assert (
        golden_batch_metric_snapshot_table["table_name"]
        == "golden_search_experiment_batch_metric_snapshots"
    )
    assert dgx_benchmark_run_table["table_name"] == "dgx_ingestion_benchmark_runs"
    assert extraction_artifact_table["table_name"] == "extraction_artifacts"
    assert extraction_quality_snapshot_table["table_name"] == "extraction_quality_snapshots"
    assert local_extraction_profile_count["count"] == 7


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
