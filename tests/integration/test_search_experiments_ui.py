from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.database import connect
from app.core.search_experiments import (
    SearchExperimentProfileRunInput,
    SearchExperimentRunInput,
    create_search_experiment_run,
    update_search_experiment_run_status,
    upsert_search_experiment_profile_run,
)
from app.main import create_app

pytestmark = pytest.mark.integration


def _pcx_admin_user_id(database_url: str) -> int:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT user_id FROM app_users WHERE login_id = 'pcx.admin'")
            return int(cursor.fetchone()["user_id"])


def _delete_experiment_run(database_url: str, experiment_run_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM search_experiment_runs WHERE experiment_run_id = %s",
                (experiment_run_id,),
            )


def test_search_experiments_page_renders_recent_runs_and_detail(
    migrated_database_url: str,
) -> None:
    actor_user_id = _pcx_admin_user_id(migrated_database_url)
    run = create_search_experiment_run(
        migrated_database_url,
        SearchExperimentRunInput(
            run_name=f"Slice 169 UI {uuid4()}",
            query_text="search experiment detail UI",
            profile_names=("bge_m3_1024", "kure_v1_1024"),
            actor_user_id=actor_user_id,
            requested_search_scope="company",
            effective_search_scope="company",
            document_group="slice-169",
            file_type=".md",
            chunk_policy_name="heading_512_64",
            strategy_name="vector_cosine_threshold",
            similarity_metric="cosine",
            top_k=3,
            score_threshold=0.25,
            status="running",
            runtime_metadata={"slice": 169, "source": "ui-test"},
            created_by="integration-test",
        ),
    )
    try:
        for profile_name, result_count, top_score in (
            ("bge_m3_1024", 2, 0.98),
            ("kure_v1_1024", 1, 0.91),
        ):
            upsert_search_experiment_profile_run(
                migrated_database_url,
                SearchExperimentProfileRunInput(
                    experiment_run_id=run.experiment_run_id,
                    profile_name=profile_name,
                    status="succeeded",
                    result_count=result_count,
                    top_score=top_score,
                    average_score=top_score - 0.02,
                    elapsed_ms=12,
                    runtime_metadata={"source": "ui-test"},
                ),
            )
        completed = update_search_experiment_run_status(
            migrated_database_url,
            run.experiment_run_id,
            status="succeeded",
            total_elapsed_ms=24,
        )
        assert completed is not None

        app = create_app(Settings(database_url=migrated_database_url))
        with TestClient(app) as client:
            response = client.get(
                f"/search/experiments?experiment_run_id={run.experiment_run_id}"
            )

        assert response.status_code == 200
        assert "검색 실험 Run" in response.text
        assert "Slice 169 UI" in response.text
        assert "vector_cosine_threshold" in response.text
        assert "bge_m3_1024" in response.text
        assert "kure_v1_1024" in response.text
        assert "Runtime metadata" in response.text
        assert f"/api/search/experiments/{run.experiment_run_id}" in response.text
    finally:
        _delete_experiment_run(migrated_database_url, run.experiment_run_id)
