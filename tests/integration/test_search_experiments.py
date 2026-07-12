from uuid import uuid4

import pytest

from app.core.database import connect
from app.core.search_experiments import (
    SearchExperimentProfileRunInput,
    SearchExperimentRunInput,
    create_search_experiment_run,
    get_search_experiment_run_detail,
    list_search_experiment_runs,
    update_search_experiment_run_status,
    upsert_search_experiment_profile_run,
)

pytestmark = pytest.mark.integration


def _cleanup_experiment_run(database_url: str, experiment_run_id: int) -> None:
    with connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM search_experiment_runs WHERE experiment_run_id = %s",
                (experiment_run_id,),
            )


def test_search_experiment_run_repository_lifecycle(migrated_database_url: str) -> None:
    suffix = str(uuid4())
    run = create_search_experiment_run(
        migrated_database_url,
        SearchExperimentRunInput(
            run_name=f"Slice 165 Search Experiment {suffix}",
            query_text="Which model retrieves the inverter guide?",
            normalized_query_text="which model retrieves the inverter guide",
            profile_names=("bge_m3_1024", "kure_v1_1024", "bge_m3_1024"),
            requested_search_scope="company",
            effective_search_scope="company",
            document_group="slice-165",
            file_type=".md",
            chunk_policy_name="heading_512_64",
            strategy_name="vector_cosine_threshold",
            similarity_metric="cosine",
            top_k=7,
            score_threshold=0.2,
            status="running",
            runtime_metadata={"slice": 165},
            created_by="integration-test",
        ),
    )
    empty_run = create_search_experiment_run(
        migrated_database_url,
        SearchExperimentRunInput(
            run_name=f"Slice 165 Empty Search Experiment {suffix}",
            query_text="Does fallback status update work?",
            profile_names=("bge_m3_1024",),
            strategy_name="vector_cosine_threshold",
        ),
    )

    try:
        succeeded_profile = upsert_search_experiment_profile_run(
            migrated_database_url,
            SearchExperimentProfileRunInput(
                experiment_run_id=run.experiment_run_id,
                profile_name="bge_m3_1024",
                status="succeeded",
                result_count=3,
                top_score=0.92,
                average_score=0.81,
                elapsed_ms=35,
                runtime_metadata={"strategy": "vector_cosine_threshold"},
            ),
        )
        failed_profile = upsert_search_experiment_profile_run(
            migrated_database_url,
            SearchExperimentProfileRunInput(
                experiment_run_id=run.experiment_run_id,
                profile_name="kure_v1_1024",
                status="failed",
                result_count=0,
                elapsed_ms=20,
                error_message="provider unavailable",
            ),
        )
        completed = update_search_experiment_run_status(
            migrated_database_url,
            run.experiment_run_id,
            status="succeeded",
            total_elapsed_ms=60,
            runtime_metadata={"completed_by": "test"},
        )
        detail = get_search_experiment_run_detail(
            migrated_database_url,
            run.experiment_run_id,
        )
        listed = list_search_experiment_runs(
            migrated_database_url,
            status="succeeded",
            strategy_name="vector_cosine_threshold",
            limit=20,
        )
        unfiltered = list_search_experiment_runs(migrated_database_url, limit=20)
        canceled_empty_run = update_search_experiment_run_status(
            migrated_database_url,
            empty_run.experiment_run_id,
            status="canceled",
            error_message="operator canceled",
        )
        missing_detail = get_search_experiment_run_detail(migrated_database_url, 999999999)

        assert run.status == "running"
        assert run.started_at is not None
        assert run.profile_names == ("bge_m3_1024", "kure_v1_1024")
        assert run.total_profile_count == 2
        assert succeeded_profile.status == "succeeded"
        assert succeeded_profile.result_count == 3
        assert succeeded_profile.finished_at is not None
        assert failed_profile.status == "failed"
        assert failed_profile.error_message == "provider unavailable"
        assert completed is not None
        assert completed.status == "succeeded"
        assert completed.completed_profile_count == 2
        assert completed.failure_count == 1
        assert completed.result_count == 3
        assert completed.total_elapsed_ms == 60
        assert completed.finished_at is not None
        assert detail is not None
        assert detail.run.experiment_run_id == run.experiment_run_id
        assert [profile.profile_name for profile in detail.profiles] == [
            "bge_m3_1024",
            "kure_v1_1024",
        ]
        assert any(item.experiment_run_id == run.experiment_run_id for item in listed)
        assert any(item.experiment_run_id == run.experiment_run_id for item in unfiltered)
        assert canceled_empty_run is not None
        assert canceled_empty_run.status == "canceled"
        assert canceled_empty_run.error_message == "operator canceled"
        assert canceled_empty_run.finished_at is not None
        assert missing_detail is None
    finally:
        _cleanup_experiment_run(migrated_database_url, run.experiment_run_id)
        _cleanup_experiment_run(migrated_database_url, empty_run.experiment_run_id)
