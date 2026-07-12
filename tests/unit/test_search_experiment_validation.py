import base64
import json
import math
from datetime import UTC, datetime

import pytest

from app.core.golden_batch_metric_snapshots import (
    InvalidGoldenBatchMetricSnapshotError,
    get_golden_batch_metric_snapshot_detail,
    list_golden_batch_metric_snapshots,
    record_golden_batch_metric_snapshot,
)
from app.core.search_experiments import (
    GoldenSearchExperimentBatchIdentity,
    InvalidSearchExperimentError,
    SearchExperimentProfileRunInput,
    SearchExperimentRunInput,
    SearchExperimentRunRecord,
    _golden_batch_identity_from_run,
    _golden_batch_prefix,
    _summary_status,
    _validate_limit,
    decode_golden_search_experiment_batch_key,
    encode_golden_search_experiment_batch_key,
    get_golden_search_experiment_batch_detail,
    list_golden_search_experiment_batch_summaries,
    validate_search_experiment_profile_run_input,
    validate_search_experiment_run_input,
)


def _encoded_payload(payload: object) -> str:
    raw_payload = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(raw_payload).decode("ascii").rstrip("=")


def _run_record(
    *,
    run_name: str = "batch / Q10",
    status: str = "succeeded",
    runtime_metadata: dict[str, object] | None = None,
) -> SearchExperimentRunRecord:
    now = datetime(2026, 7, 12, 10, 0, tzinfo=UTC)
    return SearchExperimentRunRecord(
        experiment_run_id=10,
        run_name=run_name,
        query_text="golden query",
        normalized_query_text=None,
        actor_user_id=1,
        requested_search_scope="company",
        effective_search_scope="company",
        document_group=None,
        file_type=None,
        chunk_policy_name="heading_512_64",
        strategy_name="vector_cosine",
        similarity_metric="cosine",
        top_k=5,
        score_threshold=None,
        profile_names=("bge_m3_1024",),
        status=status,
        total_profile_count=1,
        completed_profile_count=1 if status == "succeeded" else 0,
        result_count=2,
        failure_count=0,
        total_elapsed_ms=20,
        runtime_metadata=(
            runtime_metadata
            if runtime_metadata is not None
            else {
                "golden_question_batch": True,
                "question_set_id": 3,
                "question_set_name": "Golden Set",
                "question_id": 10,
            }
        ),
        error_message=None,
        created_by="unit-test",
        created_by_user_id=None,
        started_at=now,
        finished_at=now if status == "succeeded" else None,
        created_at=now,
        updated_at=now,
    )


def test_search_experiment_run_validation_deduplicates_profiles() -> None:
    validated = validate_search_experiment_run_input(
        SearchExperimentRunInput(
            run_name="  Strategy Trial  ",
            query_text="  inverter manual  ",
            profile_names=("bge_m3_1024", "bge_m3_1024", "kure_v1_1024"),
            requested_search_scope="company",
            strategy_name="vector_cosine_threshold",
            score_threshold=0.25,
            runtime_metadata={"slice": 165},
        )
    )

    assert validated.run_name == "Strategy Trial"
    assert validated.query_text == "inverter manual"
    assert validated.profile_names == ("bge_m3_1024", "kure_v1_1024")
    assert validated.requested_search_scope == "company"
    assert validated.score_threshold == 0.25


@pytest.mark.parametrize(
    ("run_input", "message"),
    [
        (
            SearchExperimentRunInput(
                run_name="trial",
                query_text=" ",
                profile_names=("bge_m3_1024",),
            ),
            "query_text must not be blank",
        ),
        (
            SearchExperimentRunInput(
                run_name="trial",
                query_text="query",
                profile_names=(),
            ),
            "profile_names must not be empty",
        ),
        (
            SearchExperimentRunInput(
                run_name="trial",
                query_text="query",
                profile_names=("bge_m3_1024",),
                requested_search_scope="everyone",
            ),
            "Unsupported requested_search_scope",
        ),
        (
            SearchExperimentRunInput(
                run_name="trial",
                query_text="query",
                profile_names=("bge_m3_1024",),
                similarity_metric="jaccard",
            ),
            "Unsupported similarity_metric",
        ),
        (
            SearchExperimentRunInput(
                run_name="trial",
                query_text="query",
                profile_names=("bge_m3_1024",),
                status="done",
            ),
            "Unsupported experiment status",
        ),
        (
            SearchExperimentRunInput(
                run_name="trial",
                query_text="query",
                profile_names=("bge_m3_1024",),
                top_k=0,
            ),
            "top_k must be greater than 0",
        ),
        (
            SearchExperimentRunInput(
                run_name="trial",
                query_text="query",
                profile_names=("bge_m3_1024",),
                actor_user_id=0,
            ),
            "actor_user_id must be greater than 0",
        ),
        (
            SearchExperimentRunInput(
                run_name="trial",
                query_text="query",
                profile_names=("bge_m3_1024",),
                runtime_metadata=[],
            ),
            "runtime_metadata must be a JSON object",
        ),
        (
            SearchExperimentRunInput(
                run_name="trial",
                query_text="query",
                profile_names=("bge_m3_1024",),
                strategy_name="vector_cosine_threshold",
                score_threshold=math.inf,
            ),
            "score_threshold must be finite",
        ),
    ],
)
def test_search_experiment_run_validation_rejects_invalid_inputs(
    run_input: SearchExperimentRunInput,
    message: str,
) -> None:
    with pytest.raises(InvalidSearchExperimentError, match=message):
        validate_search_experiment_run_input(run_input)


def test_search_experiment_profile_validation_rejects_invalid_status() -> None:
    with pytest.raises(InvalidSearchExperimentError, match="Unsupported profile status"):
        validate_search_experiment_profile_run_input(
            SearchExperimentProfileRunInput(
                experiment_run_id=1,
                profile_name="bge_m3_1024",
                status="done",
            )
        )


def test_search_experiment_profile_validation_rejects_negative_metrics() -> None:
    with pytest.raises(InvalidSearchExperimentError, match="result_count"):
        validate_search_experiment_profile_run_input(
            SearchExperimentProfileRunInput(
                experiment_run_id=1,
                profile_name="bge_m3_1024",
                result_count=-1,
            )
        )
    with pytest.raises(InvalidSearchExperimentError, match="elapsed_ms"):
        validate_search_experiment_profile_run_input(
            SearchExperimentProfileRunInput(
                experiment_run_id=1,
                profile_name="bge_m3_1024",
                elapsed_ms=-1,
            )
        )


def test_search_experiment_limit_validation_rejects_out_of_range_values() -> None:
    assert _validate_limit(10) == 10
    with pytest.raises(InvalidSearchExperimentError, match="greater than 0"):
        _validate_limit(0)
    with pytest.raises(InvalidSearchExperimentError, match="less than or equal to 5"):
        _validate_limit(6, max_limit=5)


def test_golden_search_experiment_batch_key_roundtrip() -> None:
    identity = GoldenSearchExperimentBatchIdentity(
        question_set_id=10,
        batch_prefix="Golden Set / search experiment / 20260712-101010",
        strategy_name="vector_cosine_threshold",
        top_k=5,
        score_threshold=0.25,
        chunk_policy_name="heading_512_64",
        profile_names=("bge_m3_1024", "kure_v1_1024"),
    )

    batch_key = encode_golden_search_experiment_batch_key(identity)
    decoded = decode_golden_search_experiment_batch_key(batch_key)

    assert decoded == identity


def test_golden_search_experiment_batch_key_rejects_invalid_values() -> None:
    with pytest.raises(InvalidSearchExperimentError, match="Invalid golden"):
        decode_golden_search_experiment_batch_key("not-a-valid-key")
    with pytest.raises(InvalidSearchExperimentError, match="batch_key must not be blank"):
        decode_golden_search_experiment_batch_key(" ")
    with pytest.raises(InvalidSearchExperimentError, match="Invalid golden"):
        decode_golden_search_experiment_batch_key(_encoded_payload(["not", "a", "dict"]))
    with pytest.raises(InvalidSearchExperimentError, match="Invalid golden"):
        decode_golden_search_experiment_batch_key(
            _encoded_payload({"question_set_id": 1, "top_k": 5})
        )
    with pytest.raises(InvalidSearchExperimentError, match="Invalid golden"):
        decode_golden_search_experiment_batch_key(
            _encoded_payload(
                {
                    "question_set_id": 1,
                    "top_k": 0,
                    "profile_names": ["bge_m3_1024"],
                }
            )
        )


def test_golden_search_experiment_batch_helpers_validate_before_connecting() -> None:
    with pytest.raises(InvalidSearchExperimentError, match="greater than 0"):
        list_golden_search_experiment_batch_summaries("postgresql://unused", limit=0)
    with pytest.raises(InvalidSearchExperimentError, match="Invalid golden"):
        get_golden_search_experiment_batch_detail("postgresql://unused", "not-a-valid-key")


def test_golden_batch_metric_snapshot_helpers_validate_before_connecting() -> None:
    batch_key = encode_golden_search_experiment_batch_key(
        GoldenSearchExperimentBatchIdentity(
            question_set_id=1,
            batch_prefix="snapshot batch",
            strategy_name="vector_cosine_threshold",
            top_k=5,
            score_threshold=0.1,
            chunk_policy_name="heading_512_64",
            profile_names=("bge_m3_1024",),
        )
    )

    with pytest.raises(InvalidGoldenBatchMetricSnapshotError, match="greater than 0"):
        list_golden_batch_metric_snapshots("postgresql://unused", batch_key, limit=0)
    with pytest.raises(InvalidGoldenBatchMetricSnapshotError, match="less than or equal"):
        list_golden_batch_metric_snapshots("postgresql://unused", batch_key, limit=101)
    with pytest.raises(InvalidSearchExperimentError, match="Invalid golden"):
        list_golden_batch_metric_snapshots("postgresql://unused", "not-a-valid-key")
    with pytest.raises(InvalidGoldenBatchMetricSnapshotError, match="snapshot_id"):
        get_golden_batch_metric_snapshot_detail("postgresql://unused", 0)
    with pytest.raises(InvalidGoldenBatchMetricSnapshotError, match="created_by_user_id"):
        record_golden_batch_metric_snapshot(
            "postgresql://unused",
            batch_key,
            created_by_user_id=0,
        )
    with pytest.raises(InvalidGoldenBatchMetricSnapshotError, match="created_by"):
        record_golden_batch_metric_snapshot("postgresql://unused", batch_key, created_by=" ")


def test_golden_search_experiment_batch_prefix_and_identity_helpers() -> None:
    assert _golden_batch_prefix(_run_record(run_name="prefix / Q10")) == "prefix"
    assert (
        _golden_batch_prefix(
            _run_record(run_name="prefix / Q99", runtime_metadata={"question_id": 10})
        )
        == "prefix"
    )
    assert (
        _golden_batch_prefix(
            _run_record(run_name="plain run", runtime_metadata={"question_id": False})
        )
        == "plain run"
    )
    assert _golden_batch_identity_from_run(_run_record(runtime_metadata={})) is None


def test_golden_search_experiment_batch_summary_status_priority() -> None:
    assert _summary_status([_run_record(status="succeeded")]) == "succeeded"
    assert _summary_status([_run_record(status="succeeded"), _run_record(status="failed")]) == (
        "failed"
    )
    assert _summary_status([_run_record(status="pending")]) == "running"
    assert _summary_status([_run_record(status="canceled")]) == "canceled"
    assert _summary_status([_run_record(status="skipped")]) == "failed"
