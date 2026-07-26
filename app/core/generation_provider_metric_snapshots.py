"""Generation provider metric snapshot queries."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.core.database import connect

DEFAULT_GENERATION_PROVIDER_METRIC_SNAPSHOT_LIMIT = 50
MAX_GENERATION_PROVIDER_METRIC_SNAPSHOT_LIMIT = 500


@dataclass(frozen=True)
class GenerationProviderMetricRunSnapshot:
    generation_run_id: int
    search_log_id: int
    provider_name: str
    provider_mode: str
    model_id: str
    status: str
    guardrail_status: str
    finish_reason: str | None
    input_token_count: int | None
    output_token_count: int | None
    total_token_count: int | None
    elapsed_ms: int | None
    provider_elapsed_ms: int | None
    retry_count: int
    succeeded: bool
    error_code: str | None
    error_message: str | None
    metric_present: bool
    created_at: datetime


@dataclass(frozen=True)
class GenerationProviderMetricSnapshotSummary:
    run_count: int
    metric_present_count: int
    succeeded_count: int
    failed_count: int
    no_answer_count: int
    total_token_count: int
    average_elapsed_ms: float | None
    average_provider_elapsed_ms: float | None


@dataclass(frozen=True)
class GenerationProviderMetricSnapshot:
    generated_at: datetime
    limit: int
    summary: GenerationProviderMetricSnapshotSummary
    runs: tuple[GenerationProviderMetricRunSnapshot, ...]


class InvalidGenerationProviderMetricSnapshotError(ValueError):
    """Raised when a generation provider metric snapshot query is invalid."""


def validate_generation_provider_metric_snapshot_limit(limit: int) -> int:
    if limit < 1 or limit > MAX_GENERATION_PROVIDER_METRIC_SNAPSHOT_LIMIT:
        raise InvalidGenerationProviderMetricSnapshotError(
            "limit must be between 1 and " f"{MAX_GENERATION_PROVIDER_METRIC_SNAPSHOT_LIMIT}"
        )
    return limit


def _optional_int(value: object, fallback: int | None = None) -> int | None:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _optional_text(value: object, fallback: str | None = None) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _metric_payload(row: dict[str, Any]) -> dict[str, Any]:
    response_metadata = row.get("response_metadata")
    if not isinstance(response_metadata, dict):
        return {}
    metrics = response_metadata.get("provider_metrics")
    return dict(metrics) if isinstance(metrics, dict) else {}


def _run_snapshot_from_row(row: dict[str, Any]) -> GenerationProviderMetricRunSnapshot:
    metrics = _metric_payload(row)
    metric_present = bool(metrics)
    status = str(row["status"])
    error_message = _optional_text(metrics.get("error_message"), row.get("error_message"))
    succeeded = bool(metrics.get("succeeded")) if metric_present else status == "succeeded"
    return GenerationProviderMetricRunSnapshot(
        generation_run_id=int(row["generation_run_id"]),
        search_log_id=int(row["search_log_id"]),
        provider_name=str(row["provider_name"]),
        provider_mode=str(row["provider_mode"]),
        model_id=str(row["model_id"]),
        status=status,
        guardrail_status=str(row["guardrail_status"]),
        finish_reason=_optional_text(metrics.get("finish_reason"), row.get("finish_reason")),
        input_token_count=_optional_int(
            metrics.get("input_token_count"),
            row.get("input_token_count"),
        ),
        output_token_count=_optional_int(
            metrics.get("output_token_count"),
            row.get("output_token_count"),
        ),
        total_token_count=_optional_int(
            metrics.get("total_token_count"),
            row.get("total_token_count"),
        ),
        elapsed_ms=_optional_int(metrics.get("elapsed_ms"), row.get("elapsed_ms")),
        provider_elapsed_ms=_optional_int(metrics.get("provider_elapsed_ms")),
        retry_count=_optional_int(metrics.get("retry_count"), 0) or 0,
        succeeded=succeeded,
        error_code=_optional_text(metrics.get("error_code")),
        error_message=error_message,
        metric_present=metric_present,
        created_at=row["created_at"],
    )


def _average(values: list[int]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def _summary(
    runs: tuple[GenerationProviderMetricRunSnapshot, ...],
) -> GenerationProviderMetricSnapshotSummary:
    elapsed_values = [run.elapsed_ms for run in runs if run.elapsed_ms is not None]
    provider_elapsed_values = [
        run.provider_elapsed_ms for run in runs if run.provider_elapsed_ms is not None
    ]
    return GenerationProviderMetricSnapshotSummary(
        run_count=len(runs),
        metric_present_count=sum(1 for run in runs if run.metric_present),
        succeeded_count=sum(1 for run in runs if run.succeeded),
        failed_count=sum(1 for run in runs if not run.succeeded),
        no_answer_count=sum(1 for run in runs if run.status == "no_answer"),
        total_token_count=sum(run.total_token_count or 0 for run in runs),
        average_elapsed_ms=_average(elapsed_values),
        average_provider_elapsed_ms=_average(provider_elapsed_values),
    )


def get_generation_provider_metric_snapshot(
    database_url: str,
    *,
    limit: int = DEFAULT_GENERATION_PROVIDER_METRIC_SNAPSHOT_LIMIT,
) -> GenerationProviderMetricSnapshot:
    validated_limit = validate_generation_provider_metric_snapshot_limit(limit)
    with connect(database_url) as conn:
        rows = conn.execute(
            """
            SELECT
                generation_run_id,
                search_log_id,
                provider_name,
                provider_mode,
                model_id,
                status,
                guardrail_status,
                finish_reason,
                input_token_count,
                output_token_count,
                total_token_count,
                elapsed_ms,
                response_metadata,
                error_message,
                created_at
            FROM generation_runs
            ORDER BY created_at DESC, generation_run_id DESC
            LIMIT %s
            """,
            (validated_limit,),
        ).fetchall()
    runs = tuple(_run_snapshot_from_row(dict(row)) for row in rows)
    return GenerationProviderMetricSnapshot(
        generated_at=datetime.now(UTC),
        limit=validated_limit,
        summary=_summary(runs),
        runs=runs,
    )


def generation_provider_metric_snapshot_payload(
    snapshot: GenerationProviderMetricSnapshot,
) -> dict[str, Any]:
    return {
        "generated_at": snapshot.generated_at.isoformat(),
        "limit": snapshot.limit,
        "summary": {
            "run_count": snapshot.summary.run_count,
            "metric_present_count": snapshot.summary.metric_present_count,
            "succeeded_count": snapshot.summary.succeeded_count,
            "failed_count": snapshot.summary.failed_count,
            "no_answer_count": snapshot.summary.no_answer_count,
            "total_token_count": snapshot.summary.total_token_count,
            "average_elapsed_ms": snapshot.summary.average_elapsed_ms,
            "average_provider_elapsed_ms": snapshot.summary.average_provider_elapsed_ms,
        },
        "runs": [
            {
                "generation_run_id": run.generation_run_id,
                "search_log_id": run.search_log_id,
                "provider_name": run.provider_name,
                "provider_mode": run.provider_mode,
                "model_id": run.model_id,
                "status": run.status,
                "guardrail_status": run.guardrail_status,
                "finish_reason": run.finish_reason,
                "input_token_count": run.input_token_count,
                "output_token_count": run.output_token_count,
                "total_token_count": run.total_token_count,
                "elapsed_ms": run.elapsed_ms,
                "provider_elapsed_ms": run.provider_elapsed_ms,
                "retry_count": run.retry_count,
                "succeeded": run.succeeded,
                "error_code": run.error_code,
                "error_message": run.error_message,
                "metric_present": run.metric_present,
                "created_at": run.created_at.isoformat(),
            }
            for run in snapshot.runs
        ],
    }
