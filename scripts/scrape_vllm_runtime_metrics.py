"""Scrape or parse vLLM runtime Prometheus metrics."""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.vllm_runtime_metric_snapshots import (  # noqa: E402
    InvalidVLLMRuntimeMetricSnapshotError,
    record_vllm_runtime_metric_snapshot,
    vllm_runtime_metric_snapshot_record_payload,
)
from app.core.vllm_runtime_metrics import (  # noqa: E402
    DEFAULT_VLLM_RUNTIME_METRICS_TIMEOUT_SECONDS,
    InvalidVLLMRuntimeMetricsError,
    VLLMRuntimeMetricsClient,
    scrape_vllm_runtime_metrics_from_text,
    vllm_runtime_metrics_snapshot_payload,
)

DEFAULT_DGX_VLLM_BASE_URL = "http://192.168.20.243:12000"
DEFAULT_DGX_VLLM_PROVIDER_NAME = "dgx_vllm_qwen35_122b_a10b_nvfp4"
DEFAULT_DGX_VLLM_MODEL_ID = "/home/nurivoice-dgx/models/nvidia/Qwen3.5-122B-A10B-NVFP4"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape normalized runtime metrics from a vLLM /metrics endpoint.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("NEX_PCX_VLLM_BASE_URL", DEFAULT_DGX_VLLM_BASE_URL),
    )
    parser.add_argument(
        "--provider-name",
        default=os.getenv("NEX_PCX_VLLM_PROVIDER_NAME", DEFAULT_DGX_VLLM_PROVIDER_NAME),
    )
    parser.add_argument(
        "--model-id",
        default=os.getenv("NEX_PCX_VLLM_MODEL_ID", DEFAULT_DGX_VLLM_MODEL_ID),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(
            os.getenv(
                "NEX_PCX_VLLM_METRICS_TIMEOUT_SECONDS",
                str(DEFAULT_VLLM_RUNTIME_METRICS_TIMEOUT_SECONDS),
            )
        ),
    )
    parser.add_argument(
        "--sample-file",
        default=None,
        help="Parse a local Prometheus metrics text file instead of making an HTTP request.",
    )
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--include-raw-samples", action="store_true")
    parser.add_argument(
        "--database-url",
        default=os.getenv("NEX_PCX_DATABASE_URL"),
        help="Database URL used with --persist. Defaults to NEX_PCX_DATABASE_URL.",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist the normalized vLLM runtime metrics snapshot to the database.",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    try:
        snapshot = scrape_or_parse_snapshot(args)
    except InvalidVLLMRuntimeMetricsError as exc:
        print(f"vLLM runtime metrics scrape failed: {exc}", file=sys.stderr)
        return 1

    payload = vllm_runtime_metrics_snapshot_payload(
        snapshot,
        include_raw_samples=args.include_raw_samples,
    )
    if args.persist:
        if not args.database_url:
            print(
                "vLLM runtime metrics persistence failed: database URL is required", file=sys.stderr
            )
            return 1
        try:
            record = record_vllm_runtime_metric_snapshot(
                args.database_url,
                snapshot,
                runtime_metadata={
                    "source": "scripts/scrape_vllm_runtime_metrics.py",
                    "sample_file": bool(args.sample_file),
                },
            )
        except InvalidVLLMRuntimeMetricSnapshotError as exc:
            print(f"vLLM runtime metrics persistence failed: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # pragma: no cover - defensive CLI boundary
            print(f"vLLM runtime metrics persistence failed: {exc}", file=sys.stderr)
            return 1
        payload["snapshot_record"] = vllm_runtime_metric_snapshot_record_payload(record)
    json_text = json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.json_output:
        _write_text(Path(args.json_output), json_text + "\n")
    else:
        print(json_text)

    if args.markdown_output:
        _write_text(Path(args.markdown_output), render_vllm_runtime_metrics_markdown(payload))
    return 0


def scrape_or_parse_snapshot(args: argparse.Namespace):
    if args.sample_file:
        metrics_text = Path(args.sample_file).read_text(encoding="utf-8")
        return scrape_vllm_runtime_metrics_from_text(
            metrics_text,
            provider_name=args.provider_name,
            provider_base_url=args.base_url,
            model_id=args.model_id,
            scrape_elapsed_ms=None,
        )
    client = VLLMRuntimeMetricsClient(
        args.base_url,
        provider_name=args.provider_name,
        model_id=args.model_id,
        timeout_seconds=args.timeout_seconds,
    )
    try:
        return client.scrape()
    finally:
        client.close()


def render_vllm_runtime_metrics_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# vLLM Runtime Metrics Snapshot",
        "",
        f"- `provider_name`: `{payload.get('provider_name')}`",
        f"- `provider_base_url`: `{payload.get('provider_base_url')}`",
        f"- `model_id`: `{payload.get('model_id')}`",
        f"- `sampled_at`: `{payload.get('sampled_at')}`",
        f"- `scrape_elapsed_ms`: `{payload.get('scrape_elapsed_ms')}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    metric_rows = (
        ("KV cache usage", payload.get("kv_cache_usage_percent")),
        ("CPU cache usage", payload.get("cpu_cache_usage_percent")),
        ("Running requests", payload.get("running_requests")),
        ("Waiting requests", payload.get("waiting_requests")),
        ("Swapped requests", payload.get("swapped_requests")),
        ("Prompt tokens total", payload.get("prompt_tokens_total")),
        ("Generation tokens total", payload.get("generation_tokens_total")),
        ("Prefix cache hit rate", payload.get("prefix_cache_hit_rate")),
        ("Preemptions total", payload.get("num_preemptions_total")),
        ("Avg TTFT seconds", payload.get("average_time_to_first_token_seconds")),
        ("Avg inter-token latency seconds", payload.get("average_inter_token_latency_seconds")),
        ("Avg e2e latency seconds", payload.get("average_e2e_request_latency_seconds")),
    )
    for label, value in metric_rows:
        lines.append(f"| {label} | {_markdown_value(value)} |")

    waiting_reasons = payload.get("waiting_requests_by_reason")
    if isinstance(waiting_reasons, dict) and waiting_reasons:
        lines.extend(
            ["", "## Waiting Requests by Reason", "", "| Reason | Count |", "| --- | ---: |"]
        )
        for reason, count in waiting_reasons.items():
            lines.append(f"| {reason} | {count} |")
    return "\n".join(lines) + "\n"


def _markdown_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
