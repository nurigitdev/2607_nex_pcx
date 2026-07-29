import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.core.vllm_runtime_metric_snapshots import VLLMRuntimeMetricSnapshotRecord


def _load_script_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "scrape_vllm_runtime_metrics.py"
    spec = importlib.util.spec_from_file_location("scrape_vllm_runtime_metrics_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scrape_script = _load_script_module()


def test_scrape_vllm_runtime_metrics_script_parses_sample_file_outputs(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    sample_file = tmp_path / "vllm.prom"
    sample_file.write_text(
        """
        vllm:kv_cache_usage_perc 0.64
        vllm:num_requests_running 2
        vllm:prompt_tokens_total 30
        """,
        encoding="utf-8",
    )
    json_output = tmp_path / "out" / "vllm.json"
    markdown_output = tmp_path / "out" / "vllm.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scrape_vllm_runtime_metrics.py",
            "--sample-file",
            str(sample_file),
            "--base-url",
            "http://vllm.local:12000/",
            "--provider-name",
            "pytest-vllm",
            "--model-id",
            "served-qwen",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--pretty",
        ],
    )

    assert scrape_script.main() == 0
    assert capsys.readouterr().out == ""
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    markdown = markdown_output.read_text(encoding="utf-8")
    assert payload["provider_name"] == "pytest-vllm"
    assert payload["provider_base_url"] == "http://vllm.local:12000"
    assert payload["model_id"] == "served-qwen"
    assert payload["kv_cache_usage_percent"] == 64.0
    assert payload["running_requests"] == 2
    assert "# vLLM Runtime Metrics Snapshot" in markdown
    assert "| KV cache usage | 64 |" in markdown
    assert "| Running requests | 2 |" in markdown


def test_scrape_vllm_runtime_metrics_script_prints_json_and_handles_errors(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    sample_file = tmp_path / "vllm.prom"
    sample_file.write_text("vllm:num_requests_waiting 5\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scrape_vllm_runtime_metrics.py",
            "--sample-file",
            str(sample_file),
            "--include-raw-samples",
        ],
    )

    assert scrape_script.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["waiting_requests"] == 5
    assert payload["raw_samples"][0]["name"] == "vllm:num_requests_waiting"

    bad_file = tmp_path / "bad.prom"
    bad_file.write_text("bad line", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["scrape_vllm_runtime_metrics.py", "--sample-file", str(bad_file)],
    )

    assert scrape_script.main() == 1
    captured = capsys.readouterr()
    assert "vLLM runtime metrics scrape failed" in captured.err


def test_scrape_vllm_runtime_metrics_script_persists_snapshot_when_requested(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    sample_file = tmp_path / "vllm.prom"
    sample_file.write_text("vllm:kv_cache_usage_perc 0.77\n", encoding="utf-8")
    calls = {}

    def fake_record(database_url, snapshot, *, runtime_metadata=None):
        calls["database_url"] = database_url
        calls["provider_name"] = snapshot.provider_name
        calls["runtime_metadata"] = runtime_metadata
        return VLLMRuntimeMetricSnapshotRecord(
            snapshot_id=99,
            provider_name=snapshot.provider_name,
            provider_base_url=snapshot.provider_base_url,
            model_id=snapshot.model_id,
            sampled_at=datetime(2026, 7, 29, 12, tzinfo=UTC),
            scrape_elapsed_ms=snapshot.scrape_elapsed_ms,
            raw_text_bytes=snapshot.raw_text_bytes,
            metric_count=snapshot.metric_count,
            vllm_metric_count=snapshot.vllm_metric_count,
            metric_names=snapshot.metric_names,
            kv_cache_usage_ratio=snapshot.kv_cache_usage_ratio,
            kv_cache_usage_percent=snapshot.kv_cache_usage_percent,
            cpu_cache_usage_ratio=snapshot.cpu_cache_usage_ratio,
            cpu_cache_usage_percent=snapshot.cpu_cache_usage_percent,
            running_requests=snapshot.running_requests,
            waiting_requests=snapshot.waiting_requests,
            swapped_requests=snapshot.swapped_requests,
            waiting_requests_by_reason=dict(snapshot.waiting_requests_by_reason),
            request_success_total=snapshot.request_success_total,
            prompt_tokens_total=snapshot.prompt_tokens_total,
            generation_tokens_total=snapshot.generation_tokens_total,
            prompt_tokens_cached_total=snapshot.prompt_tokens_cached_total,
            prefix_cache_hits_total=snapshot.prefix_cache_hits_total,
            prefix_cache_queries_total=snapshot.prefix_cache_queries_total,
            prefix_cache_hit_rate=snapshot.prefix_cache_hit_rate,
            num_preemptions_total=snapshot.num_preemptions_total,
            average_time_to_first_token_seconds=snapshot.average_time_to_first_token_seconds,
            average_inter_token_latency_seconds=snapshot.average_inter_token_latency_seconds,
            average_e2e_request_latency_seconds=snapshot.average_e2e_request_latency_seconds,
            average_request_queue_time_seconds=snapshot.average_request_queue_time_seconds,
            average_request_prefill_time_seconds=snapshot.average_request_prefill_time_seconds,
            average_request_decode_time_seconds=snapshot.average_request_decode_time_seconds,
            runtime_metadata={"contract_version": snapshot.contract_version},
            raw_samples=(),
            created_at=datetime(2026, 7, 29, 12, tzinfo=UTC),
        )

    monkeypatch.setattr(scrape_script, "record_vllm_runtime_metric_snapshot", fake_record)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scrape_vllm_runtime_metrics.py",
            "--sample-file",
            str(sample_file),
            "--provider-name",
            "pytest-vllm",
            "--database-url",
            "postgresql://pytest/db",
            "--persist",
        ],
    )

    assert scrape_script.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert calls == {
        "database_url": "postgresql://pytest/db",
        "provider_name": "pytest-vllm",
        "runtime_metadata": {
            "source": "scripts/scrape_vllm_runtime_metrics.py",
            "sample_file": True,
        },
    }
    assert payload["snapshot_record"]["snapshot_id"] == 99
    assert payload["snapshot_record"]["kv_cache_usage_percent"] == 77.0


def test_scrape_vllm_runtime_metrics_script_requires_database_url_for_persistence(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    sample_file = tmp_path / "vllm.prom"
    sample_file.write_text("vllm:num_requests_running 1\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scrape_vllm_runtime_metrics.py",
            "--sample-file",
            str(sample_file),
            "--database-url",
            "",
            "--persist",
        ],
    )

    assert scrape_script.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "database URL is required" in captured.err
