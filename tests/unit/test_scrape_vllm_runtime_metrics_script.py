import importlib.util
import json
import sys
from pathlib import Path


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
