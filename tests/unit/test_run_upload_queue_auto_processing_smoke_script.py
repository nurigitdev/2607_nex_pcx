import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_upload_queue_smoke_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_upload_queue_auto_processing_smoke.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_upload_queue_auto_processing_smoke_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


upload_queue_smoke = _load_upload_queue_smoke_module()


def test_main_writes_dry_run_smoke_plan(monkeypatch, tmp_path) -> None:
    json_output = tmp_path / "smoke.json"
    markdown_output = tmp_path / "smoke.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_upload_queue_auto_processing_smoke.py",
            "--dry-run",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--pretty",
        ],
    )

    exit_code = upload_queue_smoke.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["status"] == "planned"
    assert payload["plan"]["upload_url"].endswith("/api/files")
    assert "Upload Queue Auto-Processing Smoke" in markdown_output.read_text(encoding="utf-8")


def test_main_returns_ready_when_uploaded_pipeline_job_succeeds(
    monkeypatch,
    tmp_path,
) -> None:
    json_output = tmp_path / "smoke.json"
    calls = {"upload": 0}
    pipeline_payloads = [
        {"job": {"job_id": 10, "status": "pending", "stage": "queued"}},
        {
            "job": {
                "job_id": 10,
                "status": "succeeded",
                "stage": "completed",
                "progress_percent": 100,
            }
        },
    ]

    def fake_get_json(url):
        if url.endswith("/api/admin/foreground-worker-runtime"):
            return {"foreground_worker_runtime": {"status": "ready"}}
        return pipeline_payloads.pop(0)

    def fake_post_upload(**kwargs):
        calls["upload"] += 1
        return {"pipeline_job_id": 10}

    monkeypatch.setattr(upload_queue_smoke, "_get_json", fake_get_json)
    monkeypatch.setattr(upload_queue_smoke, "_post_upload", fake_post_upload)
    monkeypatch.setattr(upload_queue_smoke.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_upload_queue_auto_processing_smoke.py",
            "--poll-attempts",
            "2",
            "--poll-interval-seconds",
            "0",
            "--json-output",
            str(json_output),
        ],
    )

    exit_code = upload_queue_smoke.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert calls["upload"] == 1
    assert payload["status"] == "ready"
    assert payload["pipeline_job_id"] == 10
    assert len(payload["poll_results"]) == 2


def test_main_returns_warning_when_pipeline_job_remains_pending(
    monkeypatch,
    tmp_path,
) -> None:
    json_output = tmp_path / "smoke.json"
    monkeypatch.setattr(
        upload_queue_smoke,
        "_get_json",
        lambda url: (
            {"foreground_worker_runtime": {"status": "warning"}}
            if url.endswith("/api/admin/foreground-worker-runtime")
            else {"job": {"job_id": 11, "status": "pending", "stage": "queued"}}
        ),
    )
    monkeypatch.setattr(
        upload_queue_smoke,
        "_post_upload",
        lambda **kwargs: {"pipeline_job_id": 11},
    )
    monkeypatch.setattr(upload_queue_smoke.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_upload_queue_auto_processing_smoke.py",
            "--poll-attempts",
            "1",
            "--json-output",
            str(json_output),
        ],
    )

    exit_code = upload_queue_smoke.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["status"] == "warning"
    assert "did not reach succeeded" in payload["message"]


def test_main_returns_blocked_when_runtime_is_blocked(monkeypatch, tmp_path) -> None:
    json_output = tmp_path / "smoke.json"
    monkeypatch.setattr(
        upload_queue_smoke,
        "_get_json",
        lambda url: (
            {"foreground_worker_runtime": {"status": "blocked"}}
            if url.endswith("/api/admin/foreground-worker-runtime")
            else {"job": {"job_id": 12, "status": "pending", "stage": "queued"}}
        ),
    )
    monkeypatch.setattr(
        upload_queue_smoke,
        "_post_upload",
        lambda **kwargs: {"pipeline_job_id": 12},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_upload_queue_auto_processing_smoke.py",
            "--poll-attempts",
            "1",
            "--json-output",
            str(json_output),
        ],
    )

    exit_code = upload_queue_smoke.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert payload["status"] == "blocked"


def test_smoke_helpers_validate_inputs_and_responses() -> None:
    with pytest.raises(ValueError, match="app-url"):
        upload_queue_smoke._smoke_plan(
            app_url=" ",
            filename="a.md",
            poll_attempts=1,
            poll_interval_seconds=0,
        )
    with pytest.raises(ValueError, match="pipeline_job_id"):
        upload_queue_smoke._pipeline_job_id({})
    assert upload_queue_smoke._runtime_status({"status": "ready"}) == "ready"
