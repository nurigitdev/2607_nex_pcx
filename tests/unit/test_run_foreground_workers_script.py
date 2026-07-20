import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.core.foreground_worker_runner import (
    ForegroundWorkerRunnerPlan,
    PendingEmbeddingProfileSummary,
    ProviderHealthProbe,
    ProviderResourceGuardDecision,
    WorkerCommandResult,
)


def _load_run_foreground_workers_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_foreground_workers.py"
    spec = importlib.util.spec_from_file_location("run_foreground_workers_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_foreground_workers = _load_run_foreground_workers_module()


def test_main_writes_dry_run_evidence_without_claiming_jobs(monkeypatch, tmp_path) -> None:
    json_output = tmp_path / "runner.json"
    markdown_output = tmp_path / "runner.md"
    calls = {"pipeline": 0, "embedding": 0}
    monkeypatch.setattr(
        run_foreground_workers,
        "build_foreground_worker_runner_plan",
        lambda *args, **kwargs: _plan(),
    )
    monkeypatch.setattr(
        run_foreground_workers,
        "_run_pipeline_workers",
        lambda **kwargs: calls.update({"pipeline": calls["pipeline"] + 1}) or [],
    )
    monkeypatch.setattr(
        run_foreground_workers,
        "_run_embedding_workers",
        lambda **kwargs: calls.update({"embedding": calls["embedding"] + 1}) or [],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_foreground_workers.py",
            "--database-url",
            "postgresql://example",
            "--dry-run",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--pretty",
        ],
    )

    exit_code = run_foreground_workers.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["status"] == "planned"
    assert payload["dry_run"] is True
    assert calls == {"pipeline": 0, "embedding": 0}
    assert "Foreground Worker Runner Evidence" in markdown_output.read_text(encoding="utf-8")


def test_main_runs_bounded_pipeline_and_embedding_helpers(monkeypatch, tmp_path) -> None:
    json_output = tmp_path / "runner.json"
    markdown_output = tmp_path / "runner.md"
    monkeypatch.setattr(
        run_foreground_workers,
        "build_foreground_worker_runner_plan",
        lambda *args, **kwargs: _plan(),
    )
    monkeypatch.setattr(
        run_foreground_workers,
        "_run_pipeline_workers",
        lambda **kwargs: [
            WorkerCommandResult(
                code="pipeline_worker_1",
                command=("python", "scripts/process_pipeline_job.py"),
                exit_code=0,
                elapsed_ms=5,
                payload={"processed": True, "status": "succeeded"},
            )
        ],
    )
    monkeypatch.setattr(
        run_foreground_workers,
        "_run_embedding_workers",
        lambda **kwargs: [
            WorkerCommandResult(
                code="embedding_worker_bge_m3_1024",
                command=("python", "scripts/process_embedding_job.py"),
                exit_code=0,
                elapsed_ms=7,
                payload={"processed_count": 1, "failed_count": 0},
            )
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_foreground_workers.py",
            "--database-url",
            "postgresql://example",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ],
    )

    exit_code = run_foreground_workers.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["status"] == "completed"
    assert payload["command_count"] == 2
    assert payload["pipeline_results"][0]["code"] == "pipeline_worker_1"
    assert payload["embedding_results"][0]["code"] == "embedding_worker_bge_m3_1024"


def test_run_json_command_parses_last_stdout_line(tmp_path) -> None:
    script = tmp_path / "emit_json.py"
    script.write_text(
        "print('hello')\n" 'print(\'{"processed": true, "failed_count": 0}\')\n',
        encoding="utf-8",
    )

    result = run_foreground_workers._run_json_command(
        code="demo",
        command=(sys.executable, str(script)),
        workdir=tmp_path,
    )

    assert result.succeeded
    assert result.payload == {"processed": True, "failed_count": 0}


def test_run_json_command_redacts_database_url_from_output(tmp_path) -> None:
    database_url = "postgresql://nex_pcx_app:secret@127.0.0.1:5432/nex_pcx_app"
    script = tmp_path / "emit_secret.py"
    script.write_text(
        "import os\n"
        "print(os.environ['NEX_PCX_DATABASE_URL'])\n"
        "print('{\"processed\": true}')\n",
        encoding="utf-8",
    )

    result = run_foreground_workers._run_json_command(
        code="demo",
        command=(sys.executable, str(script)),
        workdir=tmp_path,
        database_url=database_url,
    )

    assert result.succeeded
    assert result.payload == {"processed": True}
    assert database_url not in result.stdout
    assert database_url not in result.stderr
    assert database_url not in result.shell_command
    assert "<redacted-database-url>" in result.stdout


def test_evidence_status_reports_guarded_and_partial_results() -> None:
    plan = _plan(decision="skipped")
    failed_payload_result = WorkerCommandResult(
        code="embedding_worker_bge_m3_1024",
        command=("python", "scripts/process_embedding_job.py"),
        exit_code=0,
        elapsed_ms=7,
        payload={"processed_count": 1, "failed_count": 1},
    )

    assert (
        run_foreground_workers._evidence_status(
            dry_run=False,
            plan=plan,
            pipeline_results=[],
            embedding_results=[],
        )
        == "guarded"
    )
    assert (
        run_foreground_workers._evidence_status(
            dry_run=False,
            plan=_plan(),
            pipeline_results=[],
            embedding_results=[failed_payload_result],
        )
        == "partial"
    )


def _plan(*, decision: str = "allowed") -> ForegroundWorkerRunnerPlan:
    pending = PendingEmbeddingProfileSummary(
        profile_name="bge_m3_1024",
        pending_count=1,
        max_token_count=500,
        max_char_count=1000,
        oldest_job_id=1,
        newest_job_id=1,
    )
    return ForegroundWorkerRunnerPlan(
        status="ready",
        generated_at=datetime(2026, 7, 20, tzinfo=UTC),
        workdir="/repo",
        pipeline_limit=1,
        embedding_limit_per_profile=5,
        lease_seconds=300,
        worker_name_prefix="fg",
        health_timeout_seconds=5,
        max_health_elapsed_ms=5000,
        profile_token_limits={},
        excluded_profiles=(),
        pending_profiles=(pending,),
        guard_decisions=(
            ProviderResourceGuardDecision(
                profile_name="bge_m3_1024",
                decision=decision,
                reason="ok",
                pending_count=1,
                max_token_count=500,
                max_char_count=1000,
                token_limit=None,
                route_id=1,
                provider_name="bge",
                provider_mode="remote",
                provider_base_url="http://provider",
                health=ProviderHealthProbe(
                    checked=True,
                    ready=True,
                    status="ready",
                    elapsed_ms=10,
                ),
            ),
        ),
    )
