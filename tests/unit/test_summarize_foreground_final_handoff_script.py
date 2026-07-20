import importlib.util
import json
import sys
from pathlib import Path


def _load_summarize_foreground_final_handoff_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "summarize_foreground_final_handoff.py"
    )
    spec = importlib.util.spec_from_file_location(
        "summarize_foreground_final_handoff_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


summarize_foreground_final_handoff = _load_summarize_foreground_final_handoff_module()


def test_main_writes_foreground_final_handoff_outputs(monkeypatch, tmp_path) -> None:
    _write_json(tmp_path / "artifacts" / "foreground_production_launch.json", {"status": "planned"})
    _write_json(
        tmp_path / "artifacts" / "foreground_production_shutdown.json",
        {"status": "planned"},
    )
    _write_json(
        tmp_path / "artifacts" / "foreground_operations_validation.json", {"status": "ready"}
    )
    _write_json(tmp_path / "artifacts" / "foreground_go_live_summary.json", {"status": "warning"})
    _write_json(
        tmp_path / "artifacts" / "foreground_worker_plan.json",
        {
            "commands": [
                {"code": "pipeline_worker_help", "bounded": True},
                {"code": "embedding_worker_help", "bounded": True},
                {"code": "pipeline_worker_once", "bounded": True},
                {"code": "embedding_worker_batch", "bounded": True},
            ]
        },
    )
    _write_json(
        tmp_path / "artifacts" / "foreground_worker_runner.json",
        {"status": "planned"},
    )
    _write_json(
        tmp_path / "artifacts" / "operator_handoff" / "latest" / "manifest.json",
        {"file_count": 4, "included_count": 4, "missing_required_count": 0},
    )
    json_output = tmp_path / "final-handoff.json"
    markdown_output = tmp_path / "final-handoff.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_foreground_final_handoff.py",
            "--workdir",
            str(tmp_path),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--pretty",
        ],
    )

    exit_code = summarize_foreground_final_handoff.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["status"] == "warning"
    assert payload["warning_count"] == 4
    assert "Foreground Final Handoff Checklist" in markdown_output.read_text(encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
