import importlib.util
import json
import sys
from pathlib import Path

from app.core.config import Settings


def _load_export_go_live_evidence_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "export_go_live_evidence.py"
    spec = importlib.util.spec_from_file_location("export_go_live_evidence_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


export_go_live_evidence = _load_export_go_live_evidence_module()


def make_snapshot(status: str = "ready") -> dict[str, object]:
    return {
        "version": 1,
        "status": status,
        "exported_at": "2026-07-17T04:05:06+00:00",
        "exported_at_label": "2026-07-17 04:05:06",
        "runtime": {"database_url_masked": "postgresql://user:***@example/db"},
        "provenance": {"git_commit": "abc1234", "git_branch": "master", "git_dirty": False},
        "startup_validation": {
            "status": status,
            "failed_count": 0,
            "warning_count": 0,
            "checks": [],
        },
        "go_live_readiness": {
            "status": status,
            "failed_count": 0,
            "warning_count": 0,
            "sections": [],
        },
        "summary": {},
    }


def test_main_writes_json_and_markdown_outputs(monkeypatch, tmp_path) -> None:
    captured = {}
    json_output = tmp_path / "evidence" / "go_live.json"
    markdown_output = tmp_path / "evidence" / "go_live.md"

    def fake_build_snapshot(settings, *, options, project_root):
        captured["settings"] = settings
        captured["options"] = options
        captured["project_root"] = project_root
        return make_snapshot()

    monkeypatch.setattr(
        export_go_live_evidence,
        "get_settings",
        lambda: Settings(database_url="postgresql://original/db"),
    )
    monkeypatch.setattr(
        export_go_live_evidence,
        "build_go_live_evidence_snapshot",
        fake_build_snapshot,
    )
    monkeypatch.setattr(
        export_go_live_evidence,
        "render_go_live_evidence_markdown",
        lambda snapshot: "# evidence\n",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_go_live_evidence.py",
            "--database-url",
            "postgresql://override/db",
            "--app-url",
            "http://127.0.0.1:8000",
            "--health-timeout-seconds",
            "2.5",
            "--run-provider-preflight",
            "--profile-name",
            "kure_v1",
            "--include-inactive-routes",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--pretty",
        ],
    )

    exit_code = export_go_live_evidence.main()

    assert exit_code == 0
    assert captured["settings"].database_url == "postgresql://override/db"
    startup_options = captured["options"].startup_validation
    assert startup_options.app_base_url == "http://127.0.0.1:8000"
    assert startup_options.health_timeout_seconds == 2.5
    assert startup_options.run_provider_preflight is True
    assert startup_options.profile_name == "kure_v1"
    assert startup_options.include_inactive_routes is True
    assert json.loads(json_output.read_text(encoding="utf-8"))["status"] == "ready"
    assert markdown_output.read_text(encoding="utf-8") == "# evidence\n"


def test_main_prints_json_when_output_is_omitted(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        export_go_live_evidence,
        "get_settings",
        lambda: Settings(database_url="postgresql://example/db"),
    )
    monkeypatch.setattr(
        export_go_live_evidence,
        "build_go_live_evidence_snapshot",
        lambda settings, *, options, project_root: make_snapshot(),
    )
    monkeypatch.setattr(sys, "argv", ["export_go_live_evidence.py"])

    exit_code = export_go_live_evidence.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert '"status": "ready"' in output


def test_main_returns_nonzero_for_blocked_and_strict_warning(monkeypatch) -> None:
    monkeypatch.setattr(
        export_go_live_evidence,
        "get_settings",
        lambda: Settings(database_url="postgresql://example/db"),
    )
    monkeypatch.setattr(
        export_go_live_evidence,
        "build_go_live_evidence_snapshot",
        lambda settings, *, options, project_root: make_snapshot("blocked"),
    )
    monkeypatch.setattr(sys, "argv", ["export_go_live_evidence.py"])

    assert export_go_live_evidence.main() == 1

    monkeypatch.setattr(
        export_go_live_evidence,
        "build_go_live_evidence_snapshot",
        lambda settings, *, options, project_root: make_snapshot("warning"),
    )
    monkeypatch.setattr(sys, "argv", ["export_go_live_evidence.py", "--strict"])

    assert export_go_live_evidence.main() == 1
