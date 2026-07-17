import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.core.go_live_smoke import (
    SMOKE_CHECK_PASSED,
    SMOKE_CHECK_WARNING,
    SMOKE_STATUS_BLOCKED,
    SMOKE_STATUS_WARNING,
    GoLiveSmokeCheck,
    GoLiveSmokeReport,
)


def _load_run_go_live_smoke_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_go_live_smoke.py"
    spec = importlib.util.spec_from_file_location("run_go_live_smoke_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_go_live_smoke_script = _load_run_go_live_smoke_module()


def make_report(status: str) -> GoLiveSmokeReport:
    check_status = (
        SMOKE_CHECK_WARNING if status == SMOKE_STATUS_WARNING else SMOKE_CHECK_PASSED
    )
    return GoLiveSmokeReport(
        status=status,
        checked_at=datetime(2026, 7, 17, 10, 11, 12, tzinfo=UTC),
        app_base_url="http://127.0.0.1:8000",
        checks=(
            GoLiveSmokeCheck(
                code="script_test",
                status=check_status,
                detail="script test detail",
            ),
        ),
    )


def test_main_writes_json_and_markdown_outputs(monkeypatch, tmp_path) -> None:
    captured = {}
    json_output = tmp_path / "smoke" / "go-live.json"
    markdown_output = tmp_path / "smoke" / "go-live.md"

    def fake_run(app_url, *, timeout_seconds):
        captured["app_url"] = app_url
        captured["timeout_seconds"] = timeout_seconds
        return make_report(SMOKE_STATUS_WARNING)

    monkeypatch.setattr(run_go_live_smoke_script, "run_go_live_smoke", fake_run)
    monkeypatch.setattr(
        run_go_live_smoke_script,
        "render_go_live_smoke_markdown",
        lambda payload: "# smoke\n",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_go_live_smoke.py",
            "--app-url",
            "http://app",
            "--timeout-seconds",
            "1.5",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--pretty",
        ],
    )

    exit_code = run_go_live_smoke_script.main()

    assert exit_code == 0
    assert captured == {"app_url": "http://app", "timeout_seconds": 1.5}
    assert json.loads(json_output.read_text(encoding="utf-8"))["status"] == "warning"
    assert markdown_output.read_text(encoding="utf-8") == "# smoke\n"


def test_main_prints_json_and_handles_nonzero_exit_codes(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        run_go_live_smoke_script,
        "run_go_live_smoke",
        lambda app_url, *, timeout_seconds: make_report(SMOKE_STATUS_WARNING),
    )
    monkeypatch.setattr(sys, "argv", ["run_go_live_smoke.py"])

    assert run_go_live_smoke_script.main() == 0
    assert '"status": "warning"' in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["run_go_live_smoke.py", "--strict"])
    assert run_go_live_smoke_script.main() == 1

    monkeypatch.setattr(
        run_go_live_smoke_script,
        "run_go_live_smoke",
        lambda app_url, *, timeout_seconds: make_report(SMOKE_STATUS_BLOCKED),
    )
    monkeypatch.setattr(sys, "argv", ["run_go_live_smoke.py"])
    assert run_go_live_smoke_script.main() == 1
