import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import Settings
from app.core.shutdown_drain_check import (
    DRAIN_CHECK_PASSED,
    DRAIN_CHECK_WARNING,
    DRAIN_STATUS_BLOCKED,
    DRAIN_STATUS_WARNING,
    ShutdownDrainCheck,
    ShutdownDrainReport,
)


def _load_check_shutdown_drain_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "check_shutdown_drain.py"
    spec = importlib.util.spec_from_file_location("check_shutdown_drain_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_shutdown_drain = _load_check_shutdown_drain_module()


def make_report(status: str) -> ShutdownDrainReport:
    check_status = (
        DRAIN_CHECK_WARNING if status == DRAIN_STATUS_WARNING else DRAIN_CHECK_PASSED
    )
    return ShutdownDrainReport(
        status=status,
        checked_at=datetime(2026, 7, 17, 4, 5, 6, tzinfo=UTC),
        checks=(
            ShutdownDrainCheck(
                code="script_test",
                status=check_status,
                detail="script test detail",
            ),
        ),
    )


def test_main_writes_json_and_markdown_outputs(monkeypatch, tmp_path) -> None:
    captured = {}
    json_output = tmp_path / "shutdown" / "drain.json"
    markdown_output = tmp_path / "shutdown" / "drain.md"

    def fake_build_report(settings):
        captured["settings"] = settings
        return make_report(DRAIN_STATUS_WARNING)

    monkeypatch.setattr(
        check_shutdown_drain,
        "get_settings",
        lambda: Settings(database_url="postgresql://original/db"),
    )
    monkeypatch.setattr(check_shutdown_drain, "build_shutdown_drain_report", fake_build_report)
    monkeypatch.setattr(
        check_shutdown_drain,
        "render_shutdown_drain_markdown",
        lambda payload: "# shutdown\n",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_shutdown_drain.py",
            "--database-url",
            "postgresql://override/db",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--pretty",
        ],
    )

    exit_code = check_shutdown_drain.main()

    assert exit_code == 0
    assert captured["settings"].database_url == "postgresql://override/db"
    assert json.loads(json_output.read_text(encoding="utf-8"))["status"] == "warning"
    assert markdown_output.read_text(encoding="utf-8") == "# shutdown\n"


def test_main_prints_json_when_output_is_omitted(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        check_shutdown_drain,
        "get_settings",
        lambda: Settings(database_url="postgresql://example/db"),
    )
    monkeypatch.setattr(
        check_shutdown_drain,
        "build_shutdown_drain_report",
        lambda settings: make_report(DRAIN_STATUS_WARNING),
    )
    monkeypatch.setattr(sys, "argv", ["check_shutdown_drain.py"])

    exit_code = check_shutdown_drain.main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert '"status": "warning"' in output


def test_main_returns_nonzero_for_blocked_and_strict_warning(monkeypatch) -> None:
    monkeypatch.setattr(
        check_shutdown_drain,
        "get_settings",
        lambda: Settings(database_url="postgresql://example/db"),
    )
    monkeypatch.setattr(
        check_shutdown_drain,
        "build_shutdown_drain_report",
        lambda settings: make_report(DRAIN_STATUS_BLOCKED),
    )
    monkeypatch.setattr(sys, "argv", ["check_shutdown_drain.py"])

    assert check_shutdown_drain.main() == 1

    monkeypatch.setattr(
        check_shutdown_drain,
        "build_shutdown_drain_report",
        lambda settings: make_report(DRAIN_STATUS_WARNING),
    )
    monkeypatch.setattr(sys, "argv", ["check_shutdown_drain.py", "--strict"])

    assert check_shutdown_drain.main() == 1
