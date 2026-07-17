import importlib.util
import json
import sys
from pathlib import Path


def _load_render_service_startup_templates_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "render_service_startup_templates.py"
    )
    spec = importlib.util.spec_from_file_location(
        "render_service_startup_templates_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


render_service_startup_templates = _load_render_service_startup_templates_module()


def test_main_prints_plan_without_writing_files(monkeypatch, capsys, tmp_path) -> None:
    output_dir = tmp_path / "deployment"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_service_startup_templates.py",
            "--workdir",
            str(tmp_path / "repo"),
            "--user",
            "operator",
            "--output-dir",
            str(output_dir),
            "--web-port",
            "8080",
            "--pretty",
        ],
    )

    exit_code = render_service_startup_templates.main()
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["wrote_files"] is False
    assert payload["user"] == "operator"
    assert payload["services"][0]["shell_command"].endswith("--port 8080")
    assert not output_dir.exists()


def test_main_writes_templates_and_json_output(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "deployment"
    json_output = tmp_path / "artifacts" / "service-startup.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_service_startup_templates.py",
            "--workdir",
            str(tmp_path / "repo"),
            "--user",
            "operator",
            "--group",
            "ops",
            "--output-dir",
            str(output_dir),
            "--chunk-policy-names",
            "small",
            "large",
            "--write",
            "--json-output",
            str(json_output),
        ],
    )

    exit_code = render_service_startup_templates.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["wrote_files"] is True
    assert payload["group"] == "ops"
    assert str(output_dir / "env" / "nex-pcx.env") in payload["written_files"]
    assert (output_dir / "systemd" / "nex-pcx-web.service").exists()
    assert (output_dir / "README.md").exists()
