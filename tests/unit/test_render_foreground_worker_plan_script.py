import importlib.util
import json
import sys
from pathlib import Path


def _load_render_foreground_worker_plan_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "render_foreground_worker_plan.py"
    )
    spec = importlib.util.spec_from_file_location(
        "render_foreground_worker_plan_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


render_foreground_worker_plan = _load_render_foreground_worker_plan_module()


def test_main_writes_foreground_worker_plan_outputs(monkeypatch, tmp_path) -> None:
    json_output = tmp_path / "worker-plan.json"
    markdown_output = tmp_path / "worker-plan.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_foreground_worker_plan.py",
            "--workdir",
            str(tmp_path),
            "--database-url-source",
            "${DATABASE_URL}",
            "--embedding-limit",
            "3",
            "--chunk-policy-names",
            "small",
            "large",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--pretty",
        ],
    )

    exit_code = render_foreground_worker_plan.main()
    payload = json.loads(json_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["embedding_limit"] == 3
    assert payload["chunk_policy_names"] == ["small", "large"]
    assert "Foreground Worker Command Plan" in markdown_output.read_text(encoding="utf-8")
