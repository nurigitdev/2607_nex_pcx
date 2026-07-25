from pathlib import Path


def _quality_gate_script_text() -> str:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "quality_gate.sh"
    return script_path.read_text(encoding="utf-8")


def test_quality_gate_runs_pytest_coverage_once() -> None:
    script_text = _quality_gate_script_text()

    assert script_text.count("-m pytest") == 1
    assert "tests/unit tests/smoke tests/integration tests/regression" in script_text
    assert "--cov=app" in script_text
    assert "--cov-branch" in script_text
    assert '--cov-report="json:${COVERAGE_JSON}"' in script_text
    assert "-m coverage json" not in script_text


def test_quality_gate_reports_statement_and_branch_coverage_separately() -> None:
    script_text = _quality_gate_script_text()

    assert "covered_lines" in script_text
    assert "num_statements" in script_text
    assert "covered_branches" in script_text
    assert "num_branches" in script_text
    assert "Statement coverage:" in script_text
    assert "Branch coverage:" in script_text
