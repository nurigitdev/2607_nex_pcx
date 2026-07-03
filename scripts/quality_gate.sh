#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
COVERAGE_JSON="${COVERAGE_JSON:-/tmp/nex_pcx_coverage.json}"
COVERAGE_BRANCH_FAIL_UNDER="${COVERAGE_BRANCH_FAIL_UNDER:-85}"

"${PYTHON_BIN}" -m ruff check app tests
"${PYTHON_BIN}" -m black --check app tests
"${PYTHON_BIN}" -m pytest tests/unit tests/smoke tests/integration tests/regression
"${PYTHON_BIN}" -m pytest --cov=app --cov-branch --cov-report=term-missing
"${PYTHON_BIN}" -m coverage json -o "${COVERAGE_JSON}"
"${PYTHON_BIN}" -c '
import json
import sys

coverage_path = sys.argv[1]
branch_fail_under = float(sys.argv[2])

with open(coverage_path, encoding="utf-8") as coverage_file:
    branch_coverage = float(json.load(coverage_file)["totals"]["percent_branches_covered"])

if branch_coverage < branch_fail_under:
    raise SystemExit(
        f"Branch coverage {branch_coverage:.2f}% is below required {branch_fail_under:.1f}%"
    )

print(
    f"Required branch coverage of {branch_fail_under:.1f}% reached. "
    f"Branch coverage: {branch_coverage:.2f}%"
)
' "${COVERAGE_JSON}" "${COVERAGE_BRANCH_FAIL_UNDER}"
