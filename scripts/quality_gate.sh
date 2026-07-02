#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"

"${PYTHON_BIN}" -m ruff check app tests
"${PYTHON_BIN}" -m black --check app tests
"${PYTHON_BIN}" -m pytest tests/unit tests/smoke tests/integration
"${PYTHON_BIN}" -m pytest --cov=app --cov-branch --cov-report=term-missing
