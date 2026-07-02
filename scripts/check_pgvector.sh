#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${NEX_PCX_TEST_DATABASE_URL:-}" ]]; then
  echo "NEX_PCX_TEST_DATABASE_URL is required" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"

"${PYTHON_BIN}" -m pytest tests/integration
