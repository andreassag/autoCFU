#!/usr/bin/env bash
# ==============================================================================
# Update dependencies, format code, and verify test suite for AutoCFU.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

VENV="${ROOT_DIR}/.venv"
PYTHON="${VENV}/bin/python"
UV="uv"

if [ ! -d "${VENV}" ] || [ ! -f "${PYTHON}" ]; then
    echo "Virtual environment not detected at ${VENV}. Initializing via 'make install'..."
    make install
fi

echo "=== 1. Upgrading AutoCFU Dependencies ==="
if command -v "${UV}" >/dev/null 2>&1; then
    echo "Using uv to upgrade dependencies..."
    if ! "${UV}" pip install --python "${PYTHON}" --upgrade -e ".[dev,docs]"; then
        echo "Warning: Failed to upgrade dependencies from remote package index (offline or network-restricted environment). Proceeding with installed packages..."
    fi
elif [ -f "${PYTHON}" ]; then
    echo "Using virtual environment pip to upgrade dependencies..."
    if ! "${PYTHON}" -m pip install --upgrade pip; then
        echo "Warning: Failed to upgrade pip (network restricted)."
    fi
    if ! "${PYTHON}" -m pip install --upgrade -e ".[dev,docs]"; then
        echo "Warning: Failed to upgrade dependencies from remote package index (offline or network-restricted environment). Proceeding with installed packages..."
    fi
fi

echo "=== 2. Formatting Python Code (make format) ==="
make format

echo "=== 3. Linting & Type-Checking Python Code (make lint) ==="
make lint

echo "=== 4. Running Unit Tests (make test) ==="
make test

echo "=== 5. Building Documentation (make docs) ==="
make docs

echo "=== All dependency updates and verification checks completed successfully ==="
