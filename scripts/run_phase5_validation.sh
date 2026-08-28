#!/usr/bin/env bash
set -e

if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "/home/sengar/dev/.venv/bin/activate" ]; then
        source /home/sengar/dev/.venv/bin/activate
    fi
fi

export PYTHONPATH="${PYTHONPATH:-}:."

echo "=== 1. Executing NCU Profiling Suite ==="
bash scripts/profile_ncu.sh

echo "=== 2. Running Phase 5 Manifest Verification Tests ==="
pytest -v tests/test_profiling_manifest.py

echo "=== Phase 5 Micro-Architectural Analysis Complete ==="
