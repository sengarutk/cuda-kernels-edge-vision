#!/usr/bin/env bash
set -e

if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "/home/sengar/dev/.venv/bin/activate" ]; then
        source /home/sengar/dev/.venv/bin/activate
    fi
fi

echo "=== Running Phase 1 Validation & Profiling Harness ==="
pytest -v tests/

echo "[+] Executing PyTorch Profiler..."
python profiling/run_torch_profiler.py

echo "=== Phase 1 Validation Passed Flawlessly ==="
