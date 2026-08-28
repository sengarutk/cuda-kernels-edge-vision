#!/usr/bin/env bash
set -e

if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "/home/sengar/dev/.venv/bin/activate" ]; then
        source /home/sengar/dev/.venv/bin/activate
    fi
fi

export PYTHONPATH="${PYTHONPATH:-}:."

echo "=== 1. Running Phase 3 Correctness Tests ==="
pytest -v tests/test_preprocess_triton.py tests/test_failed_kernel_parity.py

echo "=== 2. Running Isolated Preprocessing Benchmarks ==="
python benchmarks/bench_preprocess_isolated.py

echo "=== Phase 3 Validation & Benchmarks Complete ==="
