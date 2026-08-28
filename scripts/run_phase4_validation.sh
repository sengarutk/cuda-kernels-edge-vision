#!/usr/bin/env bash
set -e

if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "/home/sengar/dev/.venv/bin/activate" ]; then
        source /home/sengar/dev/.venv/bin/activate
    fi
fi

export PYTHONPATH="${PYTHONPATH:-}:."

echo "=== 1. Running Phase 4 Correctness & Parity Tests ==="
pytest -v tests/test_candidate_filter_triton.py tests/test_downstream_detection_parity.py

echo "=== 2. Running Isolated Postprocessing Benchmarks ==="
python benchmarks/bench_filter_isolated.py

echo "=== 3. Re-running Hardened Preprocessing Benchmarks ==="
python benchmarks/bench_preprocess_isolated.py

echo "=== Phase 4 Validation & Benchmarks Complete ==="
