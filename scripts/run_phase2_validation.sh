#!/usr/bin/env bash
set -e

if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "/home/sengar/dev/.venv/bin/activate" ]; then
        source /home/sengar/dev/.venv/bin/activate
    fi
fi

export PYTHONPATH="${PYTHONPATH:-}:."

echo "=== 1. Generating Deterministic Fixtures ==="
python tests/fixtures/generate_fixtures.py

echo "=== 2. Running Phase 2 Multi-Backend Baseline Validation Suite ==="
pytest -v tests/test_preprocess_baselines.py tests/test_postprocess_baselines.py tests/test_fixtures_parity.py

echo "=== Phase 2 Validation Passed Flawlessly ==="
