#!/usr/bin/env bash
set -e

if [ -f "/home/sengar/dev/.venv/bin/activate" ]; then
    source /home/sengar/dev/.venv/bin/activate
fi

export PYTHONPATH="${PYTHONPATH:-}:."

echo "=== 1. Generating Deterministic Fixtures ==="
python tests/fixtures/generate_fixtures.py

echo "=== 2. Running Full Pytest Suite with Coverage ==="
/home/sengar/dev/.venv/bin/pytest --cov=src --cov-report=term-missing -v tests/

echo "=== 3. Running Isolated Kernel Benchmarks ==="
python benchmarks/bench_preprocess_isolated.py
python benchmarks/bench_filter_isolated.py

echo "=== 4. Running Micro-Architectural NCU Suite ==="
bash scripts/profile_ncu.sh

echo "=== 5. Running End-to-End Pipeline Benchmarks ==="
python benchmarks/bench_stage_breakdown.py
python benchmarks/bench_e2e_pipeline.py

echo "=== 6. Building Research Paper PDF ==="
if [ -f "paper/build_paper.sh" ]; then
    bash paper/build_paper.sh || true
fi

echo "=== All Artifacts Generated Successfully ==="
