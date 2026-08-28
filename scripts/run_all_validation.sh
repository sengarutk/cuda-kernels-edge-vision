#!/usr/bin/env bash
set -e

if [ -f "/home/sengar/dev/.venv/bin/activate" ]; then
    source /home/sengar/dev/.venv/bin/activate
fi

export PYTHONPATH="${PYTHONPATH:-}:."

echo "=== 1. Running Full Pytest Suite with Coverage ==="
/home/sengar/dev/.venv/bin/pytest --cov=src --cov-report=term-missing -v tests/

echo "=== 2. Running Stage Breakdown & Amdahl Analysis ==="
python benchmarks/bench_stage_breakdown.py

echo "=== 3. Running End-to-End Inspection Pipeline Benchmarks ==="
python benchmarks/bench_e2e_pipeline.py

echo "=== Full Flagship 3 Systems Suite Validated Flawlessly ==="
