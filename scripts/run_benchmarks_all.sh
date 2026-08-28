#!/usr/bin/env bash
set -e

if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "/home/sengar/dev/.venv/bin/activate" ]; then
        source /home/sengar/dev/.venv/bin/activate
    fi
fi

export PYTHONPATH="${PYTHONPATH:-}:."

echo "=========================================================="
echo "RUNNING FULL BENCHMARK SUITE (KERNEL 1 & KERNEL 2)"
echo "=========================================================="
python benchmarks/bench_preprocess_isolated.py
python benchmarks/bench_filter_isolated.py
