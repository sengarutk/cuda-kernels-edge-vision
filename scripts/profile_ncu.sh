#!/usr/bin/env bash
set -e

if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "/home/sengar/dev/.venv/bin/activate" ]; then
        source /home/sengar/dev/.venv/bin/activate
    fi
fi

OUTPUT_DIR="profiling_output/ncu"
mkdir -p "$OUTPUT_DIR"

echo "=== 1. Checking Nsight Compute (ncu) Availability ==="
if command -v ncu &> /dev/null; then
    echo "[+] Running Nsight Compute capture on Triton 1D Preprocess..."
    ncu --set full \
        --target-processes all \
        --launch-count 5 \
        --launch-skip 5 \
        --export "$OUTPUT_DIR/triton_preprocess_640" \
        --force-overwrite \
        python profiling/profile_ncu_targets.py --target triton_preprocess_640 || true
else
    echo "[!] 'ncu' command not found in PATH. Proceeding with automated metrics parser..."
fi

echo "=== 2. Parsing Micro-Architectural Metrics ==="
python profiling/parse_ncu_metrics.py
