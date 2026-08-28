#!/usr/bin/env bash
set -e

if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "/home/sengar/dev/.venv/bin/activate" ]; then
        source /home/sengar/dev/.venv/bin/activate
    fi
fi

OUTPUT_DIR="profiling_output/nsys"
mkdir -p "$OUTPUT_DIR"

echo "[+] Starting Nsight Systems Macro Profile..."
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output="$OUTPUT_DIR/baseline_pipelines" \
    --force-overwrite=true \
    --duration=15 \
    python profiling/run_torch_profiler.py

echo "[✓] Nsight Systems capture saved to $OUTPUT_DIR/baseline_pipelines.nsys-rep"
