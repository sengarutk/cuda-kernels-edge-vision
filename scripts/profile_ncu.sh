#!/usr/bin/env bash
# Collects Nsight Compute metrics for every target in profiling/ncu_targets.py and parses them
# into results/ncu_summary.json. Requires GPU performance-counter access (on WSL2/Windows:
# NVIDIA Control Panel > Developer > Manage GPU Performance Counters > allow all users).
#
#   NCU=/path/to/ncu PYTHON=python bash scripts/profile_ncu.sh
#
# On WSL2 the Linux ncu binary ships inside the Windows install, e.g.
#   NCU="/mnt/c/Program Files/NVIDIA Corporation/Nsight Compute 2023.1.0/target/linux-desktop-glibc_2_11_3-x64/ncu"
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
NCU="${NCU:-ncu}"
if ! command -v "$NCU" >/dev/null 2>&1 && [ ! -x "$NCU" ]; then
    echo "Nsight Compute not found. Set NCU=/path/to/ncu." >&2
    exit 1
fi

OUT=results/ncu
REPORTS="$OUT/reports"
mkdir -p "$REPORTS"
rm -f "$OUT"/*.csv
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

METRICS="$("$PYTHON" profiling/ncu_targets.py --print-metrics)"
TARGETS="$("$PYTHON" profiling/ncu_targets.py --list)"

for target in $TARGETS; do
    echo "=== ncu: $target"
    "$NCU" --profile-from-start off --target-processes all \
        --metrics "$METRICS" \
        --export "$REPORTS/$target" --force-overwrite \
        "$PYTHON" profiling/ncu_targets.py --target "$target"
    "$NCU" --import "$REPORTS/$target.ncu-rep" --csv --page raw > "$OUT/$target.csv"
done

"$PYTHON" - "$NCU" <<'PY'
import json, subprocess, sys
version = subprocess.run([sys.argv[1], "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
meta = {
    "ncu_version": version,
    "clock_control": "base (ncu default: SM and memory clocks locked to base)",
    "cache_control": "all (ncu default: caches flushed before each kernel)",
    "replay_mode": "kernel",
}
open("results/ncu/meta.json", "w").write(json.dumps(meta, indent=2) + "\n")
PY

"$PYTHON" profiling/parse_ncu_metrics.py
