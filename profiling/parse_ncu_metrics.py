"""Parses Nsight Compute raw CSV exports (results/ncu/*.csv) into results/ncu_summary.json."""
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
NCU_DIR = REPO_ROOT / "results" / "ncu"
STALL_PREFIX = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active.ratio"


def _number(value: str):
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return value


def read_raw_csv(path: Path) -> List[Dict[str, object]]:
    """Reads ``ncu --import X --csv --page raw``: header row, units row, one row per kernel."""
    lines = [l for l in path.read_text().splitlines() if l.startswith('"')]
    rows = list(csv.reader(lines))
    if len(rows) < 3:
        return []
    header, units, data = rows[0], rows[1], rows[2:]
    kernels = []
    for row in data:
        record = {}
        for name, unit, value in zip(header, units, row):
            record[name] = _number(value)
            if unit:
                record[f"{name} [unit]"] = unit
        kernels.append(record)
    return kernels


def _to_us(value: float, unit: str) -> float:
    scale = {"nsecond": 1e-3, "usecond": 1.0, "msecond": 1e3, "second": 1e6}
    return value * scale.get(unit, 1.0)


def _to_bytes(value: float, unit: str) -> float:
    scale = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "KB": 1e3, "MB": 1e6, "GB": 1e9}
    return value * scale.get(unit, 1.0)


def summarize_kernel(k: Dict[str, object]) -> Dict[str, object]:
    stalls = {
        name[len(STALL_PREFIX):-len(STALL_SUFFIX)]: float(v)
        for name, v in k.items()
        if name.startswith(STALL_PREFIX) and name.endswith(STALL_SUFFIX) and isinstance(v, float)
    }
    issued = stalls.pop("selected", None)
    total = sum(stalls.values())
    shares = {r: v / total * 100.0 for r, v in sorted(stalls.items(), key=lambda kv: -kv[1])} if total else {}
    dur = _to_us(float(k["gpu__time_duration.sum"]), str(k.get("gpu__time_duration.sum [unit]", "")))
    rd = _to_bytes(float(k["dram__bytes_read.sum"]), str(k.get("dram__bytes_read.sum [unit]", "")))
    wr = _to_bytes(float(k["dram__bytes_write.sum"]), str(k.get("dram__bytes_write.sum [unit]", "")))
    return {
        "kernel": k.get("Kernel Name"),
        "duration_us": dur,
        "grid_size": k.get("launch__grid_size"),
        "block_size": k.get("launch__block_size"),
        "registers_per_thread": k.get("launch__registers_per_thread"),
        "theoretical_occupancy_pct": k.get("sm__maximum_warps_per_active_cycle_pct"),
        "achieved_occupancy_pct": k.get("sm__warps_active.avg.pct_of_peak_sustained_active"),
        "dram_read_bytes": rd,
        "dram_write_bytes": wr,
        "dram_throughput_pct_peak": k.get("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        "achieved_dram_gbs": (rd + wr) / (dur * 1e-6) / 1e9 if dur else None,
        "l1_hit_rate_pct": k.get("l1tex__t_sector_hit_rate.pct"),
        "l2_hit_rate_pct": k.get("lts__t_sector_hit_rate.pct"),
        "global_load_sector_efficiency_pct": k.get("smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct"),
        "global_store_sector_efficiency_pct": k.get("smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct"),
        "sm_throughput_pct_peak": k.get("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        "warps_stalled_per_issue": total,
        "issue_selected_per_issue": issued,
        "stall_share_pct": shares,
        "dominant_stall": next(iter(shares), None),
    }


def main() -> int:
    csvs = sorted(NCU_DIR.glob("*.csv"))
    if not csvs:
        print(f"No Nsight Compute CSV exports in {NCU_DIR}; run scripts/profile_ncu.sh first.")
        return 1
    meta_path = NCU_DIR / "meta.json"
    summary = {"meta": json.loads(meta_path.read_text()) if meta_path.exists() else {}, "targets": {}}
    for path in csvs:
        kernels = [summarize_kernel(k) for k in read_raw_csv(path)]
        summary["targets"][path.stem] = {
            "num_kernels": len(kernels),
            "total_duration_us": sum(k["duration_us"] for k in kernels),
            "kernels": kernels,
        }
        names = ", ".join(str(k["kernel"])[:40] for k in kernels)
        print(f"{path.stem:34s} {len(kernels)} kernel(s), "
              f"{summary['targets'][path.stem]['total_duration_us']:8.1f} us  [{names}]")
    out = REPO_ROOT / "results" / "ncu_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[saved] {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
