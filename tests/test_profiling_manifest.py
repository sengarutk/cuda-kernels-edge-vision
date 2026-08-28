import os
import json
import pytest

def test_ncu_metrics_manifest_integrity():
    manifest_path = "profiling_output/ncu/ncu_summary_metrics.json"
    if not os.path.exists(manifest_path):
        from profiling.parse_ncu_metrics import generate_ncu_summary_report
        generate_ncu_summary_report()

    assert os.path.exists(manifest_path), "NCU metrics manifest must exist!"

    with open(manifest_path, "r") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) >= 3

    required_keys = [
        "Kernel Name", "Duration (µs)", "Achieved Occupancy (%)",
        "Registers / Thread", "DRAM Throughput (%)", "L1 Cache Hit Rate (%)",
        "Dominant Warp Stall"
    ]

    for item in data:
        for key in required_keys:
            assert key in item, f"Missing key '{key}' in NCU manifest item {item}"
        assert item["Achieved Occupancy (%)"] > 0
        assert item["Registers / Thread"] > 0
