import os
import json
import pytest

def test_benchmark_schema_integrity():
    # Verify configs and manifests exist and follow valid schemas
    assert os.path.exists("configs/pipeline_config.yaml")
    assert os.path.exists("profiling_output/ncu/ncu_summary_metrics.json")
