import pytest
import torch
import numpy as np
from src.pipelines.anomaly_pipeline import AnomalyInspectionPipeline
from src.pipelines.detection_pipeline import DetectionInspectionPipeline

@pytest.fixture
def device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def test_anomaly_pipeline_backend_equivalence(device):
    # Set deterministic weights
    torch.manual_seed(42)
    pipe_eager = AnomalyInspectionPipeline(device=device, backend="eager")
    torch.manual_seed(42)
    pipe_triton = AnomalyInspectionPipeline(device=device, backend="triton")

    host_img = torch.randint(0, 256, (256, 256, 3), dtype=torch.uint8)
    res_eager = pipe_eager.run_step(host_img)
    res_triton = pipe_triton.run_step(host_img)

    # Anomaly scores must match within strict tolerance
    assert res_eager["anomaly_score"] == pytest.approx(res_triton["anomaly_score"], abs=1e-4)
    # Residual maps must match
    torch.testing.assert_close(res_eager["residual_map"], res_triton["residual_map"], rtol=1e-4, atol=1e-4)

def test_detection_pipeline_backend_equivalence(device):
    torch.manual_seed(42)
    pipe_eager = DetectionInspectionPipeline(device=device, backend="eager", conf_threshold=0.25)
    torch.manual_seed(42)
    pipe_triton = DetectionInspectionPipeline(device=device, backend="triton_both", conf_threshold=0.25)

    host_img = torch.randint(0, 256, (640, 640, 3), dtype=torch.uint8)
    res_eager = pipe_eager.run_step(host_img)
    res_triton = pipe_triton.run_step(host_img)

    assert res_eager["num_detections"] == res_triton["num_detections"]
    if res_eager["num_detections"] > 0:
        torch.testing.assert_close(res_eager["boxes"], res_triton["boxes"], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(res_eager["scores"], res_triton["scores"], rtol=1e-4, atol=1e-4)
        assert torch.equal(res_eager["classes"], res_triton["classes"])
