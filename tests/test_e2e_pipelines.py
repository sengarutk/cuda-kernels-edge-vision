import pytest
import torch
from src.pipelines.anomaly_pipeline import AnomalyInspectionPipeline
from src.pipelines.detection_pipeline import DetectionInspectionPipeline

@pytest.fixture
def device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

@pytest.mark.parametrize("backend", ["eager", "compiled", "triton"])
@pytest.mark.parametrize("use_nvtx", [False, True])
def test_anomaly_pipeline_execution(backend, use_nvtx, device):
    pipe = AnomalyInspectionPipeline(device=device, backend=backend)
    host_img = torch.randint(0, 256, (256, 256, 3), dtype=torch.uint8).pin_memory()
    res = pipe.run_step(host_img, use_nvtx=use_nvtx)
    
    assert "anomaly_score" in res
    assert isinstance(res["anomaly_score"], float)
    assert res["residual_map"].shape == (1, 1, 256, 256)

@pytest.mark.parametrize("backend", ["eager", "compiled", "triton_preprocess", "triton_postprocess", "triton_both"])
@pytest.mark.parametrize("use_nvtx", [False, True])
def test_detection_pipeline_execution(backend, use_nvtx, device):
    pipe = DetectionInspectionPipeline(device=device, backend=backend)
    host_img = torch.randint(0, 256, (640, 640, 3), dtype=torch.uint8).pin_memory()
    res = pipe.run_step(host_img, use_nvtx=use_nvtx)
    
    assert "num_detections" in res
    assert isinstance(res["num_detections"], int)

@pytest.mark.parametrize("backend", ["eager", "compiled", "triton_both"])
def test_detection_pipeline_zero_detections_branch(backend, device):
    # Test high threshold resulting in 0 candidate boxes to exercise empty NMS branches
    pipe = DetectionInspectionPipeline(device=device, backend=backend, conf_threshold=999.0)
    host_img = torch.randint(0, 256, (640, 640, 3), dtype=torch.uint8).pin_memory()
    res = pipe.run_step(host_img)
    
    assert res["num_detections"] == 0
    assert res["boxes"].shape[0] == 0
    assert res["scores"].shape[0] == 0
    assert res["classes"].shape[0] == 0
