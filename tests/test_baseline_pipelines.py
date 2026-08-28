import pytest
import torch
from src.baselines.preprocess_torch import preprocess_eager_gpu
from src.baselines.postprocess_torch import postprocess_detection_torch
from src.pipelines.anomaly_pipeline import AnomalyInspectionPipeline
from src.pipelines.detection_pipeline import DetectionInspectionPipeline

@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    return "cuda:0"

def test_preprocess_eager_shape_and_range(cuda_device):
    img = torch.randint(0, 256, (256, 256, 3), dtype=torch.uint8, device=cuda_device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=cuda_device)
    std = torch.tensor([0.229, 0.224, 0.225], device=cuda_device)
    
    out = preprocess_eager_gpu(img, mean, std)
    assert out.shape == (1, 3, 256, 256)
    assert out.dtype == torch.float32
    assert out.is_contiguous()

def test_postprocess_detection_filtering(cuda_device):
    # Create synthetic predictions [1, 100, 84]
    preds = torch.zeros(1, 100, 84, device=cuda_device)
    # Give candidate 5 a high score in class 10 (index 4 + 10 = 14)
    preds[0, 5, 14] = 0.95
    
    boxes, scores, classes = postprocess_detection_torch(preds, conf_threshold=0.5)
    assert boxes.shape[0] == 1
    assert scores[0].item() == pytest.approx(0.95, abs=1e-4)
    assert classes[0].item() == 10

def test_anomaly_pipeline_e2e_step(cuda_device):
    pipe = AnomalyInspectionPipeline(device=cuda_device)
    host_img = torch.randint(0, 256, (256, 256, 3), dtype=torch.uint8)
    result = pipe.run_step(host_img)
    assert "anomaly_score" in result
    assert isinstance(result["anomaly_score"], float)

def test_detection_pipeline_e2e_step(cuda_device):
    pipe = DetectionInspectionPipeline(device=cuda_device)
    host_img = torch.randint(0, 256, (640, 640, 3), dtype=torch.uint8)
    result = pipe.run_step(host_img)
    assert "survivor_count" in result
    assert isinstance(result["survivor_count"], int)
