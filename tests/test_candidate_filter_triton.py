import os
import pytest
import numpy as np
import torch
from src.baselines.postprocess_torch import postprocess_detection_torch_eager
from src.triton_kernels.candidate_filter import candidate_filter_triton

@pytest.fixture
def device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def test_random_distributions_parity(device):
    torch.manual_seed(123)
    for _ in range(10):
        N = np.random.choice([1000, 8400, 10000])
        C = 80
        preds = torch.randn(1, N, 4 + C, device=device)
        conf_thresh = float(np.random.uniform(0.1, 0.7))

        boxes_e, scores_e, classes_e = postprocess_detection_torch_eager(preds, conf_thresh)
        boxes_t, scores_t, classes_t, mask_t = candidate_filter_triton(preds, conf_thresh)

        assert boxes_e.shape == boxes_t.shape
        assert scores_e.shape == scores_t.shape
        assert classes_e.shape == classes_t.shape

        torch.testing.assert_close(boxes_e, boxes_t, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(scores_e, scores_t, rtol=1e-5, atol=1e-5)
        assert torch.equal(classes_e, classes_t)

def test_exact_defect_proposals_fixtures(device):
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    tensor_path = os.path.join(fixtures_dir, "yolo_candidates_8400.pt")
    preds = torch.load(tensor_path, weights_only=True).to(device)

    boxes_e, scores_e, classes_e = postprocess_detection_torch_eager(preds, conf_threshold=0.25)
    boxes_t, scores_t, classes_t, mask_t = candidate_filter_triton(preds, conf_threshold=0.25)

    # Must isolate the exact 12 synthetic defect proposals
    assert boxes_t.shape[0] == 12
    assert scores_t.shape[0] == 12
    assert classes_t.shape[0] == 12

    torch.testing.assert_close(boxes_e, boxes_t, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(scores_e, scores_t, rtol=1e-5, atol=1e-5)
    assert torch.equal(classes_e, classes_t)

@pytest.mark.parametrize("thresh", [0.0, 0.5, 0.99, 1.0])
def test_threshold_boundaries(thresh, device):
    preds = torch.rand(1, 1000, 84, device=device)
    boxes_e, scores_e, classes_e = postprocess_detection_torch_eager(preds, thresh)
    boxes_t, scores_t, classes_t, mask_t = candidate_filter_triton(preds, thresh)

    assert boxes_e.shape[0] == boxes_t.shape[0]
    torch.testing.assert_close(scores_e, scores_t, rtol=1e-5, atol=1e-5)
    assert torch.equal(classes_e, classes_t)
