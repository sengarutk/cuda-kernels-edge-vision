import pytest
import torch
import numpy as np
from src.baselines.postprocess_numpy import postprocess_detection_numpy
from src.baselines.postprocess_torch import (
    postprocess_detection_torch_eager,
    postprocess_detection_torch_compiled,
)
from src.baselines.anomaly_baselines import compute_residual_map_torch, compute_anomaly_score_topk

def test_yolo_postprocess_numpy_vs_torch(fixtures):
    preds_t = fixtures["yolo_candidates"].cuda()
    preds_np = preds_t.cpu().numpy()
    conf_thresh = 0.25

    boxes_np, scores_np, classes_np = postprocess_detection_numpy(preds_np, conf_thresh)
    boxes_t, scores_t, classes_t = postprocess_detection_torch_eager(preds_t, conf_thresh)

    assert len(boxes_np) == boxes_t.shape[0]
    torch.testing.assert_close(boxes_t, torch.from_numpy(boxes_np).cuda(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(scores_t, torch.from_numpy(scores_np).cuda(), rtol=1e-5, atol=1e-5)
    assert np.array_equal(classes_np, classes_t.cpu().numpy())

def test_yolo_torch_compiled_parity(fixtures):
    preds_t = fixtures["yolo_candidates"].cuda()
    conf_thresh = 0.25

    b_eager, s_eager, c_eager = postprocess_detection_torch_eager(preds_t, conf_thresh)
    b_comp, s_comp, c_comp = postprocess_detection_torch_compiled(preds_t, conf_thresh)

    torch.testing.assert_close(b_eager, b_comp, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(s_eager, s_comp, rtol=1e-5, atol=1e-5)
    assert torch.equal(c_eager, c_comp)

def test_yolo_postprocess_3d_batch_tensor():
    # Test [1, N, 84] 3D batch shape handling
    preds_3d = torch.randn(1, 100, 84, device="cuda")
    b_eager, s_eager, c_eager = postprocess_detection_torch_eager(preds_3d, 0.25)
    b_comp, s_comp, c_comp = postprocess_detection_torch_compiled(preds_3d, 0.25)
    torch.testing.assert_close(b_eager, b_comp, rtol=1e-5, atol=1e-5)

def test_anomaly_residual_and_topk():
    x = torch.zeros(1, 3, 256, 256, device="cuda")
    x_recon = torch.zeros(1, 3, 256, 256, device="cuda")
    x[0, :, 100:110, 100:110] = 1.0  # 100 defect pixels

    res_map = compute_residual_map_torch(x, x_recon)
    assert res_map.shape == (1, 1, 256, 256)
    assert res_map.max() > 0.0

    score = compute_anomaly_score_topk(res_map, top_k_fraction=0.01)
    assert isinstance(score, float)
    assert score > 0.0
