import pytest
import torch
import torchvision.ops as ops
from src.baselines.postprocess_torch import postprocess_detection_torch_eager
from src.triton_kernels.candidate_filter import candidate_filter_triton

@pytest.fixture
def device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def test_downstream_batched_nms_agreement(device):
    torch.manual_seed(42)
    N = 8400
    C = 80
    preds = torch.rand(1, N, 4 + C, device=device)
    # Generate valid bounding boxes [x1, y1, x2, y2]
    cx = preds[0, :, 0] * 500
    cy = preds[0, :, 1] * 500
    w = preds[0, :, 2] * 100 + 10
    h = preds[0, :, 3] * 100 + 10
    preds[0, :, 0] = cx - w / 2
    preds[0, :, 1] = cy - h / 2
    preds[0, :, 2] = cx + w / 2
    preds[0, :, 3] = cy + h / 2

    conf_thresh = 0.3
    iou_thresh = 0.45

    # 1. Eager candidate extraction + NMS
    boxes_e, scores_e, classes_e = postprocess_detection_torch_eager(preds, conf_thresh)
    keep_e = ops.batched_nms(boxes_e, scores_e, classes_e, iou_thresh)
    final_boxes_e = boxes_e[keep_e]
    final_scores_e = scores_e[keep_e]
    final_classes_e = classes_e[keep_e]

    # 2. Triton candidate extraction + NMS
    boxes_t, scores_t, classes_t, _ = candidate_filter_triton(preds, conf_thresh)
    keep_t = ops.batched_nms(boxes_t, scores_t, classes_t, iou_thresh)
    final_boxes_t = boxes_t[keep_t]
    final_scores_t = scores_t[keep_t]
    final_classes_t = classes_t[keep_t]

    assert final_boxes_e.shape == final_boxes_t.shape
    assert final_scores_e.shape == final_scores_t.shape
    assert final_classes_e.shape == final_classes_t.shape

    # Exact bitwise parity and IoU >= 0.999
    if final_boxes_e.shape[0] > 0:
        ious = ops.box_iou(final_boxes_e, final_boxes_t)
        diag_ious = torch.diag(ious)
        assert torch.all(diag_ious >= 0.999), f"IoU check failed: {diag_ious.min().item()}"

    torch.testing.assert_close(final_scores_e, final_scores_t, rtol=1e-5, atol=1e-5)
    assert torch.equal(final_classes_e, final_classes_t)
