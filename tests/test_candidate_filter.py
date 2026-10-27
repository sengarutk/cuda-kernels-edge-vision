import numpy as np
import pytest
import torch
import torchvision.ops as ops

from src.baselines.postprocess_numpy import postprocess_detection_numpy
from src.baselines.postprocess_torch import (
    postprocess_detection_torch_compiled, postprocess_detection_torch_eager,
)
from src.triton_kernels.candidate_filter import candidate_filter_triton


def _sorted(boxes, scores, classes, indices):
    order = torch.argsort(indices)
    return boxes[order], scores[order], classes[order], indices[order]


def _assert_same(a, b):
    torch.testing.assert_close(a[0], b[0], rtol=1e-6, atol=1e-5)
    torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
    assert torch.equal(a[2], b[2]) and torch.equal(a[3], b[3])


def test_numpy_reference_on_real_yolo_output(yolo_raw):
    boxes, scores, classes, idx = postprocess_detection_numpy(yolo_raw.numpy(), 0.25)
    assert len(idx) == 46 and set(classes.tolist()) == {0, 5}  # persons and a bus
    assert np.all(boxes[:, 2] > boxes[:, 0]) and np.all(boxes[:, 3] > boxes[:, 1])
    assert np.all(scores >= 0.25)


@pytest.mark.cuda
@pytest.mark.parametrize("layout", ["channels_first", "channels_last"])
def test_triton_matches_eager_and_numpy_on_real_output(yolo_raw, layout):
    preds = yolo_raw.cuda()
    if layout == "channels_last":
        preds = preds.transpose(1, 2).contiguous()
    ref = postprocess_detection_torch_eager(preds, 0.25, layout=layout)
    out = _sorted(*candidate_filter_triton(preds, 0.25, layout=layout))
    _assert_same(out, ref)
    nb, ns, nc, ni = postprocess_detection_numpy(preds.cpu().numpy(), 0.25, layout=layout)
    np.testing.assert_allclose(ref[0].cpu().numpy(), nb, rtol=1e-6, atol=1e-5)
    np.testing.assert_array_equal(ref[1].cpu().numpy(), ns)
    np.testing.assert_array_equal(ref[2].cpu().numpy(), nc)
    np.testing.assert_array_equal(ref[3].cpu().numpy(), ni)


@pytest.mark.cuda
def test_compiled_matches_eager(yolo_raw):
    preds = yolo_raw.cuda()
    _assert_same(postprocess_detection_torch_compiled(preds, 0.25), postprocess_detection_torch_eager(preds, 0.25))


@pytest.mark.cuda
@pytest.mark.parametrize("n", [1, 100, 8400, 33600])
@pytest.mark.parametrize("num_classes", [1, 3, 80])
def test_random_inputs_match_eager(n, num_classes):
    g = torch.Generator(device="cuda").manual_seed(n * 100 + num_classes)
    preds = torch.rand(1, 4 + num_classes, n, device="cuda", generator=g)
    ref = postprocess_detection_torch_eager(preds, 0.6)
    _assert_same(_sorted(*candidate_filter_triton(preds, 0.6)), ref)


@pytest.mark.cuda
@pytest.mark.parametrize("thresh", [0.0, 0.5, 1.0, 2.0])
def test_threshold_boundaries(thresh):
    preds = torch.rand(1, 84, 1000, device="cuda")
    preds[0, 4, :10] = 1.0  # exact hits on the >= boundary at thresh == 1.0
    ref = postprocess_detection_torch_eager(preds, thresh)
    out = _sorted(*candidate_filter_triton(preds, thresh))
    _assert_same(out, ref)
    expected = {0.0: 1000, 1.0: 10, 2.0: 0}
    if thresh in expected:
        assert out[0].shape == (expected[thresh], 4)


@pytest.mark.cuda
def test_xyxy_passthrough_and_half_precision_input(yolo_raw):
    preds = yolo_raw.cuda().half()
    ref = postprocess_detection_torch_eager(preds, 0.25, box_format="xyxy")
    _assert_same(_sorted(*candidate_filter_triton(preds, 0.25, box_format="xyxy")), ref)


@pytest.mark.cuda
def test_nms_results_identical(yolo_raw):
    preds = yolo_raw.cuda()
    finals = []
    for boxes, scores, classes, _ in (postprocess_detection_torch_eager(preds, 0.25),
                                      candidate_filter_triton(preds, 0.25)):
        keep = ops.batched_nms(boxes, scores, classes, 0.45)
        order = torch.argsort(scores[keep], descending=True, stable=True)
        finals.append((boxes[keep][order], scores[keep][order], classes[keep][order]))
    assert finals[0][0].shape[0] == 5
    for a, b in zip(*finals):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-5)


@pytest.mark.cuda
def test_input_validation():
    with pytest.raises(ValueError, match="CUDA"):
        candidate_filter_triton(torch.rand(1, 84, 10))
    with pytest.raises(ValueError, match="batch size 1"):
        candidate_filter_triton(torch.rand(2, 84, 10, device="cuda"))
    with pytest.raises(ValueError, match="at least one class"):
        candidate_filter_triton(torch.rand(1, 4, 10, device="cuda"))
    with pytest.raises(ValueError, match="layout"):
        candidate_filter_triton(torch.rand(1, 84, 10, device="cuda"), layout="nchw")
