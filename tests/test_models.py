import pytest
import torch

from src.models.autoencoder import build_autoencoder
from src.models.yolo import YOLOV8N_SHA256, _sha256, resolve_yolov8n_weights


def test_autoencoder_is_seeded_and_shape_preserving():
    a, b = build_autoencoder(torch.device("cpu"), 0), build_autoencoder(torch.device("cpu"), 0)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    with torch.no_grad():
        assert a(torch.rand(1, 3, 256, 256)).shape == (1, 3, 256, 256)


def test_autoencoder_does_not_touch_global_rng():
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    build_autoencoder(torch.device("cpu"), 7)
    assert torch.equal(torch.rand(3), expected)


def test_yolo_weights_checksum():
    pytest.importorskip("ultralytics")
    try:
        path = resolve_yolov8n_weights()
    except Exception as exc:  # offline without cached weights
        pytest.skip(f"weights unavailable: {exc}")
    assert _sha256(path) == YOLOV8N_SHA256


@pytest.mark.cuda
def test_yolo_traces_correctly_under_torch_compile(yolo_model, detection_frame_640):
    """Regression test: without make_compile_safe, Dynamo mis-traces C2f/SPPF (boxes off by ~465 px)."""
    x = torch.from_numpy(detection_frame_640).cuda().permute(2, 0, 1).float().div(255).unsqueeze(0)
    with torch.no_grad():
        ref = yolo_model(x)
        traced = torch.compile(yolo_model, backend="aot_eager")(x)
    assert torch.equal(ref, traced)


@pytest.mark.cuda
def test_yolo_raw_output(yolo_model, detection_frame_640, yolo_raw):
    x = torch.from_numpy(detection_frame_640).cuda().permute(2, 0, 1).float().div(255).unsqueeze(0)
    with torch.inference_mode():
        out = yolo_model(x)
    assert out.shape == (1, 84, 8400)
    assert sum(p.numel() for p in yolo_model.parameters()) < 3.2e6
    # The fixture was produced on CPU in strict FP32; cuDNN uses TF32 convolutions by default,
    # so raw values differ slightly while the confident detections are the same.
    scores_gpu, scores_cpu = out[0, 4:].cpu(), yolo_raw[0, 4:]
    assert (scores_gpu - scores_cpu).abs().max() < 0.02
    confident = scores_cpu.amax(0) > 0.5
    assert confident.sum() > 0
    torch.testing.assert_close(out[0, :4].cpu()[:, confident], yolo_raw[0, :4][:, confident], rtol=0, atol=1.0)
