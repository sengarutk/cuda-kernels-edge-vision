import pytest
import torch

from src.pipelines.anomaly_pipeline import AnomalyInspectionPipeline
from src.pipelines.detection_pipeline import DetectionInspectionPipeline

pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module")
def host_256(img_256):
    return torch.from_numpy(img_256).pin_memory()


@pytest.fixture(scope="module")
def host_640(detection_frame_640):
    return torch.from_numpy(detection_frame_640).pin_memory()


def test_anomaly_backends_agree(host_256):
    results = {}
    for pre in ("eager", "compiled", "triton"):
        out = AnomalyInspectionPipeline(preprocess=pre, seed=0).run_step(host_256, use_nvtx=pre == "triton")
        assert isinstance(out["anomaly_score"], float)
        assert out["residual_map"].shape == (1, 1, 256, 256)
        results[pre] = out
    for pre in ("compiled", "triton"):
        assert results[pre]["anomaly_score"] == pytest.approx(results["eager"]["anomaly_score"], abs=1e-5)
        torch.testing.assert_close(results[pre]["residual_map"], results["eager"]["residual_map"],
                                   rtol=1e-4, atol=1e-5)


def test_anomaly_compiled_model_matches_eager(host_256):
    a = AnomalyInspectionPipeline(preprocess="triton", seed=0).run_step(host_256)
    b = AnomalyInspectionPipeline(preprocess="triton", compile_model=True, seed=0).run_step(host_256)
    assert a["anomaly_score"] == pytest.approx(b["anomaly_score"], abs=1e-4)


def test_anomaly_seed_controls_weights(host_256):
    a = AnomalyInspectionPipeline(seed=0).run_step(host_256)["anomaly_score"]
    b = AnomalyInspectionPipeline(seed=0).run_step(host_256)["anomaly_score"]
    c = AnomalyInspectionPipeline(seed=1).run_step(host_256)["anomaly_score"]
    assert a == b and a != c


@pytest.mark.parametrize("pre,post", [("eager", "eager"), ("triton", "eager"), ("eager", "triton"),
                                      ("triton", "triton"), ("compiled", "compiled")])
def test_detection_backends_find_the_same_objects(yolo_model, host_640, pre, post):
    out = DetectionInspectionPipeline(preprocess=pre, postprocess=post, model=yolo_model).run_step(
        host_640, use_nvtx=True)
    ref = DetectionInspectionPipeline(preprocess="eager", postprocess="eager", model=yolo_model).run_step(host_640)
    assert out["num_candidates"] == ref["num_candidates"] > out["num_detections"] == ref["num_detections"] > 0
    order_a = torch.argsort(out["scores"], descending=True)
    order_b = torch.argsort(ref["scores"], descending=True)
    torch.testing.assert_close(out["boxes"][order_a], ref["boxes"][order_b], rtol=1e-4, atol=1e-2)
    assert torch.equal(out["classes"][order_a], ref["classes"][order_b])


def test_detection_compiled_model(yolo_model, host_640):
    out = DetectionInspectionPipeline(compile_model=True, model=yolo_model).run_step(host_640)
    assert out["num_detections"] == 5


def test_detection_no_survivors(yolo_model, host_640):
    out = DetectionInspectionPipeline(conf_threshold=1.01, model=yolo_model).run_step(host_640)
    assert out["num_candidates"] == out["num_detections"] == 0
    assert out["boxes"].shape == (0, 4)


@pytest.mark.parametrize("pre", ["eager", "triton"])
def test_anomaly_cuda_graph_matches_and_tracks_new_frames(host_256, pre):
    graphed = AnomalyInspectionPipeline(preprocess=pre, cuda_graph=True, seed=0)
    plain = AnomalyInspectionPipeline(preprocess=pre, seed=0)
    other = torch.flip(host_256, dims=[0]).contiguous().pin_memory()
    for frame in (host_256, other, host_256):
        a, b = graphed.run_step(frame), plain.run_step(frame)
        assert a["anomaly_score"] == pytest.approx(b["anomaly_score"], abs=1e-5)


@pytest.mark.parametrize("backend", ["eager", "triton"])
def test_detection_cuda_graph_matches(yolo_model, host_640, backend):
    graphed = DetectionInspectionPipeline(preprocess=backend, postprocess=backend, cuda_graph=True,
                                          model=yolo_model)
    plain = DetectionInspectionPipeline(preprocess=backend, postprocess=backend, model=yolo_model)
    for _ in range(2):
        a, b = graphed.run_step(host_640), plain.run_step(host_640)
        assert a["num_candidates"] == b["num_candidates"] and a["num_detections"] == b["num_detections"] == 5
        torch.testing.assert_close(torch.sort(a["scores"])[0], torch.sort(b["scores"])[0], rtol=0, atol=1e-5)


def test_cuda_graph_and_compile_are_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        AnomalyInspectionPipeline(cuda_graph=True, compile_model=True)


def test_invalid_backend_rejected():
    with pytest.raises(ValueError, match="preprocess"):
        AnomalyInspectionPipeline(preprocess="cuda")
    with pytest.raises(ValueError, match="postprocess"):
        DetectionInspectionPipeline(postprocess="numpy", model=torch.nn.Identity())
