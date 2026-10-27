import os

import numpy as np
import pytest
import torch

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def pytest_collection_modifyitems(config, items):
    """Skips tests marked ``cuda`` when no GPU is present (CPU-only CI runs the rest)."""
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="CUDA GPU required")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)


def _has_yolo() -> bool:
    try:
        import ultralytics  # noqa: F401
        from src.models.yolo import resolve_yolov8n_weights

        resolve_yolov8n_weights()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def img_256() -> np.ndarray:
    return np.load(os.path.join(FIXTURES_DIR, "image_256_uint8.npy"))


@pytest.fixture(scope="session")
def img_640() -> np.ndarray:
    return np.load(os.path.join(FIXTURES_DIR, "image_640_uint8.npy"))


@pytest.fixture(scope="session")
def expected_256():
    d = os.path.join(FIXTURES_DIR, "expected_outputs")
    return {
        "fp32": np.load(os.path.join(d, "expected_preprocessed_256_fp32.npy")),
        "fp16": np.load(os.path.join(d, "expected_preprocessed_256_fp16.npy")),
    }


@pytest.fixture(scope="session")
def yolo_raw() -> torch.Tensor:
    """Real YOLOv8n head output [1, 84, 8400] for the letterboxed bus.jpg frame (CPU)."""
    return torch.load(os.path.join(FIXTURES_DIR, "yolov8n_bus_raw_640.pt"), weights_only=True)


@pytest.fixture(scope="session")
def yolo_model():
    if not _has_yolo():
        pytest.skip("ultralytics and yolov8n.pt weights required")
    from src.models.yolo import load_yolov8n

    return load_yolov8n(torch.device("cuda:0"))


@pytest.fixture(scope="session")
def detection_frame_640() -> np.ndarray:
    if not _has_yolo():
        pytest.skip("ultralytics required for the bus.jpg asset")
    from src.frames import detection_frame

    return detection_frame(640)
