"""Regenerates the committed test fixtures (deterministic).

    python tests/fixtures/generate_fixtures.py          # images + expected outputs
    python tests/fixtures/generate_fixtures.py --yolo   # also the YOLOv8n raw-output fixture
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch

from src.baselines.preprocess_numpy import preprocess_numpy_cpu

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))
EXPECTED_DIR = os.path.join(FIXTURES_DIR, "expected_outputs")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def make_anomaly_image() -> np.ndarray:
    """256x256 gradient texture with a saturated horizontal scratch defect."""
    y, x = np.mgrid[0:256, 0:256]
    img = np.stack([(x + y) % 256, (2 * x) % 256, (3 * y) % 256], axis=-1).astype(np.uint8)
    img[120:135, 50:200, :] = 255
    return img


def generate_images() -> None:
    os.makedirs(EXPECTED_DIR, exist_ok=True)
    img_256 = make_anomaly_image()
    np.save(os.path.join(FIXTURES_DIR, "image_256_uint8.npy"), img_256)
    np.random.seed(42)
    img_640 = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    np.save(os.path.join(FIXTURES_DIR, "image_640_uint8.npy"), img_640)
    for dtype, tag in ((np.float32, "fp32"), (np.float16, "fp16")):
        expected = preprocess_numpy_cpu(img_256, MEAN, STD, dtype=dtype)
        np.save(os.path.join(EXPECTED_DIR, f"expected_preprocessed_256_{tag}.npy"), expected)
    print("[saved] image_256_uint8.npy, image_640_uint8.npy, expected_outputs/*.npy")


def generate_yolo() -> None:
    """Raw YOLOv8n head output ([1, 84, 8400]) for the letterboxed bus.jpg frame, on CPU."""
    from src.frames import detection_frame
    from src.models.yolo import load_yolov8n

    model = load_yolov8n(torch.device("cpu"))
    frame = torch.from_numpy(detection_frame(640))
    x = frame.permute(2, 0, 1).float().div(255.0).unsqueeze(0).contiguous()
    with torch.inference_mode():
        preds = model(x).clone()
    torch.save(preds, os.path.join(FIXTURES_DIR, "yolov8n_bus_raw_640.pt"))
    print(f"[saved] yolov8n_bus_raw_640.pt {tuple(preds.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yolo", action="store_true", help="also regenerate the YOLOv8n fixture")
    args = parser.parse_args()
    generate_images()
    if args.yolo:
        generate_yolo()
