import os
import pytest
import numpy as np
import torch

@pytest.fixture(scope="session")
def fixtures():
    fixture_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    img_256_path = os.path.join(fixture_dir, "image_256_uint8.npy")
    img_640_path = os.path.join(fixture_dir, "image_640_uint8.npy")
    cand_path = os.path.join(fixture_dir, "yolo_candidates_8400.pt")

    if not (os.path.exists(img_256_path) and os.path.exists(img_640_path) and os.path.exists(cand_path)):
        from tests.fixtures.generate_fixtures import generate_fixtures
        generate_fixtures()

    img_256 = np.load(img_256_path)
    img_640 = np.load(img_640_path)
    yolo_candidates = torch.load(cand_path, weights_only=True)

    return {
        "img_256": img_256,
        "img_640": img_640,
        "yolo_candidates": yolo_candidates,
    }
