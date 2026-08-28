import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
from src.baselines.preprocess_torch import preprocess_eager_gpu
from src.baselines.preprocess_numpy import preprocess_numpy_cpu

def generate_all_fixtures():
    fixtures_dir = os.path.dirname(__file__)
    expected_dir = os.path.join(fixtures_dir, "expected_outputs")
    os.makedirs(expected_dir, exist_ok=True)
    
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Generate 256x256 uint8 Anomaly Inspection Image (with structured patterns & defects)
    img_256 = np.zeros((256, 256, 3), dtype=np.uint8)
    # Background texture gradient
    for y in range(256):
        for x in range(256):
            img_256[y, x, 0] = (x + y) % 256
            img_256[y, x, 1] = (2 * x) % 256
            img_256[y, x, 2] = (3 * y) % 256
    # Artificial scratch defect
    img_256[120:135, 50:200, :] = 255
    np.save(os.path.join(fixtures_dir, "image_256_uint8.npy"), img_256)
    print("[✓] Generated image_256_uint8.npy")

    # 2. Generate 640x640 uint8 Detection Image
    img_640 = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    np.save(os.path.join(fixtures_dir, "image_640_uint8.npy"), img_640)
    print("[✓] Generated image_640_uint8.npy")

    # 3. Generate Realistic YOLO Detection Candidates [1, 8400, 84]
    # In industrial inspection, 99.8% of boxes are background (scores < 0.05), with ~10-15 true defects
    raw_candidates = np.random.uniform(0.001, 0.04, (1, 8400, 84)).astype(np.float32)
    # Box coordinates [cx, cy, w, h] in [0, 640]
    raw_candidates[0, :, :4] = np.random.uniform(10.0, 630.0, (8400, 4))
    
    # Inject 12 high-confidence defect boxes
    defect_indices = [42, 105, 512, 1024, 2048, 3100, 4200, 5500, 6200, 7100, 7800, 8350]
    for idx in defect_indices:
        target_class = (idx % 80)
        raw_candidates[0, idx, 4 + target_class] = 0.88 + (idx % 10) * 0.01

    yolo_tensor = torch.from_numpy(raw_candidates)
    torch.save(yolo_tensor, os.path.join(fixtures_dir, "yolo_candidates_8400.pt"))
    print("[✓] Generated yolo_candidates_8400.pt (with 12 validated defect proposals)")

    # 4. Generate Pre-computed Expected Outputs (FP32 & FP16)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    expected_fp32 = preprocess_numpy_cpu(img_256, mean, std, dtype=np.float32)
    np.save(os.path.join(expected_dir, "expected_preprocessed_256_fp32.npy"), expected_fp32)
    
    expected_fp16 = preprocess_numpy_cpu(img_256, mean, std, dtype=np.float16)
    np.save(os.path.join(expected_dir, "expected_preprocessed_256_fp16.npy"), expected_fp16)
    print("[✓] Generated reference outputs in expected_outputs/")

if __name__ == "__main__":
    generate_all_fixtures()


# Export alias for test suites and conftest
generate_fixtures = generate_all_fixtures
