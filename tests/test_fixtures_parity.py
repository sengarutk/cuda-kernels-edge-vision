import os
import numpy as np
import pytest

def test_precomputed_fixtures_integrity():
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    expected_dir = os.path.join(fixtures_dir, "expected_outputs")
    
    img_256 = np.load(os.path.join(fixtures_dir, "image_256_uint8.npy"))
    exp_fp32 = np.load(os.path.join(expected_dir, "expected_preprocessed_256_fp32.npy"))
    exp_fp16 = np.load(os.path.join(expected_dir, "expected_preprocessed_256_fp16.npy"))
    
    assert img_256.shape == (256, 256, 3)
    assert exp_fp32.shape == (1, 3, 256, 256)
    assert exp_fp16.shape == (1, 3, 256, 256)
    assert exp_fp32.dtype == np.float32
    assert exp_fp16.dtype == np.float16
