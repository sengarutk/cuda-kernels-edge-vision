import pytest
import torch
import numpy as np
from src.baselines.preprocess_torch import preprocess_eager_gpu
from src.triton_kernels.fused_preprocess import fused_preprocess_triton

@pytest.fixture
def device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

@pytest.mark.parametrize("shape", [
    (256, 256),
    (640, 640),
    (1280, 720),
    (384, 512),  # Non-power-of-two
    (100, 100),  # Small edge shape
])
def test_triton_fused_preprocess_correctness_fp32(shape, device):
    H, W = shape
    img = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8, device=device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)

    out_ref = preprocess_eager_gpu(img, mean, std, dtype=torch.float32)
    out_triton = fused_preprocess_triton(img, mean, std, dtype=torch.float32)

    assert out_triton.shape == (1, 3, H, W)
    assert out_triton.dtype == torch.float32
    assert out_triton.is_contiguous()

    # Numerical verification
    max_abs_err = torch.max(torch.abs(out_ref - out_triton)).item()
    mean_abs_err = torch.mean(torch.abs(out_ref - out_triton)).item()
    
    assert max_abs_err < 1e-5, f"Max abs error {max_abs_err} exceeded FP32 tolerance 1e-5"
    assert mean_abs_err < 1e-6, f"Mean abs error {mean_abs_err} exceeded 1e-6"

def test_triton_fused_preprocess_fp16(device):
    img = torch.randint(0, 256, (256, 256, 3), dtype=torch.uint8, device=device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)

    out_ref = preprocess_eager_gpu(img, mean, std, dtype=torch.float16)
    out_triton = fused_preprocess_triton(img, mean, std, dtype=torch.float16)

    assert out_triton.dtype == torch.float16
    max_abs_err = torch.max(torch.abs(out_ref - out_triton)).item()
    # FP16 arithmetic drift across 3 un-fused operations vs single-pass FMA is within 2-3 ULPs
    assert max_abs_err < 3e-3, f"Max abs error {max_abs_err} exceeded FP16 tolerance 3e-3"

def test_triton_boundary_values(device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)

    # All-zero image
    img_zeros = torch.zeros((256, 256, 3), dtype=torch.uint8, device=device)
    out_zeros = fused_preprocess_triton(img_zeros, mean, std)
    ref_zeros = preprocess_eager_gpu(img_zeros, mean, std)
    torch.testing.assert_close(out_zeros, ref_zeros, atol=1e-5, rtol=1e-4)

    # All-255 image
    img_255 = torch.full((256, 256, 3), 255, dtype=torch.uint8, device=device)
    out_255 = fused_preprocess_triton(img_255, mean, std)
    ref_255 = preprocess_eager_gpu(img_255, mean, std)
    torch.testing.assert_close(out_255, ref_255, atol=1e-5, rtol=1e-4)
