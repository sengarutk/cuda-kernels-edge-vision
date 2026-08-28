import pytest
import torch
from src.baselines.preprocess_torch import preprocess_eager_gpu
from src.triton_kernels.fused_preprocess_failed import fused_preprocess_failed_2d

def test_failed_kernel_numerical_parity():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    img = torch.randint(0, 256, (256, 256, 3), dtype=torch.uint8, device=device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)

    out_ref = preprocess_eager_gpu(img, mean, std, dtype=torch.float32)
    out_failed = fused_preprocess_failed_2d(img, mean, std)

    max_abs_err = torch.max(torch.abs(out_ref - out_failed)).item()
    assert max_abs_err < 1e-5, f"Failed kernel output diverged from reference: {max_abs_err}"
