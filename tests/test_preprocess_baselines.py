import pytest
import torch
import numpy as np
from src.baselines.preprocess_numpy import preprocess_numpy_cpu
from src.baselines.preprocess_torch import preprocess_eager_gpu, preprocess_compiled_gpu

def test_numpy_vs_torch_eager_fp32(fixtures):
    img_np = fixtures["img_640"]
    mean_np = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std_np = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_t = torch.from_numpy(img_np).cuda()
    mean_t = torch.tensor([0.485, 0.456, 0.406], device="cuda")
    std_t = torch.tensor([0.229, 0.224, 0.225], device="cuda")

    out_np = preprocess_numpy_cpu(img_np, mean_np, std_np, dtype=np.float32)
    out_torch = preprocess_eager_gpu(img_t, mean_t, std_t, dtype=torch.float32)

    out_np_t = torch.from_numpy(out_np).cuda()
    torch.testing.assert_close(out_torch, out_np_t, rtol=1e-5, atol=1e-5)

def test_torch_eager_vs_compiled_fp32(fixtures):
    img_t = torch.from_numpy(fixtures["img_640"]).cuda()
    mean_t = torch.tensor([0.485, 0.456, 0.406], device="cuda")
    std_t = torch.tensor([0.229, 0.224, 0.225], device="cuda")

    out_eager = preprocess_eager_gpu(img_t, mean_t, std_t, dtype=torch.float32)
    out_compiled = preprocess_compiled_gpu(img_t, mean_t, std_t, dtype=torch.float32)
    torch.testing.assert_close(out_eager, out_compiled, rtol=1e-5, atol=1e-5)

def test_fp16_preprocessing_tolerance(fixtures):
    img_t = torch.from_numpy(fixtures["img_640"]).cuda()
    mean_t = torch.tensor([0.485, 0.456, 0.406], device="cuda", dtype=torch.float16)
    std_t = torch.tensor([0.229, 0.224, 0.225], device="cuda", dtype=torch.float16)

    out_fp32 = preprocess_eager_gpu(img_t, mean_t.float(), std_t.float(), dtype=torch.float32)
    out_fp16 = preprocess_eager_gpu(img_t, mean_t, std_t, dtype=torch.float16)
    torch.testing.assert_close(out_fp16.float(), out_fp32, rtol=2e-3, atol=2e-3)

def test_preprocess_input_validation_errors():
    mean_np = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std_np = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    mean_t = torch.tensor([0.485, 0.456, 0.406], device="cuda")
    std_t = torch.tensor([0.229, 0.224, 0.225], device="cuda")

    # 1. Non-3D shape validation
    bad_2d_np = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(ValueError, match="Expected"):
        preprocess_numpy_cpu(bad_2d_np, mean_np, std_np)

    bad_2d_t = torch.zeros((64, 64), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="Expected"):
        preprocess_eager_gpu(bad_2d_t, mean_t, std_t)

    # 2. Non-uint8 dtype validation
    bad_float_np = np.zeros((64, 64, 3), dtype=np.float32)
    with pytest.raises(TypeError, match="Expected uint8"):
        preprocess_numpy_cpu(bad_float_np, mean_np, std_np)

    bad_float_t = torch.zeros((64, 64, 3), dtype=torch.float32, device="cuda")
    with pytest.raises(TypeError, match="Expected uint8"):
        preprocess_eager_gpu(bad_float_t, mean_t, std_t)

    # 3. CPU tensor passed to GPU preprocessor
    cpu_t = torch.zeros((64, 64, 3), dtype=torch.uint8, device="cpu")
    with pytest.raises(ValueError, match="must be on CUDA"):
        preprocess_eager_gpu(cpu_t, mean_t, std_t)
