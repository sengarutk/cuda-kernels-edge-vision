import numpy as np
import pytest
import torch

from src.baselines.preprocess_numpy import preprocess_numpy_cpu
from src.baselines.preprocess_torch import preprocess_compiled_gpu, preprocess_eager_gpu
from tests.conftest import MEAN, STD


def test_numpy_matches_committed_expected_outputs(img_256, expected_256):
    np.testing.assert_array_equal(preprocess_numpy_cpu(img_256, MEAN, STD, np.float32), expected_256["fp32"])
    np.testing.assert_array_equal(preprocess_numpy_cpu(img_256, MEAN, STD, np.float16), expected_256["fp16"])


def test_numpy_validation_errors():
    with pytest.raises(ValueError, match="Expected"):
        preprocess_numpy_cpu(np.zeros((64, 64), np.uint8))
    with pytest.raises(TypeError, match="Expected uint8"):
        preprocess_numpy_cpu(np.zeros((64, 64, 3), np.float32))


@pytest.mark.cuda
def test_eager_matches_numpy(img_640):
    mean_t, std_t = torch.tensor(MEAN, device="cuda"), torch.tensor(STD, device="cuda")
    out = preprocess_eager_gpu(torch.from_numpy(img_640).cuda(), mean_t, std_t)
    assert out.shape == (1, 3, 640, 640) and out.is_contiguous()
    ref = torch.from_numpy(preprocess_numpy_cpu(img_640, MEAN, STD)).cuda()
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.cuda
def test_compiled_matches_eager(img_640):
    mean_t, std_t = torch.tensor(MEAN, device="cuda"), torch.tensor(STD, device="cuda")
    img = torch.from_numpy(img_640).cuda()
    torch.testing.assert_close(preprocess_compiled_gpu(img, mean_t, std_t),
                               preprocess_eager_gpu(img, mean_t, std_t), rtol=1e-5, atol=1e-5)


@pytest.mark.cuda
def test_eager_fp16_within_half_precision(img_640):
    mean_t, std_t = torch.tensor(MEAN, device="cuda"), torch.tensor(STD, device="cuda")
    img = torch.from_numpy(img_640).cuda()
    out16 = preprocess_eager_gpu(img, mean_t, std_t, torch.float16)
    out32 = preprocess_eager_gpu(img, mean_t, std_t)
    assert out16.dtype == torch.float16
    torch.testing.assert_close(out16.float(), out32, rtol=0, atol=4e-3)


@pytest.mark.cuda
def test_eager_validation_errors():
    mean_t, std_t = torch.tensor(MEAN, device="cuda"), torch.tensor(STD, device="cuda")
    with pytest.raises(ValueError, match="Expected"):
        preprocess_eager_gpu(torch.zeros((64, 64), dtype=torch.uint8, device="cuda"), mean_t, std_t)
    with pytest.raises(TypeError, match="Expected uint8"):
        preprocess_eager_gpu(torch.zeros((64, 64, 3), device="cuda"), mean_t, std_t)
    with pytest.raises(ValueError, match="must be on CUDA"):
        preprocess_eager_gpu(torch.zeros((64, 64, 3), dtype=torch.uint8), mean_t, std_t)
