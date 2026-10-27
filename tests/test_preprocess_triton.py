import numpy as np
import pytest
import torch

from src.baselines.preprocess_torch import preprocess_eager_gpu
from src.triton_kernels.fused_preprocess import fused_preprocess_triton
from src.triton_kernels.fused_preprocess_2d import fused_preprocess_2d
from tests.conftest import MEAN, STD

pytestmark = pytest.mark.cuda

SHAPES = [(256, 256), (640, 640), (720, 1280), (1080, 1920), (384, 512), (100, 100), (1, 1), (17, 33)]


def _ref(img, dtype=torch.float32):
    return preprocess_eager_gpu(img, torch.tensor(MEAN, device="cuda"), torch.tensor(STD, device="cuda"), dtype)


@pytest.mark.parametrize("kernel", [fused_preprocess_triton, fused_preprocess_2d], ids=["1d", "2d"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_fp32_matches_eager(kernel, shape):
    torch.manual_seed(0)
    img = torch.randint(0, 256, (*shape, 3), dtype=torch.uint8, device="cuda")
    out = kernel(img, MEAN, STD)
    assert out.shape == (1, 3, *shape) and out.dtype == torch.float32 and out.is_contiguous()
    torch.testing.assert_close(out, _ref(img), rtol=0, atol=2e-6)


def test_matches_committed_numpy_reference(img_256, expected_256):
    out = fused_preprocess_triton(torch.from_numpy(img_256).cuda(), MEAN, STD)
    np.testing.assert_allclose(out.cpu().numpy(), expected_256["fp32"], rtol=0, atol=2e-6)


@pytest.mark.parametrize("kernel", [fused_preprocess_triton, fused_preprocess_2d], ids=["1d", "2d"])
def test_tuple_list_and_tensor_params_agree(kernel):
    img = torch.randint(0, 256, (64, 48, 3), dtype=torch.uint8, device="cuda")
    assert torch.equal(kernel(img, MEAN, STD), kernel(img, list(MEAN), list(STD)))
    # A tensor holds float32-rounded constants, so compare against those exact values.
    mean_t, std_t = torch.tensor(MEAN, device="cuda"), torch.tensor(STD, device="cuda")
    assert torch.equal(kernel(img, mean_t, std_t), kernel(img, tuple(mean_t.tolist()), tuple(std_t.tolist())))


def test_fp16_output_rounds_fp32_result():
    img = torch.randint(0, 256, (256, 256, 3), dtype=torch.uint8, device="cuda")
    out16 = fused_preprocess_triton(img, MEAN, STD, torch.float16)
    assert out16.dtype == torch.float16
    # Triton computes in FP32 and rounds once at the store: exactly the FP32 result cast to FP16.
    assert torch.equal(out16, fused_preprocess_triton(img, MEAN, STD).half())
    # Eager FP16 rounds after every op; |values| < 4, so one FP16 ULP is at most 2^-9.
    torch.testing.assert_close(out16.float(), _ref(img, torch.float16).float(), rtol=0, atol=2 * 2**-9)


def test_non_contiguous_input():
    base = torch.randint(0, 256, (128, 96, 3), dtype=torch.uint8, device="cuda")
    img = base.transpose(0, 1)
    assert not img.is_contiguous()
    torch.testing.assert_close(fused_preprocess_triton(img, MEAN, STD), _ref(img.contiguous()), rtol=0, atol=2e-6)


@pytest.mark.parametrize("value", [0, 255])
def test_constant_images(value):
    img = torch.full((256, 256, 3), value, dtype=torch.uint8, device="cuda")
    torch.testing.assert_close(fused_preprocess_triton(img, MEAN, STD), _ref(img), rtol=0, atol=2e-6)


def test_input_validation():
    with pytest.raises(ValueError, match="CUDA"):
        fused_preprocess_triton(torch.zeros((4, 4, 3), dtype=torch.uint8))
    with pytest.raises(TypeError, match="uint8"):
        fused_preprocess_triton(torch.zeros((4, 4, 3), device="cuda"))
    with pytest.raises(ValueError, match=r"\[H, W, 3\]"):
        fused_preprocess_triton(torch.zeros((4, 4, 4), dtype=torch.uint8, device="cuda"))
    with pytest.raises(TypeError, match="float32 or float16"):
        fused_preprocess_triton(torch.zeros((4, 4, 3), dtype=torch.uint8, device="cuda"), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"\[H, W, 3\]"):
        fused_preprocess_2d(torch.zeros((4, 4), dtype=torch.uint8, device="cuda"))
