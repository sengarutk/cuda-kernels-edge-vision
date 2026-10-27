import torch
import triton
import triton.language as tl

from ._params import ChannelParam, fold_normalization


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["num_pixels"],
)
@triton.jit
def _fused_preprocess_kernel(
    x_ptr,
    y_ptr,
    num_pixels,
    scale_r,
    scale_g,
    scale_b,
    bias_r,
    bias_g,
    bias_b,
    IS_FP16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused HWC uint8 -> normalized planar CHW float32/fp16 kernel.

    Each program handles BLOCK_SIZE consecutive pixels:
      Input:  interleaved RGB [R0, G0, B0, R1, G1, B1, ...] (uint8, stride-3 byte loads)
      Output: planar [R0, R1, ..., G0, G1, ..., B0, B1, ...] (contiguous stores per plane)
    Normalization ``(x / 255 - mean) / std`` is folded into one FMA per element:
    ``x * scale + bias`` with ``scale = 1 / (255 * std)`` and ``bias = -mean / std``.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_pixels

    in_ptr = x_ptr + offsets * 3
    r = tl.load(in_ptr + 0, mask=mask, other=0).to(tl.float32)
    g = tl.load(in_ptr + 1, mask=mask, other=0).to(tl.float32)
    b = tl.load(in_ptr + 2, mask=mask, other=0).to(tl.float32)

    r = r * scale_r + bias_r
    g = g * scale_g + bias_g
    b = b * scale_b + bias_b

    if IS_FP16:
        r = r.to(tl.float16)
        g = g.to(tl.float16)
        b = b.to(tl.float16)

    tl.store(y_ptr + offsets, r, mask=mask)
    tl.store(y_ptr + num_pixels + offsets, g, mask=mask)
    tl.store(y_ptr + 2 * num_pixels + offsets, b, mask=mask)


def fused_preprocess_triton(
    img_uint8_hwc: torch.Tensor,
    mean: ChannelParam = (0.485, 0.456, 0.406),
    std: ChannelParam = (0.229, 0.224, 0.225),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Fused preprocessing: [H, W, 3] uint8 CUDA tensor -> [1, 3, H, W] normalized tensor.

    ``mean`` and ``std`` may be floats, tuples, lists or tensors. Tuples avoid any
    device synchronization; a CUDA tensor costs one device-to-host read per call.
    A non-contiguous input is copied to a contiguous buffer first.
    """
    if not isinstance(img_uint8_hwc, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(img_uint8_hwc)}")
    if not img_uint8_hwc.is_cuda:
        raise ValueError("Input tensor must reside on a CUDA device")
    if img_uint8_hwc.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 tensor, got {img_uint8_hwc.dtype}")
    if img_uint8_hwc.ndim != 3 or img_uint8_hwc.shape[2] != 3:
        raise ValueError(f"Expected [H, W, 3] shape, got {tuple(img_uint8_hwc.shape)}")
    if dtype not in (torch.float32, torch.float16):
        raise TypeError(f"Output dtype must be float32 or float16, got {dtype}")

    H, W, _ = img_uint8_hwc.shape
    num_pixels = H * W
    if 3 * num_pixels >= 2**31:
        raise ValueError("Image too large for 32-bit indexing")

    (sr, sg, sb), (br, bg, bb) = fold_normalization(mean, std)
    img = img_uint8_hwc.contiguous()
    out = torch.empty((1, 3, H, W), device=img.device, dtype=dtype)

    grid = lambda meta: (triton.cdiv(num_pixels, meta["BLOCK_SIZE"]),)
    _fused_preprocess_kernel[grid](
        img, out, num_pixels, sr, sg, sb, br, bg, bb, IS_FP16=dtype == torch.float16
    )
    return out
