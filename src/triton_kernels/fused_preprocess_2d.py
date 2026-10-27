import torch
import triton
import triton.language as tl

from ._params import ChannelParam, fold_normalization


@triton.jit
def _preprocess_2d_kernel(
    x_ptr,
    y_ptr,
    H,
    W,
    scale_r,
    scale_g,
    scale_b,
    bias_r,
    bias_g,
    bias_b,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """2D-tiled variant of the fused preprocessing kernel (negative-study baseline).

    Identical arithmetic to the 1D kernel, but each program covers a BLOCK_H x BLOCK_W
    spatial tile. With BLOCK_W = 16, every tile row touches only 48 contiguous input
    bytes and 64 contiguous output bytes, so global accesses are split into short
    segments instead of long contiguous runs.
    """
    pid_w = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = (offs_h[:, None] < H) & (offs_w[None, :] < W)

    num_pixels = H * W
    pixel_idx = offs_h[:, None] * W + offs_w[None, :]

    in_ptr = x_ptr + pixel_idx * 3
    r = tl.load(in_ptr + 0, mask=mask, other=0).to(tl.float32)
    g = tl.load(in_ptr + 1, mask=mask, other=0).to(tl.float32)
    b = tl.load(in_ptr + 2, mask=mask, other=0).to(tl.float32)

    tl.store(y_ptr + pixel_idx, r * scale_r + bias_r, mask=mask)
    tl.store(y_ptr + num_pixels + pixel_idx, g * scale_g + bias_g, mask=mask)
    tl.store(y_ptr + 2 * num_pixels + pixel_idx, b * scale_b + bias_b, mask=mask)


def fused_preprocess_2d(
    img_uint8_hwc: torch.Tensor,
    mean: ChannelParam = (0.485, 0.456, 0.406),
    std: ChannelParam = (0.229, 0.224, 0.225),
    block_h: int = 16,
    block_w: int = 16,
    num_warps: int = 2,
) -> torch.Tensor:
    """Dispatcher for the 2D-tiled variant; same contract as ``fused_preprocess_triton``.

    Uses the same parameter handling as the 1D kernel, so tuples avoid any device
    synchronization and benchmarks compare the two kernels on equal terms.
    """
    if not img_uint8_hwc.is_cuda or img_uint8_hwc.dtype != torch.uint8:
        raise ValueError("Expected a uint8 CUDA tensor")
    if img_uint8_hwc.ndim != 3 or img_uint8_hwc.shape[2] != 3:
        raise ValueError(f"Expected [H, W, 3] shape, got {tuple(img_uint8_hwc.shape)}")

    H, W, _ = img_uint8_hwc.shape
    (sr, sg, sb), (br, bg, bb) = fold_normalization(mean, std)
    img = img_uint8_hwc.contiguous()
    out = torch.empty((1, 3, H, W), device=img.device, dtype=torch.float32)

    grid = (triton.cdiv(W, block_w), triton.cdiv(H, block_h))
    _preprocess_2d_kernel[grid](
        img, out, H, W, sr, sg, sb, br, bg, bb,
        BLOCK_H=block_h, BLOCK_W=block_w, num_warps=num_warps,
    )
    return out
