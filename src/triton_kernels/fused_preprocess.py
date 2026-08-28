import torch
import triton
import triton.language as tl
from typing import Union, Tuple, List

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
    num_pixels: int,
    mean_r: float,
    mean_g: float,
    mean_b: float,
    inv_std_r: float,
    inv_std_g: float,
    inv_std_b: float,
    inv_scale: float,
    IS_FP16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused HWC uint8 -> normalized planar CHW float32/fp16 kernel.
    
    Layout transformation:
      Input:  Interleaved RGB [R0, G0, B0, R1, G1, B1, ...] (uint8)
      Output: Planar [R0, R1, ..., G0, G1, ..., B0, B1, ...] (fp32 / fp16)
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_pixels

    # Compute interleaved input byte offsets
    in_r_ptr = x_ptr + offsets * 3 + 0
    in_g_ptr = x_ptr + offsets * 3 + 1
    in_b_ptr = x_ptr + offsets * 3 + 2

    # Vectorized strided loads
    r_val = tl.load(in_r_ptr, mask=mask, other=0).to(tl.float32)
    g_val = tl.load(in_g_ptr, mask=mask, other=0).to(tl.float32)
    b_val = tl.load(in_b_ptr, mask=mask, other=0).to(tl.float32)

    # Fast FMA normalization in register space
    r_norm = (r_val * inv_scale - mean_r) * inv_std_r
    g_norm = (g_val * inv_scale - mean_g) * inv_std_g
    b_norm = (b_val * inv_scale - mean_b) * inv_std_b

    # Planar output write pointers
    out_r_ptr = y_ptr + 0 * num_pixels + offsets
    out_g_ptr = y_ptr + 1 * num_pixels + offsets
    out_b_ptr = y_ptr + 2 * num_pixels + offsets

    if IS_FP16:
        tl.store(out_r_ptr, r_norm.to(tl.float16), mask=mask)
        tl.store(out_g_ptr, g_norm.to(tl.float16), mask=mask)
        tl.store(out_b_ptr, b_norm.to(tl.float16), mask=mask)
    else:
        tl.store(out_r_ptr, r_norm, mask=mask)
        tl.store(out_g_ptr, g_norm, mask=mask)
        tl.store(out_b_ptr, b_norm, mask=mask)


def fused_preprocess_triton(
    img_uint8_hwc: torch.Tensor,
    mean: Union[torch.Tensor, Tuple[float, float, float], List[float]] = (0.485, 0.456, 0.406),
    std: Union[torch.Tensor, Tuple[float, float, float], List[float]] = (0.229, 0.224, 0.225),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Python host wrapper & zero-sync dispatcher for fused preprocessing kernel.
    
    Accepts: [H, W, 3] uint8 CUDA tensor
    Returns: [1, 3, H, W] contiguous normalized tensor of requested dtype
    """
    if not isinstance(img_uint8_hwc, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(img_uint8_hwc)}")
    if not img_uint8_hwc.is_cuda:
        raise ValueError("Input tensor must reside on a CUDA device")
    if img_uint8_hwc.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 tensor, got {img_uint8_hwc.dtype}")
    if img_uint8_hwc.ndim != 3 or img_uint8_hwc.shape[2] != 3:
        raise ValueError(f"Expected [H, W, 3] shape, got {img_uint8_hwc.shape}")
    if dtype not in (torch.float32, torch.float16):
        raise TypeError(f"Output dtype must be float32 or float16, got {dtype}")

    H, W, _ = img_uint8_hwc.shape
    num_pixels = H * W
    img_contiguous = img_uint8_hwc.contiguous()

    # Fast scalar extraction avoiding device-to-host sync on lists/tuples
    if isinstance(mean, (tuple, list)):
        mr, mg, mb = float(mean[0]), float(mean[1]), float(mean[2])
    elif isinstance(mean, torch.Tensor):
        # Extract without triggering multiple cuda-to-cpu array roundtrips
        m_list = mean.tolist() if mean.dim() > 0 else [float(mean)] * 3
        mr, mg, mb = float(m_list[0]), float(m_list[1]), float(m_list[2])
    else:
        mr, mg, mb = float(mean), float(mean), float(mean)

    if isinstance(std, (tuple, list)):
        sr, sg, sb = float(std[0]), float(std[1]), float(std[2])
    elif isinstance(std, torch.Tensor):
        s_list = std.tolist() if std.dim() > 0 else [float(std)] * 3
        sr, sg, sb = float(s_list[0]), float(s_list[1]), float(s_list[2])
    else:
        sr, sg, sb = float(std), float(std), float(std)

    inv_scale = 1.0 / 255.0
    inv_std_r = 1.0 / sr
    inv_std_g = 1.0 / sg
    inv_std_b = 1.0 / sb

    # Pre-allocate contiguous [1, 3, H, W] tensor
    out = torch.empty((1, 3, H, W), device=img_uint8_hwc.device, dtype=dtype)
    is_fp16 = dtype == torch.float16

    grid = lambda META: (triton.cdiv(num_pixels, META["BLOCK_SIZE"]),)

    _fused_preprocess_kernel[grid](
        img_contiguous,
        out,
        num_pixels,
        mr,
        mg,
        mb,
        inv_std_r,
        inv_std_g,
        inv_std_b,
        inv_scale,
        IS_FP16=is_fp16,
    )
    return out
