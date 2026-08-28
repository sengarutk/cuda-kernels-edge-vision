import torch
import triton
import triton.language as tl

@triton.jit
def _failed_2d_preprocess_kernel(
    x_ptr,
    y_ptr,
    H: int,
    W: int,
    mean_r: float,
    mean_g: float,
    mean_b: float,
    inv_std_r: float,
    inv_std_g: float,
    inv_std_b: float,
    inv_scale: float,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Intentionally sub-optimal 2D-tiled preprocessing kernel.
    
    Uses 2D thread block indexing with separate channel loops. Incurs high register
    pressure, lower warp occupancy, and fragmented global memory transactions.
    """
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    offs_h = pid_y * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = pid_x * BLOCK_W + tl.arange(0, BLOCK_W)

    mask_2d = (offs_h[:, None] < H) & (offs_w[None, :] < W)
    num_pixels = H * W

    # Compute 2D pixel indices
    pixel_idx_2d = offs_h[:, None] * W + offs_w[None, :]

    # 2D Strided loads
    r_val = tl.load(x_ptr + pixel_idx_2d * 3 + 0, mask=mask_2d, other=0).to(tl.float32)
    g_val = tl.load(x_ptr + pixel_idx_2d * 3 + 1, mask=mask_2d, other=0).to(tl.float32)
    b_val = tl.load(x_ptr + pixel_idx_2d * 3 + 2, mask=mask_2d, other=0).to(tl.float32)

    r_norm = (r_val * inv_scale - mean_r) * inv_std_r
    g_norm = (g_val * inv_scale - mean_g) * inv_std_g
    b_norm = (b_val * inv_scale - mean_b) * inv_std_b

    # 2D Stores into planar output
    tl.store(y_ptr + 0 * num_pixels + pixel_idx_2d, r_norm, mask=mask_2d)
    tl.store(y_ptr + 1 * num_pixels + pixel_idx_2d, g_norm, mask=mask_2d)
    tl.store(y_ptr + 2 * num_pixels + pixel_idx_2d, b_norm, mask=mask_2d)


def fused_preprocess_failed_2d(
    img_uint8_hwc: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Dispatcher for the 2D failed optimization study kernel."""
    H, W, _ = img_uint8_hwc.shape
    img_contiguous = img_uint8_hwc.contiguous()

    mean_cpu = mean.tolist()
    std_cpu = std.tolist()
    inv_scale = 1.0 / 255.0
    inv_std_r = 1.0 / std_cpu[0]
    inv_std_g = 1.0 / std_cpu[1]
    inv_std_b = 1.0 / std_cpu[2]

    out = torch.empty((1, 3, H, W), device=img_uint8_hwc.device, dtype=torch.float32)

    BLOCK_H, BLOCK_W = 16, 16
    grid = (triton.cdiv(W, BLOCK_W), triton.cdiv(H, BLOCK_H))

    _failed_2d_preprocess_kernel[grid](
        img_contiguous,
        out,
        H,
        W,
        mean_cpu[0],
        mean_cpu[1],
        mean_cpu[2],
        inv_std_r,
        inv_std_g,
        inv_std_b,
        inv_scale,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        num_warps=2,
    )
    return out
