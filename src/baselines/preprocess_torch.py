import torch

def preprocess_eager_gpu(
    img_uint8_hwc: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Un-fused PyTorch Eager GPU preprocessing: HWC uint8 -> CHW normalized float [1, 3, H, W]."""
    if img_uint8_hwc.ndim != 3 or img_uint8_hwc.shape[2] != 3:
        raise ValueError(f"Expected [H, W, 3] tensor, got {img_uint8_hwc.shape}")
    if img_uint8_hwc.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 tensor, got {img_uint8_hwc.dtype}")
    if not img_uint8_hwc.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")

    # 1. Permute layout [H, W, 3] -> [3, H, W]
    chw = img_uint8_hwc.permute(2, 0, 1)
    # 2. Type cast & normalize to [0, 1]
    scaled = chw.to(dtype=dtype).div(255.0)
    # 3. Normalize with mean and std
    mean_t = mean.view(3, 1, 1).to(device=img_uint8_hwc.device, dtype=dtype)
    std_t = std.view(3, 1, 1).to(device=img_uint8_hwc.device, dtype=dtype)
    normalized = (scaled - mean_t) / std_t
    # 4. Make contiguous & add batch dimension [1, 3, H, W]
    return normalized.contiguous().unsqueeze(0)


# Compiled version using TorchInductor (mode='default' for robust WSL2 JIT fusion)
@torch.compile(mode="default")
def preprocess_compiled_gpu(
    img_uint8_hwc: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Compiler-fused PyTorch Inductor preprocessing baseline."""
    chw = img_uint8_hwc.permute(2, 0, 1)
    scaled = chw.to(dtype=dtype).div(255.0)
    mean_t = mean.view(3, 1, 1).to(device=img_uint8_hwc.device, dtype=dtype)
    std_t = std.view(3, 1, 1).to(device=img_uint8_hwc.device, dtype=dtype)
    normalized = (scaled - mean_t) / std_t
    return normalized.contiguous().unsqueeze(0)
