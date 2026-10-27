import torch


def preprocess_eager_gpu(
    img_uint8_hwc: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Un-fused PyTorch eager preprocessing: HWC uint8 -> CHW normalized [1, 3, H, W].

    ``mean`` and ``std`` are 3-element tensors already on the image's device, so the
    chain launches 5 kernels (cast, div, sub, div, contiguous copy) and never syncs.
    """
    if img_uint8_hwc.ndim != 3 or img_uint8_hwc.shape[2] != 3:
        raise ValueError(f"Expected [H, W, 3] tensor, got {tuple(img_uint8_hwc.shape)}")
    if img_uint8_hwc.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 tensor, got {img_uint8_hwc.dtype}")
    if not img_uint8_hwc.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")
    return _preprocess_chain(img_uint8_hwc, mean, std, dtype)


def _preprocess_chain(img_uint8_hwc, mean, std, dtype=torch.float32):
    chw = img_uint8_hwc.permute(2, 0, 1)
    scaled = chw.to(dtype=dtype).div(255.0)
    mean_t = mean.view(3, 1, 1).to(device=img_uint8_hwc.device, dtype=dtype)
    std_t = std.view(3, 1, 1).to(device=img_uint8_hwc.device, dtype=dtype)
    normalized = (scaled - mean_t) / std_t
    return normalized.contiguous().unsqueeze(0)


# TorchInductor-compiled version of the identical chain.
preprocess_compiled_gpu = torch.compile(_preprocess_chain, mode="default")
