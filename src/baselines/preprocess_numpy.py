import numpy as np

def preprocess_numpy_cpu(
    img_uint8_hwc: np.ndarray,
    mean: list[float] = [0.485, 0.456, 0.406],
    std: list[float] = [0.229, 0.224, 0.225],
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """Standard host-side CPU preprocessing: HWC uint8 -> CHW normalized float array [1, 3, H, W]."""
    if img_uint8_hwc.ndim != 3 or img_uint8_hwc.shape[2] != 3:
        raise ValueError(f"Expected [H, W, 3] image, got shape {img_uint8_hwc.shape}")
    if img_uint8_hwc.dtype != np.uint8:
        raise TypeError(f"Expected uint8 array, got {img_uint8_hwc.dtype}")

    # 1. HWC -> CHW transpose
    chw = np.transpose(img_uint8_hwc, (2, 0, 1))
    # 2. Convert to float and scale [0, 255] -> [0, 1]
    float_array = chw.astype(dtype) / 255.0
    # 3. Channel-wise mean and std normalization
    mean_arr = np.array(mean, dtype=dtype).reshape(3, 1, 1)
    std_arr = np.array(std, dtype=dtype).reshape(3, 1, 1)
    normalized = (float_array - mean_arr) / std_arr
    # 4. Add batch dimension: [1, 3, H, W]
    return np.ascontiguousarray(np.expand_dims(normalized, axis=0))
