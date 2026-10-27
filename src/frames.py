"""Benchmark input frames."""
from pathlib import Path

import numpy as np

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def letterbox(img: np.ndarray, size: int = 640, pad_value: int = 114) -> np.ndarray:
    """Resizes an HWC uint8 image to fit ``size`` x ``size`` keeping aspect ratio, then pads."""
    from PIL import Image

    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = np.asarray(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
    out = np.full((size, size, 3), pad_value, dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    out[top:top + new_h, left:left + new_w] = resized
    return out


def detection_frame(size: int = 640) -> np.ndarray:
    """Letterboxed RGB street scene (Ultralytics ``bus.jpg``) used as the detection input."""
    from PIL import Image
    import ultralytics

    path = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    return letterbox(np.asarray(Image.open(path).convert("RGB")), size)


def anomaly_frame() -> np.ndarray:
    """Synthetic 256x256 texture with a bright scratch defect (committed fixture)."""
    return np.load(FIXTURES_DIR / "image_256_uint8.npy")
