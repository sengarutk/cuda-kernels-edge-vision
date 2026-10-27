import hashlib
import os
import types
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "yolov8n.pt"
# SHA-256 of the official Ultralytics yolov8n.pt release asset (COCO, 80 classes).
YOLOV8N_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"


class YOLOv8Raw(nn.Module):
    """Wraps an Ultralytics DetectionModel and returns only the raw head tensor.

    Output: ``[1, 4 + 80, 8400]`` for a 640x640 input; rows 0-3 are (cx, cy, w, h) boxes in
    input pixels, rows 4.. are per-class sigmoid scores. No NMS is applied here.
    """

    def __init__(self, detection_model: nn.Module):
        super().__init__()
        self.model = detection_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out[0] if isinstance(out, (tuple, list)) else out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_yolov8n_weights(path: Optional[os.PathLike] = None) -> Path:
    """Returns a verified local path to yolov8n.pt, downloading it via Ultralytics if needed."""
    weights = Path(path or os.environ.get("YOLOV8N_WEIGHTS", DEFAULT_WEIGHTS))
    if not weights.exists():
        from ultralytics.utils.downloads import attempt_download_asset

        weights.parent.mkdir(parents=True, exist_ok=True)
        weights = Path(attempt_download_asset(str(weights)))
    if weights.name == "yolov8n.pt" and _sha256(weights) != YOLOV8N_SHA256:
        raise RuntimeError(f"{weights} does not match the official yolov8n.pt checksum")
    return weights


def _c2f_forward(self, x: torch.Tensor) -> torch.Tensor:
    y = list(self.cv1(x).chunk(2, 1))
    for m in self.m:
        y.append(m(y[-1]))
    return self.cv2(torch.cat(y, 1))


def _sppf_forward(self, x: torch.Tensor) -> torch.Tensor:
    y = [self.cv1(x)]
    for _ in range(getattr(self, "n", 3)):
        y.append(self.m(y[-1]))
    y = self.cv2(torch.cat(y, 1))
    return y + x if getattr(self, "add", False) else y


def make_compile_safe(model: nn.Module) -> nn.Module:
    """Rewrites C2f/SPPF ``forward`` as explicit loops (numerically identical).

    Ultralytics builds these outputs with ``y.extend(m(y[-1]) for m in ...)``, a generator
    that reads the list it is extending. PyTorch 2.5's TorchDynamo traces that pattern
    incorrectly (every block receives the same input), so a ``torch.compile``-d YOLOv8n
    silently produces wrong boxes and scores. The loop form traces correctly.
    """
    from ultralytics.nn.modules.block import SPPF, C2f

    for module in model.modules():
        if type(module) is C2f:
            module.forward = types.MethodType(_c2f_forward, module)
        elif type(module) is SPPF:
            module.forward = types.MethodType(_sppf_forward, module)
    return model


def load_yolov8n(device: torch.device, weights: Optional[os.PathLike] = None) -> YOLOv8Raw:
    """Loads pretrained YOLOv8n (3.2M parameters) with Conv+BN fused, in eval mode, FP32."""
    import ultralytics  # noqa: F401  (needed to unpickle the checkpoint)

    ckpt = torch.load(resolve_yolov8n_weights(weights), map_location="cpu", weights_only=False)
    model = (ckpt["ema"] if ckpt.get("ema") is not None else ckpt["model"]).float().fuse(verbose=False).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return YOLOv8Raw(make_compile_safe(model)).to(device).eval()
