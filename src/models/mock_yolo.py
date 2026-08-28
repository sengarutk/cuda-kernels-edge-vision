import torch
import torch.nn as nn

class IndustrialYOLONano(nn.Module):
    """Representative YOLO Nano architecture outputting [1, 8400, 84]."""
    def __init__(self, num_classes: int = 80, num_candidates: int = 8400):
        super().__init__()
        self.num_classes = num_classes
        self.num_candidates = num_candidates
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # Head projecting directly to [B, num_candidates, 4 + num_classes]
        self.head = nn.Linear(64, num_candidates * (4 + num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        feat = self.backbone(x)
        feat_flat = feat.view(b, -1)
        raw_preds = self.head(feat_flat)
        return raw_preds.view(b, self.num_candidates, 4 + self.num_classes)
