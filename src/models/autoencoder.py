import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder for 256x256 RGB reconstruction-based anomaly scoring.

    The weights are randomly initialized (seeded) and never trained: this repository
    measures latency, not detection accuracy, and the compute graph (3 strided convs,
    3 transposed convs, BatchNorm, ReLU, Sigmoid) is what matters for that.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 128, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),           # [B, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_dim, kernel_size=4, stride=2, padding=1),   # [B, latent, 32, 32]
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),          # [B, 32, 128, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1), # [B, 3, 256, 256]
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def build_autoencoder(device: torch.device, seed: int = 0) -> ConvAutoencoder:
    """Builds the autoencoder with deterministic weights, independent of global RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ConvAutoencoder()
    return model.to(device).eval()
