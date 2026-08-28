import torch

def compute_residual_map_torch(
    x: torch.Tensor,
    x_hat: torch.Tensor
) -> torch.Tensor:
    """Calculates per-pixel mean absolute error across color channels: [1, 1, H, W]."""
    residual = torch.abs(x - x_hat)
    return torch.mean(residual, dim=1, keepdim=True)

def compute_anomaly_score_topk(
    anomaly_map: torch.Tensor,
    top_k_fraction: float = 0.01
) -> float:
    """Computes mean of the top-1% anomalous pixels in the image."""
    flat = anomaly_map.view(-1)
    k = max(1, int(top_k_fraction * flat.numel()))
    top_values, _ = torch.topk(flat, k=k)
    return top_values.mean().item()
