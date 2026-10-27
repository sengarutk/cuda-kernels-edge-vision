import pytest
import torch

from src.baselines.anomaly_baselines import compute_anomaly_score_topk, compute_residual_map_torch


def test_residual_map_and_topk_score():
    x = torch.zeros(1, 3, 256, 256)
    recon = torch.zeros(1, 3, 256, 256)
    x[0, :, 100:110, 100:110] = 1.0  # 100 defective pixels, all channels off by 1.0
    res = compute_residual_map_torch(x, recon)
    assert res.shape == (1, 1, 256, 256)
    assert res.sum().item() == pytest.approx(100.0)
    # top 1% of 65536 pixels = 655 values: 100 ones and 555 zeros.
    assert compute_anomaly_score_topk(res, 0.01) == pytest.approx(100 / 655)


def test_topk_keeps_at_least_one_pixel():
    res = torch.zeros(1, 1, 4, 4)
    res[0, 0, 0, 0] = 2.0
    assert compute_anomaly_score_topk(res, 0.0) == 2.0
