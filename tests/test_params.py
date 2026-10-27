import pytest
import torch

from src.triton_kernels._params import fold_normalization, resolve_channel_params


def test_resolve_accepts_scalars_sequences_and_tensors():
    assert resolve_channel_params(0.5, "m") == (0.5, 0.5, 0.5)
    assert resolve_channel_params((1, 2, 3), "m") == (1.0, 2.0, 3.0)
    assert resolve_channel_params([1.0], "m") == (1.0, 1.0, 1.0)
    assert resolve_channel_params(torch.tensor([1.0, 2.0, 3.0]), "m") == (1.0, 2.0, 3.0)
    assert resolve_channel_params(torch.tensor(2.0), "m") == (2.0, 2.0, 2.0)


def test_resolve_rejects_wrong_length():
    with pytest.raises(ValueError, match="1 or 3 values"):
        resolve_channel_params((1.0, 2.0), "mean")


def test_fold_matches_reference_formula():
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    scale, bias = fold_normalization(mean, std)
    for x in (0.0, 17.0, 128.0, 255.0):
        for c in range(3):
            assert x * scale[c] + bias[c] == pytest.approx((x / 255.0 - mean[c]) / std[c], abs=1e-12)


def test_fold_rejects_zero_std():
    with pytest.raises(ValueError, match="non-zero"):
        fold_normalization((0.0,), (0.0,))
