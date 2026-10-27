from typing import Sequence, Tuple, Union

import torch

ChannelParam = Union[float, Sequence[float], torch.Tensor]


def resolve_channel_params(value: ChannelParam, name: str) -> Tuple[float, float, float]:
    """Converts a per-channel normalization parameter into three Python floats.

    Python scalars, tuples and lists are converted without touching the GPU. A CUDA
    tensor is accepted for convenience, but reading it back forces a device-to-host
    synchronization; latency-sensitive callers should pass tuples.
    """
    if isinstance(value, torch.Tensor):
        values = value.detach().reshape(-1).tolist()
    elif isinstance(value, (tuple, list)):
        values = [float(v) for v in value]
    else:
        values = [float(value)]

    if len(values) == 1:
        values = values * 3
    if len(values) != 3:
        raise ValueError(f"{name} must have 1 or 3 values, got {len(values)}")
    return float(values[0]), float(values[1]), float(values[2])


def fold_normalization(
    mean: ChannelParam, std: ChannelParam, input_scale: float = 255.0
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Folds ``(x / input_scale - mean) / std`` into ``x * scale + bias`` per channel."""
    m = resolve_channel_params(mean, "mean")
    s = resolve_channel_params(std, "std")
    if any(v == 0.0 for v in s):
        raise ValueError("std values must be non-zero")
    scale = tuple(1.0 / (input_scale * sv) for sv in s)
    bias = tuple(-mv / sv for mv, sv in zip(m, s))
    return scale, bias
