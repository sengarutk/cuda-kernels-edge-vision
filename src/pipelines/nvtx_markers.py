import contextlib

import torch


@contextlib.contextmanager
def nvtx_range(name: str):
    """Context manager for NVTX profiling ranges visible in Nsight Systems."""
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


def stage(name: str, enabled: bool):
    """NVTX range when ``enabled``, otherwise a no-op context."""
    return nvtx_range(name) if enabled else contextlib.nullcontext()
