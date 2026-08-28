import torch

class L2CacheFlusher:
    """Evicts GPU L2 cache by streaming across a 64MB scratch tensor between timing passes."""
    def __init__(self, size_mb: int = 64, device: str = "cuda:0"):
        num_elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32
        self.buffer = torch.empty(num_elements, dtype=torch.float32, device=device)

    def flush(self):
        # In-place write forces eviction of active L2 cache lines
        self.buffer.zero_()
