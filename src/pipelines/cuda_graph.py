from typing import Callable, Tuple

import torch


class GraphedStage:
    """Captures ``body(static_input)`` into a CUDA graph and replays it on new inputs.

    The input must keep a fixed shape and dtype. ``__call__`` copies the new input into the
    captured static buffer and replays the graph; the returned tensors are the graph's static
    outputs, so they are overwritten by the next call.
    """

    def __init__(self, body: Callable[[torch.Tensor], Tuple[torch.Tensor, ...]], example: torch.Tensor,
                 warmup: int = 3):
        self.static_input = example.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.inference_mode():
            for _ in range(warmup):  # JIT/autotune and cuDNN algorithm selection happen here
                body(self.static_input)
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.static_outputs = body(self.static_input)

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        self.static_input.copy_(x, non_blocking=True)
        self.graph.replay()
        return self.static_outputs
