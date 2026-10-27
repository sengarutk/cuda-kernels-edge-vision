from typing import Literal, Tuple

import torch
import triton
import triton.language as tl

Layout = Literal["channels_first", "channels_last"]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_N": 512}, num_warps=8),
    ],
    key=["num_candidates", "NUM_CLASSES"],
    reset_to_zero=["count_ptr"],
)
@triton.jit
def _candidate_filter_kernel(
    preds_ptr,
    stride_n,
    stride_c,
    boxes_ptr,
    scores_ptr,
    classes_ptr,
    indices_ptr,
    count_ptr,
    num_candidates,
    conf_threshold,
    NUM_CLASSES: tl.constexpr,
    BOX_XYWH: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Single-pass class reduction, threshold, box decode and stream compaction.

    Element (n, c) of the candidate matrix lives at ``preds_ptr + n * stride_n + c * stride_c``
    (columns 0-3 are the box, 4.. are class scores). With the channels-first layout that
    detectors such as YOLOv8 emit (stride_n == 1), every class-score load is coalesced
    across the block. Survivors are appended to the output buffers at positions reserved
    with one atomic add per program; their order across programs is unspecified.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = offs < num_candidates
    row = preds_ptr + offs * stride_n

    best = tl.full([BLOCK_N], float("-inf"), dtype=tl.float32)
    best_cls = tl.zeros([BLOCK_N], dtype=tl.int32)
    for c in tl.static_range(NUM_CLASSES):
        s = tl.load(row + (4 + c) * stride_c, mask=valid, other=float("-inf")).to(tl.float32)
        better = s > best
        best = tl.where(better, s, best)
        best_cls = tl.where(better, c, best_cls)

    keep = (best >= conf_threshold) & valid
    keep_i = keep.to(tl.int32)
    base = tl.atomic_add(count_ptr, tl.sum(keep_i, axis=0))
    pos = base + tl.cumsum(keep_i, axis=0) - 1

    b0 = tl.load(row, mask=keep, other=0.0).to(tl.float32)
    b1 = tl.load(row + stride_c, mask=keep, other=0.0).to(tl.float32)
    b2 = tl.load(row + 2 * stride_c, mask=keep, other=0.0).to(tl.float32)
    b3 = tl.load(row + 3 * stride_c, mask=keep, other=0.0).to(tl.float32)
    if BOX_XYWH:
        half_w = b2 * 0.5
        half_h = b3 * 0.5
        x1 = b0 - half_w
        y1 = b1 - half_h
        x2 = b0 + half_w
        y2 = b1 + half_h
    else:
        x1 = b0
        y1 = b1
        x2 = b2
        y2 = b3

    tl.store(boxes_ptr + pos * 4 + 0, x1, mask=keep)
    tl.store(boxes_ptr + pos * 4 + 1, y1, mask=keep)
    tl.store(boxes_ptr + pos * 4 + 2, x2, mask=keep)
    tl.store(boxes_ptr + pos * 4 + 3, y2, mask=keep)
    tl.store(scores_ptr + pos, best, mask=keep)
    tl.store(classes_ptr + pos, best_cls.to(tl.int64), mask=keep)
    tl.store(indices_ptr + pos, offs.to(tl.int64), mask=keep)


def candidate_matrix(preds: torch.Tensor, layout: Layout) -> torch.Tensor:
    """Returns an [N, 4 + C] (possibly strided) view of a batch-1 prediction tensor."""
    if preds.ndim == 3:
        if preds.shape[0] != 1:
            raise ValueError(f"Only batch size 1 is supported, got {preds.shape[0]}")
        preds = preds[0]
    if preds.ndim != 2:
        raise ValueError(f"Expected [1, 4 + C, N] or [1, N, 4 + C], got {tuple(preds.shape)}")
    if layout == "channels_first":
        return preds.t()
    if layout == "channels_last":
        return preds
    raise ValueError(f"Unknown layout {layout!r}")


def candidate_filter_triton(
    preds: torch.Tensor,
    conf_threshold: float = 0.25,
    layout: Layout = "channels_first",
    box_format: Literal["xywh", "xyxy"] = "xywh",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused YOLO candidate filter.

    Accepts raw batch-1 detector output, either ``[1, 4 + C, N]`` (``channels_first``,
    YOLOv8 head layout) or ``[1, N, 4 + C]`` (``channels_last``), in any strides.
    Returns ``(boxes_xyxy [K, 4], scores [K], classes [K] int64, candidate_indices [K] int64)``
    for the K candidates whose best class score is >= ``conf_threshold``. The K survivors
    come back in unspecified order; ``candidate_indices`` identifies each one. Reading K
    is the only device-to-host synchronization.
    """
    if not preds.is_cuda:
        raise ValueError("Predictions tensor must be on CUDA")
    if box_format not in ("xywh", "xyxy"):
        raise ValueError(f"Unknown box_format {box_format!r}")
    cand = candidate_matrix(preds, layout)
    num_candidates, total_cols = cand.shape
    num_classes = total_cols - 4
    if num_classes < 1:
        raise ValueError("Predictions must contain at least one class column")

    device = cand.device
    boxes = torch.empty((num_candidates, 4), dtype=torch.float32, device=device)
    scores = torch.empty(num_candidates, dtype=torch.float32, device=device)
    classes = torch.empty(num_candidates, dtype=torch.int64, device=device)
    indices = torch.empty(num_candidates, dtype=torch.int64, device=device)
    count = torch.zeros(1, dtype=torch.int32, device=device)

    grid = lambda meta: (triton.cdiv(num_candidates, meta["BLOCK_N"]),)
    _candidate_filter_kernel[grid](
        cand, cand.stride(0), cand.stride(1),
        boxes, scores, classes, indices, count,
        num_candidates, float(conf_threshold),
        NUM_CLASSES=num_classes, BOX_XYWH=box_format == "xywh",
    )

    k = int(count.item())
    return boxes[:k], scores[:k], classes[:k], indices[:k]
