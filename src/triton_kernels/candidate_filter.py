import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["num_candidates", "num_classes"],
)
@triton.jit
def _fused_candidate_filter_kernel(
    preds_ptr,          # [N, 4 + C]
    mask_ptr,           # [N]
    scores_ptr,         # [N]
    classes_ptr,        # [N]
    num_candidates: int,
    num_classes: int,
    conf_threshold: float,
    STRIDE_N: int,
    STRIDE_C: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    cand_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cand_mask = cand_offsets < num_candidates

    # Pointer to start of class scores (offset by 4 bounding box coordinates)
    score_base = preds_ptr + cand_offsets * STRIDE_N + 4 * STRIDE_C

    # Initialize register accumulators
    max_scores = tl.full([BLOCK_SIZE], -1e9, dtype=tl.float32)
    best_classes = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    # Loop over classes in unrolled chunks
    for c in range(num_classes):
        c_ptr = score_base + c * STRIDE_C
        c_scores = tl.load(c_ptr, mask=cand_mask, other=-1e9)
        
        is_greater = c_scores > max_scores
        max_scores = tl.where(is_greater, c_scores, max_scores)
        best_classes = tl.where(is_greater, c, best_classes)

    # Evaluate threshold predicate
    keep_mask = (max_scores >= conf_threshold) & cand_mask

    # Write outputs
    tl.store(mask_ptr + cand_offsets, keep_mask.to(tl.int8), mask=cand_mask)
    tl.store(scores_ptr + cand_offsets, max_scores, mask=cand_mask)
    tl.store(classes_ptr + cand_offsets, best_classes, mask=cand_mask)


def candidate_filter_triton(
    preds: torch.Tensor,
    conf_threshold: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Triton YOLO Candidate Confidence Extractor & Mask Generator.
    
    Accepts: preds tensor [1, N, 4 + C] or [N, 4 + C] on CUDA
    Returns: (filtered_boxes, filtered_scores, filtered_classes, raw_mask)
    """
    if not preds.is_cuda:
        raise ValueError("Predictions tensor must be on CUDA")
    if preds.ndim == 3:
        preds = preds.squeeze(0)
    if preds.ndim != 2:
        raise ValueError(f"Expected 2D tensor [N, 4 + C], got shape {preds.shape}")

    num_candidates, total_cols = preds.shape
    num_classes = total_cols - 4
    preds_contiguous = preds.contiguous()

    # Allocate output tensors
    mask = torch.empty(num_candidates, dtype=torch.int8, device=preds.device)
    scores = torch.empty(num_candidates, dtype=torch.float32, device=preds.device)
    classes = torch.empty(num_candidates, dtype=torch.int32, device=preds.device)

    grid = lambda META: (triton.cdiv(num_candidates, META["BLOCK_SIZE"]),)

    _fused_candidate_filter_kernel[grid](
        preds_contiguous,
        mask,
        scores,
        classes,
        num_candidates,
        num_classes,
        float(conf_threshold),
        preds_contiguous.stride(0),
        preds_contiguous.stride(1),
    )

    bool_mask = mask.to(torch.bool)
    filtered_boxes = preds_contiguous[:, :4][bool_mask]
    filtered_scores = scores[bool_mask]
    filtered_classes = classes[bool_mask].to(torch.int64)

    return filtered_boxes, filtered_scores, filtered_classes, bool_mask
