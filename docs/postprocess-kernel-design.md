# Postprocessing Kernel Design: Fused YOLO Candidate Compaction

## 1. Problem Formulation & Bottlenecks
Standard YOLO architectures (e.g. YOLOv8 Nano at $640 \times 640$) output a dense candidate tensor $P \in \mathbb{R}^{1 \times N \times (4 + C)}$ where $N = 8,400$ and $C = 80$.

In PyTorch Eager, candidate postprocessing is executed via 3 separate ATen operations:
1. `torch.max(scores, dim=1)`: Reduction along class dimension ($8,400 \times 80 = 672,000$ values), returning `(max_scores, class_indices)`.
2. `class_scores >= conf_threshold`: Boolean mask allocation across 8,400 entries.
3. Indexing `boxes[mask], scores[mask], classes[mask]`: Three non-contiguous gather operations.

In industrial defect inspection, **$>99.8\%$ of candidate boxes are low-confidence background noise ($< 0.05$)**, with only $10\text{--}20$ true defect proposals surviving.

## 2. Fused Triton Kernel Architecture (`candidate_filter.py`)
- **1D Candidate Tile Mapping**: Each thread block processes a contiguous tile of candidates ($N$).
- **Register-Level Class Reduction**:
  ```python
  max_scores = tl.full([BLOCK_SIZE], -1e9, dtype=tl.float32)
  best_classes = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
  for c in range(num_classes):
      c_scores = tl.load(score_base + c * STRIDE_C, mask=cand_mask, other=-1e9)
      is_greater = c_scores > max_scores
      max_scores = tl.where(is_greater, c_scores, max_scores)
      best_classes = tl.where(is_greater, c, best_classes)
  ```
- **Single-Pass Output Serialization**: Evaluates `keep_mask = (max_scores >= conf_threshold)` and writes `mask_ptr`, `scores_ptr`, and `classes_ptr` in a single global memory transaction.

## 3. Downstream Benefits
By compacting $8,400$ candidate proposals down to the true surviving proposals before Non-Maximum Suppression (NMS), downstream NMS memory traffic and quadratic IoU pairwise comparison overhead are reduced by $>400\times$.
