import torch

def postprocess_detection_torch_eager(
    preds: torch.Tensor,
    conf_threshold: float = 0.25
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized GPU PyTorch eager candidate filtering.
    
    Input: [1, N, 4 + C] or [N, 4 + C]
    Returns: (filtered_boxes, filtered_scores, filtered_classes)
    """
    if preds.ndim == 3:
        preds = preds.squeeze(0)
    
    boxes = preds[:, :4]
    scores = preds[:, 4:]
    
    class_scores, class_indices = torch.max(scores, dim=1)
    mask = class_scores >= conf_threshold
    
    return boxes[mask], class_scores[mask], class_indices[mask]


@torch.compile(mode="default")
def postprocess_detection_torch_compiled(
    preds: torch.Tensor,
    conf_threshold: float = 0.25
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compiler-optimized candidate score extraction and filtering."""
    if preds.ndim == 3:
        preds = preds.squeeze(0)
    boxes = preds[:, :4]
    scores = preds[:, 4:]
    class_scores, class_indices = torch.max(scores, dim=1)
    mask = class_scores >= conf_threshold
    return boxes[mask], class_scores[mask], class_indices[mask]


# Alias for backward compatibility
postprocess_detection_torch = postprocess_detection_torch_eager
