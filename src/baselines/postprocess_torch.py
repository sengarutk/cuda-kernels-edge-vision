from typing import Tuple

import torch

from src.triton_kernels.candidate_filter import Layout, candidate_matrix


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    xy = boxes[:, :2]
    half_wh = boxes[:, 2:4] * 0.5
    return torch.cat((xy - half_wh, xy + half_wh), dim=1)


def postprocess_detection_torch_eager(
    preds: torch.Tensor,
    conf_threshold: float = 0.25,
    layout: Layout = "channels_first",
    box_format: str = "xywh",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch eager candidate filter, following the Ultralytics NMS pre-filter.

    One reduction finds each candidate's best class score, a single ``nonzero`` (the only
    device-to-host synchronization) selects survivors, and the class argmax and box
    conversion run only on the K survivors. Returns
    ``(boxes_xyxy [K, 4], scores [K], classes [K] int64, candidate_indices [K] int64)``
    in ascending candidate order.
    """
    cand = candidate_matrix(preds, layout)
    keep = cand[:, 4:].amax(dim=1) >= conf_threshold
    indices = keep.nonzero().squeeze(1)
    survivors = cand.index_select(0, indices).float()
    scores, classes = survivors[:, 4:].max(dim=1)
    boxes = survivors[:, :4]
    if box_format == "xywh":
        boxes = xywh_to_xyxy(boxes)
    return boxes, scores, classes, indices


# Inductor compiles the dense reduction; the data-dependent nonzero causes a graph break.
postprocess_detection_torch_compiled = torch.compile(postprocess_detection_torch_eager, mode="default")
