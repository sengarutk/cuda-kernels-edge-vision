from typing import Tuple

import numpy as np


def postprocess_detection_numpy(
    preds: np.ndarray,
    conf_threshold: float = 0.25,
    layout: str = "channels_first",
    box_format: str = "xywh",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Host NumPy reference for the candidate filter (same contract as the PyTorch version)."""
    if preds.ndim == 3:
        preds = preds[0]
    cand = preds.T if layout == "channels_first" else preds
    keep = cand[:, 4:].max(axis=1) >= conf_threshold
    indices = np.nonzero(keep)[0]
    survivors = cand[indices].astype(np.float32)
    scores = survivors[:, 4:].max(axis=1)
    classes = survivors[:, 4:].argmax(axis=1).astype(np.int64)
    boxes = survivors[:, :4]
    if box_format == "xywh":
        half = boxes[:, 2:4] * 0.5
        boxes = np.concatenate((boxes[:, :2] - half, boxes[:, :2] + half), axis=1)
    return boxes, scores, classes, indices.astype(np.int64)
