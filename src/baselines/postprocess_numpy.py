import numpy as np

def postprocess_detection_numpy(
    preds: np.ndarray,
    conf_threshold: float = 0.25
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Host CPU NumPy detection candidate filtering."""
    if preds.ndim == 3:
        preds = preds[0]  # [8400, 84]
    
    boxes = preds[:, :4]
    scores = preds[:, 4:]
    
    class_indices = np.argmax(scores, axis=1)
    class_scores = np.max(scores, axis=1)
    
    mask = class_scores >= conf_threshold
    return boxes[mask], class_scores[mask], class_indices[mask]
