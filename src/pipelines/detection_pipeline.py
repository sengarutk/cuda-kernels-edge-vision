from typing import Literal, Dict, Any, Tuple
import torch
import torchvision.ops as ops
from src.pipelines.nvtx_markers import nvtx_range
from src.baselines.preprocess_torch import preprocess_eager_gpu, preprocess_compiled_gpu
from src.baselines.postprocess_torch import (
    postprocess_detection_torch_eager,
    postprocess_detection_torch_compiled,
)
from src.triton_kernels.fused_preprocess import fused_preprocess_triton
from src.triton_kernels.candidate_filter import candidate_filter_triton
from src.models.mock_yolo import IndustrialYOLONano

class DetectionInspectionPipeline:
    """Full single-stream batch-1 Industrial YOLO Detection Pipeline."""
    def __init__(
        self,
        device: str = "cuda:0",
        backend: Literal["eager", "compiled", "triton_preprocess", "triton_postprocess", "triton_both"] = "triton_both",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        self.device = torch.device(device)
        self.backend = backend
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        self.mean_tuple = (0.0, 0.0, 0.0)
        self.std_tuple = (1.0, 1.0, 1.0)
        self.mean_t = torch.tensor(self.mean_tuple, device=self.device)
        self.std_t = torch.tensor(self.std_tuple, device=self.device)
        
        self.model = IndustrialYOLONano().to(self.device).eval()
        if "compiled" in self.backend:
            self.model = torch.compile(self.model, mode="default")

    def run_step(self, host_image_hwc: torch.Tensor, use_nvtx: bool = False) -> Dict[str, Any]:
        """Runs batch-1 detection end-to-end: H2D -> Preprocess -> Model -> Filter -> NMS -> D2H."""
        # 1. H2D Transfer
        if use_nvtx:
            with nvtx_range("01_H2D_Transfer"):
                device_img = host_image_hwc.to(self.device, non_blocking=True)
        else:
            device_img = host_image_hwc.to(self.device, non_blocking=True)

        # 2. Preprocessing
        use_triton_pre = self.backend in ("triton_preprocess", "triton_both")
        if use_triton_pre:
            if use_nvtx:
                with nvtx_range("02_Triton_Preprocessing"):
                    input_tensor = fused_preprocess_triton(device_img, self.mean_tuple, self.std_tuple)
            else:
                input_tensor = fused_preprocess_triton(device_img, self.mean_tuple, self.std_tuple)
        elif self.backend == "compiled":
            if use_nvtx:
                with nvtx_range("02_Compiled_Preprocessing"):
                    input_tensor = preprocess_compiled_gpu(device_img, self.mean_t, self.std_t)
            else:
                input_tensor = preprocess_compiled_gpu(device_img, self.mean_t, self.std_t)
        else:
            if use_nvtx:
                with nvtx_range("02_Eager_Preprocessing"):
                    input_tensor = preprocess_eager_gpu(device_img, self.mean_t, self.std_t)
            else:
                input_tensor = preprocess_eager_gpu(device_img, self.mean_t, self.std_t)

        # 3. Model Inference
        if use_nvtx:
            with nvtx_range("03_Model_Inference"):
                with torch.no_grad():
                    preds = self.model(input_tensor)
        else:
            with torch.no_grad():
                preds = self.model(input_tensor)

        # 4. Candidate Filtering & Downstream NMS
        use_triton_post = self.backend in ("triton_postprocess", "triton_both")
        if use_triton_post:
            if use_nvtx:
                with nvtx_range("04_Triton_Candidate_Filter"):
                    boxes, scores, classes, _ = candidate_filter_triton(preds, self.conf_threshold)
            else:
                boxes, scores, classes, _ = candidate_filter_triton(preds, self.conf_threshold)
        elif self.backend == "compiled":
            if use_nvtx:
                with nvtx_range("04_Compiled_Candidate_Filter"):
                    boxes, scores, classes = postprocess_detection_torch_compiled(preds, self.conf_threshold)
            else:
                boxes, scores, classes = postprocess_detection_torch_compiled(preds, self.conf_threshold)
        else:
            if use_nvtx:
                with nvtx_range("04_Eager_Candidate_Filter"):
                    boxes, scores, classes = postprocess_detection_torch_eager(preds, self.conf_threshold)
            else:
                boxes, scores, classes = postprocess_detection_torch_eager(preds, self.conf_threshold)

        # Downstream NMS
        if boxes.shape[0] > 0:
            keep = ops.batched_nms(boxes, scores, classes, self.iou_threshold)
            final_boxes = boxes[keep]
            final_scores = scores[keep]
            final_classes = classes[keep]
        else:
            final_boxes = boxes
            final_scores = scores
            final_classes = classes

        return {
            "num_detections": final_boxes.shape[0],
            "survivor_count": final_boxes.shape[0],
            "boxes": final_boxes,
            "scores": final_scores,
            "classes": final_classes,
        }
