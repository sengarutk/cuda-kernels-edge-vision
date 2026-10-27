from typing import Any, Dict, Literal, Optional, Tuple

import torch
import torchvision.ops as ops

from src.baselines.postprocess_torch import (
    postprocess_detection_torch_compiled,
    postprocess_detection_torch_eager,
)
from src.baselines.preprocess_torch import preprocess_compiled_gpu, preprocess_eager_gpu
from src.models.yolo import load_yolov8n
from src.pipelines.cuda_graph import GraphedStage
from src.pipelines.nvtx_markers import stage
from src.triton_kernels.candidate_filter import candidate_filter_triton
from src.triton_kernels.fused_preprocess import fused_preprocess_triton

Backend = Literal["eager", "compiled", "triton"]


class DetectionInspectionPipeline:
    """Batch-1 YOLOv8n detection at 640x640.

    Stages: H2D copy -> preprocess (uint8 HWC -> float CHW / 255) -> pretrained YOLOv8n
    (eager or torch.compile) -> candidate filter (eager | compiled | triton) -> class-aware
    NMS (torchvision) -> results stay on the GPU; only the survivor count is read back.
    With ``cuda_graph=True`` the H2D copy, preprocessing and model are captured once into a
    CUDA graph and replayed per frame; the filter and NMS (data-dependent shapes) run outside.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        preprocess: Backend = "triton",
        postprocess: Backend = "triton",
        compile_model: bool = False,
        cuda_graph: bool = False,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 300,
        weights: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
    ):
        for kind, value in (("preprocess", preprocess), ("postprocess", postprocess)):
            if value not in ("eager", "compiled", "triton"):
                raise ValueError(f"Unknown {kind} backend {value!r}")
        if cuda_graph and compile_model:
            raise ValueError("cuda_graph and compile_model are mutually exclusive")
        self.device = torch.device(device)
        self.preprocess_backend = preprocess
        self.postprocess_backend = postprocess
        self.compile_model = compile_model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        # YOLOv8 expects RGB scaled to [0, 1]: mean 0, std 1.
        self.mean = (0.0, 0.0, 0.0)
        self.std = (1.0, 1.0, 1.0)
        self.mean_t = torch.tensor(self.mean, device=self.device)
        self.std_t = torch.tensor(self.std, device=self.device)
        self.model = model if model is not None else load_yolov8n(self.device, weights)
        self._forward = torch.compile(self.model, mode="default") if compile_model else self.model
        self.cuda_graph = cuda_graph
        self._graph = None

    @property
    def name(self) -> str:
        model = "compiled" if self.compile_model else "eager"
        name = f"pre={self.preprocess_backend}, model={model}, filter={self.postprocess_backend}"
        return name + (", cuda-graph" if self.cuda_graph else "")

    def h2d(self, host_image_hwc: torch.Tensor) -> torch.Tensor:
        return host_image_hwc.to(self.device, non_blocking=True)

    def preprocess(self, device_img: torch.Tensor) -> torch.Tensor:
        if self.preprocess_backend == "triton":
            return fused_preprocess_triton(device_img, self.mean, self.std)
        if self.preprocess_backend == "compiled":
            return preprocess_compiled_gpu(device_img, self.mean_t, self.std_t)
        return preprocess_eager_gpu(device_img, self.mean_t, self.std_t)

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)

    def filter(self, preds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.postprocess_backend == "triton":
            boxes, scores, classes, _ = candidate_filter_triton(preds, self.conf_threshold)
        elif self.postprocess_backend == "compiled":
            boxes, scores, classes, _ = postprocess_detection_torch_compiled(preds, self.conf_threshold)
        else:
            boxes, scores, classes, _ = postprocess_detection_torch_eager(preds, self.conf_threshold)
        return boxes, scores, classes

    def nms(self, boxes, scores, classes):
        keep = ops.batched_nms(boxes, scores, classes, self.iou_threshold)[: self.max_det]
        return boxes[keep], scores[keep], classes[keep]

    @torch.inference_mode()
    def run_step(self, host_image_hwc: torch.Tensor, use_nvtx: bool = False) -> Dict[str, Any]:
        if self.cuda_graph:
            if self._graph is None:
                body = lambda img: (self.infer(self.preprocess(img)),)
                self._graph = GraphedStage(body, self.h2d(host_image_hwc))
            with stage("01-03_CUDA_Graph", use_nvtx):
                (preds,) = self._graph(host_image_hwc)
        else:
            with stage("01_H2D", use_nvtx):
                device_img = self.h2d(host_image_hwc)
            with stage("02_Preprocess", use_nvtx):
                x = self.preprocess(device_img)
            with stage("03_Inference", use_nvtx):
                preds = self.infer(x)
        with stage("04_Filter", use_nvtx):
            boxes, scores, classes = self.filter(preds)
        num_candidates = boxes.shape[0]
        with stage("05_NMS", use_nvtx):
            boxes, scores, classes = self.nms(boxes, scores, classes)
        return {
            "num_candidates": num_candidates,
            "num_detections": boxes.shape[0],
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
        }
