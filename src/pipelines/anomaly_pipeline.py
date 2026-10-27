from typing import Any, Dict, Literal, Tuple

import torch

from src.baselines.anomaly_baselines import compute_anomaly_score_topk, compute_residual_map_torch
from src.baselines.preprocess_torch import preprocess_compiled_gpu, preprocess_eager_gpu
from src.models.autoencoder import build_autoencoder
from src.pipelines.cuda_graph import GraphedStage
from src.pipelines.nvtx_markers import stage
from src.triton_kernels.fused_preprocess import fused_preprocess_triton

PreprocessBackend = Literal["eager", "compiled", "triton"]


class AnomalyInspectionPipeline:
    """Batch-1 reconstruction-based anomaly inspection at 256x256.

    Stages: H2D copy -> preprocess (eager | compiled | triton) -> autoencoder (eager or
    torch.compile) -> residual map + top-1% score, read back to the host as a float.
    With ``cuda_graph=True`` everything before the score read-back (H2D, preprocess, model,
    residual map, top-k) is captured once into a CUDA graph and replayed for each frame.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        preprocess: PreprocessBackend = "triton",
        compile_model: bool = False,
        cuda_graph: bool = False,
        seed: int = 0,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        top_k_fraction: float = 0.01,
    ):
        if preprocess not in ("eager", "compiled", "triton"):
            raise ValueError(f"Unknown preprocess backend {preprocess!r}")
        if cuda_graph and compile_model:
            raise ValueError("cuda_graph and compile_model are mutually exclusive")
        self.device = torch.device(device)
        self.preprocess_backend = preprocess
        self.compile_model = compile_model
        self.cuda_graph = cuda_graph
        self.top_k_fraction = top_k_fraction
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.mean_t = torch.tensor(mean, device=self.device)
        self.std_t = torch.tensor(std, device=self.device)
        self.model = build_autoencoder(self.device, seed)
        self._forward = torch.compile(self.model, mode="default") if compile_model else self.model
        self._graph = None

    @property
    def name(self) -> str:
        name = f"pre={self.preprocess_backend}, model={'compiled' if self.compile_model else 'eager'}"
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

    def postprocess(self, x: torch.Tensor, recon: torch.Tensor) -> Tuple[float, torch.Tensor]:
        res_map = compute_residual_map_torch(x, recon)
        return compute_anomaly_score_topk(res_map, self.top_k_fraction), res_map

    def _graph_body(self, device_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.preprocess(device_img)
        res_map = compute_residual_map_torch(x, self.infer(x))
        flat = res_map.view(-1)
        k = max(1, int(self.top_k_fraction * flat.numel()))
        return res_map, torch.topk(flat, k=k)[0].mean()

    @torch.inference_mode()
    def run_step(self, host_image_hwc: torch.Tensor, use_nvtx: bool = False) -> Dict[str, Any]:
        if self.cuda_graph:
            if self._graph is None:
                self._graph = GraphedStage(self._graph_body, self.h2d(host_image_hwc))
            with stage("01-04_CUDA_Graph", use_nvtx):
                res_map, score = self._graph(host_image_hwc)
                return {"anomaly_score": score.item(), "residual_map": res_map}
        with stage("01_H2D", use_nvtx):
            device_img = self.h2d(host_image_hwc)
        with stage("02_Preprocess", use_nvtx):
            x = self.preprocess(device_img)
        with stage("03_Inference", use_nvtx):
            recon = self.infer(x)
        with stage("04_Postprocess", use_nvtx):
            score, res_map = self.postprocess(x, recon)
        return {"anomaly_score": score, "residual_map": res_map}
