from typing import Literal, Dict, Any, Tuple
import torch
from src.pipelines.nvtx_markers import nvtx_range
from src.baselines.preprocess_torch import preprocess_eager_gpu, preprocess_compiled_gpu
from src.triton_kernels.fused_preprocess import fused_preprocess_triton
from src.models.mock_autoencoder import IndustrialAutoencoder
from src.baselines.anomaly_baselines import compute_residual_map_torch, compute_anomaly_score_topk

class AnomalyInspectionPipeline:
    """Full single-stream batch-1 Industrial Anomaly Inspection Pipeline."""
    def __init__(
        self,
        device: str = "cuda:0",
        backend: Literal["eager", "compiled", "triton"] = "triton",
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.device = torch.device(device)
        self.backend = backend
        self.mean_tuple = mean
        self.std_tuple = std
        self.mean_t = torch.tensor(mean, device=self.device)
        self.std_t = torch.tensor(std, device=self.device)
        
        self.model = IndustrialAutoencoder().to(self.device).eval()
        if self.backend == "compiled":
            self.model = torch.compile(self.model, mode="default")

    def run_step(self, host_image_hwc: torch.Tensor, use_nvtx: bool = False) -> Dict[str, Any]:
        """Runs batch-1 anomaly inspection end-to-end: H2D -> Preprocess -> Model -> Residual -> D2H."""
        # 1. H2D Transfer
        if use_nvtx:
            with nvtx_range("01_H2D_Transfer"):
                device_img = host_image_hwc.to(self.device, non_blocking=True)
        else:
            device_img = host_image_hwc.to(self.device, non_blocking=True)

        # 2. Preprocessing
        if self.backend == "triton":
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
        else:  # eager
            if use_nvtx:
                with nvtx_range("02_Eager_Preprocessing"):
                    input_tensor = preprocess_eager_gpu(device_img, self.mean_t, self.std_t)
            else:
                input_tensor = preprocess_eager_gpu(device_img, self.mean_t, self.std_t)

        # 3. Model Forward
        if use_nvtx:
            with nvtx_range("03_Model_Inference"):
                with torch.no_grad():
                    recon = self.model(input_tensor)
        else:
            with torch.no_grad():
                recon = self.model(input_tensor)

        # 4. Postprocessing Residual & Top-K Scoring
        if use_nvtx:
            with nvtx_range("04_Postprocess_Residual"):
                res_map = compute_residual_map_torch(input_tensor, recon)
                score = compute_anomaly_score_topk(res_map, top_k_fraction=0.01)
        else:
            res_map = compute_residual_map_torch(input_tensor, recon)
            score = compute_anomaly_score_topk(res_map, top_k_fraction=0.01)

        return {"anomaly_score": score, "residual_map": res_map}
