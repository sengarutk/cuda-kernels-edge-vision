import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from tabulate import tabulate
from src.baselines.preprocess_torch import preprocess_eager_gpu
from src.triton_kernels.fused_preprocess import fused_preprocess_triton
from src.models.mock_autoencoder import IndustrialAutoencoder
from src.baselines.anomaly_baselines import compute_residual_map_torch, compute_anomaly_score_topk

@torch.inference_mode()
def measure_stage_breakdown():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Pinned host memory allocation for true asynchronous PCIe transfers
    host_img = torch.randint(0, 256, (256, 256, 3), dtype=torch.uint8).pin_memory()
    mean_tuple = (0.485, 0.456, 0.406)
    std_tuple = (0.229, 0.224, 0.225)
    mean_t = torch.tensor(mean_tuple, device=device)
    std_t = torch.tensor(std_tuple, device=device)
    model = IndustrialAutoencoder().to(device).eval()

    def time_fn(fn, *args, iters=200):
        start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for _ in range(20): 
            fn(*args)
        torch.cuda.synchronize()
        for i in range(iters):
            start[i].record()
            out = fn(*args)
            end[i].record()
        torch.cuda.synchronize()
        latencies = [s.elapsed_time(e) * 1000.0 for s, e in zip(start, end)]
        return float(np.median(latencies)), out

    # 1. H2D Transfer (Page-locked pinned PCIe copy)
    t_h2d, device_img = time_fn(lambda x: x.to(device, non_blocking=True), host_img)
    # 2. Preprocess Eager vs Triton
    t_pre_eager, in_eager = time_fn(preprocess_eager_gpu, device_img, mean_t, std_t)
    t_pre_triton, in_triton = time_fn(fused_preprocess_triton, device_img, mean_tuple, std_tuple)
    # 3. Model Forward
    t_model, recon = time_fn(lambda x: model(x), in_triton)
    # 4. Postprocessing Residual & Top-K Scoring
    def post_fn(x, x_hat):
        res = compute_residual_map_torch(x, x_hat)
        return compute_anomaly_score_topk(res)
    t_post, score = time_fn(post_fn, in_triton, recon)

    total_eager = t_h2d + t_pre_eager + t_model + t_post
    total_triton = t_h2d + t_pre_triton + t_model + t_post

    breakdown = [
        ["01_H2D_Transfer (Pinned)", f"{t_h2d:.1f} µs", f"{t_h2d/total_eager*100:.1f}%", f"{t_h2d:.1f} µs", f"{t_h2d/total_triton*100:.1f}%"],
        ["02_Preprocessing", f"{t_pre_eager:.1f} µs", f"{t_pre_eager/total_eager*100:.1f}%", f"{t_pre_triton:.1f} µs", f"{t_pre_triton/total_triton*100:.1f}%"],
        ["03_Model_Inference", f"{t_model:.1f} µs", f"{t_model/total_eager*100:.1f}%", f"{t_model:.1f} µs", f"{t_model/total_triton*100:.1f}%"],
        ["04_Postprocessing", f"{t_post:.1f} µs", f"{t_post/total_eager*100:.1f}%", f"{t_post:.1f} µs", f"{t_post/total_triton*100:.1f}%"],
        ["TOTAL PIPELINE", f"{total_eager:.1f} µs", "100.0%", f"{total_triton:.1f} µs", "100.0%"],
    ]

    headers = ["Stage Name", "Eager Time (µs)", "Eager Share (%)", "Triton Time (µs)", "Triton Share (%)"]
    divider = "=" * 85
    print("\n" + divider)
    print("STAGE-BY-STAGE LATENCY BREAKDOWN & AMDAHL RETENTION (256x256 Anomaly)")
    print(divider)
    print(tabulate(breakdown, headers=headers, tablefmt="github"))
    print(divider)

if __name__ == "__main__":
    measure_stage_breakdown()
