import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from tabulate import tabulate
from src.baselines.preprocess_torch import preprocess_eager_gpu, preprocess_compiled_gpu
from src.triton_kernels.fused_preprocess import fused_preprocess_triton
from src.triton_kernels.fused_preprocess_failed import fused_preprocess_failed_2d
from benchmarks.cache_utils import L2CacheFlusher

def measure_isolated_kernel(func, *args, warmups: int = 50, iterations: int = 500, flusher=None):
    # Warmup
    for _ in range(warmups):
        _ = func(*args)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        if flusher:
            flusher.flush()
        start_events[i].record()
        _ = func(*args)
        end_events[i].record()
    
    torch.cuda.synchronize()
    latencies_us = [s.elapsed_time(e) * 1000.0 for s, e in zip(start_events, end_events)]  # convert ms to us
    return {
        "p50_us": float(np.percentile(latencies_us, 50)),
        "p95_us": float(np.percentile(latencies_us, 95)),
        "p99_us": float(np.percentile(latencies_us, 99)),
        "mean_us": float(np.mean(latencies_us)),
        "std_us": float(np.std(latencies_us)),
    }

def run_isolated_benchmarks():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemError("CUDA required for isolated benchmarking!")

    flusher = L2CacheFlusher(size_mb=64, device=device)
    mean_t = torch.tensor([0.485, 0.456, 0.406], device=device)
    std_t = torch.tensor([0.229, 0.224, 0.225], device=device)
    mean_tuple = (0.485, 0.456, 0.406)
    std_tuple = (0.229, 0.224, 0.225)

    shapes = [
        (256, 256, "256x256 (Anomaly)"),
        (640, 640, "640x640 (Detection)"),
        (1280, 720, "1280x720 (HD Stream)"),
    ]

    results_table = []

    for H, W, label in shapes:
        img_uint8 = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8, device=device)
        num_pixels = H * W
        # Theoretical Bytes: Read 3*P bytes (uint8) + Write 3*P*4 bytes (float32)
        total_bytes = (num_pixels * 3 * 1) + (num_pixels * 3 * 4)

        # 1. PyTorch Eager
        m_eager = measure_isolated_kernel(preprocess_eager_gpu, img_uint8, mean_t, std_t, flusher=flusher)
        # 2. PyTorch Inductor (torch.compile)
        m_comp = measure_isolated_kernel(preprocess_compiled_gpu, img_uint8, mean_t, std_t, flusher=flusher)
        # 3. Triton 1D Fused (Production with Zero-Sync Tuple Dispatch)
        m_triton = measure_isolated_kernel(fused_preprocess_triton, img_uint8, mean_tuple, std_tuple, flusher=flusher)
        # 4. Triton 2D Failed Variant
        m_failed = measure_isolated_kernel(fused_preprocess_failed_2d, img_uint8, mean_t, std_t, flusher=flusher)
        # 5. Triton FP16 Fused
        m_triton_fp16 = measure_isolated_kernel(
            fused_preprocess_triton, img_uint8, mean_tuple, std_tuple, torch.float16, flusher=flusher
        )

        bw_triton_gbps = (total_bytes / (m_triton["p50_us"] * 1e-6)) / 1e9
        speedup_vs_eager = m_eager["p50_us"] / m_triton["p50_us"]
        speedup_vs_comp = m_comp["p50_us"] / m_triton["p50_us"]

        results_table.append([
            label,
            f"{m_eager['p50_us']:.1f}",
            f"{m_comp['p50_us']:.1f}",
            f"{m_failed['p50_us']:.1f}",
            f"{m_triton['p50_us']:.1f}",
            f"{m_triton_fp16['p50_us']:.1f}",
            f"{bw_triton_gbps:.1f} GB/s",
            f"{speedup_vs_eager:.2f}x",
            f"{speedup_vs_comp:.2f}x",
        ])

    headers = [
        "Resolution", "Eager p50 (µs)", "Compiled p50 (µs)", "Failed 2D (µs)",
        "Triton 1D (µs)", "Triton FP16 (µs)", "Effective BW", "Speedup vs Eager", "Speedup vs Compiled"
    ]
    
    divider = "=" * 80
    print("\n" + divider)
    print("ISOLATED PREPROCESSING KERNEL BENCHMARK RESULTS (Zero-Sync Dispatch)")
    print(divider)
    print(tabulate(results_table, headers=headers, tablefmt="github"))
    print(divider)

if __name__ == "__main__":
    run_isolated_benchmarks()
