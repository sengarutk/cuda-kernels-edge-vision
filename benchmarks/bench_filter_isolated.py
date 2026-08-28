import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from tabulate import tabulate
from src.baselines.postprocess_torch import (
    postprocess_detection_torch_eager,
    postprocess_detection_torch_compiled,
)
from src.triton_kernels.candidate_filter import candidate_filter_triton
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

def run_isolated_filter_benchmarks():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemError("CUDA required for isolated benchmarking!")

    flusher = L2CacheFlusher(size_mb=64, device=device)
    conf_threshold = 0.25

    bench_configs = [
        (8400, 80, "8,400 Candidates (640x640 YOLO Nano)"),
        (25200, 80, "25,200 Candidates (High-Res Multi-Scale)"),
    ]

    results_table = []

    for N, C, label in bench_configs:
        # Realistic candidate tensor: mostly low confidence (< 0.05) with 15 defect proposals
        raw_candidates = np.random.uniform(0.001, 0.04, (1, N, 4 + C)).astype(np.float32)
        raw_candidates[0, :, :4] = np.random.uniform(10.0, 630.0, (N, 4))
        defect_indices = np.random.choice(N, 15, replace=False)
        for idx in defect_indices:
            cls_idx = idx % C
            raw_candidates[0, idx, 4 + cls_idx] = 0.85 + (idx % 10) * 0.01

        preds_t = torch.from_numpy(raw_candidates).to(device)

        # Baseline Eager
        m_eager = measure_isolated_kernel(
            postprocess_detection_torch_eager, preds_t, conf_threshold, flusher=flusher
        )
        # Baseline Compiled
        m_comp = measure_isolated_kernel(
            postprocess_detection_torch_compiled, preds_t, conf_threshold, flusher=flusher
        )
        # Triton Fused Candidate Filter
        m_triton = measure_isolated_kernel(
            candidate_filter_triton, preds_t, conf_threshold, flusher=flusher
        )

        boxes_filt, _, _, _ = candidate_filter_triton(preds_t, conf_threshold)
        num_survivors = boxes_filt.shape[0]
        reduction_ratio = N / max(1, num_survivors)
        throughput_mcand_s = (N / (m_triton["p50_us"] * 1e-6)) / 1e6
        speedup_vs_eager = m_eager["p50_us"] / m_triton["p50_us"]
        speedup_vs_comp = m_comp["p50_us"] / m_triton["p50_us"]

        results_table.append([
            label,
            f"{m_eager['p50_us']:.1f}",
            f"{m_comp['p50_us']:.1f}",
            f"{m_triton['p50_us']:.1f}",
            f"{m_triton['p95_us']:.1f}",
            f"{reduction_ratio:.1f}x ({num_survivors} kept)",
            f"{throughput_mcand_s:.1f} M/s",
            f"{speedup_vs_eager:.2f}x",
            f"{speedup_vs_comp:.2f}x",
        ])

    headers = [
        "Workload", "Eager p50 (µs)", "Compiled p50 (µs)", "Triton p50 (µs)",
        "Triton p95 (µs)", "Reduction Ratio", "Throughput", "Speedup vs Eager", "Speedup vs Comp"
    ]

    divider = "=" * 85
    print("\n" + divider)
    print("ISOLATED DETECTION CANDIDATE FILTER BENCHMARK RESULTS (Cold L2 Cache Flush)")
    print(divider)
    print(tabulate(results_table, headers=headers, tablefmt="github"))
    print(divider)

if __name__ == "__main__":
    run_isolated_filter_benchmarks()
