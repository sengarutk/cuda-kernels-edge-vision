import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from tabulate import tabulate
from src.pipelines.anomaly_pipeline import AnomalyInspectionPipeline
from src.pipelines.detection_pipeline import DetectionInspectionPipeline
from benchmarks.cache_utils import L2CacheFlusher

@torch.inference_mode()
def benchmark_pipeline_latency(pipeline, host_input: torch.Tensor, warmups: int = 50, iterations: int = 300, flusher=None):
    for _ in range(warmups):
        _ = pipeline.run_step(host_input)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    for i in range(iterations):
        if flusher:
            flusher.flush()
        start_events[i].record()
        _ = pipeline.run_step(host_input)
        end_events[i].record()

    torch.cuda.synchronize()
    latencies_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return {
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "mean_ms": float(np.mean(latencies_ms)),
        "std_ms": float(np.std(latencies_ms)),
    }

def run_e2e_benchmarks():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemError("CUDA required for E2E benchmarking!")

    flusher = L2CacheFlusher(size_mb=64, device=device)

    # 1. Anomaly Pipeline Benchmark (256x256) with Pinned Host Memory
    host_img_256 = torch.randint(0, 256, (256, 256, 3), dtype=torch.uint8).pin_memory()
    pipe_ae_eager = AnomalyInspectionPipeline(device=device, backend="eager")
    pipe_ae_comp = AnomalyInspectionPipeline(device=device, backend="compiled")
    pipe_ae_triton = AnomalyInspectionPipeline(device=device, backend="triton")

    res_ae_eager = benchmark_pipeline_latency(pipe_ae_eager, host_img_256, flusher=flusher)
    res_ae_comp = benchmark_pipeline_latency(pipe_ae_comp, host_img_256, flusher=flusher)
    res_ae_triton = benchmark_pipeline_latency(pipe_ae_triton, host_img_256, flusher=flusher)

    # 2. Detection Pipeline Benchmark (640x640) with Pinned Host Memory
    host_img_640 = torch.randint(0, 256, (640, 640, 3), dtype=torch.uint8).pin_memory()
    pipe_det_eager = DetectionInspectionPipeline(device=device, backend="eager")
    pipe_det_comp = DetectionInspectionPipeline(device=device, backend="compiled")
    pipe_det_pre = DetectionInspectionPipeline(device=device, backend="triton_preprocess")
    pipe_det_post = DetectionInspectionPipeline(device=device, backend="triton_postprocess")
    pipe_det_both = DetectionInspectionPipeline(device=device, backend="triton_both")

    res_det_eager = benchmark_pipeline_latency(pipe_det_eager, host_img_640, flusher=flusher)
    res_det_comp = benchmark_pipeline_latency(pipe_det_comp, host_img_640, flusher=flusher)
    res_det_pre = benchmark_pipeline_latency(pipe_det_pre, host_img_640, flusher=flusher)
    res_det_post = benchmark_pipeline_latency(pipe_det_post, host_img_640, flusher=flusher)
    res_det_both = benchmark_pipeline_latency(pipe_det_both, host_img_640, flusher=flusher)

    # Table: End-to-End Latencies
    e2e_table = [
        ["Anomaly (256x256) - Eager Baseline", f"{res_ae_eager['p50_ms']:.3f} ms", f"{res_ae_eager['p95_ms']:.3f} ms", "1.00x (Baseline)"],
        ["Anomaly (256x256) - Torch Compiled", f"{res_ae_comp['p50_ms']:.3f} ms", f"{res_ae_comp['p95_ms']:.3f} ms", f"{res_ae_eager['p50_ms']/res_ae_comp['p50_ms']:.2f}x"],
        ["Anomaly (256x256) - Fused Triton Kernel 1", f"{res_ae_triton['p50_ms']:.3f} ms", f"{res_ae_triton['p95_ms']:.3f} ms", f"{res_ae_eager['p50_ms']/res_ae_triton['p50_ms']:.2f}x (+{((res_ae_eager['p50_ms'] - res_ae_triton['p50_ms'])/res_ae_eager['p50_ms'])*100:.1f}%)"],
        ["----------------------------------------", "------------", "------------", "----------------"],
        ["Detection (640x640) - Eager Baseline", f"{res_det_eager['p50_ms']:.3f} ms", f"{res_det_eager['p95_ms']:.3f} ms", "1.00x (Baseline)"],
        ["Detection (640x640) - Torch Compiled", f"{res_det_comp['p50_ms']:.3f} ms", f"{res_det_comp['p95_ms']:.3f} ms", f"{res_det_eager['p50_ms']/res_det_comp['p50_ms']:.2f}x"],
        ["Detection (640x640) - Triton Preprocess Only", f"{res_det_pre['p50_ms']:.3f} ms", f"{res_det_pre['p95_ms']:.3f} ms", f"{res_det_eager['p50_ms']/res_det_pre['p50_ms']:.2f}x"],
        ["Detection (640x640) - Triton Filter Only", f"{res_det_post['p50_ms']:.3f} ms", f"{res_det_post['p95_ms']:.3f} ms", f"{res_det_eager['p50_ms']/res_det_post['p50_ms']:.2f}x"],
        ["Detection (640x640) - Both Triton Kernels", f"{res_det_both['p50_ms']:.3f} ms", f"{res_det_both['p95_ms']:.3f} ms", f"{res_det_eager['p50_ms']/res_det_both['p50_ms']:.2f}x (+{((res_det_eager['p50_ms'] - res_det_both['p50_ms'])/res_det_eager['p50_ms'])*100:.1f}%)"],
    ]

    headers = ["Pipeline Configuration", "E2E p50 Latency", "E2E p95 Latency", "Speedup vs Eager"]

    divider = "=" * 90
    print("\n" + divider)
    print("END-TO-END INDUSTRIAL VISION PIPELINE LATENCY RESULTS (Batch Size = 1, Pinned Memory)")
    print(divider)
    print(tabulate(e2e_table, headers=headers, tablefmt="github"))
    print(divider)

if __name__ == "__main__":
    run_e2e_benchmarks()
