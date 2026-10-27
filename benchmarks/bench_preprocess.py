"""Isolated preprocessing benchmark: uint8 HWC -> normalized CHW, cold L2."""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from tabulate import tabulate

from benchmarks.harness import (
    L2CacheFlusher, device_time_us, gpu_state, host_latency_us, require_cuda, save_results,
)
from src.baselines.preprocess_torch import preprocess_compiled_gpu, preprocess_eager_gpu
from src.triton_kernels.fused_preprocess import fused_preprocess_triton
from src.triton_kernels.fused_preprocess_2d import fused_preprocess_2d

RESOLUTIONS = [(256, 256), (640, 640), (720, 1280), (1080, 1920), (2160, 3840)]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def ideal_bytes(h: int, w: int, out_bytes: int) -> int:
    """Compulsory DRAM traffic: read 3 uint8 + write 3 outputs per pixel."""
    return h * w * 3 * (1 + out_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()

    device = require_cuda()
    before = gpu_state()
    torch.manual_seed(0)
    flusher = L2CacheFlusher(64, device)
    mean_t = torch.tensor(MEAN, device=device)
    std_t = torch.tensor(STD, device=device)

    results, rows = [], []
    for h, w in RESOLUTIONS:
        img = torch.randint(0, 256, (h, w, 3), dtype=torch.uint8, device=device)
        methods = {
            "eager_fp32": (lambda: preprocess_eager_gpu(img, mean_t, std_t), 4),
            "compiled_fp32": (lambda: preprocess_compiled_gpu(img, mean_t, std_t), 4),
            "triton_fp32": (lambda: fused_preprocess_triton(img, MEAN, STD), 4),
            "triton_2d_fp32": (lambda: fused_preprocess_2d(img, MEAN, STD), 4),
            "eager_fp16": (lambda: preprocess_eager_gpu(img, mean_t, std_t, torch.float16), 2),
            "compiled_fp16": (lambda: preprocess_compiled_gpu(img, mean_t, std_t, torch.float16), 2),
            "triton_fp16": (lambda: fused_preprocess_triton(img, MEAN, STD, torch.float16), 2),
        }
        ref = preprocess_eager_gpu(img, mean_t, std_t)
        for name, (fn, out_bytes) in methods.items():
            out = fn()
            max_err = (out.float() - ref).abs().max().item()
            dev = device_time_us(fn, args.iters, args.warmup, flusher)
            host = host_latency_us(fn, args.iters, args.warmup, flusher)
            nbytes = ideal_bytes(h, w, out_bytes)
            bw = nbytes / (dev["p50"] * 1e-6) / 1e9
            results.append({
                "resolution": f"{w}x{h}", "height": h, "width": w, "method": name,
                "ideal_bytes": nbytes, "device_us": dev, "host_us": host,
                "effective_bandwidth_gbs": bw, "max_abs_err_vs_eager_fp32": max_err,
            })
            rows.append([f"{w}x{h}", name, f"{dev['p50']:.1f}", f"{dev['p95']:.1f}",
                         f"{host['p50']:.1f}", f"{bw:.1f}", f"{max_err:.1e}"])
        del img, ref
        torch.cuda.empty_cache()

    print(tabulate(rows, headers=["Resolution", "Method", "Device p50 (us)", "Device p95 (us)",
                                  "Host p50 (us)", "Eff. BW (GB/s)", "Max |err|"], tablefmt="github"))
    save_results("preprocess", {"iters": args.iters, "warmup": args.warmup, "l2_flush_mb": 64,
                                "mean": MEAN, "std": STD}, results, before)


if __name__ == "__main__":
    main()
