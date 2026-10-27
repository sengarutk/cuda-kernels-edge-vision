"""Isolated YOLO candidate-filter benchmark on real YOLOv8n outputs."""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import triton
from tabulate import tabulate

from benchmarks.harness import (
    L2CacheFlusher, device_time_us, gpu_state, host_latency_us, require_cuda, save_results,
)
from src.baselines.postprocess_torch import (
    postprocess_detection_torch_compiled, postprocess_detection_torch_eager,
)
from src.frames import detection_frame
from src.models.yolo import load_yolov8n
from src.triton_kernels.candidate_filter import (
    _candidate_filter_kernel, candidate_filter_triton, candidate_matrix,
)

CONF = 0.25


def yolo_output(model, size: int, device: str) -> torch.Tensor:
    frame = torch.from_numpy(detection_frame(size)).to(device)
    x = frame.permute(2, 0, 1).float().div(255.0).unsqueeze(0).contiguous()
    with torch.inference_mode():
        return model(x).clone()


def kernel_only(preds: torch.Tensor, layout: str):
    """Launches only the Triton kernel (no count read-back): GPU cost of the fused pass."""
    cand = candidate_matrix(preds, layout)
    n = cand.shape[0]
    bufs = (torch.empty((n, 4), device=cand.device), torch.empty(n, device=cand.device),
            torch.empty(n, dtype=torch.int64, device=cand.device),
            torch.empty(n, dtype=torch.int64, device=cand.device))
    count = torch.zeros(1, dtype=torch.int32, device=cand.device)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_N"]),)

    def run():
        count.zero_()
        _candidate_filter_kernel[grid](cand, cand.stride(0), cand.stride(1), *bufs, count, n, CONF,
                                       NUM_CLASSES=cand.shape[1] - 4, BOX_XYWH=True)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()

    device = require_cuda()
    before = gpu_state()
    flusher = L2CacheFlusher(64, device)
    model = load_yolov8n(torch.device(device))

    results, rows = [], []
    for size in (640, 1280):
        preds_cf = yolo_output(model, size, device)            # [1, 84, N] native YOLOv8 layout
        preds_cl = preds_cf.transpose(1, 2).contiguous()       # [1, N, 84] row-major layout
        n = preds_cf.shape[2]
        ref = postprocess_detection_torch_eager(preds_cf, CONF)
        k = ref[0].shape[0]
        tb, ts, tc, ti = candidate_filter_triton(preds_cf, CONF)
        order = torch.argsort(ti)
        assert torch.equal(ti[order], ref[3]) and torch.equal(tc[order], ref[2])
        torch.testing.assert_close(tb[order], ref[0])
        torch.testing.assert_close(ts[order], ref[1])

        methods = {
            "eager": (lambda: postprocess_detection_torch_eager(preds_cf, CONF), True),
            "compiled": (lambda: postprocess_detection_torch_compiled(preds_cf, CONF), True),
            "triton": (lambda: candidate_filter_triton(preds_cf, CONF), True),
            "triton_kernel_only": (kernel_only(preds_cf, "channels_first"), False),
            "triton_kernel_only_row_major": (kernel_only(preds_cl, "channels_last"), False),
        }
        for name, (fn, syncs) in methods.items():
            for cache, fl in (("cold", flusher), ("warm", None)):
                dev = device_time_us(fn, args.iters, args.warmup, fl)
                host = host_latency_us(fn, args.iters, args.warmup, fl)
                results.append({
                    "input_size": size, "num_candidates": n, "num_survivors": k, "method": name,
                    "l2": cache, "device_us": {**dev, "includes_host_sync": syncs}, "host_us": host,
                })
                rows.append([f"{size} ({n})", k, name, cache, f"{dev['p50']:.1f}",
                             f"{host['p50']:.1f}", f"{host['p95']:.1f}"])

    print(tabulate(rows, headers=["Input (cands)", "Kept", "Method", "L2", "Device p50 (us)",
                                  "Host p50 (us)", "Host p95 (us)"], tablefmt="github"))
    save_results("filter", {"iters": args.iters, "warmup": args.warmup, "conf_threshold": CONF,
                            "model": "yolov8n.pt (COCO)", "input": "bus.jpg letterboxed"},
                 results, before)


if __name__ == "__main__":
    main()
