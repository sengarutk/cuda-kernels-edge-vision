"""Workloads profiled by Nsight Compute (see scripts/profile_ncu.sh).

Each target warms up (JIT compilation, autotuning, cuDNN heuristics), then wraps exactly
one call in cudaProfilerStart/Stop, so ``ncu --profile-from-start off`` captures only the
kernels of that call.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CONFIG = Path(__file__).with_name("ncu_metrics_config.yaml")


def metric_list() -> list:
    cfg = yaml.safe_load(CONFIG.read_text())
    return [m for group in cfg["metrics"].values() for m in group]


def _image(h: int, w: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 256, (h, w, 3), dtype=torch.uint8, device="cuda")


def _yolo_preds(layout: str) -> torch.Tensor:
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "yolov8n_bus_raw_640.pt"
    preds = torch.load(path, weights_only=True).cuda()  # [1, 84, 8400]
    return preds if layout == "channels_first" else preds.transpose(1, 2).contiguous()


def build_targets():
    from src.baselines.postprocess_torch import postprocess_detection_torch_eager
    from src.baselines.preprocess_torch import preprocess_compiled_gpu, preprocess_eager_gpu
    from src.triton_kernels.candidate_filter import candidate_filter_triton
    from src.triton_kernels.fused_preprocess import fused_preprocess_triton
    from src.triton_kernels.fused_preprocess_2d import fused_preprocess_2d

    mean_t = torch.tensor(MEAN, device="cuda")
    std_t = torch.tensor(STD, device="cuda")
    targets = {}
    for h, w in ((640, 640), (1080, 1920)):
        tag = f"{w}x{h}"
        img = _image(h, w)
        targets[f"preprocess_triton_{tag}"] = lambda img=img: fused_preprocess_triton(img, MEAN, STD)
        targets[f"preprocess_triton_fp16_{tag}"] = (
            lambda img=img: fused_preprocess_triton(img, MEAN, STD, torch.float16))
        targets[f"preprocess_triton_2d_{tag}"] = lambda img=img: fused_preprocess_2d(img, MEAN, STD)
        targets[f"preprocess_eager_{tag}"] = lambda img=img: preprocess_eager_gpu(img, mean_t, std_t)
        targets[f"preprocess_compiled_{tag}"] = (
            lambda img=img: preprocess_compiled_gpu(img, mean_t, std_t))
    cf, cl = _yolo_preds("channels_first"), _yolo_preds("channels_last")
    targets["filter_triton_8400"] = lambda: candidate_filter_triton(cf, 0.25)
    targets["filter_triton_row_major_8400"] = (
        lambda: candidate_filter_triton(cl, 0.25, layout="channels_last"))
    targets["filter_eager_8400"] = lambda: postprocess_detection_torch_eager(cf, 0.25)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target")
    parser.add_argument("--list", action="store_true", help="print target names")
    parser.add_argument("--print-metrics", action="store_true", help="print comma-separated metrics")
    args = parser.parse_args()

    if args.print_metrics:
        print(",".join(metric_list()))
        return
    targets = build_targets()
    if args.list:
        print("\n".join(targets))
        return
    if args.target not in targets:
        raise SystemExit(f"Unknown target {args.target!r}; use --list")

    fn = targets[args.target]
    with torch.inference_mode():
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        fn()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
