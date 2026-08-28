import sys
import os
import argparse
import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.triton_kernels.fused_preprocess import fused_preprocess_triton
from src.triton_kernels.fused_preprocess_failed import fused_preprocess_failed_2d
from src.triton_kernels.candidate_filter import candidate_filter_triton
from src.baselines.preprocess_torch import preprocess_eager_gpu

def run_target(target_name: str, iterations: int = 15):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemError("CUDA is required for NCU target execution!")

    mean_tuple = (0.485, 0.456, 0.406)
    std_tuple = (0.229, 0.224, 0.225)
    mean_t = torch.tensor([0.485, 0.456, 0.406], device=device)
    std_t = torch.tensor([0.229, 0.224, 0.225], device=device)

    print(f"[+] Executing NCU Target: {target_name} ({iterations} iterations)...")

    if target_name == "triton_preprocess_256":
        img = torch.randint(0, 256, (256, 256, 3), dtype=torch.uint8, device=device)
        for _ in range(5):
            fused_preprocess_triton(img, mean_tuple, std_tuple)
        torch.cuda.synchronize()
        for _ in range(iterations):
            fused_preprocess_triton(img, mean_tuple, std_tuple)

    elif target_name == "triton_preprocess_640":
        img = torch.randint(0, 256, (640, 640, 3), dtype=torch.uint8, device=device)
        for _ in range(5):
            fused_preprocess_triton(img, mean_tuple, std_tuple)
        torch.cuda.synchronize()
        for _ in range(iterations):
            fused_preprocess_triton(img, mean_tuple, std_tuple)

    elif target_name == "failed_preprocess_640":
        img = torch.randint(0, 256, (640, 640, 3), dtype=torch.uint8, device=device)
        for _ in range(5):
            fused_preprocess_failed_2d(img, mean_t, std_t)
        torch.cuda.synchronize()
        for _ in range(iterations):
            fused_preprocess_failed_2d(img, mean_t, std_t)

    elif target_name == "eager_preprocess_640":
        img = torch.randint(0, 256, (640, 640, 3), dtype=torch.uint8, device=device)
        for _ in range(5):
            preprocess_eager_gpu(img, mean_t, std_t)
        torch.cuda.synchronize()
        for _ in range(iterations):
            preprocess_eager_gpu(img, mean_t, std_t)

    elif target_name == "candidate_filter_8400":
        preds = torch.randn(1, 8400, 84, device=device)
        for _ in range(5):
            candidate_filter_triton(preds, 0.25)
        torch.cuda.synchronize()
        for _ in range(iterations):
            candidate_filter_triton(preds, 0.25)

    else:
        raise ValueError(f"Unknown target name: {target_name}")

    torch.cuda.synchronize()
    print(f"[✓] Completed NCU Target: {target_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=15)
    args = parser.parse_args()
    run_target(args.target, args.iterations)
