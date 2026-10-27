"""PyTorch profiler capture of both pipelines (Triton and eager configurations).

Writes per-operator summaries to results/torch_profiler/<name>_summary.txt and Chrome
traces to results/torch_profiler/<name>_trace.json (traces are git-ignored: ~10 MB each).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.frames import anomaly_frame, detection_frame
from src.models.yolo import load_yolov8n
from src.pipelines.anomaly_pipeline import AnomalyInspectionPipeline
from src.pipelines.detection_pipeline import DetectionInspectionPipeline

OUT = Path(__file__).resolve().parents[1] / "results" / "torch_profiler"


def profile(name: str, pipe, host: torch.Tensor, steps: int = 50) -> None:
    for _ in range(20):
        pipe.run_step(host)
    torch.cuda.synchronize()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        for _ in range(steps):
            pipe.run_step(host, use_nvtx=True)
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(OUT / f"{name}_trace.json"))
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
    header = f"# {name}: {pipe.name}, {steps} steps\n"
    (OUT / f"{name}_summary.txt").write_text(header + table + "\n")
    print(header + table)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    OUT.mkdir(parents=True, exist_ok=True)
    host_ae = torch.from_numpy(anomaly_frame()).pin_memory()
    for pre in ("eager", "triton"):
        profile(f"anomaly_{pre}", AnomalyInspectionPipeline(preprocess=pre), host_ae)
    host_det = torch.from_numpy(detection_frame(640)).pin_memory()
    model = load_yolov8n(torch.device("cuda:0"))
    for backend in ("eager", "triton"):
        pipe = DetectionInspectionPipeline(preprocess=backend, postprocess=backend, model=model)
        profile(f"detection_{backend}", pipe, host_det)


if __name__ == "__main__":
    main()
