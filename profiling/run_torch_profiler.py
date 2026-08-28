import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.pipelines.anomaly_pipeline import AnomalyInspectionPipeline
from src.pipelines.detection_pipeline import DetectionInspectionPipeline

def profile_pipeline(pipeline_name: str, pipeline, dummy_input: torch.Tensor, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[+] Profiling {pipeline_name} (Warmup: 20, Timed: 50)...")
    
    # Warmup
    for _ in range(20):
        pipeline.run_step(dummy_input)
    torch.cuda.synchronize()

    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in range(50):
            pipeline.run_step(dummy_input)
            torch.cuda.synchronize()

    trace_path = os.path.join(output_dir, f"{pipeline_name}_trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"[✓] Trace exported to {trace_path}")
    
    table_str = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
    print(table_str)
    
    report_path = os.path.join(output_dir, f"{pipeline_name}_summary.txt")
    with open(report_path, "w") as f:
        f.write(table_str)

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemError("Profiling must be executed on a valid CUDA device!")

    out_dir = "profiling_output"
    
    # 1. Anomaly Pipeline (256x256 uint8)
    anomaly_pipe = AnomalyInspectionPipeline(device=device)
    dummy_anomaly_img = torch.randint(0, 256, (256, 256, 3), dtype=torch.uint8)
    profile_pipeline("anomaly_pipeline_256", anomaly_pipe, dummy_anomaly_img, out_dir)

    # 2. Detection Pipeline (640x640 uint8)
    detection_pipe = DetectionInspectionPipeline(device=device)
    dummy_detection_img = torch.randint(0, 256, (640, 640, 3), dtype=torch.uint8)
    profile_pipeline("detection_pipeline_640", detection_pipe, dummy_detection_img, out_dir)

if __name__ == "__main__":
    main()
