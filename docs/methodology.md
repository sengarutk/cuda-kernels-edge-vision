# Measurement Methodology

Every number in [results.md](results.md), the README and the paper is produced by the scripts
below and rendered from `results/*.json` by `scripts/render_report.py`. Nothing is typed by hand.
Each result file stores the environment (GPU, driver, library versions, OS, git commit) and the
GPU clock, temperature and power state before and after the run.

## Two timing modes

Implemented in [`benchmarks/harness.py`](../benchmarks/harness.py).

| Mode | What it measures | How |
| --- | --- | --- |
| **Device time** | GPU execution of the work, with CPU launch overhead hidden | CUDA events around the call. Before each call the stream is pre-filled with a spin kernel (`torch.cuda._sleep`) that runs longer than twice the measured host enqueue time, so the GPU never waits for the CPU inside the timed window. |
| **Host latency** | What an application waits for | `time.perf_counter()` from just before the call until `torch.cuda.synchronize()` returns. Includes Python dispatch, kernel launches, internal synchronizations and GPU execution. |

Device time is only meaningful for calls that never synchronize. Calls that do synchronize
(the candidate filters read back the survivor count; the anomaly score is read back with
`.item()`) are flagged `includes_host_sync`: their device time also includes GPU idle time
while the host waits.

Reporting both modes matters at batch 1. Many operators here take a few microseconds on the
GPU but tens of microseconds to dispatch from Python, so device time alone overstates what a
fused kernel saves an application, and host latency alone hides what the GPU actually does.

## Cache state

- **Cold L2** (isolated kernel benchmarks): before every timed call a 64 MB buffer is *read*
  (summed). The RTX 4050 Laptop GPU has a 24 MB L2, and the harness refuses flush buffers
  smaller than twice the L2 size. Host latency in cold mode synchronizes after the flush, so
  the flush itself is never timed.

  The flush must read, not write. The common write-based flush (`buffer.zero_()`, also used by
  `triton.testing.do_bench`) leaves L2 full of dirty lines, and the next kernel pays to write
  them back to DRAM. On this GPU that added roughly 100 µs to a 1080p preprocessing call and
  made a kernel that Nsight Compute shows to be 1.8× slower look equally fast. A read-based
  flush leaves clean lines that can be evicted for free.
- **Warm** (pipelines; the filter benchmark reports both): no flush. This is the steady state
  of a camera loop processing frames of the same shape.

## Iterations and statistics

- Isolated kernels: 30 warm-up calls, then 300 timed calls per mode and configuration.
- Pipeline stages: 50 warm-up calls, 200 timed calls per stage and mode.
- End-to-end pipelines: every configuration is warmed up (50 frames), then timed in 10
  **interleaved blocks** of 300 frames. The configuration order rotates per block, so slow drift
  in clocks or temperature affects every configuration alike. The tables report pooled p50/p95
  over the 3,000 frames and the spread of the 10 block medians, `(max - min) / median`, as a
  stability indicator.
- Reported statistics: p50, p95, p99, mean, standard deviation, min, max, n.

Warm-up absorbs Triton JIT compilation and autotuning, TorchInductor compilation, cuDNN
algorithm selection and CUDA-graph capture; none of these are timed.

## Workloads and inputs

- **Preprocessing:** random uint8 frames at 256x256, 640x640, 1280x720, 1920x1080 and
  3840x2160. Each method receives its normalization constants in its own synchronization-free
  form (device tensors for PyTorch, Python tuples for Triton). Every output is checked against
  eager FP32, and the largest deviation is stored with the result.
- **Candidate filter:** real YOLOv8n head outputs for the Ultralytics `bus.jpg` frame,
  letterboxed to 640x640 (8,400 candidates) and 1280x1280 (33,600 candidates), confidence
  threshold 0.25. The Triton output is checked against eager before timing.
- **Pipelines:** the committed 256x256 texture fixture (anomaly) and the letterboxed `bus.jpg`
  frame (detection), both in pinned host memory. The anomaly autoencoder uses seeded random
  weights; YOLOv8n uses the official pretrained COCO weights (SHA-256 verified).

## Nsight Compute

`scripts/profile_ncu.sh` profiles each target in
[`profiling/ncu_targets.py`](../profiling/ncu_targets.py). Each target is warmed up, then
exactly one call is wrapped in `cudaProfilerStart/Stop` and captured with
`ncu --profile-from-start off`. The metric list lives in
[`profiling/ncu_metrics_config.yaml`](../profiling/ncu_metrics_config.yaml). Raw reports are
exported to CSV in `results/ncu/` and parsed by
[`profiling/parse_ncu_metrics.py`](../profiling/parse_ncu_metrics.py).

Nsight Compute locks clocks to base frequency and flushes caches before each kernel by default.
Its durations are therefore longer than the benchmark's device times and are used only to
explain behaviour (occupancy, memory efficiency, stall reasons), not as latency results.

## Known sources of variance

The GPU is a laptop part under WSL2. Boost clocks depend on temperature and power, and Windows
can schedule other GPU work. The interleaved blocks and the stored GPU state make this visible;
nothing removes it. Treat differences smaller than the block spread as noise.
