# Changelog

## 1.1.0: corrected and re-measured release

v1.0.0 contained results that were hardcoded, mismeasured or inconsistent. This release removes
all of them, fixes the measurement methodology and regenerates every number from measured data.
**Do not cite numbers from v1.0.0.**

### Errata for v1.0.0

- **Nsight Compute metrics were hardcoded.** `profiling/parse_ncu_metrics.py` wrote a fixed list
  of values instead of parsing profiler output, and `ncu` never ran successfully on the test
  machine. All occupancy, register, DRAM, cache-hit and warp-stall figures in v1.0.0 (README,
  docs, paper Table 3 and the stall-breakdown figure) were not measurements.
- **The 2D-tiling "negative study" was a measurement artifact.** The 2D kernel's dispatcher read
  its CUDA-tensor `mean`/`std` with `.tolist()` inside the timed region (a device sync), while the
  1D kernel received tuples. The reported 5.03× slowdown and its register-pressure explanation are
  withdrawn.
- **The "YOLO Nano" model was not a detector.** It was three small convolutions, global pooling and
  a 45M-parameter linear layer with random weights. Its scores never exceeded the 0.25 threshold,
  so NMS never ran in any benchmark or test. It is replaced by pretrained YOLOv8n.
- **The stage breakdown reused stage times.** Only preprocessing was measured per backend; H2D,
  model and postprocessing times were copied between columns, so the reported Amdahl agreement was
  circular. The stage table was also labelled "pure GPU" although it included launch gaps.
- **"End-to-end wall-clock" numbers were GPU-event times**, partly masked by the cache-flush kernel.
- **The cold-cache flush wrote instead of read.** `buffer.zero_()` leaves L2 full of dirty lines whose
  write-back is charged to the next (measured) kernel; it is replaced by a read-based flush.
- **Inconsistent numbers:** the YOLO end-to-end gain was 4.0% in the README and 12.3% in the
  paper; 1280x720 results differed between README, paper and figure; "5.03×" was computed against
  47.6 µs but quoted next to 46.1 µs; "11.5%" was attributed to a change that is 2.3%.
- **Unsupported methodology claims:** "5 independent sessions, ±1.8% variance" had no
  implementation. `scripts/verify_evidence_integrity.py` checked hardcoded constants against
  themselves.
- **Hardware facts:** L2 was given as 32 MB (the GPU reports 24 MB); the OS as Ubuntu 22.04
  (it is 24.04); warp occupancy was explained against a 32-warp limit (sm_89 allows 48).
- **Kernel descriptions:** the candidate filter was described as "register-unrolled" and as writing
  survivors contiguously. Its class loop had a runtime bound, and compaction was done afterwards by
  three PyTorch boolean-mask gathers (three host syncs). Preprocessing loads were called "vectorized".
- **Citations:** two references could not be verified to exist (`chen2020deep`,
  `nath2010analyzing`) and one had the wrong venue (`volkov2008fitting`). They were removed.
- **Other:** the paper's repository link returned 404; the profiling traces were captured from the
  eager backend, not the code that claimed to produce them; "100% baseline/pipeline coverage" was
  false; several tests could not fail (tautological checks, expected-output fixtures never
  compared); scripts hardcoded `/home/sengar/...` paths.

### Changes

- **Kernels:** normalization folded into one FMA per element; the 2D variant uses the same
  synchronization-free dispatch as the 1D kernel (renamed `fused_preprocess_2d`). The candidate
  filter is rewritten as a single pass: a compile-time-unrolled class reduction with coalesced
  loads on YOLOv8's native `[1, 84, N]` layout, threshold, xywh→xyxy decode and in-kernel stream
  compaction (atomic reservation plus prefix sum), with one host synchronization per call.
- **Models:** pretrained YOLOv8n (Ultralytics, SHA-256 verified) replaces the mock detector. A
  workaround fixes a TorchDynamo mis-trace of Ultralytics' `C2f`/`SPPF` blocks that made
  `torch.compile`d YOLOv8n silently wrong. The autoencoder is seeded.
- **Pipelines:** per-stage backends, an optional CUDA-graph mode, real candidate and detection counts.
- **Benchmarks:** a shared harness with two clearly defined modes (device time with launch overhead
  hidden; host wall-clock latency), interleaved repeated blocks, environment and GPU-state capture,
  and JSON results in `results/`.
- **Profiling:** real Nsight Compute collection (`scripts/profile_ncu.sh`) and a CSV parser.
- **Reporting:** `scripts/render_report.py` generates `docs/results.md`, the README results block
  and the paper's numbers (`\res{...}` macros); `make check` fails on any drift.
- **Tests:** rewritten to compare against references (NumPy, eager, committed fixtures, real YOLOv8n
  outputs); GPU tests skip cleanly on CPU-only machines; CI runs the CPU subset and the report check.
