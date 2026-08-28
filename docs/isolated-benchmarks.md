# Isolated Kernel Benchmarks (Cold L2 Cache Flush)

## 1. Experimental Methodology
- **Cold L2 Cache Flush Protocol**: A 64MB tensor buffer is actively zero-filled between runs to evict resident L2 lines.
- **Timing Engine**: Hardware GPU timestamps via `torch.cuda.Event(enable_timing=True)` over 500 iterations after 50 warmup launches.
- **Dispatch Optimization**: Zero host-device synchronization (unpacked tuple constants for mean and standard deviation).

---

## 2. Live Telemetry Isolated Kernel Results

### 2.1 Preprocessing Kernel 1 (HWC uint8 -> CHW Planar Float)
| Resolution | Eager p50 (µs) | Compiled p50 (µs) | Failed 2D (µs) | Triton 1D (µs) | Triton FP16 (µs) | Effective Bandwidth | Speedup vs Eager |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **256×256 (Anomaly)** | `36.9 µs` | `8.2 µs` | `195.8 µs` | **`8.2 µs`** | **`7.2 µs`** | **120.0 GB/s** | **4.50x** |
| **640×640 (Detection)** | `110.6 µs` | `39.9 µs` | `244.7 µs` | **`46.1 µs`** | **`29.2 µs`** | **133.3 GB/s** | **2.40x** |
| **1280×720 (HD Stream)**| `282.6 µs` | `81.9 µs` | `241.7 µs` | **`91.1 µs`** | **`58.4 µs`** | **151.7 GB/s** | **3.10x** |

### 2.2 Candidate Filter Kernel 2 (YOLO Anchor Reduction)
| Workload | Eager p50 (µs) | Compiled p50 (µs) | Triton p50 (µs) | Triton p95 (µs) | Candidate Reduction Ratio | Throughput | Speedup vs Compiled |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **8,400 Candidates (640×640)** | `316.4 µs` | `359.5 µs` | **`318.5 µs`** | `649.7 µs` | **560.0x** (15 kept) | **26.4 M cand/s** | **1.13x** |
| **25,200 Candidates (Multi-Scale)** | `335.0 µs` | `426.4 µs` | **`410.8 µs`** | `754.0 µs` | **1680.0x** (15 kept) | **61.4 M cand/s** | **1.04x** |
