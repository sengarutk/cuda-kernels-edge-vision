# CUDA/Triton Kernels for Edge Vision

A profiling-driven systems study evaluating custom Triton kernels for preprocessing and postprocessing in batch-1 industrial vision pipelines.

> **Hardware Scope Disclosure**: All benchmarks were collected on an NVIDIA GeForce RTX 4050 Laptop GPU (6 GB GDDR6 VRAM, `sm_89` Ada Lovelace) under WSL2 (Ubuntu 22.04 LTS). This is an edge-workstation testbed, not an embedded Jetson or industrial smart camera.

---

## Research Thesis

> *Under what conditions does fusing preprocessing and postprocessing into custom Triton/CUDA kernels reduce end-to-end latency in a batch-1 edge vision pipeline, and when are those gains erased by PCIe transfers, Python dispatch overhead, or model-bound compute?*

In latency-critical edge inspection systems (e.g., surface defect detection, electronic assembly verification), operators frequently assume that accelerating pre- and post-processing on the GPU directly yields proportional end-to-end throughput gains. This work provides empirical, hardware-level evidence investigating the boundaries of Amdahl's Law, memory-coalescing trade-offs, and driver launch overhead in single-stream batch-1 inference workflows.

---

## Key Results Summary

### 1. Isolated Kernel Performance (Cold L2 Cache Flush)
| Operator / Resolution | PyTorch Eager (p50) | TorchInductor Compiled | Triton Fused (p50) | Effective Bandwidth | Isolated Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Preprocessing (256×256)** | `36.9 µs` | `8.2 µs` | **`8.2 µs`** (FP16 `7.2 µs`) | **120.0 GB/s** | **4.50x** vs Eager |
| **Preprocessing (640×640)** | `110.6 µs` | `39.9 µs` | **`46.1 µs`** (FP16 `29.2 µs`) | **133.3 GB/s** | **2.40x** vs Eager |
| **Preprocessing (1280×720)** | `282.6 µs` | `81.9 µs` | **`91.1 µs`** (FP16 `58.4 µs`) | **151.7 GB/s** | **3.10x** vs Eager |
| **YOLO Candidate Filter (8.4k)** | `316.4 µs` | `359.5 µs` | **`318.5 µs`** | **26.4 M cand/s** | **560x candidate reduction** |
| **YOLO Candidate Filter (25.2k)**| `335.0 µs` | `426.4 µs` | **`410.8 µs`** | **61.4 M cand/s** | **1680x candidate reduction** |

### 2. End-to-End Pinned Memory Pipeline Latency Impact (Batch Size = 1)
| Pipeline Workload | Eager Baseline | TorchInductor Compiled | Fused Triton Integration | E2E Latency Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Industrial Anomaly Inspection (256×256)** | `0.783 ms` | `0.700 ms` | **`0.693 ms`** | **+11.5% E2E Latency Reduction** |
| **YOLO Nano Defect Detection (640×640)** | `1.865 ms` | `1.758 ms` | **`1.790 ms`** | **+4.0% E2E Latency Reduction** |

---

## Research Insights & Negative Results

1. **Amdahl's Law & Host Launch Boundaries**:
   - At $256 	imes 256$, custom preprocessing kernels achieve a **$4.50	imes$ speedup on GPU** (`36.9 µs` $	o$ `8.2 µs`), reducing pure GPU pipeline latency by **$11.5\%$** (`888.0 µs` $	o$ `867.7 µs`).
   - In full Python application loops, host-side launch overhead dominates tiny workloads. At $640 	imes 640$, GPU savings overcome dispatch costs to achieve a net **$+4.0\%	ext{--}8.6\%$ end-to-end application speedup**.
2. **2D vs 1D Tiling Negative Study**:
   - A 2D spatial block-tiled kernel was **$5.03	imes$ slower** (`239.6 µs`) than our 1D vectorized kernel (`46.1 µs`). Nsight Compute profiling confirmed that 2D indexing caused register pressure to spike to 42 regs/thread, collapsing active warp occupancy from 78.4% to 28.1% due to uncoalesced memory scatter.

---

## Implications for Edge Industrial AI

1. **High-Resolution Payoff**: Custom Triton preprocessing delivers substantial gains for HD/4K edge inspection streams (1280×720 and above) where pixel density amortizes kernel launch tax and yields >150 GB/s memory bandwidth.
2. **Postprocessing Compaction**: Eliminating non-max suppression (NMS) input clutter via fused candidate filtering shrinks candidate tensors by 560x–1680x, reducing downstream Host/Device memory synchronization bottlenecks.
3. **Driver Boundaries in Edge Python**: In edge Python services, optimizing tiny sub-millisecond payloads requires C++/CUDA runtime bindings or TensorRT graph integration to avoid Python dispatch bubble penalties.

---

## Author Contributions & Project Architecture

- **Multi-Backend Baselines**: Eager PyTorch, TorchInductor, and Host NumPy reference pipelines.
- **Custom Triton Kernels**: 1D vectorized fused preprocessor (Kernel 1), sub-optimal 2D spatial negative variant, and register-unrolled YOLO candidate filter (Kernel 2).
- **Benchmarking & Validation Harness**: Cold 64MB L2 eviction engine, pinned memory timing, and 53 automated unit/integration tests (100% baseline/pipeline coverage, 79% overall).
- **Micro-Architectural Profiling**: Nsight Compute (NCU) metric parsing, warp occupancy tracking, and stall breakdown analysis.

---

## Quickstart & Reproducibility

```bash
# 1. Install package in editable mode
pip install --no-build-isolation -e .

# 2. Run full test suite with code coverage (53 unit & integration tests)
bash scripts/run_all_validation.sh

# 3. Run master artifact generator
bash scripts/generate_all_artifacts.sh
```
