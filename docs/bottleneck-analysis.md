# Phase 1: Baseline Bottleneck Analysis

## 1. Experimental Methodology
- **Workload**: Batch Size = 1 single-stream edge inference.
- **Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) under WSL2.
- **Profiling Toolchain**: PyTorch Profiler (Activity: CPU/CUDA) + Nsight Systems (NVTX range tracing).

## 2. Baseline Profiler Findings (PyTorch Eager)

### 2.1 Anomaly Pipeline (256 × 256 uint8 Input)
| Pipeline Stage | CUDA Time (%) | Operator Breakdown / Source of Overhead |
| :--- | :--- | :--- |
| **01_H2D_Transfer** | ~12.5% | Asynchronous page-locked PCIe transfer for uint8 array. |
| **02_Eager_Preprocessing** | **~28.4%** | Unfused chain: `aten::permute`, `aten::to`, `aten::div`, `aten::sub`, `aten::div`, `aten::contiguous`. Causes 4 global DRAM roundtrips for intermediate tensors. |
| **03_Model_Inference** | ~48.2% | Conv2d, BatchNorm, ReLU forward passes in Autoencoder. |
| **04_Postprocessing_Residual**| ~8.1% | `aten::abs`, `aten::mean`, `aten::topk`. |
| **05_D2H_Transfer** | ~2.8% | Scalar synchronization. |

### 2.2 Detection Pipeline (640 × 640 uint8 Input)
| Pipeline Stage | CUDA Time (%) | Operator Breakdown / Source of Overhead |
| :--- | :--- | :--- |
| **01_H2D_Transfer** | ~16.2% | Transfer of 1.2 MB uint8 image frame. |
| **02_Eager_Preprocessing** | **~24.1%** | Memory bandwidth saturation during unfused layout permute and normalization. |
| **03_Model_Inference** | ~42.5% | YOLO Nano backbone convolutions. |
| **04_Postprocessing_Filter** | **~14.7%** | `aten::max`, `aten::ge`, and boolean candidate indexing across 8,400 boxes. |
| **05_D2H_Transfer** | ~2.5% | Scalar survivor count copy. |

## 3. Systems Hypothesis & Optimization Directives
1. **Kernel 1 Target (Fused Preprocessing)**: In both pipelines, PyTorch eager preprocessing accounts for **24%–28%** of total GPU execution time due to launching 5 separate ATen kernels with intermediate memory allocations. Fusing `uint8 HWC -> float32/fp16 CHW + Normalize` into a single Triton kernel will eliminate intermediate DRAM allocations and reduce kernel launch overhead.
2. **Kernel 2 Target (Confidence Compaction)**: For candidate detection, filtering 8,400 class vectors along the channel dimension introduces significant reduction overhead (~15% of runtime). Fusing score extraction and candidate boolean masking into a single pass will prune redundant downstream NMS computation.
