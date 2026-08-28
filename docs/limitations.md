# Limitations & Scope Disclosure

## 1. Target Architecture Bounds
- **Hardware Profile**: All experiments were conducted on an NVIDIA GeForce RTX 4050 Laptop GPU (6 GB GDDR6 VRAM, 96-bit bus, 192.0 GB/s peak theoretical bandwidth, `sm_89` Ada Lovelace micro-architecture).
- **Environment**: Execution took place under WSL2 (Ubuntu 22.04 LTS). This represents an edge-workstation development environment rather than an embedded Jetson SoC (Orin Nano / Orin AGX) with unified memory architecture (UMA).

## 2. Single-Stream & Batch-1 Constraints
- **Workload Focus**: Workloads are strictly scoped to single-stream batch size 1, reflecting hard real-time latency-critical industrial inspection lines.
- **Batching Trade-offs**: High-throughput batched serving ($B \ge 8$) amortizes kernel launch overhead differently and alters cache hit ratios.

## 3. Unmodified Neural Compute Scope
- **Neural Network Weights**: Model forward passes (Autoencoder and YOLO Nano) were executed without structural weight pruning or INT8 quantization.
- **Optimization Focus**: This research intentionally isolates the non-neural pipeline boundaries (H2D transfer, normalization, planar transposition, thresholding, coordinate filtering) to assess their proportional impact under Amdahl's Law.

## 4. Future Systems Work
- **Fused Bilinear Resizing**: Integrating interpolation/resizing directly into the transposition kernel to eliminate separate resizing passes on host or GPU.
- **Native TensorRT Plugins**: Packaging custom Triton kernels as native TensorRT IPluginV2DynamicExt nodes to bypass Python driver dispatch completely.
