# Hardware & Environment Disclosure

- **Target GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB GDDR6 VRAM)
- **Compute Capability**: sm_89 (Ada Lovelace Architecture)
- **Host Environment**: Ubuntu 22.04 LTS under WSL2 (Windows 11 Host)
- **CUDA Toolkit**: 12.x
- **Deep Learning Runtime**: PyTorch >= 2.2.0 with CUDA 12.x backend
- **Kernel Compilation Engine**: OpenAI Triton >= 2.2.0
- **Memory Subsystem Specifications**:
  - Memory Bus Width: 96-bit
  - Memory Clock: 16 Gbps GDDR6
  - **Theoretical Peak DRAM Bandwidth**: **192.0 GB/s** ($96 / 8 \times 16 = 192.0\text{ GB/s}$)
  - L2 Cache Size: 32 MB (Ada Lovelace High-Capacity Cache)
- **Primary Workload Constraints**:
  - Single-stream execution (Batch Size = 1)
  - Fixed resolutions: 256x256 (Industrial Anomaly Detection) & 640x640 (Edge Detection)
  - Power / Profile: System on AC power, High Performance Power Profile

> **Scope Boundary**: This testbed represents an edge workstation / laptop GPU development environment. It is not an embedded NVIDIA Jetson, industrial smart camera, or discrete server cluster. Latency measurements reflect local driver dispatch and memory subsystem characteristics under WSL2.
