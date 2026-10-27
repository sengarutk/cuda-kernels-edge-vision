# Hardware and Software

The exact environment of every run is recorded in each `results/*.json` file under
`"environment"` and rendered at the top of [results.md](results.md).

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU (AD107, Ada Lovelace, `sm_89`) |
| SMs | 20 (max 1,536 threads = 48 warps per SM) |
| L2 cache | 24 MB (as reported by the driver, `cudaDeviceProp::l2CacheSize`) |
| Memory | 6 GB GDDR6, 96-bit bus, 16 Gbps → 192 GB/s theoretical peak |
| Host | Windows 11 + WSL2, Ubuntu 24.04 LTS |
| Driver | 581.86 (CUDA 13.0 driver API) |
| Software | Python 3.10.13, PyTorch 2.5.1 (CUDA 12.1 runtime), Triton 3.1.0, torchvision 0.20.1, Ultralytics 8.4.21 |
| Profiler | Nsight Compute 2025.3.1 (Linux build, run inside WSL2) |

Exact Python package versions are pinned in [`requirements.txt`](../requirements.txt).

## Scope

This is a laptop / edge-workstation GPU under WSL2, not an embedded Jetson module or a smart
camera. Jetson devices share DRAM between CPU and GPU, which removes the PCIe H2D copy measured
here and changes the balance between memory-bound and launch-bound stages. The qualitative
lessons may carry over; the absolute numbers do not.
