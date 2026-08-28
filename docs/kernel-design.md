# Kernel Design: Fused HWC -> CHW Preprocessing in Triton

## 1. Architectural Motivation
Edge vision pipelines frequently ingest interleaved camera buffers in `uint8 HWC` format (e.g. $[H, W, 3]$) but require planar normalized tensors in `float32/fp16 CHW` format ($[1, 3, H, W]$) for CNN/Transformer backbones.

Standard deep learning frameworks decompose this transformation into 5-6 distinct operators:
1. `permute(2, 0, 1)`: Stride metadata modification.
2. `to(float32)`: Byte-to-float upcast.
3. `div(255.0)`: Dynamic range normalization.
4. `sub(mean)`: Per-channel centering.
5. `div(std)`: Per-channel scaling.
6. `contiguous()`: Physical memory transpose into planar layout.

Each un-fused operator launches an independent GPU kernel with global memory synchronization, resulting in multiple global memory roundtrips.

## 2. Fused Triton Kernel Strategy
- **1D Linear Grid Mapping**: Pixels are linearized ($P = H \times W$) and processed in contiguous 1D tiles (`BLOCK_SIZE = 256, 512, 1024`).
- **Vectorized Strided Channel Loads**:
  - `in_r = load(ptr + offsets * 3 + 0)`
  - `in_g = load(ptr + offsets * 3 + 1)`
  - `in_b = load(ptr + offsets * 3 + 2)`
- **Register-Level Fast Math**: Normalization is performed via Fused Multiply-Add (FMA) using precalculated inverse constants:
  $$x_{\text{norm}} = (x \cdot \text{inv\_scale} - \mu) \cdot \text{inv\_std}$$
- **Direct Planar Storage**: Outputs are written directly to their respective channel sub-planes ($[0, P), [P, 2P), [2P, 3P)$) in global memory.

## 3. Zero-Sync Dispatcher Architecture
The Python host dispatcher accepts `mean` and `std` as pre-unpacked Python tuples, lists, or tensors. Scalar constants are extracted once on the host, preventing implicit `.tolist()` / `item()` device synchronization stalls during continuous streaming loops.

## 4. FP16 Precision & ULP Analysis
In FP16 (IEEE-754 half precision), 1 Unit in the Last Place (ULP) around dynamic range $\approx 2.0$ is $2^{-10} \approx 0.0009765625$. Eager PyTorch truncates each intermediate operation to FP16, whereas Triton performs intermediate arithmetic in FP32 registers and casts to FP16 at store time. The resulting maximum difference is within $\le 2\text{ ULPs} \approx 0.00195$, well within FP16 numerical tolerance.
