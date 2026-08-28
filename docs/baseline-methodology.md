# Baseline Systems Methodology & Theoretical Memory Bounds

## 1. Mathematical Specifications

### 1.1 Fused Preprocessing Transform
Given an input frame $X \in \{0, \dots, 255\}^{H \times W \times 3}$, the target normalized tensor $Y \in \mathbb{R}^{3 \times H \times W}$ is computed via:
$$Y_{c, h, w} = \frac{(X_{h, w, c} / 255.0) - \mu_c}{\sigma_c}$$
For ImageNet defaults: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.

### 1.2 Theoretical Memory Traffic (DRAM Bytes Transferred)

#### Un-Fused PyTorch Eager (Current Baseline):
For an image with $P = H \times W$ pixels:
1. `permute(2, 0, 1)`: View manipulation (0 DRAM bytes).
2. `to(float32)`: Read $3P$ bytes (`uint8`), write $12P$ bytes (`fp32`) $\to 15P$ bytes.
3. `div(255.0)`: Read $12P$ bytes, write $12P$ bytes $\to 24P$ bytes.
4. `sub(mean)`: Read $12P$ bytes, write $12P$ bytes $\to 24P$ bytes.
5. `div(std)`: Read $12P$ bytes, write $12P$ bytes $\to 24P$ bytes.
6. `contiguous()`: Read $12P$ bytes, write $12P$ bytes $\to 24P$ bytes.
**Total Un-fused Memory Traffic:** $\approx 111P$ bytes ($111 \times \text{pixels}$).

#### Fused Ideal Kernel (Phase 3 Target):
1. Single Read: $3P \times 1\text{ byte } (\text{uint8}) = 3P$ bytes.
2. Arithmetic in Registers: Multiply by $1/255.0$, subtract $\mu$, multiply by $1/\sigma$.
3. Single Write: $3P \times 4\text{ bytes } (\text{fp32}) = 12P$ bytes.
**Total Fused Memory Traffic:** $\approx 15P$ bytes.

$$\text{Theoretical DRAM Traffic Reduction} = \frac{111P}{15P} \approx \mathbf{7.4\times}$$

---

## 2. Baseline Implementations Supported
1. `NumPy CPU`: Measures host-bound ingestion + PCIe Host-to-Device transfer.
2. `PyTorch Eager GPU`: Measures standard non-fused ATen operator chain.
3. `PyTorch Inductor (torch.compile)`: Measures compiler-fused baseline with JIT C++/Triton codegen.
