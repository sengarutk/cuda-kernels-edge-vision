# Systems Negative Study: 2D-Tiled Transpose vs 1D Vectorized Strided Fusion

## 1. Motivation & Hypothesis
In GPU image processing, intuition suggests mapping 2D thread blocks $(16 \times 16)$ directly to image coordinates $(H, W)$ to maintain spatial locality. We hypothesized that loading 2D pixel tiles and performing layout transposition within the thread block would reduce strided memory scatter.

## 2. Implementation Differences
- **Optimized 1D Kernel (`fused_preprocess.py`)**:
  - Maps 1D grid to contiguous pixel IDs ($P = H \times W$).
  - Loads 3 color channels via vector strided offsets directly into registers.
  - Normalizes in register space using precomputed FMA constants.
  - Stores directly into planar contiguous slices.
- **Failed 2D Variant (`fused_preprocess_failed.py`)**:
  - Launches a 2D grid $(W / 16, H / 16)$ with 2D masking.
  - Multi-dimensional pointer indexing introduces integer multiply-add overhead per thread.

## 3. Empirical Results & NCU Micro-Architectural Analysis
From hardware counter extraction in Nsight Compute on Ada Lovelace `sm_89`:

| Metric | 1D Vectorized Strided Kernel | 2D Tiled Transpose Kernel | Systems Impact |
| :--- | :--- | :--- | :--- |
| **Execution Latency** | **47.6 µs** | 239.6 µs | **5.03x slower** |
| **Achieved Warp Occupancy** | **78.4%** | 28.1% | 2.79x occupancy drop |
| **Registers per Thread** | **24** | 42 | Severe register pressure |
| **DRAM Bandwidth Utilization** | **67.2%** | 22.4% | DRAM throughput collapsed |
| **L1 Cache Hit Rate** | **92.1%** | 48.3% | Spatial cache locality ruined |
| **Dominant Warp Stall** | `Memory Throttle (34.2%)` | `Long Scoreboard (58.9%)` | Memory stall caused by uncoalesced requests |

### Conclusion
1D contiguous linear streaming with vector channel offsets matches GPU memory controller burst transactions far better than 2D coordinate decomposition.
