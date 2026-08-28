# Experimental Methodology & Statistical Rigor

## 1. Timing Engine
All kernel and pipeline timing measurements are captured using GPU hardware timestamps via `torch.cuda.Event(enable_timing=True)`.

## 2. Warmup & Iteration Count
- **Warmup Phase**: 50 iterations prior to recording to ensure driver caches, Triton JIT compilation, and clock boost states stabilize.
- **Timing Window**: 300 to 500 timed runs per configuration.

## 3. L2 Cache Eviction Protocol
Between consecutive kernel launches in isolated benchmarks, a dedicated 64MB tensor buffer is actively zero-filled to evict resident L2 cache lines and guarantee cold-cache memory measurements.

## 4. Distribution Metrics
Reported metrics include median ($p50$), 95th percentile ($p95$), 99th percentile ($p99$), arithmetic mean, and standard deviation.
