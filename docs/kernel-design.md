# Kernel Design

All kernels are written in [Triton](https://github.com/triton-lang/triton) and compiled by Triton
to PTX for NVIDIA GPUs. There is no hand-written CUDA C++ in this repository.

## Kernel 1: fused preprocessing (`src/triton_kernels/fused_preprocess.py`)

Converts an interleaved `uint8` `[H, W, 3]` frame into a normalized planar `[1, 3, H, W]`
FP32 or FP16 tensor:

```
y[c, h, w] = (x[h, w, c] / 255 - mean[c]) / std[c]  =  x[h, w, c] * scale[c] + bias[c]
scale[c] = 1 / (255 * std[c]),   bias[c] = -mean[c] / std[c]
```

The two constants per channel are folded on the host, so each output element costs one
fused multiply-add.

- **Grid:** 1D over pixels. Each program handles `BLOCK_SIZE` consecutive pixels (autotuned
  over 256/512/1024 elements with 4 or 8 warps, keyed on the pixel count).
- **Loads:** three byte loads per pixel at stride 3 (`ptr + 3*i + c`). These are not vector
  loads. Adjacent threads touch adjacent 3-byte groups, so the requested bytes are dense in
  memory, and Nsight Compute's load sector efficiency (see [results.md](results.md)) shows how
  well the hardware merges them.
- **Stores:** three contiguous plane writes, `y[0*P + i]`, `y[1*P + i]`, `y[2*P + i]`, fully
  coalesced.
- **FP16:** arithmetic stays in FP32 and is rounded once at the store, so the FP16 output is
  exactly the FP32 result cast to FP16 (a test checks this). Eager PyTorch in FP16 rounds after
  every operation instead; with |y| < 4 the two differ by at most two FP16 ULPs (2⁻⁹ each).
- **Parameters:** `mean` and `std` may be tuples, lists, floats or tensors. Tuples cost nothing;
  a CUDA tensor costs one device-to-host read per call (documented in the dispatcher).

**Compulsory DRAM traffic.** Each pixel must read 3 bytes and write 12 bytes (FP32) or 6 bytes
(FP16): 15P bytes per frame in FP32. Eager PyTorch runs the chain `to(float32)`, `div(255)`,
`sub(mean)`, `div(std)`, `contiguous()` as 5 kernels, writing and re-reading a full FP32
intermediate each time: about 3P+12P + 3×24P + 24P = 111P bytes if nothing stays in cache.
That is 7.4× the fused minimum. For small frames the intermediates fit in the 24 MB L2, so the
real DRAM difference is smaller than this upper bound. Every method's effective throughput
(15P bytes per second of device time) is reported. It can exceed the 192 GB/s DRAM peak, because
up to 24 MB of a kernel's output may still be in L2 when the kernel finishes; Nsight Compute's
DRAM byte counters show how much actually reached DRAM.

### 2D-tiled variant (`src/triton_kernels/fused_preprocess_2d.py`)

The same arithmetic with a 2D grid of 16x16-pixel tiles (2 warps per program). Each tile row
covers only 16 pixels: 48 contiguous input bytes and 64 contiguous output bytes per plane. Long
contiguous runs are thus split into short segments. It uses the same parameter handling as the
1D kernel, so the two are benchmarked on identical terms.

> Earlier versions of this repository reported this variant as 5× slower because of register
> pressure. That result was a measurement artifact: the 2D benchmark passed CUDA tensors for
> `mean`/`std`, whose `.tolist()` forced a device synchronization inside the timed region,
> while the 1D kernel received tuples. With identical dispatch, the measured gap is far smaller
> and shrinks with resolution; see [results.md](results.md).

## Kernel 2: fused candidate filter (`src/triton_kernels/candidate_filter.py`)

Input: the raw YOLOv8 head output `[1, 4 + C, N]` (`C = 80`, `N = 8400` at 640x640). Rows 0-3
are `(cx, cy, w, h)` boxes; rows `4..` are sigmoid class scores. One program handles `BLOCK_N`
candidates (autotuned over 128/256/512):

1. **Class reduction:** a loop over the `C` classes, unrolled at compile time (`C` is a
   `constexpr`), keeps the running max and argmax in registers. With the channels-first layout
   that YOLOv8 emits, consecutive candidates are adjacent in memory, so every class load is
   coalesced. A `[N, 4 + C]` row-major input is also accepted through its strides; then the
   loads are strided by `4 + C` elements.
2. **Threshold:** `keep = max_score >= conf_threshold`.
3. **Box decode:** `(cx, cy, w, h) → (x1, y1, x2, y2)` for survivors only.
4. **Stream compaction:** each program counts its survivors (`tl.sum`) and reserves space with
   one `tl.atomic_add` on a global counter. Within-block positions come from an exclusive
   prefix sum (`tl.cumsum`). Boxes, scores, class ids and original candidate indices are
   written densely.

The host reads back the survivor count once (a single synchronization) and slices the outputs.
Survivors from different programs come back in unspecified order. Class-aware NMS
(`torchvision.ops.batched_nms`) sorts by score, so final detections are unaffected, and the
returned candidate indices allow a deterministic order when needed.

**Baseline.** The eager reference follows the Ultralytics NMS pre-filter: one `amax` over
classes, one `nonzero` (its only synchronization), then argmax and box conversion on the K
survivors only. This is a strong baseline, not a strawman: it also synchronizes once.

## CUDA graphs

`src/pipelines/cuda_graph.py` captures the static-shape part of a pipeline (H2D copy,
preprocessing, model, and for the anomaly pipeline also the residual map and top-k) into one
CUDA graph, replayed once per frame. Triton kernels are captured like any other kernel once
autotuning has finished during warm-up. The data-dependent filter and NMS stay outside the
graph.

## torch.compile and YOLOv8n

Ultralytics' `C2f` and `SPPF` blocks build their outputs with
`y.extend(m(y[-1]) for m in self.m)`, a generator that reads the list it is extending. PyTorch
2.5's TorchDynamo traces this pattern incorrectly: every block receives the same input, and the
compiled model silently outputs wrong boxes (errors of hundreds of pixels). `src/models/yolo.py`
replaces those two `forward` methods on the loaded instance with numerically identical explicit
loops. `tests/test_models.py` checks that a Dynamo-traced model matches eager bit for bit.
