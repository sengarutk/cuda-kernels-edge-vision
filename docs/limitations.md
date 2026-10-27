# Limitations and Scope

- **One GPU, one OS.** All measurements come from a single RTX 4050 Laptop GPU under WSL2. Laptop
  boost clocks vary with temperature and power, and WSL2 adds a virtualization layer to every
  kernel launch and synchronization. Expect other GPUs, native Linux, or a Jetson (shared CPU/GPU
  memory, no PCIe copy) to produce different absolute numbers and possibly different rankings.
- **Batch size 1, one stream.** Batched or multi-stream serving amortizes launch overhead
  differently.
- **Accuracy is not evaluated.** The anomaly autoencoder has seeded random, untrained weights:
  its score measures latency, not defects. YOLOv8n uses pretrained COCO weights on a street scene
  (`bus.jpg`), not an industrial dataset. Kernel outputs are verified for numerical agreement with
  PyTorch, not for detection quality.
- **Model execution is FP32.** Convolutions use cuDNN's default TF32 tensor-core math. No INT8 or
  FP16 model, TensorRT engine or pruning is evaluated.
- **Inputs.** The preprocessing kernels assume a tightly packed RGB `uint8` frame and do not
  resize, letterbox or swap channel order (BGR→RGB). The candidate filter assumes batch 1.
- **TorchInductor** is used in `mode="default"`. `max-autotune` and `reduce-overhead` (Inductor's
  own CUDA graphs) are not evaluated. Manual CUDA-graph capture of eager code is.
- **Nsight Compute** runs at locked base clocks with cache flushing (its defaults), so its
  durations are not latency results.

## Future work

- Fuse letterbox resizing and BGR→RGB conversion into the preprocessing kernel.
- Evaluate on a Jetson Orin (shared memory) and on native Linux.
- Compare against TensorRT engines with the pre- and post-processing kernels as plugins.
