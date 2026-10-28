# Reproducing the Results

## 1. Environment

```bash
git clone https://github.com/sengarutk/cuda-kernels-edge-vision.git
cd cuda-kernels-edge-vision
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
pip install --no-deps -e .
```

The extra index selects the CUDA 12.1 builds of PyTorch and torchvision that produced `results/`;
plain PyPI installs the CUDA 12.4 builds of the same versions, which work but are not the measured
configuration.

The YOLOv8n weights (`weights/yolov8n.pt`, git-ignored) are downloaded automatically by
Ultralytics on first use and checked against the official SHA-256. To use an existing copy,
place it at `weights/yolov8n.pt` or set `YOLOV8N_WEIGHTS=/path/to/yolov8n.pt`.

## 2. Tests

```bash
make test        # GPU tests are skipped automatically when no CUDA device is present
```

## 3. Benchmarks (about 10 minutes on an RTX 4050 Laptop GPU)

```bash
make bench       # writes results/preprocess.json, results/filter.json, results/pipelines.json
```

Keep the laptop on AC power and close other GPU applications while this runs.

## 4. Nsight Compute

Profiling needs access to GPU performance counters. On Windows/WSL2: NVIDIA Control Panel →
Desktop → enable Developer Settings → Developer → Manage GPU Performance Counters → "Allow access
to the GPU performance counters to all users". Applying it resets the driver; run
`wsl --shutdown` from Windows afterwards.

Nsight Compute must be a recent Linux build (2023.1 fails under WSL2 with `ERR_NVGPUCTRPERM`).
Without root, it can be unpacked from NVIDIA's CUDA repository:

```bash
curl -O https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/nsight-compute-2025.3.1_2025.3.1.4-1_amd64.deb
dpkg-deb -x nsight-compute-2025.3.1_2025.3.1.4-1_amd64.deb ~/.local/opt/ncu-2025.3.1
```

```bash
make ncu NCU=~/.local/opt/ncu-2025.3.1/opt/nvidia/nsight-compute/2025.3.1/ncu
```

## 5. Report, figures and paper

```bash
make report      # docs/results.md, the README results block, paper/generated/results.tex
make figures     # paper/figures/*.pdf from results/*.json
make paper       # paper/main.pdf (needs pdflatex + bibtex)
make check       # fails if any generated file is out of date with results/
```

`make all` runs steps 2-5 in order.

## Fixtures

`python tests/fixtures/generate_fixtures.py` regenerates the committed image fixtures
bit-for-bit. Add `--yolo` to also regenerate `yolov8n_bus_raw_640.pt` (YOLOv8n on CPU).
