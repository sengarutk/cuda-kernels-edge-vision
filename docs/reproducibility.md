# Step-by-Step Reproducibility Guide

## 1. Environment Setup
```bash
# Clone and enter directory
cd /home/sengar/cuda-kernels-for-edge-vision

# Install package in editable mode with development tools
pip install -e ".[dev]"
```

## 2. Generate Synthetic Fixtures
```bash
python tests/fixtures/generate_fixtures.py
```

## 3. Run Validation Suite
```bash
bash scripts/run_all_validation.sh
```

## 4. Run Isolated Benchmarks
```bash
bash scripts/run_benchmarks_all.sh
```

## 5. Run End-to-End Pipeline Benchmarks
```bash
bash scripts/run_e2e_benchmarks.sh
```
