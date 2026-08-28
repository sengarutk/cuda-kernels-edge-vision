import torch
import triton

def test_cuda_available():
    assert torch.cuda.is_available(), "CUDA must be available for Flagship 3!"
    device_name = torch.cuda.get_device_name(0)
    print(f"\n[+] Testing on GPU: {device_name}")
    assert "RTX" in device_name or "GeForce" in device_name or "NVIDIA" in device_name

def test_triton_installation():
    assert triton.__version__ is not None
    print(f"[+] Triton version: {triton.__version__}")
