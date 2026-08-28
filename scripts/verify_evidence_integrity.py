import os
import re
import sys

root = '/home/sengar/cuda-kernels-for-edge-vision'

print("=== Running Automated Evidence & Numerical Parity Audit ===")

# 1. Read main.tex
with open(os.path.join(root, 'paper/main.tex'), 'r', encoding='utf-8') as f:
    tex = f.read()

# 2. Check mathematical speedup consistency
# Isolated 256x256: 36.9 / 8.2 = 4.50x
eager_256 = 36.9
triton_256 = 8.2
sp_256 = round(eager_256 / triton_256, 2)
assert sp_256 == 4.50, f"Math error on 256x256 speedup: {sp_256}"
assert r"4.50\times" in tex or "4.50x" in tex, "Missing 4.50x in main.tex"
print(f"[✓] Verified 256x256 Preprocessing Speedup: {eager_256} / {triton_256} = {sp_256}x")

# 3. Check Amdahl stage calculation
# Eager: 888.0 µs, Triton: 867.7 µs -> (888.0 - 867.7) / 888.0 = 2.286% ≈ 2.3%
pure_gpu_red = round((888.0 - 867.7) / 888.0 * 100, 1)
assert pure_gpu_red == 2.3, f"Math error on pure GPU speedup: {pure_gpu_red}%"
assert r"2.3\%" in tex or "2.3%" in tex or "+2.3%" in tex, "Missing +2.3% pure-device reduction in main.tex"
print(f"[✓] Verified Amdahl Stage Retention Speedup: (888.0 - 867.7)/888.0 = +{pure_gpu_red}%")

# 4. Check E2E Python loop speedups
anom_e2e = round((0.783 - 0.693) / 0.783 * 100, 1)
yolo_e2e = round((2.173 - 1.907) / 2.173 * 100, 1)
yolo_saved = round((2.173 - 1.907) * 1000)
assert anom_e2e == 11.5, f"Math error on Anomaly E2E: {anom_e2e}%"
assert yolo_e2e in (12.2, 12.3), f"Math error on YOLO E2E: {yolo_e2e}%"
assert yolo_saved == 266, f"Math error on YOLO time saved: {yolo_saved} µs"
print(f"[✓] Verified E2E Latencies: Anomaly = +{anom_e2e}%, YOLO = +{yolo_e2e}% (saved {yolo_saved} µs)")

# 5. Check academic wording & hygiene
assert "warn warmup" not in tex, "Found typo 'warn warmup' in main.tex"
assert r"Batch Size} = 1$)" in tex or r"Batch Size = 1" in tex, "Missing closing parenthesis in Abstract"
assert "Triton 1D (FP32)" in tex, "Missing explicit 'Triton 1D (FP32)' in Table 2"
assert "production Triton kernel" not in tex, "Found disallowed 'production' claim in Sec 3.2"
assert "comparable to TorchInductor" in tex, "Missing 'comparable to TorchInductor' in Sec 4.1"
assert "less favorable memory access patterns" in tex, "Missing aligned contribution wording"
assert "does not evaluate defect-detection accuracy" in tex, "Missing industrial accuracy disclaimer in Sec 5.3"
assert "Utkarsh Sengar" in tex, "Missing author name Utkarsh Sengar in main.tex"
print("[✓] Verified Academic Wording, Typo Elimination & Scope Disclaimers")

print("=== Evidence & Numerical Parity Audit: 100% PASSED ===")
