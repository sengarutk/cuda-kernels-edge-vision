"""Shared timing utilities, environment capture and result persistence.

Two complementary measurements are used throughout the repository:

* device time: CUDA-event time on the GPU timeline. Before each timed call the stream is
  pre-filled with a spin kernel that runs longer than the host needs to enqueue the call,
  so CPU launch overhead is hidden and only GPU execution is measured. Only meaningful for
  sync-free callables; if the callable synchronizes, idle GPU time is included and the
  result is flagged ``includes_host_sync``.
* host latency: ``time.perf_counter`` wall-clock from just before the call until
  ``torch.cuda.synchronize()`` returns. Includes Python dispatch, kernel launches,
  any internal synchronizations and execution: what an application actually waits.

Either measurement can be taken with a cold L2 (a 64 MB buffer is read before each iteration;
the RTX 4050 Laptop L2 is 24 MB) or warm (no flush, steady-state streaming).
"""
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


class L2CacheFlusher:
    """Evicts the GPU L2 cache by *reading* a scratch buffer larger than L2.

    Reading leaves L2 full of clean lines that can be dropped for free. A write-based flush
    (``buffer.zero_()``, the common idiom) leaves up to a full L2 of dirty lines, and the
    measured kernel then pays for writing that unrelated data back to DRAM: ~100 us per call
    on this 24 MB L2, enough to hide real differences between kernels.
    """

    def __init__(self, size_mb: int = 64, device: str = "cuda:0"):
        l2 = torch.cuda.get_device_properties(device).L2_cache_size
        if size_mb * 1024 * 1024 < 2 * l2:
            raise ValueError(f"Flush buffer ({size_mb} MB) must be at least 2x L2 ({l2 >> 20} MB)")
        self.buffer = torch.ones(size_mb * 1024 * 1024 // 4, dtype=torch.float32, device=device)
        self._sink = torch.empty((), dtype=torch.float32, device=device)

    def flush(self) -> None:
        torch.sum(self.buffer, dim=0, out=self._sink)


def summarize(samples: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(samples, dtype=np.float64)
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(a.size),
    }


_SLEEP_CYCLES_PER_US: Optional[float] = None


def _sleep_cycles_per_us() -> float:
    """Calibrates ``torch.cuda._sleep`` (a spin kernel) in cycles per microsecond."""
    global _SLEEP_CYCLES_PER_US
    if _SLEEP_CYCLES_PER_US is None:
        cycles = 2_000_000
        torch.cuda._sleep(cycles)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda._sleep(cycles)
        end.record()
        torch.cuda.synchronize()
        _SLEEP_CYCLES_PER_US = cycles / (start.elapsed_time(end) * 1000.0)
    return _SLEEP_CYCLES_PER_US


def _enqueue_time_us(fn: Callable[[], Any], reps: int = 5) -> float:
    """Host time to issue ``fn`` (no synchronization), maximum over a few repetitions."""
    worst = 0.0
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        worst = max(worst, (time.perf_counter() - t0) * 1e6)
    torch.cuda.synchronize()
    return worst


def device_time_us(
    fn: Callable[[], Any],
    iters: int = 300,
    warmup: int = 30,
    flusher: Optional[L2CacheFlusher] = None,
) -> Dict[str, Any]:
    """GPU-timeline time per call with launch overhead hidden (see module docstring)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    prefill_us = 2.0 * _enqueue_time_us(fn) + 200.0
    prefill_cycles = int(prefill_us * _sleep_cycles_per_us())

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if flusher is not None:
            flusher.flush()
        torch.cuda._sleep(prefill_cycles)
        starts[i].record()
        fn()
        ends[i].record()
        if (i + 1) % 50 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    samples = [st.elapsed_time(en) * 1000.0 for st, en in zip(starts, ends)]
    return {**summarize(samples), "prefill_us": prefill_us}


def host_latency_us(
    fn: Callable[[], Any],
    iters: int = 300,
    warmup: int = 30,
    flusher: Optional[L2CacheFlusher] = None,
    keep_samples: bool = False,
) -> Dict[str, Any]:
    """Wall-clock time from call to completed ``torch.cuda.synchronize()``."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        if flusher is not None:
            flusher.flush()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)
    stats: Dict[str, Any] = summarize(samples)
    if keep_samples:
        stats["samples"] = samples
    return stats


def interleaved_blocks(
    fns: Dict[str, Callable[[], Any]],
    blocks: int = 5,
    iters_per_block: int = 200,
    warmup: int = 50,
) -> Dict[str, Dict[str, Any]]:
    """Host latency for several configurations, measured in rotating interleaved blocks.

    Each block times every configuration for ``iters_per_block`` calls, rotating the order
    per block so slow drift in clocks or temperature affects all configurations alike.
    Reports the pooled distribution plus per-block medians and their relative spread.
    """
    for fn in fns.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    names = list(fns)
    pooled: Dict[str, List[float]] = {n: [] for n in names}
    block_medians: Dict[str, List[float]] = {n: [] for n in names}
    for b in range(blocks):
        shift = b % len(names)
        for name in names[shift:] + names[:shift]:
            samples = host_latency_us(fns[name], iters=iters_per_block, warmup=5, keep_samples=True)["samples"]
            pooled[name].extend(samples)
            block_medians[name].append(float(np.median(samples)))
    out = {}
    for name in names:
        meds = np.asarray(block_medians[name])
        out[name] = {
            **summarize(pooled[name]),
            "block_medians": [float(m) for m in meds],
            "block_median_spread_pct": float((meds.max() - meds.min()) / np.median(meds) * 100.0),
        }
    return out


def _run(cmd: List[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def gpu_state() -> Dict[str, str]:
    fields = "clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate"
    values = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"])
    return dict(zip(fields.split(","), [v.strip() for v in values.split(",")])) if values else {}


def environment() -> Dict[str, Any]:
    """Hardware and software metadata stored with every result file."""
    import triton
    import torchvision

    props = torch.cuda.get_device_properties(0)
    os_name = ""
    if os.path.exists("/etc/os-release"):
        with open("/etc/os-release") as f:
            info = dict(line.rstrip().split("=", 1) for line in f if "=" in line)
        os_name = info.get("PRETTY_NAME", "").strip('"')
    cpu = ""
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            cpu = next((l.split(":", 1)[1].strip() for l in f if l.startswith("model name")), "")
    try:
        import ultralytics

        ultralytics_version = ultralytics.__version__
    except ImportError:
        ultralytics_version = None
    commit = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"])
    dirty = bool(_run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", "src", "benchmarks"]))
    return {
        "gpu": props.name,
        "compute_capability": f"sm_{props.major}{props.minor}",
        "sm_count": props.multi_processor_count,
        "l2_cache_mb": props.L2_cache_size / 2**20,
        "gpu_memory_gb": round(props.total_memory / 2**30, 2),
        "driver": _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "ultralytics": ultralytics_version,
        "python": platform.python_version(),
        "os": os_name,
        "kernel": platform.release(),
        "cpu": cpu,
        "git_commit": commit,
        "git_dirty": dirty,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }


def save_results(name: str, config: Dict[str, Any], results: Any, gpu_before: Dict[str, str]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    payload = {
        "benchmark": name,
        "command": " ".join([Path(sys.executable).name] + sys.argv),
        "environment": environment(),
        "gpu_state_before": gpu_before,
        "gpu_state_after": gpu_state(),
        "config": config,
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[saved] {path.relative_to(REPO_ROOT)}")
    return path


def require_cuda() -> str:
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for benchmarking.")
    return "cuda:0"
