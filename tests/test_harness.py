import pytest
import torch

from benchmarks.harness import L2CacheFlusher, device_time_us, host_latency_us, summarize


def test_summarize():
    s = summarize([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["p50"] == 3.0 and s["min"] == 1.0 and s["max"] == 100.0 and s["n"] == 5
    assert s["p50"] <= s["p95"] <= s["p99"] <= s["max"]


@pytest.mark.cuda
def test_flusher_must_exceed_l2():
    with pytest.raises(ValueError, match="at least 2x L2"):
        L2CacheFlusher(size_mb=1)


@pytest.mark.cuda
def test_flusher_reads_without_modifying_memory():
    flusher = L2CacheFlusher()
    before = flusher.buffer.clone()
    flusher.flush()
    torch.cuda.synchronize()
    assert torch.equal(flusher.buffer, before)
    assert flusher._sink.item() == flusher.buffer.numel()


@pytest.mark.cuda
def test_device_time_excludes_launch_overhead_and_host_time_includes_it():
    x = torch.zeros(1, device="cuda")
    fn = lambda: [x.add_(1) for _ in range(20)]  # 20 tiny launches: host-bound
    dev = device_time_us(fn, iters=50, warmup=5)
    host = host_latency_us(fn, iters=50, warmup=5)
    assert 0 < dev["p50"] < host["p50"]
