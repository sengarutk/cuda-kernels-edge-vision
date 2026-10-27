import json

import pytest

from profiling import parse_ncu_metrics as p

CSV = '''"ID","Kernel Name","gpu__time_duration.sum","dram__bytes_read.sum","dram__bytes_write.sum","launch__registers_per_thread","smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio","smsp__average_warps_issue_stalled_wait_per_issue_active.ratio","smsp__average_warps_issue_stalled_selected_per_issue_active.ratio"
"","","usecond","Mbyte","Mbyte","register/thread","ratio","ratio","ratio"
"0","k1","10.00","1.00","3.00","24","3.00","1.00","1.00"
"1","k2","2,000.00","0.50","0.50","16","0","0","1.00"
'''


def test_read_and_summarize(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("==PROF== noise\n" + CSV)
    kernels = [p.summarize_kernel(k) for k in p.read_raw_csv(path)]
    k1, k2 = kernels
    assert k1["kernel"] == "k1" and k1["duration_us"] == 10.0 and k1["registers_per_thread"] == 24
    assert k1["dram_read_bytes"] == 1e6 and k1["achieved_dram_gbs"] == pytest.approx(400.0)
    assert k1["stall_share_pct"] == {"long_scoreboard": 75.0, "wait": 25.0}
    assert k1["dominant_stall"] == "long_scoreboard" and k1["issue_selected_per_issue"] == 1.0
    assert k2["duration_us"] == 2000.0 and k2["stall_share_pct"] == {}


def test_main_writes_summary(tmp_path, monkeypatch):
    ncu = tmp_path / "results" / "ncu"
    ncu.mkdir(parents=True)
    (ncu / "target_a.csv").write_text(CSV)
    (ncu / "meta.json").write_text(json.dumps({"ncu_version": "x"}))
    monkeypatch.setattr(p, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(p, "NCU_DIR", ncu)
    assert p.main() == 0
    out = json.loads((tmp_path / "results" / "ncu_summary.json").read_text())
    assert out["meta"]["ncu_version"] == "x"
    assert out["targets"]["target_a"]["num_kernels"] == 2
    assert out["targets"]["target_a"]["total_duration_us"] == 2010.0
