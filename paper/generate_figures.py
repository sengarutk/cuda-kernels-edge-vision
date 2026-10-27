"""Builds the paper figures from results/*.json (no hard-coded measurements)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})
# Colorblind-safe palette (Okabe-Ito).
C = {"eager": "#999999", "compiled": "#E69F00", "triton": "#0072B2", "triton2d": "#56B4E9",
     "fp16": "#009E73", "graph": "#CC79A7"}


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


RES_LABELS = {"256x256": "256²", "640x640": "640²", "1280x720": "720p", "1920x1080": "1080p",
              "3840x2160": "4K"}


def short(config: str) -> str:
    """'pre=triton, model=eager, filter=triton, cuda-graph' -> 'pre T · model E · filter T · graph'."""
    abbrev = {"eager": "E", "compiled": "C", "triton": "T"}
    parts = []
    for item in config.split(", "):
        if "=" in item:
            key, value = item.split("=")
            parts.append(f"{key} {abbrev.get(value, value)}")
        else:
            parts.append("graph")
    return " · ".join(parts)


def fig_preprocess():
    res = load("preprocess")["results"]
    get = {(r["resolution"], r["method"]): r["device_us"]["p50"] for r in res}
    rz = list(dict.fromkeys(r["resolution"] for r in res))
    methods = [("eager_fp32", "Eager", C["eager"]), ("compiled_fp32", "Inductor", C["compiled"]),
               ("triton_fp32", "Triton 1D", C["triton"]), ("triton_2d_fp32", "Triton 2D", C["triton2d"]),
               ("triton_fp16", "Triton 1D FP16", C["fp16"])]
    x = np.arange(len(rz))
    w = 0.16
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    for i, (m, label, color) in enumerate(methods):
        ax.bar(x + (i - 2) * w, [get[(r, m)] for r in rz], w, label=label, color=color, edgecolor="none")
    ax.set_yscale("log")
    ax.set_ylabel("Device time, p50 (µs)")
    ax.set_xticks(x)
    ax.set_xticklabels([RES_LABELS.get(r, r) for r in rz])
    ax.grid(axis="y", which="major", linewidth=0.4, alpha=0.5)
    ax.legend(ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.22))
    fig.savefig(OUT / "fig_preprocess.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _color(config):
    if "cuda-graph" in config:
        return C["graph"]
    if "triton" in config:
        return C["triton"]
    if "compiled" in config:
        return C["compiled"]
    return C["eager"]


def fig_pipelines():
    data = load("pipelines")["results"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))
    for ax, kind, title in ((axes[0], "anomaly", "Anomaly (256×256 autoencoder)"),
                            (axes[1], "detection", "Detection (640×640 YOLOv8n + NMS)")):
        entries = data[kind]
        names = [short(e["config"]) for e in entries]
        p50 = np.array([e["e2e_host_us"]["p50"] for e in entries]) / 1000
        p95 = np.array([e["e2e_host_us"]["p95"] for e in entries]) / 1000
        y = np.arange(len(entries))[::-1]
        ax.barh(y, p50, color=[_color(e["config"]) for e in entries], height=0.7)
        ax.errorbar(p50, y, xerr=[np.zeros_like(p50), p95 - p50], fmt="none", ecolor="black",
                    elinewidth=0.5, capsize=1.5)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=6.5)
        ax.set_title(title, loc="left")
        ax.set_xlabel("End-to-end host latency (ms): bar p50, whisker p95")
        ax.grid(axis="x", linewidth=0.4, alpha=0.5)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig_pipelines.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_stages():
    entries = [e for e in load("pipelines")["results"]["detection"] if e["stages"]]
    stages = list(entries[0]["stages"])
    colors = ["#CCCCCC", "#0072B2", "#555555", "#E69F00", "#CC79A7"]
    fig, ax = plt.subplots(figsize=(7.0, 2.1))
    y = np.arange(len(entries))[::-1]
    for mode, offset, hatch in (("host_us", 0.19, None), ("device_us", -0.19, "////")):
        left = np.zeros(len(entries))
        for stage, color in zip(stages, colors):
            vals = np.array([e["stages"][stage][mode]["p50"] for e in entries]) / 1000
            ax.barh(y + offset, vals, 0.36, left=left, color=color, hatch=hatch, edgecolor="white",
                    linewidth=0.3, label=stage if mode == "host_us" else None)
            left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([short(e["config"]) for e in entries], fontsize=6.5)
    ax.set_xlabel("Per-stage p50 (ms): upper bar host latency, hatched bar device time")
    ax.legend(ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", linewidth=0.4, alpha=0.5)
    fig.savefig(OUT / "fig_stages.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_ncu_stalls():
    path = RESULTS / "ncu_summary.json"
    if not path.exists():
        return
    targets = json.loads(path.read_text())["targets"]
    pairs = [("preprocess_triton_1920x1080", "Triton 1D"), ("preprocess_triton_2d_1920x1080", "Triton 2D"),
             ("filter_triton_8400", "Filter [84,N]"), ("filter_triton_row_major_8400", "Filter [N,84]")]
    pairs = [(t, label) for t, label in pairs if t in targets]
    shares = {}
    for t, label in pairs:
        k = max(targets[t]["kernels"], key=lambda k: k["duration_us"])
        shares[label] = k["stall_share_pct"]
    reasons = []
    for s in shares.values():
        for r, v in sorted(s.items(), key=lambda kv: -kv[1])[:3]:
            if r not in reasons:
                reasons.append(r)
    fig, ax = plt.subplots(figsize=(3.4, 1.7))
    left = np.zeros(len(shares))
    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]
    labels = list(shares)
    for reason, color in zip(reasons, palette):
        vals = np.array([shares[lbl].get(reason, 0.0) for lbl in labels])
        ax.barh(labels[::-1], vals[::-1], left=left[::-1], color=color, label=reason.replace("_", " "), height=0.6)
        left += vals
    ax.barh(labels[::-1], (100 - left)[::-1], left=left[::-1], color="#DDDDDD", label="other", height=0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of stalled warps per issued instruction (%)")
    ax.legend(ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.32))
    fig.savefig(OUT / "fig_ncu_stalls.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_preprocess()
    fig_pipelines()
    fig_stages()
    fig_ncu_stalls()
    print("[saved] " + ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))


if __name__ == "__main__":
    main()
