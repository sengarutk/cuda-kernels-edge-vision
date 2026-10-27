"""Stage-by-stage and end-to-end latency of the two batch-1 pipelines.

Every configuration is measured directly (no stage time is reused across configurations).
Stage numbers come from running each stage in isolation on that configuration's own
intermediate tensors; end-to-end numbers come from complete ``run_step`` calls, timed in
interleaved blocks. Caches are warm (steady-state streaming of the same frame).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from tabulate import tabulate

from benchmarks.harness import (
    device_time_us, gpu_state, host_latency_us, interleaved_blocks, require_cuda, save_results,
)
from src.frames import anomaly_frame, detection_frame
from src.models.yolo import load_yolov8n
from src.pipelines.anomaly_pipeline import AnomalyInspectionPipeline
from src.pipelines.detection_pipeline import DetectionInspectionPipeline

ANOMALY_CONFIGS = [
    dict(preprocess="eager", compile_model=False),
    dict(preprocess="compiled", compile_model=False),
    dict(preprocess="triton", compile_model=False),
    dict(preprocess="eager", compile_model=True),
    dict(preprocess="compiled", compile_model=True),
    dict(preprocess="triton", compile_model=True),
    dict(preprocess="eager", cuda_graph=True),
    dict(preprocess="triton", cuda_graph=True),
]
DETECTION_CONFIGS = [
    dict(preprocess="eager", postprocess="eager", compile_model=False),
    dict(preprocess="triton", postprocess="eager", compile_model=False),
    dict(preprocess="eager", postprocess="triton", compile_model=False),
    dict(preprocess="triton", postprocess="triton", compile_model=False),
    dict(preprocess="compiled", postprocess="compiled", compile_model=True),
    dict(preprocess="triton", postprocess="triton", compile_model=True),
    dict(preprocess="eager", postprocess="eager", cuda_graph=True),
    dict(preprocess="triton", postprocess="triton", cuda_graph=True),
]


def anomaly_stages(pipe, host):
    with torch.inference_mode():
        dev_img = pipe.h2d(host)
        x = pipe.preprocess(dev_img)
        recon = pipe.infer(x)
    return {
        "h2d": (lambda: pipe.h2d(host), False),
        "preprocess": (lambda: pipe.preprocess(dev_img), False),
        "inference": (lambda: pipe.infer(x), False),
        "postprocess": (lambda: pipe.postprocess(x, recon), True),
    }


def detection_stages(pipe, host):
    with torch.inference_mode():
        dev_img = pipe.h2d(host)
        x = pipe.preprocess(dev_img)
        preds = pipe.infer(x)
        boxes, scores, classes = pipe.filter(preds)
    return {
        "h2d": (lambda: pipe.h2d(host), False),
        "preprocess": (lambda: pipe.preprocess(dev_img), False),
        "inference": (lambda: pipe.infer(x), False),
        "filter": (lambda: pipe.filter(preds), True),
        "nms": (lambda: pipe.nms(boxes, scores, classes), False),
    }


def measure(kind, pipes, host, stage_fn, args):
    results, rows = [], []
    for pipe in pipes:
        stages = {}
        # Graph-captured configurations execute their stages fused inside one graph replay,
        # so they are only measured end to end.
        stage_fns = {} if pipe.cuda_graph else stage_fn(pipe, host)
        with torch.inference_mode():
            for stage, (fn, syncs) in stage_fns.items():
                dev = device_time_us(fn, args.stage_iters, args.warmup)
                hst = host_latency_us(fn, args.stage_iters, args.warmup)
                stages[stage] = {"device_us": {**dev, "includes_host_sync": syncs}, "host_us": hst}
        results.append({"pipeline": kind, "config": pipe.name, "stages": stages})

    e2e = interleaved_blocks({p.name: (lambda p=p: p.run_step(host)) for p in pipes},
                             blocks=args.blocks, iters_per_block=args.iters, warmup=args.warmup)
    for entry, pipe in zip(results, pipes):
        entry["e2e_host_us"] = e2e[pipe.name]
        out = pipe.run_step(host)
        entry["outputs"] = ({"num_candidates": out["num_candidates"],
                             "num_detections": out["num_detections"]}
                            if kind == "detection" else {"anomaly_score": out["anomaly_score"]})
        st = entry["stages"]
        rows.append([kind, pipe.name]
                    + [f"{st[s]['host_us']['p50']:.0f}/{st[s]['device_us']['p50']:.0f}" if s in st else "-"
                       for s in results[0]["stages"]]
                    + [f"{e2e[pipe.name]['p50']:.0f}", f"{e2e[pipe.name]['p95']:.0f}",
                       f"{e2e[pipe.name]['block_median_spread_pct']:.1f}"])
    headers = ["Pipeline", "Config"] + [f"{s} host/dev" for s in results[0]["stages"]] + [
        "E2E p50", "E2E p95", "Block spread %"]
    print(tabulate(rows, headers=headers, tablefmt="github"))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--iters", type=int, default=300, help="E2E iterations per block")
    parser.add_argument("--stage-iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()

    device = require_cuda()
    before = gpu_state()

    host_ae = torch.from_numpy(anomaly_frame()).pin_memory()
    ae_pipes = [AnomalyInspectionPipeline(device, seed=0, **c) for c in ANOMALY_CONFIGS]
    anomaly = measure("anomaly", ae_pipes, host_ae, anomaly_stages, args)
    del ae_pipes
    torch.cuda.empty_cache()

    host_det = torch.from_numpy(detection_frame(640)).pin_memory()
    yolo = load_yolov8n(torch.device(device))
    det_pipes = [DetectionInspectionPipeline(device, model=yolo, **c) for c in DETECTION_CONFIGS]
    detection = measure("detection", det_pipes, host_det, detection_stages, args)

    save_results("pipelines", {**vars(args), "anomaly_input": "tests/fixtures/image_256_uint8.npy",
                               "detection_input": "bus.jpg letterboxed to 640x640",
                               "conf_threshold": 0.25, "iou_threshold": 0.45,
                               "l2": "warm (no flush)", "host_memory": "pinned"},
                 {"anomaly": anomaly, "detection": detection}, before)


if __name__ == "__main__":
    main()
