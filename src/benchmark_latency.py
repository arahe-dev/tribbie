"""Measure single-frame GPU perception and k-NN latency, including percentiles."""

from pathlib import Path
import csv
import time

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image

from encoder import build_encoder


ROOT = Path(__file__).resolve().parents[1]


def stats(name, values):
    values = np.asarray(values, dtype=float)
    return {
        "component": name,
        "mean_ms": float(values.mean()),
        "median_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "n": int(len(values)),
    }


def main():
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=500)
    args = parser.parse_args()

    meta = pd.read_csv(ROOT / "features" / "metadata.csv").to_dict("records")
    test_row = next(row for row in meta if row["split"] == "test")
    with Image.open(ROOT / Path(test_row["image_path"])) as image:
        image = image.convert("RGB")
    model, transform, device, _ = build_encoder()
    pca_models = {d: joblib.load(ROOT / "models" / f"pca_{d}.joblib") for d in (2, 8, 16, 32, 64)}
    knn_models = {d: joblib.load(ROOT / "models" / f"knn_{d}.joblib") for d in (2, 8, 16, 32, 64, 512)}

    def frame_forward():
        prep0 = time.perf_counter_ns()
        tensor = transform(image).unsqueeze(0)
        prep_ms = (time.perf_counter_ns() - prep0) / 1e6
        transfer0 = time.perf_counter_ns()
        tensor = tensor.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        transfer_ms = (time.perf_counter_ns() - transfer0) / 1e6
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        enc0 = time.perf_counter_ns()
        with torch.inference_mode():
            latent = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        encoder_ms = (time.perf_counter_ns() - enc0) / 1e6
        cpu0 = time.perf_counter_ns()
        z = latent.detach().cpu().numpy().astype(np.float32, copy=False)
        cpu_ms = (time.perf_counter_ns() - cpu0) / 1e6
        return z, prep_ms, transfer_ms, encoder_ms, cpu_ms

    # 50 warmups execute full online decision path before timing.
    for _ in range(args.warmup):
        z, *_ = frame_forward()
        z16 = pca_models[16].transform(z)
        knn_models[16].predict(z16)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    recorded = {name: [] for name in ("preprocessing", "cpu_to_device", "encoder", "device_to_cpu", "pca16", "knn16", "total_decision", "total_resnet_512", "total_raw_rgb")}
    dim_classifier = {d: [] for d in (2, 8, 16, 32, 64, 512)}
    raw_classifier = []
    raw_prep = []
    raw_knn = joblib.load(ROOT / "models" / "knn_raw.joblib")

    for _ in range(args.runs):
        total0 = time.perf_counter_ns()
        z, prep_ms, transfer_ms, encoder_ms, cpu_ms = frame_forward()
        pca0 = time.perf_counter_ns()
        z16 = pca_models[16].transform(z)
        pca_ms = (time.perf_counter_ns() - pca0) / 1e6
        knn0 = time.perf_counter_ns()
        knn_models[16].predict(z16)
        knn_ms = (time.perf_counter_ns() - knn0) / 1e6
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_ms = (time.perf_counter_ns() - total0) / 1e6

        total512_0 = time.perf_counter_ns()
        z512, prep512, transfer512, encoder512, cpu512 = frame_forward()
        knn_models[512].predict(z512)
        total512 = (time.perf_counter_ns() - total512_0) / 1e6

        raw0 = time.perf_counter_ns()
        raw_vector = np.asarray(image.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32).reshape(-1) / 255.0
        raw_prep_ms = (time.perf_counter_ns() - raw0) / 1e6
        raw0 = time.perf_counter_ns()
        raw_knn.predict(raw_vector[None, :])
        raw_knn_ms = (time.perf_counter_ns() - raw0) / 1e6
        raw_total = raw_prep_ms + raw_knn_ms
        for name, value in (("preprocessing", prep_ms), ("cpu_to_device", transfer_ms), ("encoder", encoder_ms), ("device_to_cpu", cpu_ms), ("pca16", pca_ms), ("knn16", knn_ms), ("total_decision", total_ms)):
            recorded[name].append(value)
        recorded["total_resnet_512"].append(total512)
        recorded["total_raw_rgb"].append(raw_total)
        raw_prep.append(raw_prep_ms)
        raw_classifier.append(raw_knn_ms)
        for d in (2, 8, 16, 32, 64, 512):
            vec = z if d == 512 else pca_models[d].transform(z)
            knn0 = time.perf_counter_ns()
            knn_models[d].predict(vec)
            dim_classifier[d].append((time.perf_counter_ns() - knn0) / 1e6)

    rows = [stats(name, values) for name, values in recorded.items()]
    rows.extend([stats("raw_pixels_preprocessing", raw_prep), stats("raw_pixels_knn", raw_classifier)])
    with (ROOT / "results" / "latency_breakdown.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    dim_rows = [stats(f"knn_{d}d", values) | {"dimension": d} for d, values in dim_classifier.items()]
    with (ROOT / "results" / "classifier_latency_by_dimension.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dim_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dim_rows)
    metrics_path = ROOT / "results" / "metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        metric_rows = list(csv.DictReader(stream))
        fields = list(metric_rows[0].keys()) if metric_rows else ["method", "dimension", "accuracy", "macro_f1"]
    for name in ("median_latency_ms", "p95_latency_ms"):
        if name not in fields:
            fields.append(name)
    lookup = {"Raw RGB + kNN": "total_raw_rgb", "ResNet + kNN": "total_resnet_512", "ResNet + PCA-16 + kNN": "total_decision"}
    row_stats = {row["component"]: row for row in rows}
    for row in metric_rows:
        component = lookup.get(row["method"])
        row["median_latency_ms"] = f"{row_stats[component]['median_ms']:.6f}" if component else ""
        row["p95_latency_ms"] = f"{row_stats[component]['p95_ms']:.6f}" if component else ""
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"LATENCY device={device} warmup={args.warmup} runs={args.runs}", flush=True)
    for row in rows:
        print(f"LATENCY component={row['component']} median_ms={row['median_ms']:.4f} p95_ms={row['p95_ms']:.4f} p99_ms={row['p99_ms']:.4f}", flush=True)


if __name__ == "__main__":
    main()
