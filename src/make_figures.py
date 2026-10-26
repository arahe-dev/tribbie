"""Generate latent-space, confusion, dimensionality, latency, NN, and video artifacts."""

from pathlib import Path
import csv
import json
import math
import subprocess

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["LEFT", "FORWARD", "RIGHT", "STOP"]
COLORS = {"LEFT": "#2f6fbb", "FORWARD": "#2a9d67", "RIGHT": "#e76f51", "STOP": "#845ec2"}


def ensure_dirs():
    for path in (ROOT / "artifacts" / "figures", ROOT / "artifacts" / "nearest_neighbors", ROOT / "artifacts" / "failure_cases"):
        path.mkdir(parents=True, exist_ok=True)


def latent_plot():
    features = np.load(ROOT / "features" / "all_embeddings.npy")
    meta = pd.read_csv(ROOT / "features" / "metadata.csv").to_dict("records")
    pca = joblib.load(ROOT / "models" / "pca_2.joblib")
    projected = pca.transform(features)
    split = np.array([row["split"] for row in meta])
    labels = np.array([row["label"] for row in meta])
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for label in CLASSES:
        train = (labels == label) & (split == "train")
        test = (labels == label) & (split == "test")
        ax.scatter(projected[train, 0], projected[train, 1], s=18, alpha=0.55, color=COLORS[label], label=f"{label} train")
        if np.any(test):
            ax.scatter(projected[test, 0], projected[test, 1], s=54, alpha=0.95, marker="x", linewidths=1.6, color=COLORS[label], label=f"{label} test")
    ax.set_title("ResNet-18 features projected to 2-D PCA")
    ax.set_xlabel("Training-fit PC 1")
    ax.set_ylabel("Training-fit PC 2")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8, loc="best")
    base = ROOT / "artifacts" / "figures" / "latent_space"
    fig.savefig(base.with_suffix(".png"), dpi=180)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def confusion_plot():
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
    matrix = metrics["methods"]["ResNet + PCA-16 + kNN"]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ConfusionMatrixDisplay(confusion_matrix=np.asarray(matrix), display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Held-out episode confusion matrix | PCA-16 + k-NN")
    base = ROOT / "artifacts" / "figures" / "confusion_matrix"
    fig.savefig(base.with_suffix(".png"), dpi=180)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def dimensionality_plot():
    data = pd.read_csv(ROOT / "results" / "dimensionality_metrics.csv").sort_values("dimension")
    latency_path = ROOT / "results" / "classifier_latency_by_dimension.csv"
    latency = pd.read_csv(latency_path) if latency_path.is_file() else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(data["dimension"], data["accuracy"], marker="o", linewidth=2, color="#2878b5", label="Accuracy")
    ax.plot(data["dimension"], data["macro_f1"], marker="s", linewidth=1.8, color="#d95f02", label="Macro F1")
    if not latency.empty:
        right = ax.twinx()
        by_dim = latency.set_index("dimension").loc[data["dimension"].astype(int)]
        right.plot(data["dimension"], by_dim["median_ms"], marker="^", color="#5b8e7d", linestyle="--", label="k-NN median")
        right.set_ylabel("Classifier-only median latency (ms)")
        right.legend(loc="lower right")
    ax.set_xscale("log", base=2)
    ax.set_xticks(data["dimension"])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Representation dimension (log scale)")
    ax.set_ylabel("Held-out episode score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Control-label prediction vs representation size")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    fig.savefig(ROOT / "artifacts" / "figures" / "dimension_vs_accuracy.png", dpi=180)
    plt.close(fig)


def latency_plot():
    path = ROOT / "results" / "latency_breakdown.csv"
    if not path.is_file():
        return
    data = pd.read_csv(path).set_index("component")
    names = ["preprocessing", "encoder", "device_to_cpu", "pca16", "knn16", "total_decision"]
    data = data.loc[[name for name in names if name in data.index]]
    labels = [name.replace("_", " ") for name in data.index]
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.barh(y + 0.17, data["median_ms"], height=0.32, color="#3b82a0", label="Median")
    ax.barh(y - 0.17, data["p95_ms"], height=0.32, color="#f4a261", label="p95")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Milliseconds per decision")
    ax.set_title("Single-frame pipeline latency (batch size 1)")
    ax.grid(axis="x", alpha=0.2)
    ax.legend()
    fig.savefig(ROOT / "artifacts" / "figures" / "latency_breakdown.png", dpi=180)
    plt.close(fig)


def nearest_neighbor_examples():
    meta = pd.read_csv(ROOT / "features" / "metadata.csv").to_dict("records")
    features = np.load(ROOT / "features" / "all_embeddings.npy")
    predictions = pd.read_csv(ROOT / "results" / "test_predictions.csv")
    predictions = predictions[predictions["method"] == "ResNet + PCA-16 + kNN"].copy()
    test_lookup = {row["image_path"]: i for i, row in enumerate(meta)}
    train_indices = [i for i, row in enumerate(meta) if row["split"] == "train"]
    pca = joblib.load(ROOT / "models" / "pca_16.joblib")
    knn = joblib.load(ROOT / "models" / "knn_16.joblib")
    if predictions.empty:
        return
    wrong = predictions[predictions["true_label"] != predictions["predicted_label"]]
    chosen = (wrong if not wrong.empty else predictions).head(8)
    fail_rows = []
    for number, row in enumerate(chosen.to_dict("records"), start=1):
        query_index = test_lookup[row["image_path"]]
        query_z = pca.transform(features[query_index : query_index + 1])
        distances, neighbors = knn.kneighbors(query_z, n_neighbors=4)
        tile_w, tile_h = 236, 206
        sheet = Image.new("RGB", (tile_w * 5, tile_h), "white")
        draw = ImageDraw.Draw(sheet)
        query_image = Image.open(ROOT / Path(row["image_path"])).convert("RGB").resize((220, 165))
        sheet.paste(query_image, (8, 8))
        draw.text((8, 176), f"QUERY true={row['true_label']} pred={row['predicted_label']}", fill="black")
        for rank, (distance, local_index) in enumerate(zip(distances[0], neighbors[0]), start=1):
            train_row = meta[train_indices[int(local_index)]]
            neighbor_image = Image.open(ROOT / Path(train_row["image_path"])).convert("RGB").resize((220, 165))
            x = tile_w * rank
            sheet.paste(neighbor_image, (x + 8, 8))
            draw.text((x + 8, 176), f"NN{rank} {train_row['label']} d={distance:.3f}", fill="black")
        destination = ROOT / "artifacts" / "nearest_neighbors" / f"query_{number:02d}.png"
        sheet.save(destination)
        if row["true_label"] != row["predicted_label"]:
            failure_path = ROOT / "artifacts" / "failure_cases" / f"offline_{number:02d}_{row['true_label']}_as_{row['predicted_label']}.png"
            sheet.save(failure_path)
            fail_rows.append({"image_path": row["image_path"], "true_label": row["true_label"], "predicted_label": row["predicted_label"], "confidence": row["confidence"], "nearest_label": meta[train_indices[int(neighbors[0][0])]]["label"], "nearest_distance": float(distances[0][0]), "artifact": failure_path.relative_to(ROOT).as_posix()})
    with (ROOT / "results" / "failure_cases.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(fail_rows[0].keys()) if fail_rows else ["image_path", "true_label", "predicted_label", "confidence", "nearest_label", "nearest_distance", "artifact"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(fail_rows)
    (ROOT / "results" / "failure_analysis.json").write_text(json.dumps({"offline_misclassifications": len(wrong), "examples_saved": len(chosen), "failure_artifacts": len(fail_rows)}, indent=2), encoding="utf-8")


def closed_loop_failure_examples():
    episodes_path = ROOT / "results" / "episode_results.csv"
    frames_path = ROOT / "results" / "demo_frames.csv"
    if not episodes_path.is_file() or not frames_path.is_file():
        return
    episodes = pd.read_csv(episodes_path)
    frames = pd.read_csv(frames_path)
    failures = episodes[episodes["success"] == 0]
    run_summary = json.loads((ROOT / "results" / "closed_loop_summary.json").read_text(encoding="utf-8"))
    policy = run_summary.get("policy", "unknown")
    for _, episode in failures.iterrows():
        samples = frames[frames["episode_id"] == episode["episode_id"]]
        if samples.empty:
            continue
        chosen_indices = sorted(set((0, len(samples) // 2, len(samples) - 1)))
        chosen = samples.iloc[chosen_indices]
        canvas = Image.new("RGB", (960, 284), "white")
        draw = ImageDraw.Draw(canvas)
        for col, (_, sample) in enumerate(chosen.iterrows()):
            image = Image.open(ROOT / Path(sample["front_path"])).convert("RGB").resize((320, 240))
            canvas.paste(image, (col * 320, 0))
            draw.text((col * 320 + 6, 246), f"decision={int(sample['decision'])} action={sample['action']}", fill="black")
        draw.text((6, 267), f"{episode['episode_id']} failed | final distance={episode['final_distance_m']:.2f} m, bearing={episode['final_bearing_rad']:.2f} rad", fill="black")
        canvas.save(ROOT / "artifacts" / "failure_cases" / f"closed_loop_{policy}_{episode['episode_id']}.png")


def compose_video():
    frame_manifest = ROOT / "results" / "demo_frames.csv"
    if not frame_manifest.is_file():
        return
    rows = pd.read_csv(frame_manifest).to_dict("records")
    if not rows:
        return
    try:
        import imageio_ffmpeg
    except ImportError:
        print("VIDEO_SKIPPED imageio-ffmpeg is not installed", flush=True)
        return

    width, panel_height, banner_height = 1280, 480, 60
    fps = 12
    repeats = max(1, int(round(720 / len(rows))))
    repeats = min(repeats, max(1, int(1080 / len(rows))))
    repeats = min(repeats, 8)
    destination = ROOT / "artifacts" / "demo.mp4"
    temp_path = ROOT / "artifacts" / "demo_tmp.mp4"
    policy = str(rows[0].get("policy", "pca16"))
    policy_banner = {
        "raw": "64x64 RGB -> distance-weighted k-NN",
        "pca16": "ResNet-18 -> PCA-16 -> distance-weighted k-NN",
        "pca64": "ResNet-18 -> PCA-64 -> distance-weighted k-NN",
        "moe16": "RGB -> learned 16-D latent -> 4-expert MoE",
    }.get(policy, policy)

    def encode(codec):
        writer = imageio_ffmpeg.write_frames(
            str(temp_path),
            (width, panel_height + banner_height),
            fps=fps,
            codec=codec,
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=1,
            ffmpeg_log_level="warning",
            output_params=["-preset", "fast", "-crf", "20"] if codec == "libx264" else ["-q:v", "5"],
        )
        writer.send(None)
        try:
            for row in rows:
                front = np.asarray(Image.open(ROOT / Path(row["front_path"])).convert("RGB").resize((640, panel_height)))
                overhead = np.asarray(Image.open(ROOT / Path(row["overhead_path"])).convert("RGB").resize((640, panel_height)))
                canvas = np.full((panel_height + banner_height, width, 3), 25, dtype=np.uint8)
                canvas[:banner_height] = (24, 32, 44)
                canvas[banner_height:, :640] = front
                canvas[banner_height:, 640:] = overhead
                bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
                cv2.putText(bgr, f"Isaac Sim | {policy_banner}", (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (242, 246, 250), 1, cv2.LINE_AA)
                text = f"{row['episode_id']}  raw={row['raw_prediction']}  action={row['action']}  conf={row['confidence']:.2f}  decision={row['latency_ms']:.2f} ms"
                cv2.putText(bgr, text, (16, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (120, 220, 180), 1, cv2.LINE_AA)
                cv2.putText(bgr, "Front camera RGB", (12, banner_height + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(bgr, "Overhead simulator camera", (652, banner_height + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                for _ in range(repeats):
                    writer.send(frame.tobytes())
        finally:
            writer.close()
        temp_path.replace(destination)

    codec = "libx264"
    try:
        encode(codec)
    except Exception as exc:
        print(f"VIDEO_H264_FAILED {exc}; retrying MPEG-4 Part 2", flush=True)
        temp_path.unlink(missing_ok=True)
        codec = "mpeg4"
        encode(codec)

    binary = imageio_ffmpeg.get_ffmpeg_exe()
    probe = subprocess.run([binary, "-hide_banner", "-i", str(destination), "-f", "null", "-"], capture_output=True, text=True)
    probe_text = probe.stderr + probe.stdout
    duration = (len(rows) * repeats) / fps
    codec_detected = "h264" if "Video: h264" in probe_text else ("mpeg4" if "Video: mpeg4" in probe_text else "unknown")
    if probe.returncode != 0 or destination.stat().st_size == 0:
        raise RuntimeError("Encoded MP4 did not decode successfully")
    info = {"path": destination.relative_to(ROOT).as_posix(), "codec": codec_detected, "frames": len(rows) * repeats, "decision_frames": len(rows), "fps": fps, "duration_seconds": duration, "size_bytes": destination.stat().st_size, "h264_validation": codec_detected == "h264"}
    (ROOT / "results" / "video_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("VIDEO", json.dumps(info, sort_keys=True), flush=True)


def main():
    ensure_dirs()
    latent_plot()
    confusion_plot()
    dimensionality_plot()
    latency_plot()
    nearest_neighbor_examples()
    closed_loop_failure_examples()
    compose_video()
    print("FIGURES_READY", flush=True)


if __name__ == "__main__":
    main()
