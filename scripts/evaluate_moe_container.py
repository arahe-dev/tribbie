"""Container-side transfer and latency check for the trained MoE policy.

This is deliberately a small local robot-task check, not the HM3D-OVON
navigation benchmark. It evaluates the saved policy on the pre-existing,
episode-held-out image split and times the production resize/tensor/model path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from moe_latent import MoELatentPolicy  # noqa: E402


CLASS_ORDER = ["LEFT", "FORWARD", "RIGHT", "STOP"]


def percentile_stats(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(np.mean(values)),
    }


def tensor_from_rgb(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    resized = Image.fromarray(rgb, mode="RGB").resize((64, 64), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.to(device)


def load_test_frames(csv_path: Path) -> list[tuple[np.ndarray, str, str]]:
    frames = []
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["split"] != "test":
                continue
            image_path = ROOT / row["image_path"]
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            frames.append((rgb, row["label"], row["episode_id"]))
    if not frames:
        raise RuntimeError(f"No test frames found in {csv_path}")
    return frames


def macro_f1(confusion: np.ndarray) -> float:
    scores = []
    for i in range(len(CLASS_ORDER)):
        tp = int(confusion[i, i])
        fp = int(confusion[:, i].sum() - tp)
        fn = int(confusion[i, :].sum() - tp)
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/autoresearch_spatial_moe_v3.pt")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/labels.csv")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--output", type=Path, default=ROOT / "results/moe_v3_container_crossdataset.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This run is intended to verify the CUDA container path; CUDA is unavailable")
    torch.set_num_threads(1)
    device = torch.device("cuda")
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    if config.get("class_order") != CLASS_ORDER:
        raise RuntimeError(f"Unexpected model action order: {config.get('class_order')}")
    model = MoELatentPolicy(
        latent_dim=int(config["latent_dim"]),
        num_experts=int(config["num_experts"]),
        num_classes=int(config["num_classes"]),
        dropout=float(config["dropout"]),
        spatial_pool=int(config.get("spatial_pool", 1)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    frames = load_test_frames(args.labels.resolve())
    confusion = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=np.int64)
    with torch.inference_mode():
        for rgb, label, _episode in frames:
            scores = model(tensor_from_rgb(rgb, device)).exp()[0].cpu().numpy()
            confusion[CLASS_ORDER.index(label), int(np.argmax(scores))] += 1

    # Model-only timing on a GPU-resident tensor.
    timing_input = tensor_from_rgb(frames[0][0], device)
    for _ in range(50):
        with torch.inference_mode():
            model(timing_input)
    torch.cuda.synchronize(device)
    model_ms: list[float] = []
    for _ in range(args.iterations):
        torch.cuda.synchronize(device)
        started = time.perf_counter_ns()
        with torch.inference_mode():
            model(timing_input)
        torch.cuda.synchronize(device)
        model_ms.append((time.perf_counter_ns() - started) / 1e6)

    # Full in-process request path: 320x240 RGB -> bilinear 64x64 -> H2D ->
    # MoE -> D2H/argmax. File I/O, localhost IPC, rendering, and actuation are
    # intentionally excluded, matching the server's per-request computation.
    request_ms: list[float] = []
    with torch.inference_mode():
        for i in range(args.iterations + 50):
            rgb = frames[i % len(frames)][0]
            torch.cuda.synchronize(device)
            started = time.perf_counter_ns()
            tensor = tensor_from_rgb(rgb, device)
            scores = model(tensor).exp()[0].cpu().numpy()
            int(np.argmax(scores))
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if i >= 50:
                request_ms.append(elapsed)

    correct = int(confusion.trace())
    result = {
        "scope": "local Jetbot red-pole task only; not HM3D-OVON",
        "evaluation": {
            "dataset_csv": str(args.labels.resolve()),
            "split": "episode-held-out test split (ep09, ep10)",
            "samples": len(frames),
            "episodes": sorted({episode for _, _, episode in frames}),
            "class_order": CLASS_ORDER,
            "class_support": {
                name: int(confusion[i, :].sum()) for i, name in enumerate(CLASS_ORDER)
            },
            "accuracy": correct / len(frames),
            "macro_f1": macro_f1(confusion),
            "confusion_matrix_rows_true_cols_pred": confusion.tolist(),
        },
        "model": {
            "checkpoint": str(checkpoint_path),
            "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "latent_dim": int(config["latent_dim"]),
            "experts": int(config["num_experts"]),
            "spatial_pool": int(config.get("spatial_pool", 1)),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "latency_iterations": args.iterations,
            "model_forward_gpu_resident": percentile_stats(model_ms),
            "resize_transfer_model_output_in_process": percentile_stats(request_ms),
            "excluded_from_request_timing": ["disk I/O", "socket IPC", "Isaac rendering/physics", "actuation"],
        },
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"WROTE {output_path}")


if __name__ == "__main__":
    main()
