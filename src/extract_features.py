"""Extract cached 512-D ResNet-18 embeddings from labels.csv."""

from pathlib import Path
import csv
import json
import time

import numpy as np
import torch
from PIL import Image

from encoder import build_encoder


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "data" / "labels.csv"
    out = ROOT / "features"
    out.mkdir(parents=True, exist_ok=True)
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"No samples found in {source}")

    model, transform, device, weights = build_encoder()
    batch_size = 32
    embeddings = []
    t0 = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        tensors = []
        for row in batch_rows:
            image_path = ROOT / Path(row["image_path"])
            with Image.open(image_path) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(tensors, dim=0).to(device)
        with torch.inference_mode():
            batch_features = model(batch)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        embeddings.append(batch_features.detach().cpu().numpy().astype(np.float32))
        print(f"EXTRACTED {min(start + len(batch_rows), len(rows))}/{len(rows)}", flush=True)

    x = np.concatenate(embeddings, axis=0)
    if x.shape != (len(rows), 512):
        raise RuntimeError(f"Expected N x 512 features, got {x.shape}")
    np.save(out / "all_embeddings.npy", x)

    metadata_fields = list(rows[0].keys())
    with (out / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metadata_fields)
        writer.writeheader()
        writer.writerows(rows)
    for split in ("train", "val", "test"):
        indices = np.array([i for i, row in enumerate(rows) if row["split"] == split], dtype=int)
        np.save(out / f"{split}_embeddings.npy", x[indices])
        np.save(out / f"{split}_labels.npy", np.array([rows[i]["label"] for i in indices], dtype=str))
        np.save(out / f"{split}_indices.npy", indices)

    elapsed = time.perf_counter() - t0
    info = {
        "encoder": "torchvision ResNet-18 / ImageNet-1K V1, frozen fc=Identity",
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "device": str(device),
        "weights": weights.url,
        "input_transform": str(transform),
        "feature_shape": list(x.shape),
        "samples": len(rows),
        "batch_size": batch_size,
        "elapsed_seconds": round(elapsed, 4),
        "throughput_frames_per_second": round(len(rows) / max(elapsed, 1e-9), 3),
        "cuda_available": torch.cuda.is_available(),
    }
    (out / "extraction_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("FEATURES", json.dumps(info, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
