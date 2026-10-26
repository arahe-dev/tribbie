"""Fit a frozen ResNet-18 + PCA-16 + k-NN comparator on the MoE rollout split."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from encoder import build_encoder
from weighted_knn import ClassWeightedKNeighborsClassifier


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["LEFT", "FORWARD", "RIGHT", "STOP"]
K_VALUES = (1, 3, 5, 7, 11)


def load_features(rows, model, transform, device, batch_size=32):
    chunks = []
    for start in range(0, len(rows), batch_size):
        images = []
        for row in rows[start : start + batch_size]:
            with Image.open(ROOT / Path(row["image_path"])) as source:
                images.append(transform(source.convert("RGB")))
        batch = torch.stack(images).to(device)
        with torch.inference_mode():
            features = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        chunks.append(features.detach().cpu().numpy().astype(np.float32, copy=False))
        print(f"EMBEDDED {min(start + len(images), len(rows))}/{len(rows)}", flush=True)
    return np.concatenate(chunks, axis=0)


def summarize(y_true, y_pred, y_prob):
    report = classification_report(
        y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in CLASSES
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
        "mean_confidence": float(np.max(y_prob, axis=1).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=ROOT / "data" / "reactive_stop_v2" / "metadata.csv")
    parser.add_argument("--moe-metrics", type=Path, default=ROOT / "results" / "random_stop_moe_v2_metrics.json")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models" / "random_stop_pca16_v2")
    parser.add_argument("--report", type=Path, default=ROOT / "results" / "random_stop_baseline_v2.json")
    args = parser.parse_args()

    metadata_path = args.metadata if args.metadata.is_absolute() else ROOT / args.metadata
    moe_metrics_path = args.moe_metrics if args.moe_metrics.is_absolute() else ROOT / args.moe_metrics
    models_dir = args.models_dir if args.models_dir.is_absolute() else ROOT / args.models_dir
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    rows = list(csv.DictReader(metadata_path.open(newline="", encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"No training images in {metadata_path}")

    splits = {name: [row for row in rows if row["split"] == name] for name in ("train", "val", "test")}
    if any(not split_rows for split_rows in splits.values()):
        raise RuntimeError(f"Every split must be non-empty: { {k: len(v) for k, v in splits.items()} }")
    episode_sets = {name: {row["episode_id"] for row in split_rows} for name, split_rows in splits.items()}
    if (episode_sets["train"] & episode_sets["val"] or episode_sets["train"] & episode_sets["test"]
            or episode_sets["val"] & episode_sets["test"]):
        raise RuntimeError("Episode leakage across the shared MoE train/validation/test split")

    moe_metrics = json.loads(moe_metrics_path.read_text(encoding="utf-8"))
    moe_split = moe_metrics["split"]
    expected = {
        "train": set(moe_split["train_episodes"]),
        "val": set(moe_split["validation_episodes"]),
        "test": set(moe_split["test_episodes"]),
    }
    if episode_sets != expected:
        raise RuntimeError(f"Baseline split does not match the MoE split: baseline={episode_sets}, MoE={expected}")

    torch.set_num_threads(1)
    model, transform, device, weights = build_encoder()
    started = time.perf_counter()
    features = {name: load_features(split_rows, model, transform, device) for name, split_rows in splits.items()}
    labels = {name: np.asarray([row["label"] for row in split_rows], dtype=str) for name, split_rows in splits.items()}

    pca = PCA(n_components=16, svd_solver="full", random_state=0)
    z_train = pca.fit_transform(features["train"])
    z_val = pca.transform(features["val"])
    z_test = pca.transform(features["test"])

    best = None
    for k in K_VALUES:
        classifier = ClassWeightedKNeighborsClassifier(
            n_neighbors=k, weights="distance", algorithm="brute", n_jobs=1, class_weight_power=0.5
        )
        classifier.fit(z_train, labels["train"])
        val_prob = classifier.predict_proba(z_val)
        val_pred = classifier.classes_[val_prob.argmax(axis=1)]
        score = (
            f1_score(labels["val"], val_pred, labels=CLASSES, average="macro", zero_division=0),
            accuracy_score(labels["val"], val_pred),
            -k,
        )
        if best is None or score > best[0]:
            best = (score, classifier, k, val_prob, val_pred)

    _, classifier, selected_k, val_prob, val_pred = best
    test_prob = classifier.predict_proba(z_test)
    test_pred = classifier.classes_[test_prob.argmax(axis=1)]
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, models_dir / "pca_16.joblib")
    joblib.dump(classifier, models_dir / "knn_16.joblib")

    report = {
        "method": "Frozen ImageNet ResNet-18 + training-fit PCA-16 + class-reweighted distance-weighted k-NN",
        "purpose": "slower fixed-representation comparator for the randomized stop benchmark",
        "input_source": str(metadata_path.relative_to(ROOT).as_posix()),
        "shared_split_with": "models/random_stop_moe_v2.pt",
        "encoder": "torchvision ResNet-18 / ImageNet-1K V1, frozen fc=Identity",
        "encoder_weights": weights.url,
        "device": str(device),
        "feature_shape": [512],
        "latent_dimension": 16,
        "pca_fit_split": "train only",
        "knn_selection": {
            "candidates": list(K_VALUES),
            "criterion": "validation macro F1, then validation accuracy, then smaller k; inverse-square-root class factors match the MoE training loss",
            "class_weight_power": 0.5,
            "class_factors": classifier.class_factors_,
            "selected_k": int(selected_k),
            "validation": summarize(labels["val"], val_pred, val_prob),
        },
        "test": summarize(labels["test"], test_pred, test_prob),
        "split_episodes": {name: sorted(values) for name, values in episode_sets.items()},
        "split_samples": {name: len(values) for name, values in splits.items()},
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "feature_extraction_and_fit_seconds": round(time.perf_counter() - started, 3),
        "artifacts": {
            "pca": str((models_dir / "pca_16.joblib").relative_to(ROOT).as_posix()),
            "knn": str((models_dir / "knn_16.joblib").relative_to(ROOT).as_posix()),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("BASELINE", json.dumps({k: report[k] for k in ("device", "split_samples", "knn_selection", "test", "feature_extraction_and_fit_seconds")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
