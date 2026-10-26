"""Evaluate held-out episodes for raw pixels and latent k-NN variants."""

from pathlib import Path
import csv
import json
import time

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["LEFT", "FORWARD", "RIGHT", "STOP"]


def raw_features(rows):
    values = []
    for row in rows:
        with Image.open(ROOT / Path(row["image_path"])) as image:
            values.append(np.asarray(image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32).reshape(-1) / 255.0)
    return np.stack(values)


def main():
    meta = pd.read_csv(ROOT / "features" / "metadata.csv").to_dict("records")
    z = np.load(ROOT / "features" / "all_embeddings.npy")
    test_ids = np.flatnonzero([row["split"] == "test" for row in meta])
    train_episode_ids = {row["episode_id"] for row in meta if row["split"] == "train"}
    test_episode_ids = {meta[i]["episode_id"] for i in test_ids}
    if train_episode_ids & test_episode_ids:
        raise RuntimeError("Episode leakage detected in evaluation")
    test_rows = [meta[i] for i in test_ids]
    y_true = np.array([row["label"] for row in test_rows])
    result_rows = []
    full_reports = {}
    prediction_rows = []

    methods = [
        ("Raw RGB + kNN", 12288, joblib.load(ROOT / "models" / "knn_raw.joblib"), raw_features(test_rows)),
        ("ResNet + kNN", 512, joblib.load(ROOT / "models" / "knn_512.joblib"), z[test_ids]),
        ("ResNet + PCA-16 + kNN", 16, joblib.load(ROOT / "models" / "knn_16.joblib"), joblib.load(ROOT / "models" / "pca_16.joblib").transform(z[test_ids])),
    ]
    for method, dim, clf, x_test in methods:
        t0 = time.perf_counter_ns()
        predicted = clf.predict(x_test)
        elapsed = (time.perf_counter_ns() - t0) / 1e6
        report = classification_report(y_true, predicted, labels=CLASSES, output_dict=True, zero_division=0)
        matrix = confusion_matrix(y_true, predicted, labels=CLASSES)
        full_reports[method] = {
            "dimension": dim,
            "accuracy": float(accuracy_score(y_true, predicted)),
            "macro_f1": float(f1_score(y_true, predicted, labels=CLASSES, average="macro", zero_division=0)),
            "per_class": {name: {"precision": report[name]["precision"], "recall": report[name]["recall"], "f1": report[name]["f1-score"], "support": report[name]["support"]} for name in CLASSES},
            "confusion_matrix": matrix.tolist(),
            "predict_batch_ms": elapsed,
            "test_samples": len(y_true),
        }
        result_rows.append({"method": method, "dimension": dim, "accuracy": full_reports[method]["accuracy"], "macro_f1": full_reports[method]["macro_f1"], "test_samples": len(y_true), "batch_prediction_ms": round(elapsed, 5)})
        probabilities = clf.predict_proba(x_test)
        for row, truth, pred, probs in zip(test_rows, y_true, predicted, probabilities):
            prediction_rows.append({"method": method, "image_path": row["image_path"], "episode_id": row["episode_id"], "true_label": truth, "predicted_label": pred, "confidence": float(np.max(probs))})

    train_policy_metrics = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
    summary = {
        "evaluation": "held-out episodes only",
        "class_order": CLASSES,
        "split": train_policy_metrics["split"],
        "label_distribution": train_policy_metrics["label_distribution"],
        "feature_shape": list(z.shape),
        "methods": full_reports,
        "pca_explained_variance": {str(dim): float(joblib.load(ROOT / "models" / f"pca_{dim}.joblib").explained_variance_ratio_.sum()) for dim in (2, 8, 16, 32, 64)},
    }
    with (ROOT / "results" / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=result_rows[0].keys())
        writer.writeheader()
        writer.writerows(result_rows)
    with (ROOT / "results" / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["method", "class", "precision", "recall", "f1", "support"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for method, metrics in full_reports.items():
            for label in CLASSES:
                writer.writerow({"method": method, "class": label, **metrics["per_class"][label]})
    (ROOT / "results" / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (ROOT / "results" / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=prediction_rows[0].keys())
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(f"OFFLINE_TEST episodes={sorted(test_episode_ids)} samples={len(y_true)}", flush=True)
    for row in result_rows:
        print(f"OFFLINE method={row['method']} dim={row['dimension']} accuracy={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
