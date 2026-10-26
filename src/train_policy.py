"""Train grouped-split k-NN policies and PCA compressions."""

from pathlib import Path
import csv
import json
import time

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["LEFT", "FORWARD", "RIGHT", "STOP"]
DIMS = [2, 8, 16, 32, 64, 512]
K_VALUES = [1, 3, 5, 7, 11]


def raw_pixel_features(rows):
    values = []
    for row in rows:
        path = ROOT / Path(row["image_path"])
        with Image.open(path) as image:
            image = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
            values.append(np.asarray(image, dtype=np.float32).reshape(-1) / 255.0)
    return np.stack(values, axis=0)


def fit_best_knn(x_train, y_train, x_val, y_val):
    best = None
    for k in K_VALUES:
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance", algorithm="brute", n_jobs=1)
        clf.fit(x_train, y_train)
        predictions = clf.predict(x_val)
        macro_f1 = f1_score(y_val, predictions, labels=CLASSES, average="macro", zero_division=0)
        accuracy = accuracy_score(y_val, predictions)
        score = (macro_f1, accuracy, -k)
        if best is None or score > best[0]:
            best = (score, clf, k, macro_f1, accuracy)
    return best[1:]


def evaluate_classifier(method, dim, classifier, x_test, y_test, test_rows):
    predictions = classifier.predict(x_test)
    report = classification_report(y_test, predictions, labels=CLASSES, output_dict=True, zero_division=0)
    metrics = {
        "method": method,
        "dimension": int(dim),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, labels=CLASSES, average="macro", zero_division=0)),
        "test_samples": int(len(y_test)),
        "per_class": {
            name: {"precision": report[name]["precision"], "recall": report[name]["recall"], "f1": report[name]["f1-score"], "support": report[name]["support"]}
            for name in CLASSES
        },
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=CLASSES).tolist(),
    }
    pred_rows = []
    for row, truth, pred, probs in zip(test_rows, y_test, predictions, classifier.predict_proba(x_test)):
        pred_rows.append(
            {
                "method": method,
                "image_path": row["image_path"],
                "episode_id": row["episode_id"],
                "true_label": truth,
                "predicted_label": pred,
                "confidence": float(np.max(probs)),
            }
        )
    return metrics, pred_rows


def main():
    meta = pd.read_csv(ROOT / "features" / "metadata.csv").to_dict("records")
    features = np.load(ROOT / "features" / "all_embeddings.npy")
    if len(meta) != len(features):
        raise RuntimeError("metadata/features row count mismatch")
    labels = np.array([row["label"] for row in meta], dtype=str)
    splits = np.array([row["split"] for row in meta], dtype=str)
    train_idx = np.flatnonzero(splits == "train")
    val_idx = np.flatnonzero(splits == "val")
    test_idx = np.flatnonzero(splits == "test")
    train_rows = [meta[i] for i in train_idx]
    val_rows = [meta[i] for i in val_idx]
    test_rows = [meta[i] for i in test_idx]
    x_train, y_train = features[train_idx], labels[train_idx]
    x_val, y_val = features[val_idx], labels[val_idx]
    x_test, y_test = features[test_idx], labels[test_idx]
    if len(set(row["episode_id"] for row in train_rows) & set(row["episode_id"] for row in test_rows)):
        raise RuntimeError("Episode leakage between train and test")
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)

    # Raw 64x64 RGB baseline.
    raw_all = raw_pixel_features(meta)
    np.save(ROOT / "features" / "raw_pixels_64.npy", raw_all)
    raw_train, raw_val, raw_test = raw_all[train_idx], raw_all[val_idx], raw_all[test_idx]
    raw_clf, raw_k, raw_val_f1, raw_val_acc = fit_best_knn(raw_train, y_train, raw_val, y_val)
    joblib.dump(raw_clf, ROOT / "models" / "knn_raw.joblib")

    metric_rows = []
    reports = {"split": {"train_episodes": sorted(set(row["episode_id"] for row in train_rows)), "val_episodes": sorted(set(row["episode_id"] for row in val_rows)), "test_episodes": sorted(set(row["episode_id"] for row in test_rows))}, "class_order": CLASSES, "methods": {}}
    prediction_rows = []
    raw_metric, raw_predictions = evaluate_classifier("Raw RGB + kNN", 12288, raw_clf, raw_test, y_test, test_rows)
    raw_metric.update({"validation_k": raw_k, "validation_macro_f1": raw_val_f1, "validation_accuracy": raw_val_acc})
    metric_rows.append(raw_metric)
    reports["methods"]["Raw RGB + kNN"] = raw_metric
    prediction_rows.extend(raw_predictions)

    sweep_started = time.perf_counter()
    for dim in DIMS:
        if dim == 512:
            z_train, z_val, z_test = x_train, x_val, x_test
            pca = None
            method = "ResNet + kNN"
        else:
            pca = PCA(n_components=dim, svd_solver="full", random_state=0)
            z_train = pca.fit_transform(x_train)
            z_val = pca.transform(x_val)
            z_test = pca.transform(x_test)
            method = "ResNet + PCA + kNN" if dim == 16 else f"ResNet + PCA-{dim} + kNN"
            joblib.dump(pca, ROOT / "models" / f"pca_{dim}.joblib")
        clf, k, val_f1, val_acc = fit_best_knn(z_train, y_train, z_val, y_val)
        joblib.dump(clf, ROOT / "models" / f"knn_{dim}.joblib")
        metric, preds = evaluate_classifier(method, dim, clf, z_test, y_test, test_rows)
        metric.update({"validation_k": k, "validation_macro_f1": val_f1, "validation_accuracy": val_acc})
        if pca is not None:
            metric["explained_variance_ratio_sum"] = float(np.sum(pca.explained_variance_ratio_))
        metric_rows.append(metric)
        reports["methods"][method] = metric
        prediction_rows.extend(preds)
        print(f"TRAINED dim={dim} k={k} val_f1={val_f1:.3f} test_f1={metric['macro_f1']:.3f}", flush=True)

    reports["dimensionality_sweep_seconds"] = round(time.perf_counter() - sweep_started, 4)
    reports["sample_counts"] = {"train": int(len(train_idx)), "validation": int(len(val_idx)), "test": int(len(test_idx))}
    reports["label_distribution"] = {split: {label: int(sum(meta[i]["label"] == label for i in idx)) for label in CLASSES} for split, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx))}

    fieldnames = list(metric_rows[0].keys())
    with (ROOT / "results" / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metric_rows)
    sweep_rows = [row for row in metric_rows if row["dimension"] in DIMS]
    sweep_fields = ["method", "dimension", "accuracy", "macro_f1", "test_samples", "validation_k", "validation_macro_f1", "validation_accuracy", "explained_variance_ratio_sum"]
    with (ROOT / "results" / "dimensionality_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sweep_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sweep_rows)
    (ROOT / "results" / "metrics.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    with (ROOT / "results" / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=prediction_rows[0].keys())
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(f"SPLIT train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} episodes={reports['split']}", flush=True)
    print(f"RAW_BEST_K={raw_k} VALIDATION_MACRO_F1={raw_val_f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
