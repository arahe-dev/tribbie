"""Train and evaluate the compact image-to-latent MoE policy."""

from pathlib import Path
import argparse
import csv
import json
import random
import time

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
from torch.nn import functional as F

from moe_latent import MoELatentPolicy, gate_balance_loss


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["LEFT", "FORWARD", "RIGHT", "STOP"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(CLASSES)}
FLIP_LABEL = torch.tensor([2, 1, 0, 3], dtype=torch.long)


def load_images(rows):
    images = []
    for row in rows:
        path = ROOT / Path(row["image_path"])
        with Image.open(path) as source:
            image = source.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
            images.append(np.asarray(image, dtype=np.uint8))
    if not images:
        return np.empty((0, 64, 64, 3), dtype=np.uint8)
    return np.stack(images)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def augment_batch(images, labels):
    """Mild color changes plus label-correct horizontal reflection."""
    count = images.shape[0]
    device = images.device
    flip_mask = torch.rand(count, device=device) < 0.5
    if flip_mask.any():
        images = images.clone()
        images[flip_mask] = images[flip_mask].flip(-1)
        labels = labels.clone()
        labels[flip_mask] = FLIP_LABEL.to(device)[labels[flip_mask]]

    brightness = torch.empty((count, 1, 1, 1), device=device).uniform_(0.78, 1.22)
    contrast = torch.empty((count, 1, 1, 1), device=device).uniform_(0.80, 1.20)
    saturation = torch.empty((count, 1, 1, 1), device=device).uniform_(0.82, 1.18)
    images = images * brightness
    means = images.mean(dim=(2, 3), keepdim=True)
    images = (images - means) * contrast + means
    gray = images.mean(dim=1, keepdim=True)
    images = (images - gray) * saturation + gray
    return images.clamp_(0.0, 1.0), labels


def score_model(model, images, labels, batch_size=64):
    model.eval()
    outputs = []
    gates = []
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            log_probs, weights, _ = model(images[start : start + batch_size], return_aux=True)
            outputs.append(log_probs.exp().cpu().numpy())
            gates.append(weights.cpu().numpy())
    probabilities = np.concatenate(outputs, axis=0)
    gate_weights = np.concatenate(gates, axis=0)
    predictions = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    truth = np.asarray(CLASSES)[labels.cpu().numpy()]
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, labels=CLASSES, average="macro", zero_division=0)),
        "predictions": predictions,
        "truth": truth,
        "probabilities": probabilities,
        "gate_weights": gate_weights,
    }


def train_one_seed(train_images, train_labels, val_images, val_labels, seed, args, device, class_weights):
    seed_everything(seed)
    model = MoELatentPolicy(
        latent_dim=args.latent_dim,
        num_experts=args.experts,
        dropout=args.dropout,
        spatial_pool=args.spatial_pool,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state = None
    best_key = None
    best_epoch = 0
    best_metrics = None
    stale_epochs = 0
    epoch_rows = []
    n = len(train_labels)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(n, device=device)
        loss_sum = 0.0
        seen = 0
        for start in range(0, n, args.batch_size):
            indices = order[start : start + args.batch_size]
            batch, targets = augment_batch(train_images[indices], train_labels[indices])
            log_probs, gates, _ = model(batch, return_aux=True)
            loss = F.nll_loss(log_probs, targets, weight=class_weights) + args.balance_weight * gate_balance_loss(gates)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            seen += len(indices)

        val = score_model(model, val_images, val_labels)
        val_loss = float(F.nll_loss(
            torch.from_numpy(np.log(np.maximum(val["probabilities"], 1e-8))).to(device), val_labels
        ).item())
        key = (val["macro_f1"], val["accuracy"], -val_loss)
        epoch_rows.append({
            "seed": seed,
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "validation_loss": val_loss,
            "validation_accuracy": val["accuracy"],
            "validation_macro_f1": val["macro_f1"],
            "validation_mean_gate_usage": val["gate_weights"].mean(axis=0).tolist(),
        })
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_metrics = {"accuracy": val["accuracy"], "macro_f1": val["macro_f1"], "loss": val_loss}
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch % 25 == 0 or epoch == 1:
            print(f"MOE seed={seed} epoch={epoch} val_f1={val['macro_f1']:.4f} val_acc={val['accuracy']:.4f}", flush=True)
        if stale_epochs >= args.patience:
            break

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "state_dict": best_state,
        "history": epoch_rows,
    }


def classification_metrics(truth, predictions, probabilities):
    report = classification_report(truth, predictions, labels=CLASSES, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, labels=CLASSES, average="macro", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in CLASSES
        },
        "confusion_matrix": confusion_matrix(truth, predictions, labels=CLASSES).tolist(),
        "mean_confidence": float(np.max(probabilities, axis=1).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="17,23,42,73,101")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--spatial-pool", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=0.0015)
    parser.add_argument("--weight-decay", type=float, default=0.003)
    parser.add_argument("--balance-weight", type=float, default=0.015)
    parser.add_argument("--class-weight-power", type=float, default=0.0,
                        help="Use (train_count / class_count)^power in NLL loss; 0 disables class weighting")
    parser.add_argument("--metadata-path", type=Path, default=ROOT / "features" / "metadata.csv")
    parser.add_argument("--model-name", default="moe_latent16")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one random seed is required")

    metadata_path = args.metadata_path if args.metadata_path.is_absolute() else ROOT / args.metadata_path
    metadata_path = metadata_path.resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")
    if not args.model_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("model-name may contain only letters, numbers, underscores, and hyphens")
    metadata = pd.read_csv(metadata_path).to_dict("records")
    split_ids = {name: [i for i, row in enumerate(metadata) if row["split"] == name] for name in ("train", "val", "test")}
    train_rows = [metadata[i] for i in split_ids["train"]]
    val_rows = [metadata[i] for i in split_ids["val"]]
    test_rows = [metadata[i] for i in split_ids["test"]]
    train_episodes = {row["episode_id"] for row in train_rows}
    val_episodes = {row["episode_id"] for row in val_rows}
    test_episodes = {row["episode_id"] for row in test_rows}
    if train_episodes & val_episodes or train_episodes & test_episodes or val_episodes & test_episodes:
        raise RuntimeError("Episode leakage between train, validation, and test splits")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"MOE device={device} train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}", flush=True)
    train_images_np = load_images(train_rows)
    val_images_np = load_images(val_rows)
    test_images_np = load_images(test_rows)
    def image_tensor(array):
        return torch.from_numpy(array).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).contiguous() / 255.0
    train_images = image_tensor(train_images_np)
    val_images = image_tensor(val_images_np)
    test_images = image_tensor(test_images_np)
    train_labels = torch.tensor([LABEL_TO_ID[row["label"]] for row in train_rows], dtype=torch.long, device=device)
    val_labels = torch.tensor([LABEL_TO_ID[row["label"]] for row in val_rows], dtype=torch.long, device=device)
    test_labels = torch.tensor([LABEL_TO_ID[row["label"]] for row in test_rows], dtype=torch.long, device=device)

    class_counts = torch.bincount(train_labels, minlength=len(CLASSES)).to(torch.float32)
    if torch.any(class_counts == 0):
        missing = [CLASSES[index] for index, count in enumerate(class_counts.tolist()) if count == 0]
        raise RuntimeError(f"Training split has no examples for classes: {missing}")
    class_weights = (len(train_labels) / (len(CLASSES) * class_counts)).pow(args.class_weight_power)
    class_weights = class_weights / class_weights.mean()
    print("MOE train_class_counts=" + json.dumps({name: int(class_counts[index]) for index, name in enumerate(CLASSES)}), flush=True)
    print("MOE class_weights=" + json.dumps({name: float(class_weights[index]) for index, name in enumerate(CLASSES)}), flush=True)

    starts_at = time.perf_counter()
    seed_results = [
        train_one_seed(train_images, train_labels, val_images, val_labels, seed, args, device, class_weights)
        for seed in seeds
    ]
    seed_results.sort(
        key=lambda result: (
            result["best_validation"]["macro_f1"],
            result["best_validation"]["accuracy"],
            -result["best_validation"]["loss"],
        ),
        reverse=True,
    )
    selected = seed_results[0]
    model = MoELatentPolicy(args.latent_dim, args.experts, dropout=args.dropout, spatial_pool=args.spatial_pool).to(device)
    model.load_state_dict(selected["state_dict"])
    test_result = score_model(model, test_images, test_labels)
    test_metrics = classification_metrics(test_result["truth"], test_result["predictions"], test_result["probabilities"])
    gate_mean = test_result["gate_weights"].mean(axis=0)
    gate_entropy = -np.sum(test_result["gate_weights"] * np.log(np.maximum(test_result["gate_weights"], 1e-8)), axis=1)

    # Model-only latency on preloaded 64x64 frames. It excludes capture, resizing, and IPC.
    model.eval()
    sample = test_images[:1]
    for _ in range(50):
        with torch.inference_mode():
            model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings = []
    for _ in range(500):
        started = time.perf_counter_ns()
        with torch.inference_mode():
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter_ns() - started) / 1e6)
    latency = {
        "scope": "MoE model forward on preloaded 64x64 RGB tensor; excludes capture, image resize, and IPC",
        "iterations": len(timings),
        "median_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "device": str(device),
    }

    # Compare against the existing PCA-16+kNN reference on the same untouched test rows.
    pca_metrics = None
    if metadata_path == (ROOT / "features" / "metadata.csv").resolve():
        embeddings = np.load(ROOT / "features" / "all_embeddings.npy")
        test_indices = np.asarray(split_ids["test"], dtype=int)
        pca = joblib.load(ROOT / "models" / "pca_16.joblib")
        knn = joblib.load(ROOT / "models" / "knn_16.joblib")
        pca_test = pca.transform(embeddings[test_indices])
        pca_pred = knn.predict(pca_test)
        pca_metrics = classification_metrics(np.asarray([row["label"] for row in test_rows]), pca_pred, knn.predict_proba(pca_test))

    model_dir = ROOT / "models"
    result_dir = ROOT / "results"
    model_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    variant = args.model_name if args.experts == 4 else f"{args.model_name}_e{args.experts}"
    checkpoint_path = model_dir / f"{variant}.pt"
    torch.save({
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "config": {
            "input": "RGB 64x64 scaled to [0,1]",
            "latent_dim": args.latent_dim,
            "num_experts": args.experts,
            "num_classes": len(CLASSES),
            "dropout": args.dropout,
            "spatial_pool": args.spatial_pool,
            "class_order": CLASSES,
            "selected_seed": selected["seed"],
            "selected_epoch": selected["best_epoch"],
        },
    }, checkpoint_path)

    prediction_path = result_dir / f"{variant}_test_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_path", "episode_id", "true_label", "predicted_label", "confidence", "expert_0", "expert_1", "expert_2", "expert_3"])
        writer.writeheader()
        for row, truth, pred, probs, weights in zip(
            test_rows, test_result["truth"], test_result["predictions"], test_result["probabilities"], test_result["gate_weights"]
        ):
            writer.writerow({
                "image_path": row["image_path"],
                "episode_id": row["episode_id"],
                "true_label": truth,
                "predicted_label": pred,
                "confidence": float(np.max(probs)),
                **{f"expert_{i}": float(value) for i, value in enumerate(weights)},
            })

    history_path = result_dir / f"{variant}_validation_history.csv"
    history_rows = selected["history"]
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", "epoch", "train_loss", "validation_loss", "validation_accuracy", "validation_macro_f1", "validation_mean_gate_usage"])
        writer.writeheader()
        writer.writerows(history_rows)

    metrics = {
        "model": "Compact image encoder -> learned 16-D latent -> 4-expert gated action mixture",
        "task": "supervised four-action prediction; geometric simulator labels; pixels-only policy input",
        "architecture": {
            "input_shape": [3, 64, 64],
            "latent_dim": args.latent_dim,
            "num_experts": args.experts,
            "num_classes": len(CLASSES),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "spatial_pool": args.spatial_pool,
            "checkpoint": str(checkpoint_path.relative_to(ROOT).as_posix()),
        },
        "selection": {
            "criterion": "highest validation macro F1; then accuracy; then lower validation loss",
            "selected_seed": selected["seed"],
            "selected_epoch": selected["best_epoch"],
            "selected_validation": selected["best_validation"],
            "seeds": [{"seed": r["seed"], "best_epoch": r["best_epoch"], **r["best_validation"]} for r in seed_results],
            "hyperparameters": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "train_class_counts": {name: int(class_counts[index]) for index, name in enumerate(CLASSES)},
            "train_class_weights": {name: float(class_weights[index]) for index, name in enumerate(CLASSES)},
            "metadata_path": metadata_path.relative_to(ROOT).as_posix() if metadata_path.is_relative_to(ROOT) else str(metadata_path),
        },
        "split": {
            "train_episodes": sorted(train_episodes),
            "validation_episodes": sorted(val_episodes),
            "test_episodes": sorted(test_episodes),
            "samples": {"train": len(train_rows), "validation": len(val_rows), "test": len(test_rows)},
        },
        "test": test_metrics,
        "test_expert_usage_mean": [float(value) for value in gate_mean],
        "test_gate_entropy_mean": float(gate_entropy.mean()),
        "test_mean_gate_usage_by_true_class": {
            label: test_result["gate_weights"][test_result["truth"] == label].mean(axis=0).tolist()
            for label in CLASSES
        },
        "pca16_knn_reference_same_test": pca_metrics,
        "latency": latency,
        "training_seconds": round(time.perf_counter() - starts_at, 3),
        "limitations": [
            "small single-simulator dataset and only two held-out collection episodes",
            "test metrics are preliminary and do not establish physical-robot or scene generalization",
            "timing is model-forward only, not camera-to-actuator latency",
        ],
    }
    (result_dir / f"{variant}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("MOE_TEST", json.dumps({
        "selected_seed": selected["seed"],
        "selected_epoch": selected["best_epoch"],
        "validation": selected["best_validation"],
        "test": test_metrics,
        "pca16_knn_reference": pca_metrics,
        "latency": latency,
        "expert_usage": gate_mean.tolist(),
        "training_seconds": metrics["training_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
